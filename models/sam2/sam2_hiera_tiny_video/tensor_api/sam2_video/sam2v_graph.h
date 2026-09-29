// SAM2.1 hiera-tiny VIDEO tracking path authored on the C++ Tensor API.
//
// Adds the three per-frame memory graphs of the SAM2 tracking loop to the
// image-path encoder (reused from ../sam2_image at image_size=1024 with
// fold_no_mem_embed cleared, so its output IS the raw top-level feature map):
//
//   memcond{N}: memory attention over a FIXED bank of N spatial memory
//     slots (4096 tokens x 64ch each) + 64 object-pointer tokens. Unused
//     slots/pointers are masked with an additive key mask, which matches
//     the reference's variable-length bank exactly. Single-head 256-dim
//     rank-4 attention; rotate-half RoPE with the head-dim permutation
//     baked into the checkpoint's q/k projections at export time.
//   decode: the video mask decoder — sparse prompt as an INPUT (2x256:
//     click row + pad, or the baked tracking row), a nomem scalar input
//     that adds the no-memory embedding on the conditioning frame, ALL
//     four mask tokens emitted plus iou (4, sigmoid), object pointers
//     (4x256) and the object score. The host picks argmax iou over 1..3.
//   memorize: the memory encoder — mask downsampler (4x stride-2 convs,
//     torch padding = explicit pad + VALID), feature projection, 2
//     ConvNeXt fuser blocks (depthwise 7x7), 1x1 projection to 64ch,
//     plus occ * the occlusion embedding. Output token-major [1,4096,64].
//
// Layouts are NHWC / token-major end to end. The memory bank, temporal
// position rows, pointer tokens/positions and key mask are per-frame host
// inputs (the reference pipeline's contract); the chained-state variants
// (in-graph DynamicUpdateSlice / odml.cache_update at a runtime slot) are
// built on top of the same graphs.

#ifndef MODELS_SAM2_SAM2_HIERA_TINY_VIDEO_TENSOR_API_SAM2_VIDEO_SAM2V_GRAPH_H_
#define MODELS_SAM2_SAM2_HIERA_TINY_VIDEO_TENSOR_API_SAM2_VIDEO_SAM2V_GRAPH_H_

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image/sam2_config.h"
#include "models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image/sam2_graph.h"

namespace litert::tensor::examples::sam2_video {

using ::litert::tensor::examples::sam2::Sam2Config;
using ::litert::tensor::examples::sam2::TfTensor;
using ::litert::tensor::examples::sam2::WeightMap;

// Constants baked from the weights at graph-build time (pre-scaled
// projections, sign-baked RoPE tables, position grids, ...). Pass ONE cache
// to every Build* call that goes into the same ModelFactory so the signatures
// share a single copy of each constant in the flatbuffer. Use a fresh cache
// per WeightMap: entries are keyed by name, not by weight contents. Build*
// calls without a cache (nullptr) bake their own copies, which is correct but
// duplicates the constants across signatures.
class ConstCache {
 public:
  const TfTensor& GetOrCreate(const std::string& key,
                              const std::function<TfTensor()>& make);

 private:
  std::map<std::string, TfTensor> entries_;
};

struct Sam2VideoConfig {
  Sam2Config image;  // image_size = 1024 for the video pipeline
  int mem_ch = 64;   // memory channel dim
  int hidden = 256;  // memory-attention hidden (single head)
  int ma_layers = 4;
  int ff_hidden = 2048;
  int num_ptr_frames = 16;  // object-pointer frames
  int ptr_split = 4;        // 256-dim pointer -> 4 tokens of 64
  float ln_eps = 1e-5f;     // torch nn.LayerNorm default (memory attention)
  float ln_eps_2d = 1e-6f;  // Sam2VideoLayerNorm (memory encoder)
  float mask_neg = -1e9f;   // additive key-mask fill for unused slots

  Sam2VideoConfig() { image.image_size = 1024; }

  int hw() const {  // top-level token count (64x64 at 1024)
    int g = image.embed_grid();
    return g * g;
  }
  int n_ptr() const { return num_ptr_frames * ptr_split; }  // 64
  int mem_len(int nmm) const { return nmm * hw() + n_ptr(); }
};

struct MemCondInputs {
  TfTensor pix_raw;   // [1, G, G, 256] raw top-level features (NHWC)
  TfTensor mem_bank;  // [1, N, HW, 64] spatial memories, token-major
  TfTensor slot_tpe;  // [1, N, 1, 64] temporal position row per slot
  TfTensor ptr_tok;   // [1, 1, 64, 64] object-pointer tokens
  TfTensor ptr_pos;   // [1, 1, 64, 64] pointer temporal positions
  TfTensor key_mask;  // [1, 1, 1, N*HW+64] additive (0 / mask_neg)
  std::vector<TfTensor> AsList() const {
    return {pix_raw, mem_bank, slot_tpe, ptr_tok, ptr_pos, key_mask};
  }
};

struct VideoDecoderInputs {
  TfTensor pix_feat;  // [1, G, G, 256] (memcond output, or pix_raw + nomem)
  TfTensor feat_s1;   // [1, 4G, 4G ... see image spec] high-res skip 1
  TfTensor feat_s0;   // high-res skip 0
  TfTensor sparse;    // [1, 2, 256] sparse prompt rows
  TfTensor nomem;     // [1, 1, 1, 1] 1.0 on the conditioning frame else 0.0
  std::vector<TfTensor> AsList() const {
    return {pix_feat, feat_s1, feat_s0, sparse, nomem};
  }
};

struct VideoDecoderOutputs {
  TfTensor masks;         // [1, 4, S/4, S/4] all mask tokens' logits
  TfTensor iou_scores;    // [1, 4] (sigmoid)
  TfTensor obj_ptr;       // [1, 4, 256] object pointer per mask token
  TfTensor object_score;  // [1, 1]
  std::vector<TfTensor> AsList() const {
    return {masks, iou_scores, obj_ptr, object_score};
  }
};

struct MemorizeInputs {
  TfTensor pix_raw;       // [1, G, G, 256]
  TfTensor mask_for_mem;  // [1, S, S, 1] sigmoid(hi-res)*20-10 (host-built)
  TfTensor occ;           // [1, 1, 1] 1 - is_obj_appearing
  std::vector<TfTensor> AsList() const {
    return {pix_raw, mask_for_mem, occ};
  }
};

// Everything the host needs after one object step, produced in-graph (the
// fused step_* signatures): the host loop of sam2v_main.cc between decode and
// memorize — best of mask tokens 1..3 by iou (first max wins), no-object
// handling (mask -> -1024, pointer -> no_obj_ptr, occ = 1), the 4x bilinear
// upsample (align_corners=false) and mask_for_mem (binarized when prompted,
// sigmoid otherwise; *20 - 10) — then the memory encoder.
struct StepOutputs {
  TfTensor mem;           // [1, HW, 64] memory for the bank
  TfTensor ptr;           // [1, 256] object pointer for the bank
  TfTensor low_mask;      // [1, S/4, S/4] chosen logits (display)
  TfTensor object_score;  // [1, 1]
  TfTensor iou;           // [1, 1] predicted iou of the chosen mask
  std::vector<TfTensor> AsList() const {
    return {mem, ptr, low_mask, object_score, iou};
  }
};

// decode -> in-graph post -> memorize. The decoder's nomem input doubles as
// the prompted flag (1.0 on the conditioning frame, 0.0 on tracked frames);
// keep it a graph INPUT — baked as a constant it creates constant-constant
// MUL/SUB nodes the GPU delegate rejects.
//
// `multimask` follows SAM 2's rule (multimask_max_pt_num = 1): true for 0-1
// prompt points (tracked frames, single click) -> best of tokens 1..3 by iou;
// false for 2+ points -> token 0 unless its stability score (area of
// logits > 0.05 / area > -0.05) is below 0.98, then the best of 1..3; the
// object pointer is always token 0's (HF Sam2VideoModel semantics).
StepOutputs BuildStep(const Sam2VideoConfig& config,
                      const VideoDecoderInputs& dec_in,
                      const TfTensor& pix_raw, const WeightMap& weights,
                      bool multimask = true, int binarize_mode = -1,
                      ConstCache* cache = nullptr);

MemCondInputs MakeMemCondInputs(const Sam2VideoConfig& config, int nmm);
VideoDecoderInputs MakeVideoDecoderInputs(const Sam2VideoConfig& config);
MemorizeInputs MakeMemorizeInputs(const Sam2VideoConfig& config);

// pix_feat [1, G, G, 256] token-major output, named "pix_feat".
TfTensor BuildMemCond(const Sam2VideoConfig& config, int nmm,
                      const MemCondInputs& inputs, const WeightMap& weights,
                      ConstCache* cache = nullptr);
// mask_mode: 0 = all 4 masks and object pointers (the classic `decode`
// signature); 1 = masks/pointers 1..3 only (BuildStep, multimask); 2 = all 4
// masks, pointer 0 only (BuildStep, single mask).
VideoDecoderOutputs BuildVideoDecoder(const Sam2VideoConfig& config,
                                      const VideoDecoderInputs& inputs,
                                      const WeightMap& weights,
                                      int mask_mode = 0,
                                      ConstCache* cache = nullptr);
// mem [1, 4096, 64] token-major output, named "mem".
TfTensor BuildMemorize(const Sam2VideoConfig& config,
                       const MemorizeInputs& inputs, const WeightMap& weights,
                       ConstCache* cache = nullptr);

}  // namespace litert::tensor::examples::sam2_video

#endif  // MODELS_SAM2_SAM2_HIERA_TINY_VIDEO_TENSOR_API_SAM2_VIDEO_SAM2V_GRAPH_H_
