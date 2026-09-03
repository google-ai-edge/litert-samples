// SAM2.1 hiera-tiny VIDEO tracking path authored on the C++ Tensor API.
//
// Adds the three per-frame memory graphs of the SAM2 tracking loop to the
// image-path encoder (reused from ../sam2_image at image_size=1024 with a
// zeroed no_mem_embed, so its output IS the raw top-level feature map):
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

#include <string>
#include <vector>

#include "models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image/sam2_config.h"
#include "models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image/sam2_graph.h"

namespace litert::tensor::examples::sam2_video {

using ::litert::tensor::examples::sam2::Sam2Config;
using ::litert::tensor::examples::sam2::TfTensor;
using ::litert::tensor::examples::sam2::WeightMap;

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

MemCondInputs MakeMemCondInputs(const Sam2VideoConfig& config, int nmm);
VideoDecoderInputs MakeVideoDecoderInputs(const Sam2VideoConfig& config);
MemorizeInputs MakeMemorizeInputs(const Sam2VideoConfig& config);

// pix_feat [1, G, G, 256] token-major output, named "pix_feat".
TfTensor BuildMemCond(const Sam2VideoConfig& config, int nmm,
                      const MemCondInputs& inputs, const WeightMap& weights);
VideoDecoderOutputs BuildVideoDecoder(const Sam2VideoConfig& config,
                                      const VideoDecoderInputs& inputs,
                                      const WeightMap& weights);
// mem [1, 4096, 64] token-major output, named "mem".
TfTensor BuildMemorize(const Sam2VideoConfig& config,
                       const MemorizeInputs& inputs, const WeightMap& weights);

}  // namespace litert::tensor::examples::sam2_video

#endif  // MODELS_SAM2_SAM2_HIERA_TINY_VIDEO_TENSOR_API_SAM2_VIDEO_SAM2V_GRAPH_H_
