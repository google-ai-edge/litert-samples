// SAM2.1 hiera-tiny image path authored on the C++ Tensor API.
//
// Two signatures in one flatbuffer:
//   encode_image: pixels [1,512,512,3] NHWC (ImageNet-normalized) ->
//     image_embeddings [1,32,32,256] (no_mem_embed folded in),
//     feat_s1 [1,64,64,64], feat_s0 [1,128,128,32] (conv_s1/conv_s0 folded
//     into the encoder — the decoder-ready layout the published converted
//     models use).
//   decode_mask: image_embeddings + feat_s1 + feat_s0 + point_coords
//     [1,1,2] (x,y in image space, one positive point; the not_a_point pad
//     token is a baked constant) -> masks [1,3,128,128] multimask logits,
//     iou_scores [1,3], object_score [1,1].
//
// Layout is NHWC end to end (TFLite conv layout; the MLX checkpoint's conv
// weights are already OHWI so every weight loads without transposition).
// Constants baked at build time: the 128-grid re-composed pos_embed_full,
// the dense image positional-encoding grid, the no-mask dense prompt row,
// the output tokens, and the prompt-encoder point/pad embeddings. All
// attention is rank-4; window partition/unpartition stays <=4-D (the
// order-equivariant split-H-then-W form proven in the converted-path work).

#ifndef MODELS_SAM2_SAM2_HIERA_TINY_VIDEO_TENSOR_API_SAM2_IMAGE_SAM2_GRAPH_H_
#define MODELS_SAM2_SAM2_HIERA_TINY_VIDEO_TENSOR_API_SAM2_IMAGE_SAM2_GRAPH_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image/sam2_config.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples::sam2 {

using TfTensor = ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>;
using WeightMap = absl::flat_hash_map<std::string, TfTensor>;

struct EncoderInputs {
  TfTensor pixels;  // [1, S, S, 3] kFP32 NHWC, ImageNet-normalized
  std::vector<TfTensor> AsList() const { return {pixels}; }
};

struct EncoderOutputs {
  TfTensor image_embeddings;  // [1, S/16, S/16, 256]
  TfTensor feat_s1;           // [1, S/8, S/8, 64]
  TfTensor feat_s0;           // [1, S/4, S/4, 32]
  std::vector<TfTensor> AsList() const {
    return {image_embeddings, feat_s1, feat_s0};
  }
};

struct DecoderInputs {
  TfTensor image_embeddings;  // [1, S/16, S/16, 256]
  TfTensor feat_s1;           // [1, S/8, S/8, 64]
  TfTensor feat_s0;           // [1, S/4, S/4, 32]
  TfTensor point_coords;      // [1, 1, 2] kFP32 (x, y) in image space
  std::vector<TfTensor> AsList() const {
    return {image_embeddings, feat_s1, feat_s0, point_coords};
  }
};

struct DecoderOutputs {
  TfTensor masks;         // [1, 3, S/4, S/4] multimask logits
  TfTensor iou_scores;    // [1, 3]
  TfTensor object_score;  // [1, 1]
  std::vector<TfTensor> AsList() const {
    return {masks, iou_scores, object_score};
  }
};

// Every weight the two graphs consume, under the checkpoint's MLX naming
// (trunk./neck./sam_prompt_encoder./sam_mask_decoder./no_mem_embed).
// Single source of truth for the synthetic generator and the loader.
struct WeightSpec {
  std::string name;
  std::vector<int> shape;
  float init_scale;  // synthetic init only
};
std::vector<WeightSpec> GetWeightSpecs(const Sam2Config& config);

WeightMap MakeSyntheticWeights(const Sam2Config& config, unsigned seed);

EncoderInputs MakeEncoderInputs(const Sam2Config& config);
DecoderInputs MakeDecoderInputs(const Sam2Config& config);

// odml.runtime_bmm control inputs created during the LAST Build*() call
// (one int32 [1,1,1,7] graph input per distinct bound length S, named
// "rbmm_s<S>"). The caller appends the tensors to that signature's input
// list and feeds each with seven int32 copies of S at runtime (the proven
// attn_bench fill pattern). Returns and clears the registry.
std::vector<std::pair<int, TfTensor>> TakeRbmmParams();

EncoderOutputs BuildEncoder(const Sam2Config& config,
                            const EncoderInputs& inputs,
                            const WeightMap& weights);
DecoderOutputs BuildDecoder(const Sam2Config& config,
                            const DecoderInputs& inputs,
                            const WeightMap& weights);

}  // namespace litert::tensor::examples::sam2

#endif  // MODELS_SAM2_SAM2_HIERA_TINY_VIDEO_TENSOR_API_SAM2_IMAGE_SAM2_GRAPH_H_
