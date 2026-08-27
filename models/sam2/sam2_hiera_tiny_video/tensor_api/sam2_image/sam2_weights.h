// Real-checkpoint weight loading for the SAM2 image-path graphs.
//
// Loads the 512-baked MLX-naming safetensors set
// (mlboydaisuke/SAM2-hiera-tiny-LiteRT mlx/sam2_tiny_512.safetensors,
// fp16 -> fp32; trunk.pos_embed_full already re-composed at the 128x128
// token grid — the corr-0.9999 fix). Every tensor is shape-checked against
// GetWeightSpecs. No SAM2 key matches the gemma3 loader's Gemma-norm
// (+1.0) predicate, so weights load unmodified.

#ifndef MODELS_SAM2_SAM2_HIERA_TINY_VIDEO_TENSOR_API_SAM2_IMAGE_SAM2_WEIGHTS_H_
#define MODELS_SAM2_SAM2_HIERA_TINY_VIDEO_TENSOR_API_SAM2_IMAGE_SAM2_WEIGHTS_H_

#include <string>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image/sam2_config.h"
#include "models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image/sam2_graph.h"

namespace litert::tensor::examples::sam2 {

absl::StatusOr<WeightMap> LoadCheckpointWeights(const Sam2Config& config,
                                                const std::string& path);

}  // namespace litert::tensor::examples::sam2

#endif  // MODELS_SAM2_SAM2_HIERA_TINY_VIDEO_TENSOR_API_SAM2_IMAGE_SAM2_WEIGHTS_H_
