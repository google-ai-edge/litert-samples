// Video-stack weight loading (memory attention / memory encoder / pointer
// heads / baked tables) for the SAM2 video example. The image-path weights
// load through ../sam2_image's loader; this adds the video-only keys from
// the same safetensors file into the same WeightMap.

#ifndef MODELS_SAM2_SAM2_HIERA_TINY_VIDEO_TENSOR_API_SAM2_VIDEO_SAM2V_WEIGHTS_H_
#define MODELS_SAM2_SAM2_HIERA_TINY_VIDEO_TENSOR_API_SAM2_VIDEO_SAM2V_WEIGHTS_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_video/sam2v_graph.h"

namespace litert::tensor::examples::sam2_video {

using ::litert::tensor::examples::sam2::WeightSpec;

// Every video-only weight the graphs consume (checkpoint naming + the
// "tables." entries the export derives), shapes for image_size = 1024.
std::vector<WeightSpec> GetVideoWeightSpecs(const Sam2VideoConfig& config);

// Loads GetVideoWeightSpecs into `weights` (which already holds the image
// keys). Compensates the safetensor loader's Gemma-norm +1.0 heuristic on
// the three video keys that match its predicate.
absl::Status LoadVideoWeights(const Sam2VideoConfig& config,
                              const std::string& path, WeightMap& weights);

// Synthetic video weights for a shape/route check (image keys come from
// sam2_image's generator).
void MakeSyntheticVideoWeights(const Sam2VideoConfig& config, unsigned seed,
                               WeightMap& weights);

}  // namespace litert::tensor::examples::sam2_video

#endif  // MODELS_SAM2_SAM2_HIERA_TINY_VIDEO_TENSOR_API_SAM2_VIDEO_SAM2V_WEIGHTS_H_
