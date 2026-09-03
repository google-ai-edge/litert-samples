// Real-checkpoint weight loading. See sam2_weights.h.

#include "models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image/sam2_weights.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "tensor/examples/gemma3/safetensor_loader.h"
#include "models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image/sam2_config.h"
#include "models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image/sam2_graph.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples::sam2 {

namespace {

// Guard against the gemma3 loader's Gemma-norm +1.0 heuristic silently
// firing on a future key (none of the current SAM2 keys match it).
bool LoaderWouldAddGemmaOffset(const std::string& name) {
  return absl::StrContains(name, "layernorm") ||
         absl::EndsWith(name, "norm.weight");
}

}  // namespace

absl::StatusOr<WeightMap> LoadCheckpointWeights(const Sam2Config& config,
                                                const std::string& path) {
  auto loader_or = SafetensorLoader::Load(path);
  if (!loader_or.ok()) return loader_or.status();
  auto loader = std::move(*loader_or);

  WeightMap weights;
  for (const WeightSpec& spec : GetWeightSpecs(config)) {
    if (LoaderWouldAddGemmaOffset(spec.name)) {
      return absl::FailedPreconditionError(absl::StrCat(
          spec.name,
          ": matches the gemma3 loader's norm-offset predicate; a +1.0 "
          "would corrupt this weight — handle explicitly before loading"));
    }
    auto handle_or = loader.LoadTensor(
        spec.name, SafetensorLoader::QuantizedLoadMode::kDequantizeToFp32);
    if (!handle_or.ok()) {
      return absl::NotFoundError(absl::StrCat("checkpoint missing ",
                                              spec.name, ": ",
                                              handle_or.status().message()));
    }
    TfTensor tensor(*handle_or);
    const auto& loaded_shape = tensor.GetShape();
    std::vector<int> loaded(loaded_shape.begin(), loaded_shape.end());
    if (loaded != spec.shape) {
      return absl::FailedPreconditionError(absl::StrCat(
          spec.name, ": checkpoint shape [", absl::StrJoin(loaded, ","),
          "] != expected [", absl::StrJoin(spec.shape, ","), "]"));
    }
    tensor.SetName(spec.name);
    weights[spec.name] = std::move(tensor);
  }
  return weights;
}

}  // namespace litert::tensor::examples::sam2
