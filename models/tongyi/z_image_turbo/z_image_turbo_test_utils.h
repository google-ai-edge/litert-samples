// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MODELS_TONGYI_Z_IMAGE_TURBO_Z_IMAGE_TURBO_TEST_UTILS_H_
#define MODELS_TONGYI_Z_IMAGE_TURBO_Z_IMAGE_TURBO_TEST_UTILS_H_

#include <cstdint>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <string>
#include <vector>

#include "models/tongyi/z_image_turbo/z_image_turbo_model_config.h"
#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "runtime/proto/text2image_metadata.pb.h"
#include "tflite/schema/schema_generated.h"

namespace litert::omni::text2image {

// Tensor descriptor used by `BuildMinimalTfliteModelBuffer`.
struct TensorSpec {
  std::string name;
  std::vector<int32_t> dims;
  tflite::TensorType type;
  // Optional float32 constant values for output tensors. If empty, outputs
  // zeros. If non-empty, `type` must be `tflite::TensorType_FLOAT32` and the
  // size must match the tensor element count.
  std::vector<float> float_values = {};
};

// Parsed fields from a PNG file's `IHDR` chunk.
struct ParsedPngHeader {
  uint32_t width = 0;
  uint32_t height = 0;
  uint8_t bit_depth = 0;
  uint8_t color_type = 0;
};

// Options controlling the synthetic `.litertlm` bundle written by
// `WriteZImageTurboBundle`.
struct ZImageTurboTestBundleOptions {
  // Optional metadata override; when null, built from `config`.
  const lm::proto::Text2ImageMetadata* absl_nullable custom_meta = nullptr;
  // Tiny model configuration governing all submodel shapes.
  ZImageTurboModelConfig config;
  // Number of sharded main transformer blocks (`1..6`) to include. Defaults to
  // 6 so both `TFLiteModel` (`0..3`) and `GenericBinaryData` (`4..5`) paths are
  // exercised.
  int num_main_shards = 6;
  // Constant values emitted by the synthetic text encoder, final layer, and VAE
  // decoder TFLite models.
  float prompt_embed_value = 0.25f;
  float final_velocity_value = 0.25f;
  float vae_rgb_value = 0.5f;
};

// Returns a tiny valid `ZImageTurboModelConfig` suitable for fast hermetic
// tests (`seq_len = 8`, `img_size = 16`, `in_channels = 2`, `dim = 12`,
// `prompt_dim = 8`, `axes_dims = {4, 4, 4}`, compact tokenizer IDs).
ZImageTurboModelConfig MakeTinyZImageTurboConfig();

// Allocates a host-memory float32 `TensorBuffer` initialized to zeros.
absl::StatusOr<TensorBuffer> MakeFloatTensorBuffer(Environment& env,
                                                   absl::Span<const int> dims);

// Builds a minimal valid single-signature TFLite FlatBuffer whose outputs emit
// either zeros or `TensorSpec::float_values` via constant `ADD` ops.
std::string BuildMinimalTfliteModelBuffer(absl::Span<const TensorSpec> inputs,
                                          absl::Span<const TensorSpec> outputs);

// Reads the full contents of `path` into a byte string.
absl::StatusOr<std::string> ReadFileBytes(const std::filesystem::path& path);

// Validates the 8-byte PNG signature and parses the `IHDR` chunk from `path`.
absl::StatusOr<ParsedPngHeader> ParsePngHeader(
    const std::filesystem::path& path);

// Builds `Text2ImageMetadata` packing `prototypes.litert_models.z_image_turbo`
// `ZImageTurbo` fields derived from `config` into `custom_model`.
absl::StatusOr<lm::proto::Text2ImageMetadata> BuildDefaultZImageTurboMetadata(
    const ZImageTurboModelConfig& config);

// Writes a Z-Image-Turbo `.litertlm` test bundle named `filename` inside
// `::testing::TempDir()` and returns its full path.
absl::StatusOr<std::string> WriteZImageTurboBundle(
    absl::string_view filename, const ZImageTurboTestBundleOptions& options);

// Convenience overload using `MakeTinyZImageTurboConfig()` and optional
// metadata override.
absl::StatusOr<std::string> WriteZImageTurboBundle(
    absl::string_view filename,
    const lm::proto::Text2ImageMetadata* absl_nullable custom_meta = nullptr);

}  // namespace litert::omni::text2image

#endif  // MODELS_TONGYI_Z_IMAGE_TURBO_Z_IMAGE_TURBO_TEST_UTILS_H_
