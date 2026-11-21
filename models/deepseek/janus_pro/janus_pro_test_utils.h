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

#ifndef MODELS_DEEPSEEK_JANUS_PRO_JANUS_PRO_TEST_UTILS_H_
#define MODELS_DEEPSEEK_JANUS_PRO_JANUS_PRO_TEST_UTILS_H_

#include <cstdint>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "runtime/proto/any2any_metadata.pb.h"
#include "tflite/schema/schema_generated.h"

namespace litert::omni::any2any {

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

// Options controlling the synthetic `.litertlm` bundle written by
// `WriteJanusProBundle`.
struct JanusProTestBundleOptions {
  // Optional metadata override; when null, `BuildDefaultTestMetadata(options)`
  // is used.
  const lm::proto::Any2AnyMetadata* absl_nullable custom_meta = nullptr;
  // Whether to include `tf_lite_vision_encoder`.
  bool include_vision_encoder = true;
  // Whether to include `tf_lite_vq_image_embedder`, `tf_lite_gen_head`, and
  // `tf_lite_vq_image_decoder`.
  bool include_image_gen_models = true;
  int llm_hidden_size = 4;
  int llm_vocab_size = 32;
  int vq_codebook_size = 8;
  int vq_codebook_embed_dim = 4;
  int image_size = 16;
  int patch_size = 8;
  int num_image_tokens = 4;
  bool generate_image = false;
  int max_new_tokens = 2;
  float cfg_weight = 5.0f;
  float image_temperature = 0.0f;
  // Deterministic VQ codebook index peaked by the synthetic `tf_lite_gen_head`.
  int gen_head_preferred_code = 3;
  // Constant RGB channel values in `[-1.0, 1.0]` emitted by the synthetic
  // `tf_lite_vq_image_decoder`.
  float vq_decoder_r = 1.0f;
  float vq_decoder_g = 0.0f;
  float vq_decoder_b = -1.0f;
};

// Builds a minimal valid single-signature TFLite FlatBuffer whose outputs emit
// either zeros or `TensorSpec::float_values` via constant `ADD` ops.
std::string BuildMinimalTfliteModelBuffer(absl::Span<const TensorSpec> inputs,
                                          absl::Span<const TensorSpec> outputs);

// Reads the full contents of `path` into a byte string.
absl::StatusOr<std::string> ReadFileBytes(const std::filesystem::path& path);

// Encodes an interleaved pixel buffer (`channels` in `{1, 3, 4}`) to PNG bytes.
absl::StatusOr<std::vector<uint8_t>> EncodePngBytes(
    absl::Span<const uint8_t> pixels, int width, int height, int channels);

// Writes a PNG file to `path` from an interleaved pixel buffer.
absl::Status WriteTestPngFile(const std::filesystem::path& path,
                              absl::Span<const uint8_t> pixels, int width,
                              int height, int channels);

// Builds default `Any2AnyMetadata` packing `JanusPro` custom metadata matching
// `options` and the `checksum_fake_lm_*` testdata models.
absl::StatusOr<lm::proto::Any2AnyMetadata> BuildDefaultTestMetadata(
    const JanusProTestBundleOptions& options = {});

// Writes a Janus-Pro `.litertlm` test bundle named `filename` inside
// `::testing::TempDir()` and returns its full path.
absl::StatusOr<std::string> WriteJanusProBundle(
    absl::string_view filename, const JanusProTestBundleOptions& options = {});

// Convenience overload preserving the custom-metadata-only call pattern.
absl::StatusOr<std::string> WriteJanusProBundle(
    absl::string_view filename,
    const lm::proto::Any2AnyMetadata* absl_nullable custom_meta);

}  // namespace litert::omni::any2any

#endif  // MODELS_DEEPSEEK_JANUS_PRO_JANUS_PRO_TEST_UTILS_H_
