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

#ifndef MODELS_OMNIGEN_OMNIGEN2_RUNNER_OMNIGEN2_TEST_UTILS_H_
#define MODELS_OMNIGEN_OMNIGEN2_RUNNER_OMNIGEN2_TEST_UTILS_H_

#include <array>
#include <cstdint>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <string>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "runtime/proto/text2image_metadata.pb.h"
#include "tflite/schema/schema_generated.h"

namespace litert::omni::text2image::omnigen2 {

// Parsed PNG signature + IHDR metadata for verifying generated PNG outputs.
struct ParsedPngHeader {
  uint32_t width = 0;
  uint32_t height = 0;
  uint8_t bit_depth = 0;
  uint8_t color_type = 0;
};

// Options controlling the synthetic `.litertlm` OmniGen2 bundle written by
// `WriteOmniGen2Bundle`.
struct OmniGen2TestBundleOptions {
  const lm::proto::Text2ImageMetadata* absl_nullable custom_meta = nullptr;
  bool include_metadata_section = true;
  bool include_tokenizer_section = true;
  // Optional section key (`model_type` for TFLiteModel sections or `name` for
  // GenericBinaryData sections) to omit from the generated bundle.
  std::string omit_section;
  int seq_len = 32;
  int img_size = 32;
  int in_channels = 16;
  int patch_size = 2;
  int text_hidden_size = 8;
  int dit_hidden_size = 48;
  std::array<int, 3> axes_dims_rope = {4, 6, 6};
  int temb_dim = 16;
  int steps = 2;
  float text_guidance_scale = 2.0f;
  float vae_output_value = 0.0f;
  tflite::TensorType text_input_type = tflite::TensorType_INT32;
};

// Reads all bytes from `path`.
absl::StatusOr<std::string> ReadFileBytes(const std::filesystem::path& path);

// Validates a PNG file signature + IHDR chunk and returns parsed dimensions.
absl::StatusOr<ParsedPngHeader> ParsePngHeader(
    const std::filesystem::path& path);

// Builds default `Text2ImageMetadata` packing an `OmniGen2` custom_model proto
// matching `options`.
absl::StatusOr<lm::proto::Text2ImageMetadata> BuildDefaultTestMetadata(
    const OmniGen2TestBundleOptions& options = {});

// Writes a self-contained synthetic OmniGen2 `.litertlm` bundle (7 standard
// `TFLiteModel` sections + 6 named `GenericBinaryData` shards + zlib-compressed
// HF tokenizer + `Text2ImageMetadata` proto) into `testing::TempDir()`.
absl::StatusOr<std::string> WriteOmniGen2Bundle(
    absl::string_view filename, const OmniGen2TestBundleOptions& options = {});

}  // namespace litert::omni::text2image::omnigen2

#endif  // MODELS_OMNIGEN_OMNIGEN2_RUNNER_OMNIGEN2_TEST_UTILS_H_
