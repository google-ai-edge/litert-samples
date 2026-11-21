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

#ifndef MODELS_LFM_LFM2_5_AUDIO_LFM2_AUDIO_TEST_UTILS_H_
#define MODELS_LFM_LFM2_5_AUDIO_LFM2_AUDIO_TEST_UTILS_H_

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

// Options controlling the test `.litertlm` bundle written by
// `WriteLfm2AudioBundle`.
struct Lfm2AudioTestBundleOptions {
  // Optional metadata override; when null, `BuildDefaultTestMetadata()` is
  // used.
  const lm::proto::Any2AnyMetadata* absl_nullable custom_meta = nullptr;
  // When true, includes `tf_lite_tts_projector`, `tf_lite_tts_decoder`, and
  // `tf_lite_audio_vocoder` submodels in addition to the default text embedder,
  // LLM backbone, and audio encoder.
  bool include_tts_models = false;
  int llm_hidden_size = 4;
  int tts_hidden_size = 4;
  int tts_vocab_size = 8;
  int tts_num_codebooks = 8;
  int vocoder_chunk_frames = 2;
  int vocoder_samples_per_frame = 8;
  // Constant PCM amplitude emitted by the test vocoder model when
  // `include_tts_models` is true.
  float vocoder_output_amplitude = 0.25f;
};

// Builds a minimal valid single-signature TFLite FlatBuffer whose outputs emit
// either zeros or `TensorSpec::float_values` via constant `ADD` ops.
std::string BuildMinimalTfliteModelBuffer(absl::Span<const TensorSpec> inputs,
                                          absl::Span<const TensorSpec> outputs);

// Reads the full contents of `path` into a byte string.
absl::StatusOr<std::string> ReadFileBytes(const std::filesystem::path& path);

// Builds default `Any2AnyMetadata` packing `Lfm2Audio` custom metadata matching
// the `checksum_fake_lm_*` testdata models.
absl::StatusOr<lm::proto::Any2AnyMetadata> BuildDefaultTestMetadata();

// Writes an LFM2-Audio `.litertlm` test bundle named `filename` inside
// `::testing::TempDir()` and returns its full path.
absl::StatusOr<std::string> WriteLfm2AudioBundle(
    absl::string_view filename, const Lfm2AudioTestBundleOptions& options = {});

// Convenience overload preserving the custom-metadata-only call pattern.
absl::StatusOr<std::string> WriteLfm2AudioBundle(
    absl::string_view filename,
    const lm::proto::Any2AnyMetadata* absl_nullable custom_meta);

// Writes mono 16-bit PCM WAV data to `path` from normalized float samples in
// `[-1.0, 1.0]`.
absl::Status WriteTestWavFile(const std::filesystem::path& path,
                              absl::Span<const float> pcm_samples,
                              int sample_rate_hz);

}  // namespace litert::omni::any2any

#endif  // MODELS_LFM_LFM2_5_AUDIO_LFM2_AUDIO_TEST_UTILS_H_
