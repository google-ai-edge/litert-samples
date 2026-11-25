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

#include "models/lfm/lfm2_5_audio/runner/lfm2_audio_test_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <ios>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "models/lfm/lfm2_5_audio/runner/lfm2_audio.pb.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "omni/any2any/any2any_test_utils.h"
#include "runtime/proto/any2any_metadata.pb.h"
#include "runtime/proto/any2any_model_type.pb.h"
#include "schema/core/litertlm_header_schema_generated.h"
#include "tflite/schema/schema_generated.h"

namespace litert::omni::any2any {
namespace {

constexpr absl::string_view kMinimalTokenizerJson = R"json({
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "<|pad|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 1, "content": "<|startoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 6, "content": "<|im_start|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 7, "content": "<|im_end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 5, "content": "\n", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
      "<|pad|>": 0,
      "<|startoftext|>": 1,
      "t2": 2,
      "Hello": 3,
      "world": 4,
      "\n": 5,
      "<|im_start|>": 6,
      "<|im_end|>": 7,
      "t8": 8, "t9": 9, "t10": 10, "t11": 11, "t12": 12,
      "t13": 13, "t14": 14, "t15": 15, "t16": 16, "t17": 17, "t18": 18,
      "t19": 19, "t20": 20, "t21": 21, "t22": 22, "t23": 23, "t24": 24,
      "t25": 25, "t26": 26, "t27": 27, "t28": 28, "t29": 29, "t30": 30,
      "t31": 31
    },
    "unk_token": "<|pad|>"
  }
})json";

void AppendLe16(std::string& out, uint16_t value) {
  out.push_back(static_cast<char>(value & 0xff));
  out.push_back(static_cast<char>((value >> 8) & 0xff));
}

void AppendLe32(std::string& out, uint32_t value) {
  out.push_back(static_cast<char>(value & 0xff));
  out.push_back(static_cast<char>((value >> 8) & 0xff));
  out.push_back(static_cast<char>((value >> 16) & 0xff));
  out.push_back(static_cast<char>((value >> 24) & 0xff));
}

}  // namespace

absl::StatusOr<std::string> ReadFileBytes(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    return absl::InternalError(absl::StrCat("Failed to open: ", path.string()));
  }
  return std::string(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
}

absl::StatusOr<lm::proto::Any2AnyMetadata> BuildDefaultTestMetadata() {
  lm::proto::Any2AnyMetadata any2any_meta;
  any2any_meta.set_input_sample_rate_hz(16000);
  any2any_meta.set_output_sample_rate_hz(24000);
  any2any_meta.set_system_prompt("Respond with interleaved text and audio.");
  any2any_meta.add_supported_languages("en-US");
  lm::proto::Lfm2Audio lfm2;
  lfm2.set_llm_hidden_size(4);
  lfm2.set_llm_vocab_size(32);
  lfm2.set_tts_hidden_size(4);
  lfm2.set_tts_vocab_size(8);
  lfm2.set_tts_num_codebooks(8);
  lfm2.set_audio_mel_bins(128);
  lfm2.set_audio_subsampling_factor(8);
  lfm2.set_generate_audio(false);
  lfm2.set_max_new_tokens(1);
  lfm2.set_max_tts_tokens(1);
  lfm2.set_voice("lfm2_voice");
  lfm2.set_language("en-US");
  if (!any2any_meta.mutable_any2any_model_type()
           ->mutable_custom_model()
           ->PackFrom(lfm2)) {
    return absl::InternalError("Failed to pack Lfm2Audio test metadata.");
  }
  return any2any_meta;
}

absl::StatusOr<std::string> WriteLfm2AudioBundle(
    absl::string_view filename, const Lfm2AudioTestBundleOptions& options) {
  lm::proto::Any2AnyMetadata any2any_meta;
  if (options.custom_meta != nullptr) {
    any2any_meta = *options.custom_meta;
  } else {
    ABSL_ASSIGN_OR_RETURN(any2any_meta, BuildDefaultTestMetadata());
  }

  std::vector<BundleSectionPayload> extra_sections;
  extra_sections.push_back(BundleSectionPayload{
      .data_type = lm::schema::AnySectionDataType_TFLiteModel,
      .model_type = "tf_lite_audio_encoder",
      .bytes = BuildMinimalTfliteModelBuffer(
          {{"mel", {1, 128, 16}, tflite::TensorType_FLOAT32}},
          {{"features",
            {1, 2, options.llm_hidden_size},
            tflite::TensorType_FLOAT32}}),
  });

  if (options.include_tts_models) {
    // Deterministic Depthformer logits: strong peak on code 0 (never EOAudio =
    // tts_vocab_size - 1) so sampling produces valid audio frames even when
    // `tts_temperature > 0`.
    std::vector<float> decoder_logits(options.tts_vocab_size, -50.0f);
    if (!decoder_logits.empty()) {
      decoder_logits[0] = 50.0f;
    }
    const int vocoder_out_samples =
        options.vocoder_chunk_frames * options.vocoder_samples_per_frame;
    std::vector<float> vocoder_waveform(vocoder_out_samples,
                                        options.vocoder_output_amplitude);

    extra_sections.push_back(BundleSectionPayload{
        .data_type = lm::schema::AnySectionDataType_TFLiteModel,
        .model_type = "tf_lite_tts_projector",
        .bytes = BuildMinimalTfliteModelBuffer(
            {{"llm_hidden",
              {1, 1, options.llm_hidden_size},
              tflite::TensorType_FLOAT32},
             {"audio_codes",
              {1, options.tts_num_codebooks},
              tflite::TensorType_INT32}},
            {{"depthformer_in",
              {1, options.tts_num_codebooks, options.tts_hidden_size},
              tflite::TensorType_FLOAT32},
             {"audio_emb",
              {1, 1, options.llm_hidden_size},
              tflite::TensorType_FLOAT32}}),
    });
    extra_sections.push_back(BundleSectionPayload{
        .data_type = lm::schema::AnySectionDataType_TFLiteModel,
        .model_type = "tf_lite_tts_decoder",
        .bytes = BuildMinimalTfliteModelBuffer(
            {{"cond_emb",
              {1, 1, options.tts_hidden_size},
              tflite::TensorType_FLOAT32},
             {"prev_audio_code", {1, 1}, tflite::TensorType_INT32},
             {"step_idx", {1}, tflite::TensorType_INT32},
             {"kv_cache",
              {1, 2, options.tts_num_codebooks, options.tts_hidden_size},
              tflite::TensorType_FLOAT32}},
            {{"logits",
              {1, 1, options.tts_vocab_size},
              tflite::TensorType_FLOAT32,
              std::move(decoder_logits)},
             {"new_kv_cache",
              {1, 2, options.tts_num_codebooks, options.tts_hidden_size},
              tflite::TensorType_FLOAT32}}),
    });
    extra_sections.push_back(BundleSectionPayload{
        .data_type = lm::schema::AnySectionDataType_TFLiteModel,
        .model_type = "tf_lite_audio_vocoder",
        .bytes = BuildMinimalTfliteModelBuffer(
            {{"audio_codes",
              {1, options.tts_num_codebooks, options.vocoder_chunk_frames},
              tflite::TensorType_INT32}},
            {{"waveform",
              {1, vocoder_out_samples},
              tflite::TensorType_FLOAT32,
              std::move(vocoder_waveform)}}),
    });
  }

  Any2AnyBundleOptions bundle_options{
      .metadata = std::move(any2any_meta),
      .tokenizer_json = std::string(kMinimalTokenizerJson),
      .extra_sections = std::move(extra_sections),
  };
  return WriteAny2AnyTestBundle(filename, std::move(bundle_options));
}

absl::StatusOr<std::string> WriteLfm2AudioBundle(
    absl::string_view filename,
    const lm::proto::Any2AnyMetadata* absl_nullable custom_meta) {
  Lfm2AudioTestBundleOptions options;
  options.custom_meta = custom_meta;
  return WriteLfm2AudioBundle(filename, options);
}

absl::Status WriteTestWavFile(const std::filesystem::path& path,
                              absl::Span<const float> pcm_samples,
                              int sample_rate_hz) {
  if (sample_rate_hz <= 0) {
    return absl::InvalidArgumentError("sample_rate_hz must be positive.");
  }
  constexpr uint16_t kNumChannels = 1;
  constexpr uint16_t kBitsPerSample = 16;
  const uint32_t byte_rate = static_cast<uint32_t>(sample_rate_hz) *
                             kNumChannels * (kBitsPerSample / 8);
  const uint16_t block_align = kNumChannels * (kBitsPerSample / 8);
  const uint32_t data_size =
      static_cast<uint32_t>(pcm_samples.size() * sizeof(int16_t));

  std::string wav;
  wav.reserve(44 + data_size);
  wav.append("RIFF", 4);
  AppendLe32(wav, 36 + data_size);
  wav.append("WAVE", 4);
  wav.append("fmt ", 4);
  AppendLe32(wav, 16);  // PCM fmt chunk size.
  AppendLe16(wav, 1);   // PCM format.
  AppendLe16(wav, kNumChannels);
  AppendLe32(wav, static_cast<uint32_t>(sample_rate_hz));
  AppendLe32(wav, byte_rate);
  AppendLe16(wav, block_align);
  AppendLe16(wav, kBitsPerSample);
  wav.append("data", 4);
  AppendLe32(wav, data_size);

  for (float s : pcm_samples) {
    const float clamped = std::clamp(s, -1.0f, 1.0f);
    const int16_t quantized =
        static_cast<int16_t>(std::lround(clamped * 32767.0f));
    AppendLe16(wav, static_cast<uint16_t>(quantized));
  }

  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    return absl::InternalError(
        absl::StrCat("Failed to open WAV file for writing: ", path.string()));
  }
  out.write(wav.data(), wav.size());
  if (!out.good()) {
    return absl::InternalError(
        absl::StrCat("Failed to write WAV file: ", path.string()));
  }
  return absl::OkStatus();
}

}  // namespace litert::omni::any2any
