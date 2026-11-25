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

#include "models/deepseek/janus_pro/runner/janus_pro_test_utils.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <ios>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "models/deepseek/janus_pro/runner/janus_pro.pb.h"
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
#include "stb_image_write.h"
#include "tflite/schema/schema_generated.h"

namespace litert::omni::any2any {
namespace {

// Minimal WordLevel tokenizer containing all Janus-Pro control tokens inside
// IDs [0, 27] so both `llm_vocab_size=32` and `llm_vocab_size=28` (28 logits +
// 4 hidden state dims) remain strictly in-bounds.
constexpr absl::string_view kMinimalJanusProTokenizerJson = R"json({
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "<｜▁pad▁｜>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 1, "content": "<｜begin▁of▁sentence｜>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 2, "content": "<｜end▁of▁sentence｜>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 5, "content": "<begin_of_image>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 6, "content": "<end_of_image>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 7, "content": "<image_placeholder>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 8, "content": "<|User|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 9, "content": "<|Assistant|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
      "<｜▁pad▁｜>": 0,
      "<｜begin▁of▁sentence｜>": 1,
      "<｜end▁of▁sentence｜>": 2,
      "Hello": 3,
      "world": 4,
      "<begin_of_image>": 5,
      "<end_of_image>": 6,
      "<image_placeholder>": 7,
      "<|User|>": 8,
      "<|Assistant|>": 9,
      "\n": 10,
      ":": 11,
      "Describe": 12,
      "this": 13,
      "image": 14,
      "fox": 15,
      "t16": 16, "t17": 17, "t18": 18, "t19": 19, "t20": 20, "t21": 21,
      "t22": 22, "t23": 23, "t24": 24, "t25": 25, "t26": 26, "t27": 27,
      "t28": 28, "t29": 29, "t30": 30, "t31": 31
    },
    "unk_token": "<｜▁pad▁｜>"
  }
})json";

}  // namespace

absl::StatusOr<std::string> ReadFileBytes(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    return absl::InternalError(absl::StrCat("Failed to open: ", path.string()));
  }
  return std::string(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
}

absl::StatusOr<std::vector<uint8_t>> EncodePngBytes(
    absl::Span<const uint8_t> pixels, int width, int height, int channels) {
  if (width <= 0 || height <= 0 || channels <= 0) {
    return absl::InvalidArgumentError(
        "PNG width, height, and channels must be positive.");
  }
  const size_t expected_size = static_cast<size_t>(width) * height * channels;
  if (pixels.size() != expected_size) {
    return absl::InvalidArgumentError(
        "Pixel buffer size does not match width * height * channels.");
  }
  std::vector<uint8_t> png_bytes;
  const int ok = stbi_write_png_to_func(
      +[](void* context, void* data, int size) {
        auto* out = static_cast<std::vector<uint8_t>*>(context);
        const auto* bytes = static_cast<const uint8_t*>(data);
        out->insert(out->end(), bytes, bytes + size);
      },
      &png_bytes, width, height, channels, pixels.data(), width * channels);
  if (ok == 0 || png_bytes.empty()) {
    return absl::InternalError("stbi_write_png_to_func failed.");
  }
  return png_bytes;
}

absl::Status WriteTestPngFile(const std::filesystem::path& path,
                              absl::Span<const uint8_t> pixels, int width,
                              int height, int channels) {
  ABSL_ASSIGN_OR_RETURN(std::vector<uint8_t> png_bytes,
                        EncodePngBytes(pixels, width, height, channels));
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    return absl::InternalError(
        absl::StrCat("Failed to open PNG file for writing: ", path.string()));
  }
  out.write(reinterpret_cast<const char*>(png_bytes.data()), png_bytes.size());
  if (!out.good()) {
    return absl::InternalError(
        absl::StrCat("Failed to write PNG file: ", path.string()));
  }
  return absl::OkStatus();
}

absl::StatusOr<lm::proto::Any2AnyMetadata> BuildDefaultTestMetadata(
    const JanusProTestBundleOptions& options) {
  lm::proto::Any2AnyMetadata any2any_meta;
  any2any_meta.set_system_prompt("Describe images clearly.");
  any2any_meta.add_supported_languages("en-US");
  any2any_meta.add_supported_languages("zh-CN");
  lm::proto::JanusPro janus_pro;
  janus_pro.set_llm_hidden_size(options.llm_hidden_size);
  janus_pro.set_llm_vocab_size(options.llm_vocab_size);
  janus_pro.set_vq_codebook_size(options.vq_codebook_size);
  janus_pro.set_vq_codebook_embed_dim(options.vq_codebook_embed_dim);
  janus_pro.set_image_size(options.image_size);
  janus_pro.set_patch_size(options.patch_size);
  janus_pro.set_num_image_tokens(options.num_image_tokens);
  janus_pro.set_cfg_weight(options.cfg_weight);
  janus_pro.set_image_temperature(options.image_temperature);
  janus_pro.set_generate_image(options.generate_image);
  janus_pro.set_max_new_tokens(options.max_new_tokens);
  if (!any2any_meta.mutable_any2any_model_type()
           ->mutable_custom_model()
           ->PackFrom(janus_pro)) {
    return absl::InternalError("Failed to pack JanusPro test metadata.");
  }
  return any2any_meta;
}

absl::StatusOr<std::string> WriteJanusProBundle(
    absl::string_view filename, const JanusProTestBundleOptions& options) {
  lm::proto::Any2AnyMetadata any2any_meta;
  if (options.custom_meta != nullptr) {
    any2any_meta = *options.custom_meta;
  } else {
    ABSL_ASSIGN_OR_RETURN(any2any_meta, BuildDefaultTestMetadata(options));
  }

  std::vector<BundleSectionPayload> extra_sections;
  if (options.include_vision_encoder) {
    extra_sections.push_back(BundleSectionPayload{
        .data_type = lm::schema::AnySectionDataType_TFLiteModel,
        .model_type = "tf_lite_vision_encoder",
        .bytes = BuildMinimalTfliteModelBuffer(
            {{"pixel_values",
              {1, 3, options.image_size, options.image_size},
              tflite::TensorType_FLOAT32}},
            {{"vision_embeds",
              {1, options.num_image_tokens, options.llm_hidden_size},
              tflite::TensorType_FLOAT32}}),
    });
  }

  if (options.include_image_gen_models) {
    std::vector<float> gen_head_logits(options.vq_codebook_size, -10.0f);
    if (options.gen_head_preferred_code >= 0 &&
        options.gen_head_preferred_code < options.vq_codebook_size) {
      gen_head_logits[options.gen_head_preferred_code] = 20.0f;
    }

    const size_t plane =
        static_cast<size_t>(options.image_size) * options.image_size;
    std::vector<float> vq_decoder_nchw(3 * plane, 0.0f);
    for (size_t i = 0; i < plane; ++i) {
      vq_decoder_nchw[i] = options.vq_decoder_r;
      vq_decoder_nchw[plane + i] = options.vq_decoder_g;
      vq_decoder_nchw[2 * plane + i] = options.vq_decoder_b;
    }

    extra_sections.push_back(BundleSectionPayload{
        .data_type = lm::schema::AnySectionDataType_TFLiteModel,
        .model_type = "tf_lite_vq_image_embedder",
        .bytes = BuildMinimalTfliteModelBuffer(
            {{"token_ids", {1, 1}, tflite::TensorType_INT32}},
            {{"embeds",
              {1, 1, options.llm_hidden_size},
              tflite::TensorType_FLOAT32}}),
    });
    extra_sections.push_back(BundleSectionPayload{
        .data_type = lm::schema::AnySectionDataType_TFLiteModel,
        .model_type = "tf_lite_gen_head",
        .bytes =
            BuildMinimalTfliteModelBuffer({{"hidden_states",
                                            {1, 1, options.llm_hidden_size},
                                            tflite::TensorType_FLOAT32}},
                                          {{"logits",
                                            {1, options.vq_codebook_size},
                                            tflite::TensorType_FLOAT32,
                                            std::move(gen_head_logits)}}),
    });
    extra_sections.push_back(BundleSectionPayload{
        .data_type = lm::schema::AnySectionDataType_TFLiteModel,
        .model_type = "tf_lite_vq_image_decoder",
        .bytes = BuildMinimalTfliteModelBuffer(
            {{"vq_codes",
              {1, options.num_image_tokens},
              tflite::TensorType_INT32}},
            {{"rgb_image",
              {1, 3, options.image_size, options.image_size},
              tflite::TensorType_FLOAT32,
              std::move(vq_decoder_nchw)}}),
    });
  }

  Any2AnyBundleOptions bundle_options{
      .metadata = std::move(any2any_meta),
      .tokenizer_json = std::string(kMinimalJanusProTokenizerJson),
      .extra_sections = std::move(extra_sections),
  };
  return WriteAny2AnyTestBundle(filename, std::move(bundle_options));
}

absl::StatusOr<std::string> WriteJanusProBundle(
    absl::string_view filename,
    const lm::proto::Any2AnyMetadata* absl_nullable custom_meta) {
  JanusProTestBundleOptions options;
  options.custom_meta = custom_meta;
  return WriteJanusProBundle(filename, options);
}

}  // namespace litert::omni::any2any
