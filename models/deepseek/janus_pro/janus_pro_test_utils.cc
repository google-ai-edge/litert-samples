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

#include "models/deepseek/janus_pro/janus_pro_test_utils.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <ios>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "models/deepseek/janus_pro/janus_pro.pb.h"
#include <gtest/gtest.h>
#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "flatbuffers/buffer.h"
#include "flatbuffers/flatbuffer_builder.h"
#include <cstring>
#include "runtime/proto/any2any_metadata.pb.h"
#include "runtime/proto/any2any_model_type.pb.h"
#include "schema/core/litertlm_header.h"
#include "schema/core/litertlm_header_schema_generated.h"
#include "schema/core/litertlm_section.h"
#include "stb_image_write.h"
#include "tflite/schema/schema_generated.h"

namespace litert::omni::any2any {
namespace {

constexpr absl::string_view kTestdataDir =
    "litert_lm/runtime/testdata";

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

std::string BuildMinimalTfliteModelBuffer(
    absl::Span<const TensorSpec> inputs, absl::Span<const TensorSpec> outputs) {
  flatbuffers::FlatBufferBuilder builder;

  auto opcode =
      tflite::CreateOperatorCode(builder, tflite::BuiltinOperator_ADD);
  auto opcodes_vec = builder.CreateVector({opcode});

  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
  buffers.push_back(tflite::CreateBuffer(builder));

  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors;
  std::vector<int32_t> sg_inputs;
  std::vector<int32_t> sg_outputs;
  std::vector<flatbuffers::Offset<tflite::TensorMap>> inputs_map;
  std::vector<flatbuffers::Offset<tflite::TensorMap>> outputs_map;
  std::vector<flatbuffers::Offset<tflite::Operator>> ops;

  for (const auto& in : inputs) {
    int32_t idx = static_cast<int32_t>(tensors.size());
    tensors.push_back(
        tflite::CreateTensor(builder, builder.CreateVector(in.dims), in.type,
                             /*buffer=*/0, builder.CreateString(in.name)));
    sg_inputs.push_back(idx);
    inputs_map.push_back(
        tflite::CreateTensorMap(builder, builder.CreateString(in.name), idx));
  }

  for (size_t j = 0; j < outputs.size(); ++j) {
    const auto& out = outputs[j];
    size_t num_elems = 1;
    for (int32_t d : out.dims) {
      num_elems *= static_cast<size_t>(d);
    }
    std::vector<uint8_t> const_a_bytes(num_elems * sizeof(float), 0);
    std::vector<uint8_t> const_b_bytes(num_elems * sizeof(float), 0);
    if (!out.float_values.empty()) {
      ABSL_CHECK_EQ(out.type, tflite::TensorType_FLOAT32);
      ABSL_CHECK_EQ(out.float_values.size(), num_elems);
      std::memcpy(const_a_bytes.data(), out.float_values.data(), num_elems * sizeof(float));
    }
    uint32_t buf_a_idx = static_cast<uint32_t>(buffers.size());
    buffers.push_back(
        tflite::CreateBuffer(builder, builder.CreateVector(const_a_bytes)));
    uint32_t buf_b_idx = static_cast<uint32_t>(buffers.size());
    buffers.push_back(
        tflite::CreateBuffer(builder, builder.CreateVector(const_b_bytes)));

    int32_t const_a_idx = static_cast<int32_t>(tensors.size());
    tensors.push_back(tflite::CreateTensor(
        builder, builder.CreateVector(out.dims), out.type, buf_a_idx,
        builder.CreateString(absl::StrCat("const_a_", j))));
    int32_t const_b_idx = static_cast<int32_t>(tensors.size());
    tensors.push_back(tflite::CreateTensor(
        builder, builder.CreateVector(out.dims), out.type, buf_b_idx,
        builder.CreateString(absl::StrCat("const_b_", j))));
    int32_t out_idx = static_cast<int32_t>(tensors.size());
    tensors.push_back(
        tflite::CreateTensor(builder, builder.CreateVector(out.dims), out.type,
                             /*buffer=*/0, builder.CreateString(out.name)));
    sg_outputs.push_back(out_idx);
    outputs_map.push_back(tflite::CreateTensorMap(
        builder, builder.CreateString(out.name), out_idx));

    std::vector<int32_t> op_inputs = {const_a_idx, const_b_idx};
    std::vector<int32_t> op_outputs = {out_idx};
    ops.push_back(tflite::CreateOperator(builder, /*opcode_index=*/0,
                                         builder.CreateVector(op_inputs),
                                         builder.CreateVector(op_outputs)));
  }

  auto subgraph = tflite::CreateSubGraph(
      builder, builder.CreateVector(tensors), builder.CreateVector(sg_inputs),
      builder.CreateVector(sg_outputs), builder.CreateVector(ops),
      builder.CreateString("main"));
  auto sig_def =
      tflite::CreateSignatureDef(builder, builder.CreateVector(inputs_map),
                                 builder.CreateVector(outputs_map),
                                 builder.CreateString("serving_default"),
                                 /*subgraph_index=*/0);
  auto model = tflite::CreateModel(
      builder, /*version=*/3, opcodes_vec, builder.CreateVector({subgraph}),
      builder.CreateString("test_model"), builder.CreateVector(buffers),
      /*metadata_buffer=*/0, /*metadata=*/0, builder.CreateVector({sig_def}));
  tflite::FinishModelBuffer(builder, model);

  return std::string(reinterpret_cast<const char*>(builder.GetBufferPointer()),
                     builder.GetSize());
}

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
  prototypes::litert_models::janus_pro::JanusPro janus_pro;
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
  const auto testdata_dir =
      std::filesystem::path(::testing::SrcDir()) / std::string(kTestdataDir);
  ABSL_ASSIGN_OR_RETURN(
      std::string llm_backbone_bytes,
      ReadFileBytes(testdata_dir / "checksum_fake_lm_embedding_input.tflite"));
  ABSL_ASSIGN_OR_RETURN(
      std::string text_embedder_bytes,
      ReadFileBytes(testdata_dir / "checksum_fake_lm_embedder.tflite"));

  lm::schema::ZlibBackendedSectionStream zlib_tok_stream(
      std::make_unique<lm::schema::StringBackedSectionStream>(
          std::string(kMinimalJanusProTokenizerJson)));
  ABSL_RETURN_IF_ERROR(zlib_tok_stream.Prepare());
  std::stringstream tok_ss;
  tok_ss << zlib_tok_stream.GetStream().rdbuf();
  std::string compressed_tokenizer_bytes = tok_ss.str();

  lm::proto::Any2AnyMetadata any2any_meta;
  if (options.custom_meta != nullptr) {
    any2any_meta = *options.custom_meta;
  } else {
    ABSL_ASSIGN_OR_RETURN(any2any_meta, BuildDefaultTestMetadata(options));
  }
  std::string any2any_meta_bytes = any2any_meta.SerializeAsString();

  struct SectionPayload {
    lm::schema::AnySectionDataType data_type;
    std::string model_type;
    std::string bytes;
  };
  std::vector<SectionPayload> payloads = {
      {lm::schema::AnySectionDataType_Any2AnyMetadataProto, "",
       std::move(any2any_meta_bytes)},
      {lm::schema::AnySectionDataType_HF_Tokenizer_Zlib, "",
       std::move(compressed_tokenizer_bytes)},
      {lm::schema::AnySectionDataType_TFLiteModel, "tf_lite_text_embedder",
       std::move(text_embedder_bytes)},
      {lm::schema::AnySectionDataType_TFLiteModel, "tf_lite_llm_backbone",
       std::move(llm_backbone_bytes)},
  };

  if (options.include_vision_encoder) {
    payloads.push_back(
        {lm::schema::AnySectionDataType_TFLiteModel, "tf_lite_vision_encoder",
         BuildMinimalTfliteModelBuffer(
             {{"pixel_values",
               {1, 3, options.image_size, options.image_size},
               tflite::TensorType_FLOAT32}},
             {{"vision_embeds",
               {1, options.num_image_tokens, options.llm_hidden_size},
               tflite::TensorType_FLOAT32}})});
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

    payloads.push_back({lm::schema::AnySectionDataType_TFLiteModel,
                        "tf_lite_vq_image_embedder",
                        BuildMinimalTfliteModelBuffer(
                            {{"token_ids", {1, 1}, tflite::TensorType_INT32}},
                            {{"embeds",
                              {1, 1, options.llm_hidden_size},
                              tflite::TensorType_FLOAT32}})});
    payloads.push_back(
        {lm::schema::AnySectionDataType_TFLiteModel, "tf_lite_gen_head",
         BuildMinimalTfliteModelBuffer({{"hidden_states",
                                         {1, 1, options.llm_hidden_size},
                                         tflite::TensorType_FLOAT32}},
                                       {{"logits",
                                         {1, options.vq_codebook_size},
                                         tflite::TensorType_FLOAT32,
                                         std::move(gen_head_logits)}})});
    payloads.push_back({lm::schema::AnySectionDataType_TFLiteModel,
                        "tf_lite_vq_image_decoder",
                        BuildMinimalTfliteModelBuffer(
                            {{"vq_codes",
                              {1, options.num_image_tokens},
                              tflite::TensorType_INT32}},
                            {{"rgb_image",
                              {1, 3, options.image_size, options.image_size},
                              tflite::TensorType_FLOAT32,
                              std::move(vq_decoder_nchw)}})});
  }

  constexpr uint64_t kDataStartOffset = 16384;
  flatbuffers::FlatBufferBuilder builder(2048);
  std::vector<flatbuffers::Offset<lm::schema::SectionObject>> section_objects;
  uint64_t cursor = kDataStartOffset;
  for (const auto& p : payloads) {
    std::vector<flatbuffers::Offset<lm::schema::KeyValuePair>> items;
    if (!p.model_type.empty()) {
      items.push_back(
          lm::schema::CreateKeyValuePair(builder, "model_type", p.model_type));
    }
    auto items_vec = items.empty() ? 0 : builder.CreateVector(items);
    const uint64_t begin = cursor;
    const uint64_t end = begin + p.bytes.size();
    section_objects.push_back(lm::schema::CreateSectionObject(
        builder, items_vec, begin, end, p.data_type));
    cursor = end;
  }
  auto section_metadata = lm::schema::CreateSectionMetadata(
      builder, builder.CreateVector(section_objects));
  auto metadata =
      lm::schema::CreateLiteRTLMMetaData(builder, 0, section_metadata);
  builder.Finish(metadata);

  const auto out_path =
      std::filesystem::path(::testing::TempDir()) / std::string(filename);
  std::ofstream file(out_path, std::ios::binary);
  if (!file.good()) {
    return absl::InternalError("Failed to open test .litertlm file.");
  }
  file.write("LITERTLM", 8);
  const uint32_t major_version = 1;
  const uint32_t minor_version = 0;
  const uint32_t patch_version = 0;
  const uint32_t padding = 0;
  file.write(reinterpret_cast<const char*>(&major_version), 4);
  file.write(reinterpret_cast<const char*>(&minor_version), 4);
  file.write(reinterpret_cast<const char*>(&patch_version), 4);
  file.write(reinterpret_cast<const char*>(&padding), 4);
  const uint64_t header_end_offset = 32 + builder.GetSize();
  if (header_end_offset > kDataStartOffset) {
    return absl::InternalError(
        "Test .litertlm header exceeds kDataStartOffset.");
  }
  file.write(reinterpret_cast<const char*>(&header_end_offset), 8);
  file.write(reinterpret_cast<const char*>(builder.GetBufferPointer()),
             builder.GetSize());
  std::string header_pad(kDataStartOffset - header_end_offset, '\0');
  file.write(header_pad.data(), header_pad.size());
  for (const auto& p : payloads) {
    file.write(p.bytes.data(), p.bytes.size());
  }
  if (!file.good()) {
    return absl::InternalError("Failed to write test .litertlm file.");
  }
  return out_path.string();
}

absl::StatusOr<std::string> WriteJanusProBundle(
    absl::string_view filename,
    const lm::proto::Any2AnyMetadata* absl_nullable custom_meta) {
  JanusProTestBundleOptions options;
  options.custom_meta = custom_meta;
  return WriteJanusProBundle(filename, options);
}

}  // namespace litert::omni::any2any
