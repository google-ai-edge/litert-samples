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

#include "models/tongyi/z_image_turbo/z_image_turbo_test_utils.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <ios>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "models/tongyi/z_image_turbo/z_image_turbo.pb.h"
#include "models/tongyi/z_image_turbo/z_image_turbo_model_config.h"
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
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_types.h"
#include "runtime/proto/image_gen_model_type.pb.h"
#include "runtime/proto/text2image_metadata.pb.h"
#include "schema/core/litertlm_header.h"
#include "schema/core/litertlm_header_schema_generated.h"
#include "schema/core/litertlm_section.h"
#include "tflite/schema/schema_generated.h"

namespace litert::omni::text2image {
namespace {

constexpr absl::string_view kMinimalTokenizerJson = R"json({
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "<|endoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 1, "content": "<|im_start|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 2, "content": "<|im_end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 3, "content": "\n", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
      "<|endoftext|>": 0,
      "<|im_start|>": 1,
      "<|im_end|>": 2,
      "\n": 3,
      "assistant": 4,
      "user": 5,
      "a": 6,
      "red": 7,
      "fox": 8,
      "snow": 9,
      "sunset": 10,
      "cat": 11,
      "dog": 12,
      "hello": 13,
      "world": 14,
      "image": 15
    },
    "unk_token": "<|endoftext|>"
  }
})json";

uint32_t ReadBe32(const char* ptr) {
  return (static_cast<uint32_t>(static_cast<uint8_t>(ptr[0])) << 24) |
         (static_cast<uint32_t>(static_cast<uint8_t>(ptr[1])) << 16) |
         (static_cast<uint32_t>(static_cast<uint8_t>(ptr[2])) << 8) |
         static_cast<uint32_t>(static_cast<uint8_t>(ptr[3]));
}

uint64_t AlignUp(const uint64_t value, const uint64_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

}  // namespace

ZImageTurboModelConfig MakeTinyZImageTurboConfig() {
  ZImageTurboModelConfig cfg;
  cfg.seq_len = 8;
  cfg.img_size = 16;
  cfg.in_channels = 2;
  cfg.patch_size = 2;
  cfg.dim = 12;
  cfg.prompt_dim = 8;
  cfg.steps = 2;
  cfg.axes_dims = {4, 4, 4};
  cfg.rope_theta = 256.0f;
  cfg.vae_scaling_factor = 0.5f;
  cfg.vae_shift_factor = 0.1f;
  cfg.pad_token_id = 0;
  cfg.prefix_token_ids = {1};
  cfg.suffix_token_ids = {2, 3, 1, 4, 3};
  return cfg;
}

absl::StatusOr<TensorBuffer> MakeFloatTensorBuffer(Environment& env,
                                                   absl::Span<const int> dims) {
  size_t total = 1;
  for (int d : dims) {
    total *= static_cast<size_t>(d);
  }
  RankedTensorType tensor_type(ElementType::Float32,
                               Layout(Dimensions(dims.begin(), dims.end())));
  LITERT_ASSIGN_OR_RETURN(
      TensorBuffer buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory,
                                  tensor_type, total * sizeof(float)));
  std::vector<float> zeros(total, 0.0f);
  LITERT_RETURN_IF_ERROR(buf.Write<float>(zeros));
  return buf;
}

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

  for (const TensorSpec& in : inputs) {
    const int32_t idx = static_cast<int32_t>(tensors.size());
    tensors.push_back(
        tflite::CreateTensor(builder, builder.CreateVector(in.dims), in.type,
                             /*buffer=*/0, builder.CreateString(in.name)));
    sg_inputs.push_back(idx);
    inputs_map.push_back(
        tflite::CreateTensorMap(builder, builder.CreateString(in.name), idx));
  }

  for (size_t j = 0; j < outputs.size(); ++j) {
    const TensorSpec& out = outputs[j];
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
    const uint32_t buf_a_idx = static_cast<uint32_t>(buffers.size());
    buffers.push_back(
        tflite::CreateBuffer(builder, builder.CreateVector(const_a_bytes)));
    const uint32_t buf_b_idx = static_cast<uint32_t>(buffers.size());
    buffers.push_back(
        tflite::CreateBuffer(builder, builder.CreateVector(const_b_bytes)));

    const int32_t const_a_idx = static_cast<int32_t>(tensors.size());
    tensors.push_back(tflite::CreateTensor(
        builder, builder.CreateVector(out.dims), out.type, buf_a_idx,
        builder.CreateString(absl::StrCat("const_a_", j))));
    const int32_t const_b_idx = static_cast<int32_t>(tensors.size());
    tensors.push_back(tflite::CreateTensor(
        builder, builder.CreateVector(out.dims), out.type, buf_b_idx,
        builder.CreateString(absl::StrCat("const_b_", j))));
    const int32_t out_idx = static_cast<int32_t>(tensors.size());
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

absl::StatusOr<ParsedPngHeader> ParsePngHeader(
    const std::filesystem::path& path) {
  ABSL_ASSIGN_OR_RETURN(std::string bytes, ReadFileBytes(path));
  constexpr char kPngSignature[8] = {'\x89', 'P',  'N',    'G',
                                     '\r',   '\n', '\x1a', '\n'};
  if (bytes.size() < 33 || std::memcmp(bytes.data(), kPngSignature, 8) != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("File is not a valid PNG: ", path.string()));
  }
  const uint32_t ihdr_len = ReadBe32(bytes.data() + 8);
  if (ihdr_len != 13 || absl::string_view(bytes).substr(12, 4) != "IHDR") {
    return absl::InvalidArgumentError(
        absl::StrCat("PNG file missing valid IHDR chunk: ", path.string()));
  }
  ParsedPngHeader header;
  header.width = ReadBe32(bytes.data() + 16);
  header.height = ReadBe32(bytes.data() + 20);
  header.bit_depth = static_cast<uint8_t>(bytes[24]);
  header.color_type = static_cast<uint8_t>(bytes[25]);
  return header;
}

absl::StatusOr<lm::proto::Text2ImageMetadata> BuildDefaultZImageTurboMetadata(
    const ZImageTurboModelConfig& config) {
  prototypes::litert_models::z_image_turbo::ZImageTurbo z_proto;
  z_proto.set_seq_len(config.seq_len);
  z_proto.set_img_size(config.img_size);
  z_proto.set_in_channels(config.in_channels);
  z_proto.set_patch_size(config.patch_size);
  z_proto.set_dim(config.dim);
  z_proto.set_prompt_dim(config.prompt_dim);
  z_proto.set_default_steps(config.steps);
  z_proto.set_vae_scaling_factor(config.vae_scaling_factor);
  z_proto.set_vae_shift_factor(config.vae_shift_factor);
  for (int axis_dim : config.axes_dims) {
    z_proto.add_axes_dims(axis_dim);
  }
  z_proto.set_rope_theta(config.rope_theta);
  z_proto.set_shift(config.shift);
  z_proto.set_use_dynamic_shifting(config.use_dynamic_shifting);
  z_proto.set_pad_token_id(config.pad_token_id);
  for (int32_t tok : config.prefix_token_ids) {
    z_proto.add_prefix_token_ids(tok);
  }
  for (int32_t tok : config.suffix_token_ids) {
    z_proto.add_suffix_token_ids(tok);
  }

  lm::proto::Text2ImageMetadata meta;
  if (!meta.mutable_image_gen_model_type()->mutable_custom_model()->PackFrom(
          z_proto)) {
    return absl::InternalError("Failed to pack ZImageTurbo test metadata.");
  }
  return meta;
}

absl::StatusOr<std::string> WriteZImageTurboBundle(
    absl::string_view filename, const ZImageTurboTestBundleOptions& options) {
  const ZImageTurboModelConfig& cfg = options.config;
  const int seq_len = cfg.seq_len;
  const int img_tokens = cfg.ImageTokens();
  const int total_tokens = cfg.TotalTokens();
  const int dim = cfg.dim;
  const int prompt_dim = cfg.prompt_dim;
  const int packed_ch = cfg.PackedChannels();
  const int head_dim = cfg.HeadDim();
  const int adaln_dim = cfg.AdalnDim();
  const int latent_size = cfg.LatentSize();

  lm::schema::ZlibBackendedSectionStream zlib_tok_stream(
      std::make_unique<lm::schema::StringBackedSectionStream>(
          std::string(kMinimalTokenizerJson)));
  ABSL_RETURN_IF_ERROR(zlib_tok_stream.Prepare());
  std::stringstream tok_ss;
  tok_ss << zlib_tok_stream.GetStream().rdbuf();
  std::string compressed_tokenizer_bytes = tok_ss.str();

  lm::proto::Text2ImageMetadata t2i_meta;
  if (options.custom_meta != nullptr) {
    t2i_meta = *options.custom_meta;
  } else {
    ABSL_ASSIGN_OR_RETURN(t2i_meta, BuildDefaultZImageTurboMetadata(cfg));
  }
  std::string t2i_meta_bytes = t2i_meta.SerializeAsString();

  struct SectionPayload {
    lm::schema::AnySectionDataType data_type;
    std::string meta_key;
    std::string meta_value;
    std::string bytes;
  };

  std::vector<float> prompt_embed_vals(
      static_cast<size_t>(seq_len) * static_cast<size_t>(prompt_dim),
      options.prompt_embed_value);
  std::vector<float> final_vel_vals(
      static_cast<size_t>(img_tokens) * static_cast<size_t>(packed_ch),
      options.final_velocity_value);
  std::vector<float> vae_rgb_vals(static_cast<size_t>(3) *
                                      static_cast<size_t>(cfg.img_size) *
                                      static_cast<size_t>(cfg.img_size),
                                  options.vae_rgb_value);

  std::vector<SectionPayload> payloads = {
      {lm::schema::AnySectionDataType_Text2ImageMetadataProto, "", "",
       std::move(t2i_meta_bytes)},
      {lm::schema::AnySectionDataType_HF_Tokenizer_Zlib, "", "",
       std::move(compressed_tokenizer_bytes)},
      {lm::schema::AnySectionDataType_TFLiteModel, "model_type",
       "tf_lite_text_encoder",
       BuildMinimalTfliteModelBuffer(
           {{"input_ids", {1, seq_len}, tflite::TensorType_INT32},
            {"attention_mask", {1, seq_len}, tflite::TensorType_INT32}},
           {{"prompt_embeds",
             {1, seq_len, prompt_dim},
             tflite::TensorType_FLOAT32,
             std::move(prompt_embed_vals)}})},
      {lm::schema::AnySectionDataType_GenericBinaryData, "name",
       std::string(z_image_turbo::kCapEmbedderSectionName),
       BuildMinimalTfliteModelBuffer(
           {{"cap_feats",
             {1, seq_len, prompt_dim},
             tflite::TensorType_FLOAT32}},
           {{"cap", {1, seq_len, dim}, tflite::TensorType_FLOAT32}})},
      {lm::schema::AnySectionDataType_GenericBinaryData, "name",
       std::string(z_image_turbo::kContextRefinerSectionName),
       BuildMinimalTfliteModelBuffer(
           {{"cap", {1, seq_len, dim}, tflite::TensorType_FLOAT32},
            {"cos", {1, seq_len, 1, head_dim}, tflite::TensorType_FLOAT32},
            {"sin", {1, seq_len, 1, head_dim}, tflite::TensorType_FLOAT32}},
           {{"cap_refined", {1, seq_len, dim}, tflite::TensorType_FLOAT32}})},
      {lm::schema::AnySectionDataType_GenericBinaryData, "name",
       std::string(z_image_turbo::kXEmbedderSectionName),
       BuildMinimalTfliteModelBuffer(
           {{"x", {1, img_tokens, packed_ch}, tflite::TensorType_FLOAT32},
            {"t", {1}, tflite::TensorType_FLOAT32}},
           {{"x_emb", {1, img_tokens, dim}, tflite::TensorType_FLOAT32},
            {"adaln", {1, adaln_dim}, tflite::TensorType_FLOAT32}})},
      {lm::schema::AnySectionDataType_GenericBinaryData, "name",
       std::string(z_image_turbo::kNoiseRefinerSectionName),
       BuildMinimalTfliteModelBuffer(
           {{"x_emb", {1, img_tokens, dim}, tflite::TensorType_FLOAT32},
            {"cos", {1, img_tokens, 1, head_dim}, tflite::TensorType_FLOAT32},
            {"sin", {1, img_tokens, 1, head_dim}, tflite::TensorType_FLOAT32},
            {"adaln", {1, adaln_dim}, tflite::TensorType_FLOAT32}},
           {{"x_refined", {1, img_tokens, dim}, tflite::TensorType_FLOAT32}})},
  };

  const int num_shards = options.num_main_shards;
  for (int i = 0; i < num_shards && i < 6; ++i) {
    std::string main_bytes = BuildMinimalTfliteModelBuffer(
        {{"unified", {1, total_tokens, dim}, tflite::TensorType_FLOAT32},
         {"cos", {1, total_tokens, 1, head_dim}, tflite::TensorType_FLOAT32},
         {"sin", {1, total_tokens, 1, head_dim}, tflite::TensorType_FLOAT32},
         {"adaln", {1, adaln_dim}, tflite::TensorType_FLOAT32}},
        {{"unified_out", {1, total_tokens, dim}, tflite::TensorType_FLOAT32}});
    if (i < 4) {
      payloads.push_back(
          {lm::schema::AnySectionDataType_TFLiteModel, "model_type",
           absl::StrCat("tf_lite_diffusion_transformer_single_block_", i),
           std::move(main_bytes)});
    } else {
      payloads.push_back(
          {lm::schema::AnySectionDataType_GenericBinaryData, "name",
           std::string(i == 4 ? z_image_turbo::kSingleBlock4SectionName
                              : z_image_turbo::kSingleBlock5SectionName),
           std::move(main_bytes)});
    }
  }

  payloads.push_back(
      {lm::schema::AnySectionDataType_TFLiteModel, "model_type",
       "tf_lite_diffusion_transformer_final",
       BuildMinimalTfliteModelBuffer(
           {{"unified", {1, total_tokens, dim}, tflite::TensorType_FLOAT32},
            {"adaln", {1, adaln_dim}, tflite::TensorType_FLOAT32}},
           {{"model_out",
             {1, img_tokens, packed_ch},
             tflite::TensorType_FLOAT32,
             std::move(final_vel_vals)}})});
  payloads.push_back({lm::schema::AnySectionDataType_TFLiteModel, "model_type",
                      "tf_lite_image_decoder",
                      BuildMinimalTfliteModelBuffer(
                          {{"latents",
                            {1, cfg.in_channels, latent_size, latent_size},
                            tflite::TensorType_FLOAT32}},
                          {{"rgb",
                            {1, 3, cfg.img_size, cfg.img_size},
                            tflite::TensorType_FLOAT32,
                            std::move(vae_rgb_vals)}})});

  constexpr uint64_t kDataStartOffset = 16384;
  constexpr uint64_t kSectionAlignment = 16;
  flatbuffers::FlatBufferBuilder builder(4096);
  std::vector<flatbuffers::Offset<lm::schema::SectionObject>> section_objects;
  std::vector<uint64_t> section_begins;
  section_begins.reserve(payloads.size());

  uint64_t cursor = kDataStartOffset;
  for (const SectionPayload& p : payloads) {
    cursor = AlignUp(cursor, kSectionAlignment);
    std::vector<flatbuffers::Offset<lm::schema::KeyValuePair>> items;
    if (!p.meta_key.empty()) {
      items.push_back(
          lm::schema::CreateKeyValuePair(builder, p.meta_key, p.meta_value));
    }
    auto items_vec = items.empty() ? 0 : builder.CreateVector(items);
    const uint64_t begin = cursor;
    const uint64_t end = begin + p.bytes.size();
    section_begins.push_back(begin);
    section_objects.push_back(lm::schema::CreateSectionObject(
        builder, items_vec, begin, end, p.data_type));
    cursor = end;
  }
  auto section_metadata = lm::schema::CreateSectionMetadata(
      builder, builder.CreateVector(section_objects));
  auto metadata =
      lm::schema::CreateLiteRTLMMetaData(builder, 0, section_metadata);
  builder.Finish(metadata);

  const std::filesystem::path out_path =
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
  uint64_t file_pos = header_end_offset;
  for (size_t i = 0; i < payloads.size(); ++i) {
    if (section_begins[i] > file_pos) {
      std::string pad(section_begins[i] - file_pos, '\0');
      file.write(pad.data(), pad.size());
      file_pos = section_begins[i];
    }
    file.write(payloads[i].bytes.data(), payloads[i].bytes.size());
    file_pos += payloads[i].bytes.size();
  }
  if (!file.good()) {
    return absl::InternalError("Failed to write test .litertlm file.");
  }
  return out_path.string();
}

absl::StatusOr<std::string> WriteZImageTurboBundle(
    absl::string_view filename,
    const lm::proto::Text2ImageMetadata* absl_nullable custom_meta) {
  ZImageTurboTestBundleOptions options;
  options.config = MakeTinyZImageTurboConfig();
  options.custom_meta = custom_meta;
  return WriteZImageTurboBundle(filename, options);
}

}  // namespace litert::omni::text2image
