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

#include "models/omnigen/omnigen2/runner/omnigen2_test_utils.h"

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

#include "models/omnigen/omnigen2/runner/omnigen2.pb.h"
#include "models/omnigen/omnigen2/runner/omnigen2_model_config.h"
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "flatbuffers/buffer.h"
#include "flatbuffers/flatbuffer_builder.h"
#include "omni/any2any/any2any_test_utils.h"
#include "omni/base/custom_model_utils.h"
#include "runtime/proto/image_gen_model_type.pb.h"
#include "runtime/proto/text2image_metadata.pb.h"
#include "schema/core/litertlm_header.h"
#include "schema/core/litertlm_header_schema_generated.h"
#include "schema/core/litertlm_section.h"
#include "tflite/schema/schema_generated.h"

namespace litert::omni::text2image::omnigen2 {
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
      "system": 4,
      "user": 5,
      "You": 6,
      "are": 7,
      "a": 8,
      "helpful": 9,
      "assistant": 10,
      "that": 11,
      "generates": 12,
      "high-quality": 13,
      "images": 14,
      "based": 15,
      "on": 16,
      "instructions.": 17,
      "cat": 18,
      "robot": 19,
      "painting": 20,
      "canvas": 21,
      "sunset": 22,
      "mountain": 23
    },
    "unk_token": "<|endoftext|>"
  }
})json";

uint32_t ReadBe32(const char* p) {
  return (static_cast<uint32_t>(static_cast<uint8_t>(p[0])) << 24) |
         (static_cast<uint32_t>(static_cast<uint8_t>(p[1])) << 16) |
         (static_cast<uint32_t>(static_cast<uint8_t>(p[2])) << 8) |
         static_cast<uint32_t>(static_cast<uint8_t>(p[3]));
}

uint64_t AlignUp(uint64_t offset, uint64_t alignment) {
  return (offset + alignment - 1) & ~(alignment - 1);
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

absl::StatusOr<ParsedPngHeader> ParsePngHeader(
    const std::filesystem::path& path) {
  ABSL_ASSIGN_OR_RETURN(std::string bytes, ReadFileBytes(path));
  static constexpr unsigned char kPngSig[8] = {0x89, 'P',  'N',  'G',
                                               0x0d, 0x0a, 0x1a, 0x0a};
  if (bytes.size() < 33 || std::memcmp(bytes.data(), kPngSig, 8) != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("File does not have a valid PNG header: ", path.string()));
  }
  const uint32_t ihdr_len = ReadBe32(bytes.data() + 8);
  if (ihdr_len < 13 || std::memcmp(bytes.data() + 12, "IHDR", 4) != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("PNG missing valid IHDR chunk: ", path.string()));
  }
  ParsedPngHeader header;
  header.width = ReadBe32(bytes.data() + 16);
  header.height = ReadBe32(bytes.data() + 20);
  header.bit_depth = static_cast<uint8_t>(bytes[24]);
  header.color_type = static_cast<uint8_t>(bytes[25]);
  return header;
}

absl::StatusOr<lm::proto::Text2ImageMetadata> BuildDefaultTestMetadata(
    const OmniGen2TestBundleOptions& options) {
  lm::proto::Text2ImageMetadata meta;
  lm::proto::OmniGen2 omnigen2;
  omnigen2.set_seq_len(options.seq_len);
  omnigen2.set_img_size(options.img_size);
  omnigen2.set_in_channels(options.in_channels);
  omnigen2.set_patch_size(options.patch_size);
  omnigen2.set_hidden_size(options.dit_hidden_size);
  omnigen2.set_prompt_dim(options.text_hidden_size);
  omnigen2.set_default_steps(options.steps);
  omnigen2.set_vae_scaling_factor(0.3611f);
  omnigen2.set_vae_shift_factor(0.1159f);
  omnigen2.set_timestep_scale(1000.0f);
  for (int axis_dim : options.axes_dims_rope) {
    omnigen2.add_axes_dim_rope(axis_dim);
  }
  omnigen2.set_dynamic_time_shift(true);
  omnigen2.set_text_guidance_scale(options.text_guidance_scale);

  ABSL_RETURN_IF_ERROR(EnsureDefaultCustomModel<lm::proto::OmniGen2>(
      *meta.mutable_image_gen_model_type(), omnigen2));
  return meta;
}

absl::StatusOr<std::string> WriteOmniGen2Bundle(
    absl::string_view filename, const OmniGen2TestBundleOptions& options) {
  const int latent_dim = options.img_size / 8;
  const int grid_dim = latent_dim / options.patch_size;
  const int img_tokens = grid_dim * grid_dim;
  const int joint_tokens = options.seq_len + img_tokens;
  const int packed_ch =
      options.in_channels * options.patch_size * options.patch_size;
  const int head_dim = options.axes_dims_rope[0] + options.axes_dims_rope[1] +
                       options.axes_dims_rope[2];

  struct SectionPayload {
    lm::schema::AnySectionDataType data_type;
    std::string kv_key;
    std::string kv_val;
    std::string bytes;
  };
  std::vector<SectionPayload> payloads;

  auto maybe_add = [&](lm::schema::AnySectionDataType data_type,
                       std::string kv_key, std::string kv_val,
                       std::string bytes) {
    if (!options.omit_section.empty() && kv_val == options.omit_section) {
      return;
    }
    payloads.push_back(
        {data_type, std::move(kv_key), std::move(kv_val), std::move(bytes)});
  };

  if (options.include_metadata_section) {
    lm::proto::Text2ImageMetadata meta;
    if (options.custom_meta != nullptr) {
      meta = *options.custom_meta;
    } else {
      ABSL_ASSIGN_OR_RETURN(meta, BuildDefaultTestMetadata(options));
    }
    maybe_add(lm::schema::AnySectionDataType_Text2ImageMetadataProto, "", "",
              meta.SerializeAsString());
  }

  if (options.include_tokenizer_section) {
    lm::schema::ZlibBackendedSectionStream zlib_tok_stream(
        std::make_unique<lm::schema::StringBackedSectionStream>(
            std::string(kMinimalTokenizerJson)));
    ABSL_RETURN_IF_ERROR(zlib_tok_stream.Prepare());
    std::stringstream tok_ss;
    tok_ss << zlib_tok_stream.GetStream().rdbuf();
    maybe_add(lm::schema::AnySectionDataType_HF_Tokenizer_Zlib, "", "",
              tok_ss.str());
  }

  // 1. Standard TFLiteModel section: `tf_lite_text_encoder`.
  std::vector<float> text_out_vals(
      static_cast<size_t>(options.seq_len) * options.text_hidden_size, 0.25f);
  maybe_add(lm::schema::AnySectionDataType_TFLiteModel, "model_type",
            "tf_lite_text_encoder",
            any2any::BuildMinimalTfliteModelBuffer(
                {{"args_0", {1, options.seq_len}, options.text_input_type},
                 {"args_1", {1, options.seq_len}, options.text_input_type}},
                {{"prompt_embeds",
                  {1, options.seq_len, options.text_hidden_size},
                  tflite::TensorType_FLOAT32,
                  std::move(text_out_vals)}}));

  // 2. GenericBinaryData shard: `dit_cap_embedder`.
  maybe_add(lm::schema::AnySectionDataType_GenericBinaryData, "name",
            std::string(kSectionDitCapEmbedder),
            any2any::BuildMinimalTfliteModelBuffer(
                {{"args_0",
                  {1, options.seq_len, options.text_hidden_size},
                  tflite::TensorType_FLOAT32}},
                {{"cap_emb",
                  {1, options.seq_len, options.dit_hidden_size},
                  tflite::TensorType_FLOAT32}}));

  // 3. GenericBinaryData shard: `dit_context_refiner`.
  maybe_add(
      lm::schema::AnySectionDataType_GenericBinaryData, "name",
      std::string(kSectionDitContextRefiner),
      any2any::BuildMinimalTfliteModelBuffer(
          {{"args_0",
            {1, options.seq_len, options.dit_hidden_size},
            tflite::TensorType_FLOAT32},
           {"args_1",
            {1, options.seq_len, 1, head_dim},
            tflite::TensorType_FLOAT32},
           {"args_2",
            {1, options.seq_len, 1, head_dim},
            tflite::TensorType_FLOAT32},
           {"args_3", {1, 1, 1, options.seq_len}, tflite::TensorType_FLOAT32}},
          {{"cap_refined",
            {1, options.seq_len, options.dit_hidden_size},
            tflite::TensorType_FLOAT32}}));

  // 4. GenericBinaryData shard: `dit_x_embedder`.
  maybe_add(
      lm::schema::AnySectionDataType_GenericBinaryData, "name",
      std::string(kSectionDitXEmbedder),
      any2any::BuildMinimalTfliteModelBuffer(
          {{"args_0", {1, img_tokens, packed_ch}, tflite::TensorType_FLOAT32},
           {"args_1", {1}, tflite::TensorType_FLOAT32}},
          {{"x_emb",
            {1, img_tokens, options.dit_hidden_size},
            tflite::TensorType_FLOAT32},
           {"temb", {1, options.temb_dim}, tflite::TensorType_FLOAT32}}));

  // 5. GenericBinaryData shard: `dit_noise_refiner`.
  maybe_add(
      lm::schema::AnySectionDataType_GenericBinaryData, "name",
      std::string(kSectionDitNoiseRefiner),
      any2any::BuildMinimalTfliteModelBuffer(
          {{"args_0",
            {1, img_tokens, options.dit_hidden_size},
            tflite::TensorType_FLOAT32},
           {"args_1", {1, img_tokens, 1, head_dim}, tflite::TensorType_FLOAT32},
           {"args_2", {1, img_tokens, 1, head_dim}, tflite::TensorType_FLOAT32},
           {"args_3", {1, options.temb_dim}, tflite::TensorType_FLOAT32}},
          {{"x_refined",
            {1, img_tokens, options.dit_hidden_size},
            tflite::TensorType_FLOAT32}}));

  // 6. Main blocks 0..3 (standard TFLiteModel) and 4..5 (GenericBinaryData).
  const std::string main_block_tflite = any2any::BuildMinimalTfliteModelBuffer(
      {{"args_0",
        {1, joint_tokens, options.dit_hidden_size},
        tflite::TensorType_FLOAT32},
       {"args_1", {1, joint_tokens, 1, head_dim}, tflite::TensorType_FLOAT32},
       {"args_2", {1, joint_tokens, 1, head_dim}, tflite::TensorType_FLOAT32},
       {"args_3", {1, 1, 1, joint_tokens}, tflite::TensorType_FLOAT32},
       {"args_4", {1, options.temb_dim}, tflite::TensorType_FLOAT32}},
      {{"joint_out",
        {1, joint_tokens, options.dit_hidden_size},
        tflite::TensorType_FLOAT32}});
  for (int i = 0; i < 4; ++i) {
    maybe_add(lm::schema::AnySectionDataType_TFLiteModel, "model_type",
              absl::StrCat("tf_lite_diffusion_transformer_single_block_", i),
              main_block_tflite);
  }
  maybe_add(lm::schema::AnySectionDataType_GenericBinaryData, "name",
            std::string(kSectionDitSingleBlock4), main_block_tflite);
  maybe_add(lm::schema::AnySectionDataType_GenericBinaryData, "name",
            std::string(kSectionDitSingleBlock5), main_block_tflite);

  // 7. Standard TFLiteModel section: `tf_lite_diffusion_transformer_final`.
  std::vector<float> vel_vals(static_cast<size_t>(img_tokens) * packed_ch,
                              0.1f);
  maybe_add(lm::schema::AnySectionDataType_TFLiteModel, "model_type",
            "tf_lite_diffusion_transformer_final",
            any2any::BuildMinimalTfliteModelBuffer(
                {{"args_0",
                  {1, joint_tokens, options.dit_hidden_size},
                  tflite::TensorType_FLOAT32},
                 {"args_1", {1, options.temb_dim}, tflite::TensorType_FLOAT32}},
                {{"velocity",
                  {1, img_tokens, packed_ch},
                  tflite::TensorType_FLOAT32,
                  std::move(vel_vals)}}));

  // 8. Standard TFLiteModel section: `tf_lite_image_decoder`.
  std::vector<float> vae_vals(
      static_cast<size_t>(3) * options.img_size * options.img_size,
      options.vae_output_value);
  maybe_add(lm::schema::AnySectionDataType_TFLiteModel, "model_type",
            "tf_lite_image_decoder",
            any2any::BuildMinimalTfliteModelBuffer(
                {{"args_0",
                  {1, options.in_channels, latent_dim, latent_dim},
                  tflite::TensorType_FLOAT32}},
                {{"image",
                  {1, 3, options.img_size, options.img_size},
                  tflite::TensorType_FLOAT32,
                  std::move(vae_vals)}}));

  constexpr uint64_t kDataStartOffset = 16384;
  constexpr uint64_t kSectionAlignment = 64;
  flatbuffers::FlatBufferBuilder builder(4096);
  std::vector<flatbuffers::Offset<lm::schema::SectionObject>> section_objects;
  std::vector<uint64_t> section_offsets;
  section_objects.reserve(payloads.size());
  section_offsets.reserve(payloads.size());

  uint64_t cursor = kDataStartOffset;
  for (const auto& p : payloads) {
    cursor = AlignUp(cursor, kSectionAlignment);
    std::vector<flatbuffers::Offset<lm::schema::KeyValuePair>> items;
    if (!p.kv_key.empty()) {
      items.push_back(
          lm::schema::CreateKeyValuePair(builder, p.kv_key, p.kv_val));
    }
    auto items_vec = items.empty() ? 0 : builder.CreateVector(items);
    const uint64_t begin = cursor;
    const uint64_t end = begin + p.bytes.size();
    section_objects.push_back(lm::schema::CreateSectionObject(
        builder, items_vec, begin, end, p.data_type));
    section_offsets.push_back(begin);
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

  uint64_t written = header_end_offset;
  for (size_t i = 0; i < payloads.size(); ++i) {
    const uint64_t pad_len = section_offsets[i] - written;
    if (pad_len > 0) {
      std::string pad(pad_len, '\0');
      file.write(pad.data(), pad.size());
      written += pad_len;
    }
    file.write(payloads[i].bytes.data(), payloads[i].bytes.size());
    written += payloads[i].bytes.size();
  }
  if (!file.good()) {
    return absl::InternalError("Failed to write test .litertlm file.");
  }
  return out_path.string();
}

}  // namespace litert::omni::text2image::omnigen2
