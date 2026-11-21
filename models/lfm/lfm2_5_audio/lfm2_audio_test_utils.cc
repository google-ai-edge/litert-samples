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

#include "models/lfm/lfm2_5_audio/lfm2_audio_test_utils.h"

#include <algorithm>
#include <cmath>
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

#include "models/lfm/lfm2_5_audio/lfm2_audio.pb.h"
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
#include "tflite/schema/schema_generated.h"

namespace litert::omni::any2any {
namespace {

constexpr absl::string_view kTestdataDir =
    "litert_lm/runtime/testdata";

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

absl::StatusOr<lm::proto::Any2AnyMetadata> BuildDefaultTestMetadata() {
  lm::proto::Any2AnyMetadata any2any_meta;
  any2any_meta.set_input_sample_rate_hz(16000);
  any2any_meta.set_output_sample_rate_hz(24000);
  any2any_meta.set_system_prompt("Respond with interleaved text and audio.");
  any2any_meta.add_supported_languages("en-US");
  prototypes::litert_models::lfm2_audio::Lfm2Audio lfm2;
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
          std::string(kMinimalTokenizerJson)));
  ABSL_RETURN_IF_ERROR(zlib_tok_stream.Prepare());
  std::stringstream tok_ss;
  tok_ss << zlib_tok_stream.GetStream().rdbuf();
  std::string compressed_tokenizer_bytes = tok_ss.str();

  lm::proto::Any2AnyMetadata any2any_meta;
  if (options.custom_meta != nullptr) {
    any2any_meta = *options.custom_meta;
  } else {
    ABSL_ASSIGN_OR_RETURN(any2any_meta, BuildDefaultTestMetadata());
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
      {lm::schema::AnySectionDataType_TFLiteModel, "tf_lite_audio_encoder",
       BuildMinimalTfliteModelBuffer(
           {{"mel", {1, 128, 16}, tflite::TensorType_FLOAT32}},
           {{"features",
             {1, 2, options.llm_hidden_size},
             tflite::TensorType_FLOAT32}})},
  };

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

    payloads.push_back(
        {lm::schema::AnySectionDataType_TFLiteModel, "tf_lite_tts_projector",
         BuildMinimalTfliteModelBuffer(
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
               tflite::TensorType_FLOAT32}})});
    payloads.push_back(
        {lm::schema::AnySectionDataType_TFLiteModel, "tf_lite_tts_decoder",
         BuildMinimalTfliteModelBuffer(
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
               tflite::TensorType_FLOAT32}})});
    payloads.push_back(
        {lm::schema::AnySectionDataType_TFLiteModel, "tf_lite_audio_vocoder",
         BuildMinimalTfliteModelBuffer(
             {{"audio_codes",
               {1, options.tts_num_codebooks, options.vocoder_chunk_frames},
               tflite::TensorType_INT32}},
             {{"waveform",
               {1, vocoder_out_samples},
               tflite::TensorType_FLOAT32,
               std::move(vocoder_waveform)}})});
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
