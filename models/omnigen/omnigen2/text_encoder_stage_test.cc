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

#include "models/omnigen/omnigen2/text_encoder_stage.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "models/omnigen/omnigen2/io_types.h"
#include "models/omnigen/omnigen2/omnigen2_test_utils.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_types.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/mock_litert_runner.h"
#include "omni/base/model_utils.h"
#include "omni/text2image/prompt_source.h"
#include "runtime/executor/executor_settings_base.h"
#include "support/tokenizer/tokenizer.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK
#include "tflite/schema/schema_generated.h"

namespace litert::omni::text2image::omnigen2 {
namespace {

using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::IsEmpty;

class StubTokenizer : public support::Tokenizer {
 public:
  explicit StubTokenizer(support::TokenIds positive_ids,
                         support::TokenIds negative_ids = {})
      : positive_ids_(std::move(positive_ids)),
        negative_ids_(std::move(negative_ids)) {}

  support::TokenizerType GetTokenizerType() const override {
    return support::TokenizerType::kHuggingFace;
  }

  absl::StatusOr<support::TokenIds> TextToTokenIds(
      absl::string_view text) override {
    seen_texts_.push_back(std::string(text));
    if (text == "user\n" || text.empty() || text == "user\nlow quality") {
      return negative_ids_;
    }
    return positive_ids_;
  }

  absl::StatusOr<int> TokenToId(absl::string_view /*token*/) override {
    return absl::NotFoundError("not implemented");
  }

  absl::StatusOr<std::string> TokenIdsToText(
      absl::Span<const int> /*token_ids*/,
      bool /*skip_special_tokens*/) override {
    return "";
  }

  std::vector<std::string> GetTokens() const override { return {}; }
  int GetVocabSize() const override { return 1000; }

  const std::vector<std::string>& seen_texts() const { return seen_texts_; }

 private:
  support::TokenIds positive_ids_;
  support::TokenIds negative_ids_;
  std::vector<std::string> seen_texts_;
};

class OmniGen2TextEncoderStageTest : public ::testing::Test {
 protected:
  struct BufferSpec {
    ElementType element_type;
    std::vector<int> dims;
    size_t elem_size;
  };

  void SetUp() override {
    auto env = Environment::Create({});
    ASSERT_TRUE(env.HasValue());
    env_ = std::make_unique<Environment>(std::move(*env));
  }

  absl::StatusOr<TensorBuffer> CreateBuffer(ElementType element_type,
                                            const std::vector<int>& dims,
                                            size_t elem_size) {
    RankedTensorType type(element_type,
                          Layout(Dimensions(dims.begin(), dims.end())));
    size_t num_elements = 1;
    for (int d : dims) num_elements *= d;
    LITERT_ASSIGN_OR_RETURN(
        auto buf,
        TensorBuffer::CreateManaged(*env_, TensorBufferType::kHostMemory,
                                    std::move(type), num_elements * elem_size));
    return buf;
  }

  void ExpectBuffers(MockLiteRtRunner& runner,
                     std::vector<BufferSpec> input_specs,
                     std::vector<BufferSpec> output_specs) {
    EXPECT_CALL(runner, CreateInputBuffers(absl::string_view("")))
        .WillOnce(
            [this, input_specs = std::move(input_specs)](absl::string_view)
                -> absl::StatusOr<std::vector<TensorBuffer>> {
              std::vector<TensorBuffer> bufs;
              bufs.reserve(input_specs.size());
              for (const auto& spec : input_specs) {
                ABSL_ASSIGN_OR_RETURN(
                    TensorBuffer buf,
                    CreateBuffer(spec.element_type, spec.dims, spec.elem_size));
                bufs.push_back(std::move(buf));
              }
              return bufs;
            });
    EXPECT_CALL(runner, CreateOutputBuffers(absl::string_view("")))
        .WillOnce(
            [this, output_specs = std::move(output_specs)](absl::string_view)
                -> absl::StatusOr<std::vector<TensorBuffer>> {
              std::vector<TensorBuffer> bufs;
              bufs.reserve(output_specs.size());
              for (const auto& spec : output_specs) {
                ABSL_ASSIGN_OR_RETURN(
                    TensorBuffer buf,
                    CreateBuffer(spec.element_type, spec.dims, spec.elem_size));
                bufs.push_back(std::move(buf));
              }
              return bufs;
            });
  }

  std::unique_ptr<Environment> env_;
};

TEST_F(OmniGen2TextEncoderStageTest,
       FramesPadsTruncatesAndSupportsCfgWithSwappedIndices) {
  constexpr int kSeqLen = 8;
  constexpr int kHidden = 4;

  PushPromptSource prompt_source(ImageGenInputMetadata{
      .width = 64, .height = 64, .num_inference_steps = 3, .seed = 99});

  auto runner = std::make_unique<MockLiteRtRunner>();
  ExpectBuffers(*runner,
                {{ElementType::Int32, {1, kSeqLen}, 4},
                 {ElementType::Int32, {1, kSeqLen}, 4}},
                {{ElementType::Float32, {1, kSeqLen, kHidden}, 4}});

  std::vector<std::vector<int32_t>> captured_ids;
  std::vector<std::vector<int32_t>> captured_masks;
  int call_count = 0;
  EXPECT_CALL(*runner, Run(absl::string_view(""), _, _))
      .Times(2)
      .WillRepeatedly([&](absl::string_view,
                          absl::Span<const TensorBuffer> inputs,
                          absl::Span<const TensorBuffer> outputs) {
        // Swapped indices: attention_mask = 0, input_ids = 1.
        std::vector<int32_t> mask(kSeqLen, -1);
        std::vector<int32_t> ids(kSeqLen, -1);
        EXPECT_TRUE(const_cast<TensorBuffer&>(inputs[0])
                        .Read<int32_t>(absl::MakeSpan(mask))
                        .HasValue());
        EXPECT_TRUE(const_cast<TensorBuffer&>(inputs[1])
                        .Read<int32_t>(absl::MakeSpan(ids))
                        .HasValue());
        captured_masks.push_back(std::move(mask));
        captured_ids.push_back(std::move(ids));

        const float fill = (call_count++ == 0) ? 1.25f : -0.75f;
        std::vector<float> embeds(kSeqLen * kHidden, fill);
        EXPECT_TRUE(const_cast<TensorBuffer&>(outputs[0])
                        .Write<float>(embeds)
                        .HasValue());
        return absl::OkStatus();
      });

  OmniGen2TextEncoderStage::Config config;
  config.seq_len = kSeqLen;
  config.pad_token_id = 999;
  config.prompt_prefix = "user\n";
  config.prefix_token_ids = {101, 102};
  config.suffix_token_ids = {103};
  config.encode_negative_prompt = true;
  config.negative_prompt = "low quality";
  config.input_indices.input_ids = 1;
  config.input_indices.attention_mask = 0;

  // Positive prompt has 7 user tokens -> truncated to 5 (8 - 2 - 1).
  // Negative prompt has 2 user tokens -> active_len = 2 + 2 + 1 = 5.
  auto tokenizer = std::make_unique<StubTokenizer>(
      support::TokenIds{10, 11, 12, 13, 14, 15, 16}, support::TokenIds{20, 21});
  const StubTokenizer* tokenizer_ptr = tokenizer.get();

  ASSERT_OK_AND_ASSIGN(auto stage, OmniGen2TextEncoderStage::Create(
                                       &prompt_source, std::move(tokenizer),
                                       std::move(runner), config));

  EXPECT_FALSE(stage->NeedSchedule());
  ASSERT_OK(prompt_source.PushPrompt("a painting"));
  EXPECT_TRUE(stage->NeedSchedule());
  ASSERT_OK(stage->Schedule());
  ASSERT_TRUE(stage->HasOutput());

  ASSERT_OK_AND_ASSIGN(OmniGen2TextEncoderOutput out, stage->GetOutput());
  EXPECT_THAT(tokenizer_ptr->seen_texts(),
              ElementsAre("user\na painting", "user\nlow quality"));
  EXPECT_EQ(out.metadata.seed, 99);
  EXPECT_EQ(out.active_seq_len, 8);
  EXPECT_EQ(out.negative_active_seq_len, 5);
  EXPECT_EQ(out.prompt_embeds.size(), static_cast<size_t>(kSeqLen * kHidden));
  EXPECT_FLOAT_EQ(out.prompt_embeds[0], 1.25f);
  EXPECT_EQ(out.negative_prompt_embeds.size(),
            static_cast<size_t>(kSeqLen * kHidden));
  EXPECT_FLOAT_EQ(out.negative_prompt_embeds[0], -0.75f);

  ASSERT_EQ(captured_ids.size(), 2u);
  EXPECT_THAT(captured_ids[0], ElementsAre(101, 102, 10, 11, 12, 13, 14, 103));
  EXPECT_THAT(captured_masks[0], ElementsAre(1, 1, 1, 1, 1, 1, 1, 1));
  EXPECT_THAT(captured_ids[1],
              ElementsAre(101, 102, 20, 21, 103, 999, 999, 999));
  EXPECT_THAT(captured_masks[1], ElementsAre(1, 1, 1, 1, 1, 0, 0, 0));
}

TEST_F(OmniGen2TextEncoderStageTest,
       RunsWithRealCompiledModelAndInt64InputBuffersWithoutCfg) {
  constexpr int kSeqLen = 6;
  constexpr int kHidden = 4;
  std::vector<float> expected_embeds(kSeqLen * kHidden, 0.5f);
  const std::string tflite_buf = BuildMinimalTfliteModelBuffer(
      {{"args_0", {1, kSeqLen}, tflite::TensorType_INT64},
       {"args_1", {1, kSeqLen}, tflite::TensorType_INT64}},
      {{"prompt_embeds",
        {1, kSeqLen, kHidden},
        tflite::TensorType_FLOAT32,
        expected_embeds}});

  ModelOptions options;
  options.backend = lm::Backend::CPU;
  options.num_threads = 1;
  options.cache_dir = ":nocache";
  ASSERT_OK_AND_ASSIGN(
      CompiledModel compiled,
      CreateCompiledModelFromBuffer(*env_, options, tflite_buf, "test_te"));

  PushPromptSource prompt_source(ImageGenInputMetadata{});
  OmniGen2TextEncoderStage::Config config;
  config.seq_len = kSeqLen;
  config.prefix_token_ids = {1};
  config.suffix_token_ids = {2};
  config.encode_negative_prompt = false;

  ASSERT_OK_AND_ASSIGN(
      auto stage, OmniGen2TextEncoderStage::Create(
                      &prompt_source,
                      std::make_unique<StubTokenizer>(support::TokenIds{7, 8}),
                      std::make_unique<LiteRtRunnerImpl>(&compiled), config));

  ASSERT_OK(prompt_source.PushPrompt("hello"));
  ASSERT_OK(stage->Schedule());
  ASSERT_OK_AND_ASSIGN(OmniGen2TextEncoderOutput out, stage->GetOutput());
  EXPECT_EQ(out.active_seq_len, 4);
  EXPECT_EQ(out.prompt_embeds, expected_embeds);
  EXPECT_THAT(out.negative_prompt_embeds, IsEmpty());
  EXPECT_EQ(out.negative_active_seq_len, 0);
}

TEST_F(OmniGen2TextEncoderStageTest, RejectsInvalidConfigAndBuffers) {
  PushPromptSource prompt_source(ImageGenInputMetadata{});

  // seq_len <= prefix + suffix.
  OmniGen2TextEncoderStage::Config bad_seq;
  bad_seq.seq_len = 3;
  bad_seq.prefix_token_ids = {1, 2};
  bad_seq.suffix_token_ids = {3};
  EXPECT_THAT(
      OmniGen2TextEncoderStage::Create(
          &prompt_source, std::make_unique<StubTokenizer>(support::TokenIds{5}),
          std::make_unique<MockLiteRtRunner>(), bad_seq),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // Duplicate input indices.
  OmniGen2TextEncoderStage::Config dup_idx;
  dup_idx.seq_len = 8;
  dup_idx.prefix_token_ids = {1};
  dup_idx.suffix_token_ids = {2};
  dup_idx.input_indices.input_ids = 0;
  dup_idx.input_indices.attention_mask = 0;
  auto dup_runner = std::make_unique<MockLiteRtRunner>();
  ExpectBuffers(
      *dup_runner,
      {{ElementType::Int32, {1, 8}, 4}, {ElementType::Int32, {1, 8}, 4}},
      {{ElementType::Float32, {1, 8, 4}, 4}});
  EXPECT_THAT(
      OmniGen2TextEncoderStage::Create(
          &prompt_source, std::make_unique<StubTokenizer>(support::TokenIds{5}),
          std::move(dup_runner), dup_idx),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // Unsupported input buffer dtype (Float32 instead of Int32/Int64).
  OmniGen2TextEncoderStage::Config ok_cfg;
  ok_cfg.seq_len = 8;
  ok_cfg.prefix_token_ids = {1};
  ok_cfg.suffix_token_ids = {2};
  auto bad_dtype_runner = std::make_unique<MockLiteRtRunner>();
  ExpectBuffers(
      *bad_dtype_runner,
      {{ElementType::Float32, {1, 8}, 4}, {ElementType::Int32, {1, 8}, 4}},
      {{ElementType::Float32, {1, 8, 4}, 4}});
  EXPECT_THAT(
      OmniGen2TextEncoderStage::Create(
          &prompt_source, std::make_unique<StubTokenizer>(support::TokenIds{5}),
          std::move(bad_dtype_runner), ok_cfg),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::omni::text2image::omnigen2
