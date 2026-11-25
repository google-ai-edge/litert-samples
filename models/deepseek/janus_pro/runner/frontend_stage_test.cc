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

#include "models/deepseek/janus_pro/runner/frontend_stage.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "models/deepseek/janus_pro/runner/fan_out_stage.h"
#include "models/deepseek/janus_pro/runner/image_stage.h"
#include "models/deepseek/janus_pro/runner/io_types.h"
#include "models/deepseek/janus_pro/runner/janus_pro_model_config.h"
#include "models/deepseek/janus_pro/runner/text_stage.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "omni/base/mock_litert_runner.h"
#include "omni/omni_session.h"
#include "support/tokenizer/tokenizer.h"
#include "support/util/convert_tensor_buffer.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni::any2any {
namespace {

using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::NiceMock;

class FakeTokenizer : public support::Tokenizer {
 public:
  support::TokenizerType GetTokenizerType() const override {
    return support::TokenizerType::kHuggingFace;
  }

  absl::StatusOr<support::TokenIds> TextToTokenIds(
      absl::string_view text) override {
    support::TokenIds ids;
    ids.reserve(text.size());
    for (unsigned char c : text) {
      ids.push_back(static_cast<int>(c) % 24 + 3);
    }
    return ids;
  }

  absl::StatusOr<int> TokenToId(absl::string_view token) override {
    if (token == janus_pro::kBosToken) return janus_pro::kBosTokenId;
    if (token == janus_pro::kEosToken) return janus_pro::kEosTokenId;
    if (token == janus_pro::kPadToken) return janus_pro::kPadTokenId;
    if (token == janus_pro::kBeginOfImageToken) {
      return janus_pro::kBeginOfImageTokenId;
    }
    if (token == janus_pro::kEndOfImageToken) {
      return janus_pro::kEndOfImageTokenId;
    }
    if (token == janus_pro::kImagePlaceholderToken) {
      return janus_pro::kImagePlaceholderTokenId;
    }
    if (token == janus_pro::kUserToken) return janus_pro::kUserTokenId;
    if (token == janus_pro::kAssistantToken) {
      return janus_pro::kAssistantTokenId;
    }
    return absl::NotFoundError("not a special token");
  }

  absl::StatusOr<std::string> TokenIdsToText(
      absl::Span<const int> token_ids, bool /*skip_special_tokens*/) override {
    return std::string(token_ids.size(), 'x');
  }

  std::vector<std::string> GetTokens() const override { return {}; }
  int GetVocabSize() const override { return 102400; }
};

std::shared_ptr<NiceMock<MockLiteRtRunner>> MakeCountingEmbedderRunner(
    int hidden_size, int* run_count = nullptr) {
  auto runner = std::make_shared<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*runner, CreateInputBuffers(_))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto in_buf, support::CreateTensorBuffer<int32_t>({1, 1}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(in_buf));
            return bufs;
          });
  ON_CALL(*runner, CreateOutputBuffers(_))
      .WillByDefault([hidden_size](absl::string_view)
                         -> absl::StatusOr<std::vector<TensorBuffer>> {
        LITERT_ASSIGN_OR_RETURN(
            auto out_buf,
            support::CreateTensorBuffer<float>({1, 1, hidden_size}));
        std::vector<TensorBuffer> bufs;
        bufs.push_back(std::move(out_buf));
        return bufs;
      });
  ON_CALL(*runner, Run(_, _, _))
      .WillByDefault(
          [hidden_size, run_count](
              absl::string_view, absl::Span<const TensorBuffer> input_buffers,
              absl::Span<const TensorBuffer> output_buffers) -> absl::Status {
            if (run_count != nullptr) {
              ++(*run_count);
            }
            int32_t tok = 0;
            LITERT_RETURN_IF_ERROR(const_cast<TensorBuffer&>(input_buffers[0])
                                       .Read<int32_t>(absl::MakeSpan(&tok, 1)));
            std::vector<float> out_vals(hidden_size);
            for (int d = 0; d < hidden_size; ++d) {
              out_vals[d] =
                  static_cast<float>(tok) * 0.01f + static_cast<float>(d);
            }
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(output_buffers[0])
                    .Write<float>(absl::MakeConstSpan(out_vals)));
            return absl::OkStatus();
          });
  return runner;
}

std::unique_ptr<NiceMock<MockLiteRtRunner>> MakeMockVisionRunner(
    int image_size, int num_image_tokens, int hidden_size) {
  auto runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*runner, CreateInputBuffers(_))
      .WillByDefault([image_size](absl::string_view)
                         -> absl::StatusOr<std::vector<TensorBuffer>> {
        LITERT_ASSIGN_OR_RETURN(
            auto in_buf,
            support::CreateTensorBuffer<float>({1, 3, image_size, image_size}));
        std::vector<TensorBuffer> bufs;
        bufs.push_back(std::move(in_buf));
        return bufs;
      });
  ON_CALL(*runner, CreateOutputBuffers(_))
      .WillByDefault([num_image_tokens, hidden_size](absl::string_view)
                         -> absl::StatusOr<std::vector<TensorBuffer>> {
        LITERT_ASSIGN_OR_RETURN(auto out_buf,
                                support::CreateTensorBuffer<float>(
                                    {1, num_image_tokens, hidden_size}));
        std::vector<TensorBuffer> bufs;
        bufs.push_back(std::move(out_buf));
        return bufs;
      });
  ON_CALL(*runner, Run(_, _, _))
      .WillByDefault(
          [num_image_tokens, hidden_size](
              absl::string_view, absl::Span<const TensorBuffer>,
              absl::Span<const TensorBuffer> output_buffers) -> absl::Status {
            std::vector<float> out_vals(num_image_tokens * hidden_size, 7.0f);
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(output_buffers[0])
                    .Write<float>(absl::MakeConstSpan(out_vals)));
            return absl::OkStatus();
          });
  return runner;
}

TEST(JanusProFrontendStageTest,
     FormatsMultiTurnUnderstandingWithoutRuntimeEmbedderCalls) {
  PushInputSource input_source;
  ASSERT_OK_AND_ASSIGN(auto fan_out,
                       JanusProInputFanOutStage::Create(&input_source, true));
  auto tokenizer = std::make_shared<FakeTokenizer>();

  JanusProModelConfig cfg;
  cfg.llm_hidden_size = 4;
  cfg.llm_vocab_size = 200000;
  cfg.image_size = 16;
  cfg.vision_patch_size = 8;
  cfg.num_image_tokens = 4;
  cfg.generate_image = false;

  int frontend_embedder_calls = 0;
  auto frontend_embedder =
      MakeCountingEmbedderRunner(/*hidden_size=*/4, &frontend_embedder_calls);

  ASSERT_OK_AND_ASSIGN(auto text_stage,
                       JanusProTextStage::Create(
                           fan_out->text_queue(), fan_out->end_of_turn_queue(),
                           fan_out->end_of_input_queue(), tokenizer,
                           MakeCountingEmbedderRunner(/*hidden_size=*/4), cfg));
  ASSERT_OK_AND_ASSIGN(
      auto image_stage,
      JanusProImageStage::Create(
          fan_out->image_queue(),
          MakeMockVisionRunner(/*image_size=*/16, /*num_image_tokens=*/4,
                               /*hidden_size=*/4),
          cfg));
  ASSERT_OK_AND_ASSIGN(auto frontend, JanusProFrontendStage::Create(
                                          text_stage.get(), image_stage.get(),
                                          tokenizer, frontend_embedder, cfg));

  const int calls_after_create = frontend_embedder_calls;
  ASSERT_GT(calls_after_create, 0);

  // Turn 1: Multimodal understanding (Image + Text).
  ASSERT_OK(input_source.PushInput(OmniSession::ImageInput{
      .image_bytes = std::vector<uint8_t>(4 * 4 * 3, 100),
      .width = 4,
      .height = 4,
      .channels = 3,
  }));
  ASSERT_OK(input_source.PushInput(OmniSession::TextInput{.text = "Describe"}));
  ASSERT_OK(input_source.PushInput(OmniSession::EndOfInputTurn{}));

  while (fan_out->NeedSchedule()) ASSERT_OK(fan_out->Schedule());
  while (image_stage->NeedSchedule()) ASSERT_OK(image_stage->Schedule());
  while (text_stage->NeedSchedule()) ASSERT_OK(text_stage->Schedule());
  while (!frontend->HasOutput()) ASSERT_OK(frontend->Schedule());

  // Zero runtime embedder calls from `frontend`.
  EXPECT_EQ(frontend_embedder_calls, calls_after_create);

  ASSERT_OK_AND_ASSIGN(JanusProFrontendChunk turn1, frontend->GetOutput());
  EXPECT_FALSE(turn1.generate_image);
  ASSERT_FALSE(turn1.token_ids.empty());
  EXPECT_EQ(turn1.token_ids.front(), janus_pro::kBosTokenId);
  EXPECT_EQ(turn1.cond_embeddings.size(),
            static_cast<size_t>(turn1.prompt_len) * 4u);
  EXPECT_TRUE(turn1.uncond_token_ids.empty());
  EXPECT_TRUE(turn1.uncond_embeddings.empty());

  // Turn 2: Follow-up text turn starts with EOS token.
  ASSERT_OK(input_source.PushInput(OmniSession::TextInput{.text = "More"}));
  ASSERT_OK(input_source.PushInput(OmniSession::EndOfInputTurn{}));
  while (fan_out->NeedSchedule()) ASSERT_OK(fan_out->Schedule());
  while (text_stage->NeedSchedule()) ASSERT_OK(text_stage->Schedule());
  while (!frontend->HasOutput()) ASSERT_OK(frontend->Schedule());

  EXPECT_EQ(frontend_embedder_calls, calls_after_create);
  ASSERT_OK_AND_ASSIGN(JanusProFrontendChunk turn2, frontend->GetOutput());
  ASSERT_FALSE(turn2.token_ids.empty());
  EXPECT_EQ(turn2.token_ids.front(), janus_pro::kEosTokenId);
}

TEST(JanusProFrontendStageTest,
     FormatsVisualGenerationPromptWithPrecomputedUncondCfgEmbeddings) {
  PushInputSource input_source;
  ASSERT_OK_AND_ASSIGN(auto fan_out,
                       JanusProInputFanOutStage::Create(&input_source, false));
  auto tokenizer = std::make_shared<FakeTokenizer>();

  JanusProModelConfig gen_cfg;
  gen_cfg.llm_hidden_size = 4;
  gen_cfg.llm_vocab_size = 200000;
  gen_cfg.generate_image = true;

  int frontend_embedder_calls = 0;
  auto frontend_embedder =
      MakeCountingEmbedderRunner(/*hidden_size=*/4, &frontend_embedder_calls);

  ASSERT_OK_AND_ASSIGN(
      auto text_stage,
      JanusProTextStage::Create(
          fan_out->text_queue(), fan_out->end_of_turn_queue(),
          fan_out->end_of_input_queue(), tokenizer,
          MakeCountingEmbedderRunner(/*hidden_size=*/4), gen_cfg));
  ASSERT_OK_AND_ASSIGN(
      auto frontend,
      JanusProFrontendStage::Create(text_stage.get(), /*image_stage=*/nullptr,
                                    tokenizer, frontend_embedder, gen_cfg));

  const int calls_after_create = frontend_embedder_calls;
  ASSERT_GT(calls_after_create, 0);

  ASSERT_OK(input_source.PushInput(OmniSession::TextInput{.text = "A fox"}));
  ASSERT_OK(input_source.PushInput(OmniSession::EndOfInputTurn{}));
  while (fan_out->NeedSchedule()) ASSERT_OK(fan_out->Schedule());
  while (text_stage->NeedSchedule()) ASSERT_OK(text_stage->Schedule());
  while (!frontend->HasOutput()) ASSERT_OK(frontend->Schedule());

  EXPECT_EQ(frontend_embedder_calls, calls_after_create);
  ASSERT_OK_AND_ASSIGN(JanusProFrontendChunk gen_turn, frontend->GetOutput());
  EXPECT_TRUE(gen_turn.generate_image);
  ASSERT_GE(gen_turn.token_ids.size(), 3u);
  EXPECT_EQ(gen_turn.token_ids.front(), janus_pro::kBosTokenId);
  EXPECT_EQ(gen_turn.token_ids.back(), janus_pro::kBeginOfImageTokenId);

  // Unconditional CFG prompt matches exact length: [BOS, PAD..., BOI].
  ASSERT_EQ(gen_turn.uncond_token_ids.size(), gen_turn.token_ids.size());
  EXPECT_EQ(gen_turn.uncond_token_ids.front(), janus_pro::kBosTokenId);
  EXPECT_EQ(gen_turn.uncond_token_ids.back(), janus_pro::kBeginOfImageTokenId);
  for (size_t i = 1; i + 1 < gen_turn.uncond_token_ids.size(); ++i) {
    EXPECT_EQ(gen_turn.uncond_token_ids[i], janus_pro::kPadTokenId);
  }
  ASSERT_EQ(gen_turn.uncond_embeddings.size(),
            gen_turn.uncond_token_ids.size() * 4u);
  EXPECT_FLOAT_EQ(gen_turn.uncond_embeddings[0],
                  janus_pro::kBosTokenId * 0.01f);
  EXPECT_FLOAT_EQ(gen_turn.uncond_embeddings[4],
                  janus_pro::kPadTokenId * 0.01f);
  EXPECT_FLOAT_EQ(gen_turn.uncond_embeddings[(gen_turn.prompt_len - 1) * 4],
                  janus_pro::kBeginOfImageTokenId * 0.01f);
}

TEST(JanusProFrontendStageTest, RejectsInvalidConfigAtCreate) {
  PushInputSource input_source;
  ASSERT_OK_AND_ASSIGN(auto fan_out,
                       JanusProInputFanOutStage::Create(&input_source, false));
  auto tokenizer = std::make_shared<FakeTokenizer>();
  JanusProModelConfig valid_cfg;
  valid_cfg.llm_hidden_size = 4;
  valid_cfg.llm_vocab_size = 200000;
  ASSERT_OK_AND_ASSIGN(
      auto text_stage,
      JanusProTextStage::Create(
          fan_out->text_queue(), fan_out->end_of_turn_queue(),
          fan_out->end_of_input_queue(), tokenizer,
          MakeCountingEmbedderRunner(/*hidden_size=*/4), valid_cfg));

  JanusProModelConfig bad_cfg = valid_cfg;
  bad_cfg.llm_hidden_size = 0;
  EXPECT_THAT(JanusProFrontendStage::Create(
                  text_stage.get(), /*image_stage=*/nullptr, tokenizer,
                  MakeCountingEmbedderRunner(/*hidden_size=*/4), bad_cfg),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::omni::any2any
