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

#include "models/deepseek/janus_pro/runner/llm_stage.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "models/deepseek/janus_pro/runner/io_types.h"
#include "models/deepseek/janus_pro/runner/janus_pro_model_config.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "omni/any2any/any2any_test_utils.h"
#include "omni/base/litert_runner.h"
#include "omni/base/mock_litert_lm_runner.h"
#include "omni/base/mock_litert_runner.h"
#include "omni/base/model_utils.h"
#include "omni/base/stage.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "support/tokenizer/tokenizer.h"
#include "support/util/convert_tensor_buffer.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep
#include "tflite/schema/schema_generated.h"
#include "tflite/types/half.h"

namespace litert::omni::any2any {
namespace {

using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::NiceMock;

class FakeTokenizer : public support::Tokenizer {
 public:
  support::TokenizerType GetTokenizerType() const override {
    return support::TokenizerType::kHuggingFace;
  }

  absl::StatusOr<support::TokenIds> TextToTokenIds(
      absl::string_view /*text*/) override {
    return support::TokenIds{3};
  }

  absl::StatusOr<int> TokenToId(absl::string_view token) override {
    if (token == janus_pro::kEosToken) return 2;
    return absl::NotFoundError("not a special token");
  }

  absl::StatusOr<std::string> TokenIdsToText(
      absl::Span<const int> token_ids, bool /*skip_special_tokens*/) override {
    std::string out = "  ";
    for (int id : token_ids) {
      if (id == 3) {
        out.append("Hello");
      } else if (id == 4) {
        out.append(" world");
      } else {
        out.push_back(static_cast<char>('a' + (id % 26)));
      }
    }
    return out;
  }

  std::vector<std::string> GetTokens() const override { return {}; }
  int GetVocabSize() const override { return 32; }
};

class FakeFrontendSourceStage
    : public SingleThreadedStageWithDeque<JanusProFrontendChunk> {
 public:
  void PushTurn(JanusProFrontendChunk turn) { PushOutput(std::move(turn)); }

 protected:
  bool NeedScheduleInternal() const override { return false; }
  absl::Status ScheduleInternal() override { return absl::OkStatus(); }
};

std::unique_ptr<NiceMock<MockLiteRtRunner>> MakeMockVqEmbedRunner(
    int hidden_size, std::vector<int32_t>* received_codes = nullptr) {
  auto runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
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
          [hidden_size, received_codes](
              absl::string_view, absl::Span<const TensorBuffer> input_buffers,
              absl::Span<const TensorBuffer> output_buffers) -> absl::Status {
            int32_t code = 0;
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(input_buffers[0])
                    .Read<int32_t>(absl::MakeSpan(&code, 1)));
            if (received_codes != nullptr) {
              received_codes->push_back(code);
            }
            std::vector<float> out_vals(hidden_size, static_cast<float>(code));
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(output_buffers[0])
                    .Write<float>(absl::MakeConstSpan(out_vals)));
            return absl::OkStatus();
          });
  return runner;
}

std::unique_ptr<NiceMock<MockLiteRtRunner>> MakeCfgTestingGenHeadRunner(
    int hidden_size, int vq_codebook_size) {
  auto runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*runner, CreateInputBuffers(_))
      .WillByDefault([hidden_size](absl::string_view)
                         -> absl::StatusOr<std::vector<TensorBuffer>> {
        LITERT_ASSIGN_OR_RETURN(auto in_buf, support::CreateTensorBuffer<float>(
                                                 {1, 1, hidden_size}));
        std::vector<TensorBuffer> bufs;
        bufs.push_back(std::move(in_buf));
        return bufs;
      });
  ON_CALL(*runner, CreateOutputBuffers(_))
      .WillByDefault([vq_codebook_size](absl::string_view)
                         -> absl::StatusOr<std::vector<TensorBuffer>> {
        LITERT_ASSIGN_OR_RETURN(
            auto out_buf,
            support::CreateTensorBuffer<float>({1, vq_codebook_size}));
        std::vector<TensorBuffer> bufs;
        bufs.push_back(std::move(out_buf));
        return bufs;
      });
  ON_CALL(*runner, Run(_, _, _))
      .WillByDefault(
          [hidden_size, vq_codebook_size](
              absl::string_view, absl::Span<const TensorBuffer> input_buffers,
              absl::Span<const TensorBuffer> output_buffers) -> absl::Status {
            std::vector<float> in_vec(hidden_size, 0.0f);
            LITERT_RETURN_IF_ERROR(const_cast<TensorBuffer&>(input_buffers[0])
                                       .Read<float>(absl::MakeSpan(in_vec)));
            const bool is_cond = in_vec[0] > 0.0f;
            // Without CFG, cond alone peaks at index 1 (9.0 > 8.0) and uncond
            // peaks at index 1 (10.0 > 5.0). With cfg_weight = 5.0:
            //   cfg[1] = 10.0 + 5.0 * (9.0 - 10.0) = 5.0
            //   cfg[3] =  5.0 + 5.0 * (8.0 -  5.0) = 20.0 -> argmax = 3.
            std::vector<float> logits(vq_codebook_size, -10.0f);
            if (is_cond) {
              logits[1] = 9.0f;
              logits[3] = 8.0f;
            } else {
              logits[1] = 10.0f;
              logits[3] = 5.0f;
            }
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(output_buffers[0])
                    .Write<float>(absl::MakeConstSpan(logits)));
            return absl::OkStatus();
          });
  return runner;
}

TEST(JanusProLlmStageTest, CreateValidatesConfigAndSubmodelBuffers) {
  FakeFrontendSourceStage frontend;
  auto tokenizer = std::make_shared<FakeTokenizer>();
  auto llm = std::make_shared<NiceMock<MockLiteRtLmRunner>>();

  JanusProModelConfig bad_hidden;
  bad_hidden.llm_hidden_size = 0;
  bad_hidden.llm_vocab_size = 32;
  EXPECT_THAT(JanusProLlmStage::Create(&frontend, tokenizer, llm, nullptr,
                                       nullptr, nullptr, bad_hidden),
              StatusIs(absl::StatusCode::kInvalidArgument));

  JanusProModelConfig bad_vocab;
  bad_vocab.llm_hidden_size = 4;
  bad_vocab.llm_vocab_size = 0;
  EXPECT_THAT(JanusProLlmStage::Create(&frontend, tokenizer, llm, nullptr,
                                       nullptr, nullptr, bad_vocab),
              StatusIs(absl::StatusCode::kInvalidArgument));

  JanusProModelConfig bad_codebook;
  bad_codebook.llm_hidden_size = 4;
  bad_codebook.llm_vocab_size = 32;
  bad_codebook.vq_codebook_size = 0;
  EXPECT_THAT(
      JanusProLlmStage::Create(&frontend, tokenizer, llm, nullptr, nullptr,
                               MakeCfgTestingGenHeadRunner(4, 8), bad_codebook),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // Mismatched vq_image_embedder output size (8 != llm_hidden_size 4).
  JanusProModelConfig cfg;
  cfg.llm_hidden_size = 4;
  cfg.llm_vocab_size = 32;
  cfg.vq_codebook_size = 8;
  EXPECT_THAT(JanusProLlmStage::Create(&frontend, tokenizer, llm, nullptr,
                                       MakeMockVqEmbedRunner(/*hidden_size=*/8),
                                       nullptr, cfg),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(JanusProLlmStageTest, RunsUnderstandingTurnGreedyStopsOnEosAndTrimsText) {
  JanusProModelConfig cfg;
  cfg.llm_hidden_size = 4;
  cfg.llm_vocab_size = 32;
  cfg.max_new_tokens = 8;
  cfg.do_sample = false;

  FakeFrontendSourceStage frontend;
  auto tokenizer = std::make_shared<FakeTokenizer>();
  auto llm = std::make_shared<NiceMock<MockLiteRtLmRunner>>();

  ON_CALL(*llm, Prefill(_)).WillByDefault([](const lm::ExecutorInputs&) {
    return absl::OkStatus();
  });
  int decode_calls = 0;
  const std::vector<int> scripted_tokens = {3, 4, 2};  // "Hello", " world", EOS
  ON_CALL(*llm, Decode(_))
      .WillByDefault(
          [&](const lm::ExecutorInputs&) -> absl::StatusOr<TensorBuffer> {
            const int tok = scripted_tokens[decode_calls++];
            std::vector<float> logits(32, -5.0f);
            logits[tok] = 10.0f;
            LITERT_ASSIGN_OR_RETURN(
                auto buf,
                support::CopyToTensorBuffer<float>(logits, {1, 1, 32}));
            return buf;
          });

  ASSERT_OK_AND_ASSIGN(
      auto stage, JanusProLlmStage::Create(&frontend, tokenizer, llm, nullptr,
                                           nullptr, nullptr, cfg));

  JanusProFrontendChunk turn;
  turn.generate_image = false;
  turn.prompt_len = 2;
  turn.token_ids = {1, 3};
  turn.cond_embeddings.assign(2 * 4, 0.25f);
  frontend.PushTurn(std::move(turn));

  ASSERT_OK(stage->Schedule());
  ASSERT_TRUE(stage->HasOutput());
  ASSERT_OK_AND_ASSIGN(JanusProLlmOutput out, stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<std::string>(out.payload));
  EXPECT_EQ(std::get<std::string>(out.payload), "Hello world");
  // Stopped on EOS (3 decode calls, no extra commit call).
  EXPECT_EQ(decode_calls, 3);
}

TEST(JanusProLlmStageTest, AcceptsFloat16LogitsAndRejectsInvalidOutputs) {
  JanusProModelConfig cfg;
  cfg.llm_hidden_size = 4;
  cfg.llm_vocab_size = 32;
  cfg.max_new_tokens = 1;
  cfg.do_sample = false;

  FakeFrontendSourceStage frontend;
  auto tokenizer = std::make_shared<FakeTokenizer>();
  auto llm = std::make_shared<NiceMock<MockLiteRtLmRunner>>();
  ON_CALL(*llm, Prefill(_)).WillByDefault([](const lm::ExecutorInputs&) {
    return absl::OkStatus();
  });

  int decode_calls = 0;
  ON_CALL(*llm, Decode(_))
      .WillByDefault(
          [&](const lm::ExecutorInputs&) -> absl::StatusOr<TensorBuffer> {
            ++decode_calls;
            std::vector<float> logits(32, -2.0f);
            logits[3] = 5.0f;
            LITERT_ASSIGN_OR_RETURN(
                auto buf,
                (support::ConvertAndCopyToTensorBuffer<tflite::half, float>(
                    logits, {1, 1, 32})));
            return buf;
          });

  ASSERT_OK_AND_ASSIGN(
      auto stage, JanusProLlmStage::Create(&frontend, tokenizer, llm, nullptr,
                                           nullptr, nullptr, cfg));

  JanusProFrontendChunk turn;
  turn.prompt_len = 1;
  turn.token_ids = {1};
  turn.cond_embeddings = {1.0f, 2.0f, 3.0f, 4.0f};
  frontend.PushTurn(turn);

  ASSERT_OK(stage->Schedule());
  ASSERT_OK_AND_ASSIGN(JanusProLlmOutput out, stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<std::string>(out.payload));
  EXPECT_EQ(std::get<std::string>(out.payload), "Hello");
  // 1 decode step + 1 commit step for the non-EOS final token.
  EXPECT_EQ(decode_calls, 2);

  // Int32 decode output is rejected with kInvalidArgument.
  ON_CALL(*llm, Decode(_))
      .WillByDefault(
          [](const lm::ExecutorInputs&) -> absl::StatusOr<TensorBuffer> {
            std::vector<int32_t> bad_out(32, 0);
            LITERT_ASSIGN_OR_RETURN(
                auto buf,
                support::CopyToTensorBuffer<int32_t>(bad_out, {1, 1, 32}));
            return buf;
          });
  frontend.PushTurn(turn);
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kInvalidArgument));

  // Mismatched embedding size is rejected with kInvalidArgument.
  JanusProFrontendChunk bad_turn;
  bad_turn.prompt_len = 2;
  bad_turn.cond_embeddings = {1.0f, 2.0f};
  frontend.PushTurn(std::move(bad_turn));
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(JanusProLlmStageTest, RunsVisualGenerationWithCfgLogitsAndVqFeedback) {
  JanusProModelConfig cfg;
  cfg.llm_hidden_size = 4;
  cfg.llm_vocab_size = 32;
  cfg.vq_codebook_size = 8;
  cfg.image_size = 16;
  cfg.num_image_tokens = 4;
  cfg.generate_image = true;
  cfg.cfg_weight = 5.0f;
  cfg.image_temperature = 0.0f;
  cfg.max_new_tokens = 2;

  FakeFrontendSourceStage frontend;
  auto tokenizer = std::make_shared<FakeTokenizer>();
  auto cond_llm = std::make_shared<NiceMock<MockLiteRtLmRunner>>();
  auto uncond_llm = std::make_shared<NiceMock<MockLiteRtLmRunner>>();

  int cond_reset_calls = 0;
  int uncond_reset_calls = 0;
  ON_CALL(*cond_llm, Reset()).WillByDefault([&]() {
    ++cond_reset_calls;
    return absl::OkStatus();
  });
  ON_CALL(*uncond_llm, Reset()).WillByDefault([&]() {
    ++uncond_reset_calls;
    return absl::OkStatus();
  });
  ON_CALL(*cond_llm, Prefill(_)).WillByDefault([](const lm::ExecutorInputs&) {
    return absl::OkStatus();
  });
  ON_CALL(*uncond_llm, Prefill(_)).WillByDefault([](const lm::ExecutorInputs&) {
    return absl::OkStatus();
  });
  ON_CALL(*cond_llm, Decode(_))
      .WillByDefault(
          [](const lm::ExecutorInputs&) -> absl::StatusOr<TensorBuffer> {
            std::vector<float> out(32 + 4, 0.0f);
            out[2] = 10.0f;  // EOS logit for follow-up understanding turn
            out[32] = 1.0f;  // Positive hidden_state[0] -> conditional
            LITERT_ASSIGN_OR_RETURN(
                auto buf, support::CopyToTensorBuffer<float>(out, {1, 1, 36}));
            return buf;
          });
  ON_CALL(*uncond_llm, Decode(_))
      .WillByDefault(
          [](const lm::ExecutorInputs&) -> absl::StatusOr<TensorBuffer> {
            std::vector<float> out(32 + 4, 0.0f);
            out[32] = -1.0f;  // Negative hidden_state[0] -> unconditional
            LITERT_ASSIGN_OR_RETURN(
                auto buf, support::CopyToTensorBuffer<float>(out, {1, 1, 36}));
            return buf;
          });

  std::vector<int32_t> vq_embed_codes;
  ASSERT_OK_AND_ASSIGN(
      auto stage, JanusProLlmStage::Create(
                      &frontend, tokenizer, cond_llm, uncond_llm,
                      MakeMockVqEmbedRunner(/*hidden_size=*/4, &vq_embed_codes),
                      MakeCfgTestingGenHeadRunner(/*hidden_size=*/4,
                                                  /*vq_codebook_size=*/8),
                      cfg));

  JanusProFrontendChunk gen_turn;
  gen_turn.generate_image = true;
  gen_turn.prompt_len = 3;
  gen_turn.uncond_prompt_len = 3;
  gen_turn.cond_embeddings.assign(3 * 4, 0.5f);
  gen_turn.uncond_embeddings.assign(3 * 4, 0.1f);
  frontend.PushTurn(std::move(gen_turn));

  ASSERT_OK(stage->Schedule());
  ASSERT_TRUE(stage->HasOutput());
  ASSERT_OK_AND_ASSIGN(JanusProLlmOutput out, stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<JanusProVqCodeGrid>(out.payload));
  const auto& grid = std::get<JanusProVqCodeGrid>(out.payload);
  EXPECT_EQ(grid.image_width, 16);
  EXPECT_EQ(grid.image_height, 16);
  EXPECT_THAT(grid.vq_codes, ElementsAre(3, 3, 3, 3));
  // Steps 1, 2, 3 feed back the previously sampled VQ code (3).
  EXPECT_THAT(vq_embed_codes, ElementsAre(3, 3, 3));
  EXPECT_EQ(cond_reset_calls, 1);
  EXPECT_EQ(uncond_reset_calls, 1);

  // Subsequent understanding turn resets `cond_llm` before prefill because
  // `last_turn_was_image_gen_` was set.
  JanusProFrontendChunk text_turn;
  text_turn.generate_image = false;
  text_turn.prompt_len = 2;
  text_turn.cond_embeddings.assign(2 * 4, 0.2f);
  frontend.PushTurn(std::move(text_turn));
  ASSERT_OK(stage->Schedule());
  EXPECT_EQ(cond_reset_calls, 2);

  stage->Reset();
  EXPECT_EQ(cond_reset_calls, 3);
  EXPECT_EQ(uncond_reset_calls, 2);
}

TEST(JanusProLlmStageTest, VisualGenerationRejectsMissingRunnersOrHiddenState) {
  JanusProModelConfig cfg;
  cfg.llm_hidden_size = 4;
  cfg.llm_vocab_size = 32;
  cfg.vq_codebook_size = 8;
  cfg.num_image_tokens = 2;
  cfg.generate_image = true;

  FakeFrontendSourceStage frontend;
  auto tokenizer = std::make_shared<FakeTokenizer>();
  auto cond_llm = std::make_shared<NiceMock<MockLiteRtLmRunner>>();
  auto uncond_llm = std::make_shared<NiceMock<MockLiteRtLmRunner>>();
  ON_CALL(*cond_llm, Reset()).WillByDefault([]() { return absl::OkStatus(); });
  ON_CALL(*uncond_llm, Reset()).WillByDefault([]() {
    return absl::OkStatus();
  });
  ON_CALL(*cond_llm, Prefill(_)).WillByDefault([](const lm::ExecutorInputs&) {
    return absl::OkStatus();
  });
  ON_CALL(*uncond_llm, Prefill(_)).WillByDefault([](const lm::ExecutorInputs&) {
    return absl::OkStatus();
  });

  JanusProFrontendChunk gen_turn;
  gen_turn.generate_image = true;
  gen_turn.prompt_len = 2;
  gen_turn.uncond_prompt_len = 2;
  gen_turn.cond_embeddings.assign(2 * 4, 0.5f);
  gen_turn.uncond_embeddings.assign(2 * 4, 0.1f);

  // 1. Missing uncond/vq/gen_head runners -> kFailedPrecondition.
  ASSERT_OK_AND_ASSIGN(auto stage_no_gen, JanusProLlmStage::Create(
                                              &frontend, tokenizer, cond_llm,
                                              nullptr, nullptr, nullptr, cfg));
  frontend.PushTurn(gen_turn);
  EXPECT_THAT(stage_no_gen->Schedule(),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  // 2. Decode returning only vocab_size floats (no hidden state) ->
  //    kFailedPrecondition.
  ON_CALL(*cond_llm, Decode(_))
      .WillByDefault(
          [](const lm::ExecutorInputs&) -> absl::StatusOr<TensorBuffer> {
            std::vector<float> logits_only(32, 0.0f);
            LITERT_ASSIGN_OR_RETURN(
                auto buf,
                support::CopyToTensorBuffer<float>(logits_only, {1, 1, 32}));
            return buf;
          });
  ON_CALL(*uncond_llm, Decode(_))
      .WillByDefault(
          [](const lm::ExecutorInputs&) -> absl::StatusOr<TensorBuffer> {
            std::vector<float> logits_only(32, 0.0f);
            LITERT_ASSIGN_OR_RETURN(
                auto buf,
                support::CopyToTensorBuffer<float>(logits_only, {1, 1, 32}));
            return buf;
          });

  ASSERT_OK_AND_ASSIGN(
      auto stage_full,
      JanusProLlmStage::Create(&frontend, tokenizer, cond_llm, uncond_llm,
                               MakeMockVqEmbedRunner(/*hidden_size=*/4),
                               MakeCfgTestingGenHeadRunner(
                                   /*hidden_size=*/4, /*vq_codebook_size=*/8),
                               cfg));
  frontend.PushTurn(gen_turn);
  EXPECT_THAT(stage_full->Schedule(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(JanusProLlmStageTest,
     RunsVisualGenerationWithRealCompiledTfliteSubmodels) {
  Expected<Environment> env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  const ModelOptions cpu_opts{
      .cache_dir = ":nocache",
      .backend = lm::Backend::CPU,
      .num_threads = 1,
  };

  std::string vq_emb_buf = BuildMinimalTfliteModelBuffer(
      {{"token_ids", {1, 1}, tflite::TensorType_INT32}},
      {{"embeds", {1, 1, 4}, tflite::TensorType_FLOAT32}});
  std::vector<float> head_logits(8, -5.0f);
  head_logits[6] = 15.0f;
  std::string gen_head_buf = BuildMinimalTfliteModelBuffer(
      {{"hidden_states", {1, 1, 4}, tflite::TensorType_FLOAT32}},
      {{"logits", {1, 8}, tflite::TensorType_FLOAT32, head_logits}});

  ASSERT_OK_AND_ASSIGN(CompiledModel vq_emb_model,
                       CreateCompiledModelFromBuffer(*env, cpu_opts, vq_emb_buf,
                                                     "vq_image_embedder"));
  ASSERT_OK_AND_ASSIGN(
      CompiledModel gen_head_model,
      CreateCompiledModelFromBuffer(*env, cpu_opts, gen_head_buf, "gen_head"));

  JanusProModelConfig cfg;
  cfg.llm_hidden_size = 4;
  cfg.llm_vocab_size = 32;
  cfg.vq_codebook_size = 8;
  cfg.image_size = 16;
  cfg.num_image_tokens = 3;
  cfg.generate_image = true;
  cfg.image_temperature = 0.0f;

  FakeFrontendSourceStage frontend;
  auto tokenizer = std::make_shared<FakeTokenizer>();
  auto cond_llm = std::make_shared<NiceMock<MockLiteRtLmRunner>>();
  auto uncond_llm = std::make_shared<NiceMock<MockLiteRtLmRunner>>();
  auto ok_status = []() { return absl::OkStatus(); };
  auto ok_prefill = [](const lm::ExecutorInputs&) { return absl::OkStatus(); };
  auto hidden_decode =
      [](const lm::ExecutorInputs&) -> absl::StatusOr<TensorBuffer> {
    std::vector<float> out(36, 0.5f);
    LITERT_ASSIGN_OR_RETURN(
        auto buf, support::CopyToTensorBuffer<float>(out, {1, 1, 36}));
    return buf;
  };
  ON_CALL(*cond_llm, Reset()).WillByDefault(ok_status);
  ON_CALL(*uncond_llm, Reset()).WillByDefault(ok_status);
  ON_CALL(*cond_llm, Prefill(_)).WillByDefault(ok_prefill);
  ON_CALL(*uncond_llm, Prefill(_)).WillByDefault(ok_prefill);
  ON_CALL(*cond_llm, Decode(_)).WillByDefault(hidden_decode);
  ON_CALL(*uncond_llm, Decode(_)).WillByDefault(hidden_decode);

  ASSERT_OK_AND_ASSIGN(
      auto stage,
      JanusProLlmStage::Create(
          &frontend, tokenizer, cond_llm, uncond_llm,
          std::make_unique<LiteRtRunnerImpl>(&vq_emb_model),
          std::make_unique<LiteRtRunnerImpl>(&gen_head_model), cfg));

  JanusProFrontendChunk turn;
  turn.generate_image = true;
  turn.prompt_len = 2;
  turn.uncond_prompt_len = 2;
  turn.cond_embeddings.assign(8, 0.2f);
  turn.uncond_embeddings.assign(8, 0.1f);
  frontend.PushTurn(std::move(turn));

  ASSERT_OK(stage->Schedule());
  ASSERT_OK_AND_ASSIGN(JanusProLlmOutput out, stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<JanusProVqCodeGrid>(out.payload));
  EXPECT_THAT(std::get<JanusProVqCodeGrid>(out.payload).vq_codes,
              ElementsAre(6, 6, 6));
}

}  // namespace
}  // namespace litert::omni::any2any
