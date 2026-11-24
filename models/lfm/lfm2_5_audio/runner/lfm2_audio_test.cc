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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <numbers>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "models/lfm/lfm2_5_audio/runner/audio_stage.h"
#include "models/lfm/lfm2_5_audio/runner/frontend_stage.h"
#include "models/lfm/lfm2_5_audio/runner/generation_stage.h"
#include "models/lfm/lfm2_5_audio/runner/input_fan_out_stage.h"
#include "models/lfm/lfm2_5_audio/runner/io_types.h"
#include "models/lfm/lfm2_5_audio/runner/lfm2_audio.pb.h"
#include "models/lfm/lfm2_5_audio/runner/lfm2_audio_factory.h"
#include "models/lfm/lfm2_5_audio/runner/lfm2_audio_model_config.h"
#include "models/lfm/lfm2_5_audio/runner/lfm2_audio_test_utils.h"
#include "models/lfm/lfm2_5_audio/runner/text_stage.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "omni/any2any/any2any_session.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_lm_runner.h"
#include "omni/base/mock_litert_runner.h"
#include "omni/base/model_resources.h"
#include "omni/base/model_utils.h"
#include "omni/base/sequenced.h"
#include "omni/omni_engine.h"
#include "omni/omni_registry.h"
#include "omni/omni_session.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/proto/any2any_metadata.pb.h"
#include "runtime/proto/any2any_model_type.pb.h"
#include "runtime/proto/any2any_session_config.pb.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "support/tokenizer/tokenizer.h"
#include "support/util/convert_tensor_buffer.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

namespace litert::omni::any2any {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::NiceMock;

class FakeTokenizer : public support::Tokenizer {
 public:
  explicit FakeTokenizer(int out_of_range_token_id = -1)
      : out_of_range_token_id_(out_of_range_token_id) {}

  absl::StatusOr<std::vector<int>> TextToTokenIds(
      absl::string_view text) override {
    if (text == "OUT_OF_RANGE" && out_of_range_token_id_ != -1) {
      return std::vector<int>{out_of_range_token_id_};
    }
    if (text == "\n") {
      return std::vector<int>{5};
    }
    return std::vector<int>{3};
  }

  absl::StatusOr<int> TokenToId(absl::string_view token) override {
    if (token == "<|pad|>") return 0;
    if (token == "<|startoftext|>") return 1;
    if (token == "<|im_start|>") return 6;
    if (token == "<|im_end|>") return 7;
    if (token == "<|audio_start|>") return 128;
    if (token == "<|text_end|>") return 130;
    return absl::NotFoundError("Unknown token");
  }

  absl::StatusOr<std::string> TokenIdsToText(
      absl::Span<const int> token_ids, bool /*skip_special_tokens*/) override {
    std::string out;
    for (int id : token_ids) {
      if (id == 3) {
        absl::StrAppend(&out, "Hello");
      } else if (id == 4) {
        absl::StrAppend(&out, " world");
      }
    }
    return out;
  }

  support::TokenizerType GetTokenizerType() const override {
    return support::TokenizerType::kHuggingFace;
  }

  std::vector<std::string> GetTokens() const override { return {}; }

  int GetVocabSize() const override { return 256; }

 private:
  int out_of_range_token_id_;
};

class ScriptedLiteRtLmRunner : public LiteRtLmRunner {
 public:
  struct StepSpec {
    int top_token_id = 7;
    std::vector<float> hidden_state;
  };

  ScriptedLiteRtLmRunner(int vocab_size, int hidden_dim,
                         std::vector<StepSpec> steps)
      : vocab_size_(vocab_size),
        hidden_dim_(hidden_dim),
        steps_(std::move(steps)) {}

  absl::Status Prefill(const lm::ExecutorInputs& inputs) override {
    ++prefill_calls_;
    auto emb_ptr = inputs.GetProjectedAudioEmbeddingsPtr();
    if (!emb_ptr.ok() || *emb_ptr == nullptr) {
      return absl::InvalidArgumentError(
          "Expected continuous prompt embeddings in Prefill.");
    }
    return absl::OkStatus();
  }

  absl::StatusOr<TensorBuffer> Decode(
      const lm::ExecutorInputs& inputs) override {
    const bool has_audio_emb = inputs.GetAudioDataPtr().ok();
    decode_used_audio_emb_.push_back(has_audio_emb);
    if (has_audio_emb) {
      ABSL_ASSIGN_OR_RETURN(const TensorBuffer* emb_buf,
                            inputs.GetProjectedAudioEmbeddingsPtr());
      std::vector<float> emb(hidden_dim_, 0.0f);
      LITERT_RETURN_IF_ERROR(
          const_cast<TensorBuffer*>(emb_buf)->Read<float>(absl::MakeSpan(emb)));
      received_audio_embeddings_.push_back(std::move(emb));
    } else {
      ABSL_ASSIGN_OR_RETURN(const TensorBuffer* tok_buf,
                            inputs.GetTextTokenIdsPtr());
      int32_t tok = -999;
      LITERT_RETURN_IF_ERROR(const_cast<TensorBuffer*>(tok_buf)->Read<int32_t>(
          absl::MakeSpan(&tok, 1)));
      received_text_tokens_.push_back(tok);
    }

    StepSpec spec;
    if (step_cursor_ < steps_.size()) {
      spec = steps_[step_cursor_++];
    } else {
      spec.top_token_id = 7;
      spec.hidden_state.assign(hidden_dim_, 0.0f);
    }

    std::vector<float> out(vocab_size_ + hidden_dim_, -10.0f);
    if (spec.top_token_id >= 0 && spec.top_token_id < vocab_size_) {
      out[spec.top_token_id] = 10.0f;
    }
    for (int i = 0;
         i < hidden_dim_ && i < static_cast<int>(spec.hidden_state.size());
         ++i) {
      out[vocab_size_ + i] = spec.hidden_state[i];
    }

    LITERT_ASSIGN_OR_RETURN(
        TensorBuffer buf,
        support::CreateTensorBuffer<float>({1, 1, vocab_size_ + hidden_dim_}));
    LITERT_RETURN_IF_ERROR(buf.Write<float>(absl::MakeConstSpan(out)));
    return buf;
  }

  absl::Status Reset() override {
    step_cursor_ = 0;
    return absl::OkStatus();
  }

  int prefill_calls() const { return prefill_calls_; }
  const std::vector<bool>& decode_used_audio_emb() const {
    return decode_used_audio_emb_;
  }
  const std::vector<int32_t>& received_text_tokens() const {
    return received_text_tokens_;
  }
  const std::vector<std::vector<float>>& received_audio_embeddings() const {
    return received_audio_embeddings_;
  }

 private:
  int vocab_size_;
  int hidden_dim_;
  std::vector<StepSpec> steps_;
  size_t step_cursor_ = 0;
  int prefill_calls_ = 0;
  std::vector<bool> decode_used_audio_emb_;
  std::vector<int32_t> received_text_tokens_;
  std::vector<std::vector<float>> received_audio_embeddings_;
};

TEST(Lfm2AudioStageTest,
     ExtractsNemoLogMelWithPerFeatureNormalizationAndRunsFastConformer) {
  // Arrange
  Lfm2AudioModelConfig config;
  config.llm_hidden_size = 4;
  config.audio_mel_bins = 128;
  config.audio_subsampling_factor = 8;

  // Verify default LFM2.5-Audio-1.5B architecture constants.
  EXPECT_EQ(lfm2_audio::kTtsHiddenSize, 1024);
  EXPECT_EQ(lfm2_audio::kTtsVocabSize, 2049);
  EXPECT_EQ(lfm2_audio::kTtsNumLayers, 6);
  EXPECT_EQ(lfm2_audio::kTtsNumCodebooks, 8);
  EXPECT_EQ(lfm2_audio::kDetokenizerUpsampleSamples, 1920);
  EXPECT_EQ(lfm2_audio::kBosTokenId, 1);
  EXPECT_EQ(lfm2_audio::kImStartTokenId, 6);
  EXPECT_EQ(lfm2_audio::kImEndTokenId, 7);
  EXPECT_EQ(lfm2_audio::kAudioStartTokenId, 128);
  EXPECT_EQ(lfm2_audio::kTextEndTokenId, 130);

  // First half (800 samples): 440 Hz tone; second half (800 samples): 4000 Hz
  // tone, so low mel bins peak in the first half and high mel bins peak in the
  // second half after per-feature normalization.
  std::vector<float> pcm(1600);
  for (size_t i = 0; i < pcm.size(); ++i) {
    const float freq = (i < 800) ? 440.0f : 4000.0f;
    pcm[i] = 0.5f * std::sin(2.0f * std::numbers::pi_v<float> * freq *
                             static_cast<float>(i) / 16000.0f);
  }

  int valid_frames = 0;
  ASSERT_OK_AND_ASSIGN(
      std::vector<float> log_mel,
      ExtractNemoLogMelSpectrogram(pcm, config, &valid_frames));
  // NeMo FilterbankFeatures.get_seq_len(1600) = floor(1600 / 160) = 10 frames.
  EXPECT_EQ(valid_frames, 10);
  ASSERT_EQ(log_mel.size(), 10u * 128u);

  // Verify per-feature mean is ~0 and Bessel sample std is ~1 across active
  // mel bins (including Slaney mel bins 18 ~444 Hz and 99 ~3975 Hz).
  for (int m : {18, 20, 97, 99}) {
    double sum = 0.0;
    for (int t = 0; t < 10; ++t) {
      sum += log_mel[t * 128 + m];
    }
    const double mean = sum / 10.0;
    EXPECT_NEAR(mean, 0.0, 1e-4) << "mel bin " << m;
    double sq_sum = 0.0;
    for (int t = 0; t < 10; ++t) {
      const double diff = log_mel[t * 128 + m] - mean;
      sq_sum += diff * diff;
    }
    const double bessel_std = std::sqrt(sq_sum / 9.0);
    EXPECT_NEAR(bessel_std, 1.0, 1e-3) << "mel bin " << m;
  }

  // Verify frequency selectivity: in a 128-bin Slaney filterbank over
  // [0, 8000 Hz], 440 Hz maps to bin 18 (~444 Hz) and 4000 Hz maps to bin 99
  // (~3975 Hz). Early frames [1..3] carry 440 Hz; late frames [6..8] carry
  // 4000 Hz.
  double low_early = 0.0;
  double low_late = 0.0;
  double high_early = 0.0;
  double high_late = 0.0;
  for (int t = 1; t <= 3; ++t) {
    low_early += log_mel[t * 128 + 18];
    high_early += log_mel[t * 128 + 99];
  }
  for (int t = 6; t <= 8; ++t) {
    low_late += log_mel[t * 128 + 18];
    high_late += log_mel[t * 128 + 99];
  }
  EXPECT_GT(low_early, low_late + 2.0);
  EXPECT_GT(high_late, high_early + 2.0);

  // Verify exact boundary: 1280 samples = 8 frames (not 9), and <160 samples
  // yields 0 valid frames and an empty spectrogram.
  int exact_8_frames = -1;
  ASSERT_OK_AND_ASSIGN(
      std::vector<float> mel_8,
      ExtractNemoLogMelSpectrogram(absl::MakeConstSpan(pcm).subspan(0, 1280),
                                   config, &exact_8_frames));
  EXPECT_EQ(exact_8_frames, 8);
  EXPECT_EQ(mel_8.size(), 8u * 128u);

  int short_frames = -1;
  ASSERT_OK_AND_ASSIGN(
      std::vector<float> mel_short,
      ExtractNemoLogMelSpectrogram(absl::MakeConstSpan(pcm).subspan(0, 100),
                                   config, &short_frames));
  EXPECT_EQ(short_frames, 0);
  EXPECT_TRUE(mel_short.empty());

  PushInputSource input_source;
  ASSERT_OK_AND_ASSIGN(auto fan_out,
                       Lfm2AudioInputFanOutStage::Create(&input_source));

  int32_t observed_mel_len = -1;
  auto mock_runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*mock_runner, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto mel_buf, support::CreateTensorBuffer<float>({1, 128, 16}));
            LITERT_ASSIGN_OR_RETURN(auto len_buf,
                                    support::CreateTensorBuffer<int32_t>({1}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(mel_buf));
            bufs.push_back(std::move(len_buf));
            return bufs;
          });
  ON_CALL(*mock_runner, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto out_buf, support::CreateTensorBuffer<float>({1, 2, 4}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(out_buf));
            return bufs;
          });
  ON_CALL(*mock_runner, Run(absl::string_view(""), _, _))
      .WillByDefault(
          [&](absl::string_view, absl::Span<const TensorBuffer> inputs,
              absl::Span<const TensorBuffer> outputs) -> absl::Status {
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(inputs[1]).Read<int32_t>(
                    absl::MakeSpan(&observed_mel_len, 1)));
            std::vector<float> fake_out = {1.0f, 2.0f, 3.0f, 4.0f,
                                           5.0f, 6.0f, 7.0f, 8.0f};
            auto& out_buf = const_cast<TensorBuffer&>(outputs[0]);
            LITERT_RETURN_IF_ERROR(
                out_buf.Write<float>(absl::MakeConstSpan(fake_out)));
            return absl::OkStatus();
          });

  ASSERT_OK_AND_ASSIGN(
      auto audio_stage,
      Lfm2AudioStage::Create(fan_out->audio_queue(),
                             fan_out->audio_input_metadata_queue(),
                             std::move(mock_runner), config));

  // Act
  ASSERT_OK(input_source.PushInput(AudioInput{.pcm_samples = std::move(pcm)}));
  ASSERT_OK(input_source.PushInput(OmniSession::EndOfInputTurn{}));
  ASSERT_OK(input_source.Schedule());
  ASSERT_OK(fan_out->Schedule());
  EXPECT_TRUE(audio_stage->NeedSchedule());
  ASSERT_OK(audio_stage->Schedule());

  // Assert: `mel2emb_len(10) = ceil(10 / 8) = 2` frames with placeholder token
  // ID `-100` (no `<|audio_start|>`/`<|audio_end|>` wrapping around input
  // audio).
  EXPECT_EQ(observed_mel_len, 10);
  ASSERT_TRUE(audio_stage->HasOutput());
  ASSERT_OK_AND_ASSIGN(Sequenced<Lfm2AudioEmbeddingSegment> out,
                       audio_stage->GetOutput());
  EXPECT_EQ(out.seqnum, 0u);
  EXPECT_EQ(out.item.num_tokens, 2);
  EXPECT_THAT(out.item.token_ids,
              ElementsAre(lfm2_audio::kAudioEmbedPlaceholderTokenId,
                          lfm2_audio::kAudioEmbedPlaceholderTokenId));
  EXPECT_EQ(out.item.embeddings.size(), 8u);
}

TEST(Lfm2AudioStageTest, CoalescesConsecutiveAudioChunksBitForBit) {
  Lfm2AudioModelConfig config;
  config.llm_hidden_size = 4;
  config.audio_mel_bins = 128;
  config.audio_subsampling_factor = 8;

  std::vector<float> full_pcm(3200);
  for (size_t i = 0; i < full_pcm.size(); ++i) {
    full_pcm[i] = 0.4f * std::sin(2.0f * std::numbers::pi_v<float> * 300.0f *
                                  static_cast<float>(i) / 16000.0f) +
                  0.3f * std::cos(2.0f * std::numbers::pi_v<float> * 1200.0f *
                                  static_cast<float>(i) / 16000.0f *
                                  static_cast<float>(i + 1) / 3200.0f);
  }

  struct RunCapture {
    int encoder_calls = 0;
    int32_t mel_len = -1;
    std::vector<float> mel_input;
    Lfm2AudioEmbeddingSegment segment;
  };

  auto run_chunks = [&](int num_chunks) -> absl::StatusOr<RunCapture> {
    PushInputSource input_source;
    ABSL_ASSIGN_OR_RETURN(auto fan_out,
                          Lfm2AudioInputFanOutStage::Create(&input_source));
    RunCapture capture;
    auto mock_runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
    ON_CALL(*mock_runner, CreateInputBuffers(absl::string_view("")))
        .WillByDefault(
            [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
              LITERT_ASSIGN_OR_RETURN(
                  auto mel_buf,
                  support::CreateTensorBuffer<float>({1, 128, 32}));
              LITERT_ASSIGN_OR_RETURN(
                  auto len_buf, support::CreateTensorBuffer<int32_t>({1}));
              std::vector<TensorBuffer> bufs;
              bufs.push_back(std::move(mel_buf));
              bufs.push_back(std::move(len_buf));
              return bufs;
            });
    ON_CALL(*mock_runner, CreateOutputBuffers(absl::string_view("")))
        .WillByDefault(
            [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
              LITERT_ASSIGN_OR_RETURN(
                  auto out_buf, support::CreateTensorBuffer<float>({1, 4, 4}));
              std::vector<TensorBuffer> bufs;
              bufs.push_back(std::move(out_buf));
              return bufs;
            });
    ON_CALL(*mock_runner, Run(absl::string_view(""), _, _))
        .WillByDefault(
            [&](absl::string_view, absl::Span<const TensorBuffer> inputs,
                absl::Span<const TensorBuffer> outputs) -> absl::Status {
              ++capture.encoder_calls;
              capture.mel_input.assign(128 * 32, 0.0f);
              LITERT_RETURN_IF_ERROR(
                  const_cast<TensorBuffer&>(inputs[0]).Read<float>(
                      absl::MakeSpan(capture.mel_input)));
              LITERT_RETURN_IF_ERROR(
                  const_cast<TensorBuffer&>(inputs[1]).Read<int32_t>(
                      absl::MakeSpan(&capture.mel_len, 1)));
              std::vector<float> out(16, 0.0f);
              for (int i = 0; i < 16; ++i) {
                out[i] = capture.mel_input[i * 128 + (i % 20)];
              }
              LITERT_RETURN_IF_ERROR(
                  const_cast<TensorBuffer&>(outputs[0])
                      .Write<float>(absl::MakeConstSpan(out)));
              return absl::OkStatus();
            });

    ABSL_ASSIGN_OR_RETURN(
        auto audio_stage,
        Lfm2AudioStage::Create(fan_out->audio_queue(),
                               fan_out->audio_input_metadata_queue(),
                               std::move(mock_runner), config));

    const size_t chunk_size = full_pcm.size() / num_chunks;
    for (int c = 0; c < num_chunks; ++c) {
      const size_t begin = c * chunk_size;
      const size_t end =
          (c + 1 == num_chunks) ? full_pcm.size() : (c + 1) * chunk_size;
      std::vector<float> slice(full_pcm.begin() + begin,
                               full_pcm.begin() + end);
      ABSL_RETURN_IF_ERROR(
          input_source.PushInput(AudioInput{.pcm_samples = std::move(slice)}));
    }
    ABSL_RETURN_IF_ERROR(input_source.PushInput(OmniSession::EndOfInputTurn{}));
    ABSL_RETURN_IF_ERROR(input_source.Schedule());
    ABSL_RETURN_IF_ERROR(fan_out->Schedule());
    ABSL_RETURN_IF_ERROR(audio_stage->Schedule());
    ABSL_ASSIGN_OR_RETURN(Sequenced<Lfm2AudioEmbeddingSegment> out,
                          audio_stage->GetOutput());
    capture.segment = std::move(out.item);
    return capture;
  };

  ASSERT_OK_AND_ASSIGN(RunCapture one_chunk, run_chunks(/*num_chunks=*/1));
  ASSERT_OK_AND_ASSIGN(RunCapture four_chunks, run_chunks(/*num_chunks=*/4));

  EXPECT_EQ(one_chunk.encoder_calls, 1);
  EXPECT_EQ(four_chunks.encoder_calls, 1);
  EXPECT_EQ(one_chunk.mel_len, 20);
  EXPECT_EQ(four_chunks.mel_len, 20);
  EXPECT_EQ(one_chunk.mel_input, four_chunks.mel_input);
  EXPECT_EQ(one_chunk.segment.num_tokens, 3);
  EXPECT_EQ(one_chunk.segment.embeddings, four_chunks.segment.embeddings);
  EXPECT_EQ(one_chunk.segment.token_ids, four_chunks.segment.token_ids);
}

TEST(Lfm2AudioStageTest,
     OrdersMetadataBySeqnumRejectsNonMonoAndSkipsShortAudio) {
  Lfm2AudioModelConfig config;
  config.llm_hidden_size = 4;
  config.audio_mel_bins = 128;
  config.audio_subsampling_factor = 8;

  PushInputSource input_source;
  ASSERT_OK_AND_ASSIGN(auto fan_out,
                       Lfm2AudioInputFanOutStage::Create(&input_source));

  std::vector<int32_t> observed_mel_lens;
  auto mock_runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*mock_runner, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto mel_buf, support::CreateTensorBuffer<float>({1, 128, 32}));
            LITERT_ASSIGN_OR_RETURN(auto len_buf,
                                    support::CreateTensorBuffer<int32_t>({1}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(mel_buf));
            bufs.push_back(std::move(len_buf));
            return bufs;
          });
  ON_CALL(*mock_runner, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto out_buf, support::CreateTensorBuffer<float>({1, 4, 4}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(out_buf));
            return bufs;
          });
  ON_CALL(*mock_runner, Run(absl::string_view(""), _, _))
      .WillByDefault(
          [&](absl::string_view, absl::Span<const TensorBuffer> inputs,
              absl::Span<const TensorBuffer> outputs) -> absl::Status {
            int32_t mel_len = 0;
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(inputs[1]).Read<int32_t>(
                    absl::MakeSpan(&mel_len, 1)));
            observed_mel_lens.push_back(mel_len);
            std::vector<float> fake_out(16, 1.0f);
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(outputs[0])
                    .Write<float>(absl::MakeConstSpan(fake_out)));
            return absl::OkStatus();
          });

  ASSERT_OK_AND_ASSIGN(
      auto audio_stage,
      Lfm2AudioStage::Create(fan_out->audio_queue(),
                             fan_out->audio_input_metadata_queue(),
                             std::move(mock_runner), config));

  // 1. Metadata ordering: first audio at default 16 kHz (1600 samples = 10
  //    frames), then metadata switching to 32 kHz, then 3200 samples at 32 kHz
  //    (resamples to 1600 samples = 10 frames).
  ASSERT_OK(input_source.PushInput(
      AudioInput{.pcm_samples = std::vector<float>(1600, 0.1f)}));
  ASSERT_OK(input_source.PushInput(
      AudioInputMetadata{.sample_rate_hz = 32000, .num_channels = 1}));
  ASSERT_OK(input_source.PushInput(
      AudioInput{.pcm_samples = std::vector<float>(3200, 0.2f)}));
  ASSERT_OK(input_source.PushInput(OmniSession::EndOfInputTurn{}));
  ASSERT_OK(input_source.Schedule());
  ASSERT_OK(fan_out->Schedule());
  ASSERT_OK(audio_stage->Schedule());

  // Both audio segments must have mel_len == 10 (the later 32 kHz metadata
  // must NOT apply retroactively to the first audio segment).
  EXPECT_THAT(observed_mel_lens, ElementsAre(10, 10));
  ASSERT_OK_AND_ASSIGN(auto seg0, audio_stage->GetOutput());
  EXPECT_EQ(seg0.seqnum, 0u);
  EXPECT_EQ(seg0.item.num_tokens, 2);
  ASSERT_OK_AND_ASSIGN(auto seg1, audio_stage->GetOutput());
  EXPECT_EQ(seg1.seqnum, 1u);  // AudioInputMetadata placeholder
  EXPECT_EQ(seg1.item.num_tokens, 0);
  ASSERT_OK_AND_ASSIGN(auto seg2, audio_stage->GetOutput());
  EXPECT_EQ(seg2.seqnum, 2u);
  EXPECT_EQ(seg2.item.num_tokens, 2);

  // 2. Empty and <160-sample audio emit empty segments without running encoder.
  fan_out->Reset();
  audio_stage->Reset();
  observed_mel_lens.clear();
  ASSERT_OK(input_source.PushInput(AudioInput{.pcm_samples = {}}));
  ASSERT_OK(input_source.PushInput(OmniSession::EndOfInputTurn{}));
  ASSERT_OK(input_source.PushInput(
      AudioInput{.pcm_samples = std::vector<float>(100, 0.5f)}));
  ASSERT_OK(input_source.PushInput(OmniSession::EndOfInputTurn{}));
  ASSERT_OK(input_source.Schedule());
  ASSERT_OK(fan_out->Schedule());
  ASSERT_OK(audio_stage->Schedule());
  EXPECT_TRUE(observed_mel_lens.empty());
  ASSERT_OK_AND_ASSIGN(auto empty_seg, audio_stage->GetOutput());
  EXPECT_EQ(empty_seg.item.num_tokens, 0);
  ASSERT_OK_AND_ASSIGN(auto short_seg, audio_stage->GetOutput());
  EXPECT_EQ(short_seg.item.num_tokens, 0);

  // 3. Non-mono audio metadata (num_channels == 2) is rejected.
  fan_out->Reset();
  audio_stage->Reset();
  ASSERT_OK(input_source.PushInput(
      AudioInputMetadata{.sample_rate_hz = 16000, .num_channels = 2}));
  ASSERT_OK(input_source.Schedule());
  ASSERT_OK(fan_out->Schedule());
  EXPECT_THAT(audio_stage->Schedule(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

std::shared_ptr<NiceMock<MockLiteRtRunner>> CreateMockTextEmbedderRunner(
    int hidden_dim) {
  auto mock_emb = std::make_shared<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*mock_emb, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto in_buf, support::CreateTensorBuffer<int32_t>({1, 1}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(in_buf));
            return bufs;
          });
  ON_CALL(*mock_emb, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault([hidden_dim](absl::string_view)
                         -> absl::StatusOr<std::vector<TensorBuffer>> {
        LITERT_ASSIGN_OR_RETURN(
            auto out_buf,
            support::CreateTensorBuffer<float>({1, 1, hidden_dim}));
        std::vector<TensorBuffer> bufs;
        bufs.push_back(std::move(out_buf));
        return bufs;
      });
  ON_CALL(*mock_emb, Run(absl::string_view(""), _, _))
      .WillByDefault(
          [hidden_dim](absl::string_view, absl::Span<const TensorBuffer> inputs,
                       absl::Span<const TensorBuffer> outputs) -> absl::Status {
            int32_t tok = 0;
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(inputs[0]).Read<int32_t>(
                    absl::MakeSpan(&tok, 1)));
            std::vector<float> emb(hidden_dim, static_cast<float>(tok) * 0.01f);
            LITERT_RETURN_IF_ERROR(const_cast<TensorBuffer&>(outputs[0])
                                       .Write<float>(absl::MakeConstSpan(emb)));
            return absl::OkStatus();
          });
  return mock_emb;
}

TEST(Lfm2AudioTextAndFrontendTest,
     BuildsChatStatePromptAndRejectsOutOfRangeTokensAndImageInputs) {
  // Arrange
  Lfm2AudioModelConfig config;
  config.llm_hidden_size = 4;
  config.llm_vocab_size = 256;

  PushInputSource input_source;
  ASSERT_OK_AND_ASSIGN(auto fan_out_no_audio,
                       Lfm2AudioInputFanOutStage::Create(
                           &input_source, /*has_audio_encoder=*/false));
  // Rejects AudioInput when has_audio_encoder is false.
  ASSERT_OK(input_source.PushInput(AudioInput{.pcm_samples = {0.1f, 0.2f}}));
  ASSERT_OK(input_source.Schedule());
  EXPECT_THAT(fan_out_no_audio->Schedule(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  fan_out_no_audio->Reset();

  ASSERT_OK_AND_ASSIGN(auto fan_out,
                       Lfm2AudioInputFanOutStage::Create(&input_source));

  // Rejects ImageInput with kInvalidArgument.
  ASSERT_OK(input_source.PushInput(ImageInput{}));
  ASSERT_OK(input_source.Schedule());
  EXPECT_THAT(fan_out->Schedule(),
              StatusIs(absl::StatusCode::kInvalidArgument));
  fan_out->Reset();

  auto tokenizer =
      std::make_shared<FakeTokenizer>(/*out_of_range_token_id=*/300);
  auto embedder_runner = CreateMockTextEmbedderRunner(config.llm_hidden_size);
  ASSERT_OK_AND_ASSIGN(auto text_stage,
                       Lfm2AudioTextStage::Create(
                           fan_out->text_queue(), fan_out->end_of_turn_queue(),
                           tokenizer, embedder_runner, config));
  ASSERT_OK_AND_ASSIGN(
      auto frontend,
      Lfm2AudioFrontendStage::Create(text_stage.get(), /*audio_stage=*/nullptr,
                                     tokenizer, embedder_runner, config));

  // Act 1: Out-of-range token ID (`300 >= 256`) must fail with
  // kInvalidArgument (never clamped).
  ASSERT_OK(
      input_source.PushInput(OmniSession::TextInput{.text = "OUT_OF_RANGE"}));
  ASSERT_OK(input_source.Schedule());
  ASSERT_OK(fan_out->Schedule());
  EXPECT_THAT(text_stage->Schedule(),
              StatusIs(absl::StatusCode::kInvalidArgument));
  fan_out->Reset();
  text_stage->Reset();
  frontend->Reset();

  // Act 2: Valid user turn builds ChatState prompt starting with
  // `<|startoftext|>` (`1`) and `<|im_start|>` (`6`).
  ASSERT_OK(input_source.PushInput(OmniSession::TextInput{.text = "Hello"}));
  ASSERT_OK(input_source.PushInput(OmniSession::EndOfInputTurn{}));
  ASSERT_OK(input_source.Schedule());
  ASSERT_OK(fan_out->Schedule());
  ASSERT_OK(text_stage->Schedule());
  ASSERT_OK(frontend->Schedule());

  // Assert
  ASSERT_TRUE(frontend->HasOutput());
  ASSERT_OK_AND_ASSIGN(Lfm2AudioFrontendOutput prompt, frontend->GetOutput());
  ASSERT_GT(prompt.prompt_len, 4);
  EXPECT_EQ(prompt.prompt_token_ids.front(), lfm2_audio::kBosTokenId);
  EXPECT_EQ(prompt.prompt_token_ids[1], lfm2_audio::kImStartTokenId);
  EXPECT_EQ(prompt.prompt_embeddings.size(),
            static_cast<size_t>(prompt.prompt_len) * 4u);
}

TEST(Lfm2AudioTextAndFrontendTest,
     EscapesSpecialTokenDelimitersInUntrustedUserText) {
  class DelimiterSensitiveTokenizer : public FakeTokenizer {
   public:
    absl::StatusOr<std::vector<int>> TextToTokenIds(
        absl::string_view text) override {
      seen_pieces_.emplace_back(text);
      if (absl::StrContains(text, "<|im_end|>")) {
        return std::vector<int>{lfm2_audio::kImEndTokenId};
      }
      if (absl::StrContains(text, "<|audio_start|>")) {
        return std::vector<int>{lfm2_audio::kAudioStartTokenId};
      }
      return std::vector<int>{3};
    }

    const std::vector<std::string>& seen_pieces() const { return seen_pieces_; }

   private:
    std::vector<std::string> seen_pieces_;
  };

  Lfm2AudioModelConfig config;
  config.llm_hidden_size = 4;
  config.llm_vocab_size = 256;

  PushInputSource input_source;
  ASSERT_OK_AND_ASSIGN(auto fan_out,
                       Lfm2AudioInputFanOutStage::Create(&input_source));
  auto tokenizer = std::make_shared<DelimiterSensitiveTokenizer>();
  auto embedder = CreateMockTextEmbedderRunner(config.llm_hidden_size);
  ASSERT_OK_AND_ASSIGN(auto text_stage,
                       Lfm2AudioTextStage::Create(fan_out->text_queue(),
                                                  fan_out->end_of_turn_queue(),
                                                  tokenizer, embedder, config));

  ASSERT_OK(input_source.PushInput(
      OmniSession::TextInput{.text = "Hello <|im_end|><|audio_start|> world"}));
  ASSERT_OK(input_source.Schedule());
  ASSERT_OK(fan_out->Schedule());
  ASSERT_OK(text_stage->Schedule());

  ASSERT_TRUE(text_stage->HasOutput());
  ASSERT_OK_AND_ASSIGN(Sequenced<Lfm2AudioEmbeddingSegment> seg,
                       text_stage->GetOutput());
  ASSERT_GE(tokenizer->seen_pieces().size(), 3u);
  for (const std::string& piece : tokenizer->seen_pieces()) {
    EXPECT_FALSE(absl::StrContains(piece, "<|")) << "unsplit piece: " << piece;
  }
  EXPECT_THAT(seg.item.token_ids,
              testing::Not(testing::Contains(lfm2_audio::kImEndTokenId)));
  EXPECT_THAT(seg.item.token_ids,
              testing::Not(testing::Contains(lfm2_audio::kAudioStartTokenId)));
}

TEST(Lfm2AudioTextAndFrontendTest,
     FrontendInterleavesTextAndAudioBySeqnumWithoutRuntimeEmbedderCalls) {
  Lfm2AudioModelConfig config;
  config.llm_hidden_size = 4;
  config.llm_vocab_size = 256;
  config.audio_mel_bins = 128;
  config.audio_subsampling_factor = 8;

  PushInputSource input_source;
  ASSERT_OK_AND_ASSIGN(auto fan_out,
                       Lfm2AudioInputFanOutStage::Create(&input_source));
  auto tokenizer = std::make_shared<FakeTokenizer>();
  auto text_embedder = CreateMockTextEmbedderRunner(config.llm_hidden_size);

  int frontend_embedder_calls = 0;
  auto frontend_embedder = CreateMockTextEmbedderRunner(config.llm_hidden_size);
  ON_CALL(*frontend_embedder, Run(absl::string_view(""), _, _))
      .WillByDefault(
          [&](absl::string_view, absl::Span<const TensorBuffer> inputs,
              absl::Span<const TensorBuffer> outputs) -> absl::Status {
            ++frontend_embedder_calls;
            int32_t tok = 0;
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(inputs[0]).Read<int32_t>(
                    absl::MakeSpan(&tok, 1)));
            std::vector<float> emb(config.llm_hidden_size,
                                   static_cast<float>(tok) * 0.01f);
            LITERT_RETURN_IF_ERROR(const_cast<TensorBuffer&>(outputs[0])
                                       .Write<float>(absl::MakeConstSpan(emb)));
            return absl::OkStatus();
          });

  auto mock_audio_runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*mock_audio_runner, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto mel_buf, support::CreateTensorBuffer<float>({1, 128, 16}));
            LITERT_ASSIGN_OR_RETURN(auto len_buf,
                                    support::CreateTensorBuffer<int32_t>({1}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(mel_buf));
            bufs.push_back(std::move(len_buf));
            return bufs;
          });
  ON_CALL(*mock_audio_runner, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto out_buf, support::CreateTensorBuffer<float>({1, 2, 4}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(out_buf));
            return bufs;
          });
  ON_CALL(*mock_audio_runner, Run(absl::string_view(""), _, _))
      .WillByDefault(
          [](absl::string_view, absl::Span<const TensorBuffer> /*inputs*/,
             absl::Span<const TensorBuffer> outputs) -> absl::Status {
            std::vector<float> fake_out(8, 0.75f);
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(outputs[0])
                    .Write<float>(absl::MakeConstSpan(fake_out)));
            return absl::OkStatus();
          });

  ASSERT_OK_AND_ASSIGN(auto text_stage,
                       Lfm2AudioTextStage::Create(
                           fan_out->text_queue(), fan_out->end_of_turn_queue(),
                           tokenizer, text_embedder, config));
  ASSERT_OK_AND_ASSIGN(
      auto audio_stage,
      Lfm2AudioStage::Create(fan_out->audio_queue(),
                             fan_out->audio_input_metadata_queue(),
                             std::move(mock_audio_runner), config));
  ASSERT_OK_AND_ASSIGN(
      auto frontend,
      Lfm2AudioFrontendStage::Create(text_stage.get(), audio_stage.get(),
                                     tokenizer, frontend_embedder, config));

  // All control-token embeddings must be precomputed during Create(), with
  // zero runtime calls to `frontend_embedder` during Schedule().
  const int calls_after_create = frontend_embedder_calls;
  ASSERT_GT(calls_after_create, 0);

  // Turn 1: text -> 2 coalesced audio chunks -> text -> EndOfInputTurn.
  ASSERT_OK(input_source.PushInput(OmniSession::TextInput{.text = "Hello"}));
  ASSERT_OK(input_source.PushInput(
      AudioInput{.pcm_samples = std::vector<float>(800, 0.1f)}));
  ASSERT_OK(input_source.PushInput(
      AudioInput{.pcm_samples = std::vector<float>(800, 0.2f)}));
  ASSERT_OK(input_source.PushInput(OmniSession::TextInput{.text = "world"}));
  ASSERT_OK(input_source.PushInput(OmniSession::EndOfInputTurn{}));
  ASSERT_OK(input_source.Schedule());
  ASSERT_OK(fan_out->Schedule());
  ASSERT_OK(text_stage->Schedule());
  ASSERT_OK(audio_stage->Schedule());
  ASSERT_OK(frontend->Schedule());

  EXPECT_EQ(frontend_embedder_calls, calls_after_create);
  ASSERT_TRUE(frontend->HasOutput());
  ASSERT_OK_AND_ASSIGN(Lfm2AudioFrontendOutput turn1, frontend->GetOutput());
  EXPECT_EQ(turn1.prompt_token_ids.front(), lfm2_audio::kBosTokenId);
  // Find the two `-100` audio placeholder tokens and verify they are bracketed
  // by the preceding and trailing text tokens (`3`).
  auto audio_it =
      std::find(turn1.prompt_token_ids.begin(), turn1.prompt_token_ids.end(),
                lfm2_audio::kAudioEmbedPlaceholderTokenId);
  ASSERT_NE(audio_it, turn1.prompt_token_ids.end());
  ASSERT_GE(std::distance(turn1.prompt_token_ids.begin(), audio_it), 1);
  ASSERT_GE(std::distance(audio_it, turn1.prompt_token_ids.end()), 3);
  EXPECT_EQ(*(audio_it - 1), 3);
  EXPECT_EQ(*audio_it, lfm2_audio::kAudioEmbedPlaceholderTokenId);
  EXPECT_EQ(*(audio_it + 1), lfm2_audio::kAudioEmbedPlaceholderTokenId);
  EXPECT_EQ(*(audio_it + 2), 3);

  // Turn 2: audio-only turn must NOT repeat BOS/system prefix and must still
  // make zero runtime calls to `frontend_embedder`.
  ASSERT_OK(input_source.PushInput(
      AudioInputMetadata{.sample_rate_hz = 16000, .num_channels = 1}));
  ASSERT_OK(input_source.PushInput(
      AudioInput{.pcm_samples = std::vector<float>(1600, 0.3f)}));
  ASSERT_OK(input_source.PushInput(OmniSession::EndOfInputTurn{}));
  ASSERT_OK(input_source.Schedule());
  ASSERT_OK(fan_out->Schedule());
  ASSERT_OK(text_stage->Schedule());
  ASSERT_OK(audio_stage->Schedule());
  ASSERT_OK(frontend->Schedule());

  EXPECT_EQ(frontend_embedder_calls, calls_after_create);
  ASSERT_TRUE(frontend->HasOutput());
  ASSERT_OK_AND_ASSIGN(Lfm2AudioFrontendOutput turn2, frontend->GetOutput());
  ASSERT_GT(turn2.prompt_len, 2);
  EXPECT_EQ(turn2.prompt_token_ids.front(), lfm2_audio::kImStartTokenId);
  EXPECT_THAT(turn2.prompt_token_ids,
              testing::Not(testing::Contains(lfm2_audio::kBosTokenId)));
  EXPECT_THAT(turn2.prompt_token_ids,
              testing::Contains(lfm2_audio::kAudioEmbedPlaceholderTokenId));
}

TEST(Lfm2AudioGenerationStageTest,
     InterleavedTextAndAudioOutWithDepthformerAndDetokenizer) {
  // Arrange
  Lfm2AudioModelConfig config;
  config.llm_hidden_size = 4;
  config.llm_vocab_size = 256;
  config.tts_hidden_size = 4;
  config.tts_vocab_size = 32;  // EOAudio = 31, valid detok codes in [0, 30]
  config.tts_num_layers = 1;
  config.tts_num_codebooks = 8;
  config.tts_max_cache_len = 8;
  config.vocoder_chunk_frames = 2;
  config.detokenizer_upsample_samples = 16;
  config.generate_audio = true;
  config.tts_temperature = 0.0f;
  config.max_new_tokens = 8;

  PushInputSource input_source;
  ASSERT_OK_AND_ASSIGN(auto fan_out,
                       Lfm2AudioInputFanOutStage::Create(&input_source));
  auto tokenizer = std::make_shared<FakeTokenizer>();
  auto embedder_runner = CreateMockTextEmbedderRunner(config.llm_hidden_size);
  ASSERT_OK_AND_ASSIGN(auto text_stage,
                       Lfm2AudioTextStage::Create(
                           fan_out->text_queue(), fan_out->end_of_turn_queue(),
                           tokenizer, embedder_runner, config));
  ASSERT_OK_AND_ASSIGN(
      auto frontend,
      Lfm2AudioFrontendStage::Create(text_stage.get(), /*audio_stage=*/nullptr,
                                     tokenizer, embedder_runner, config));

  // Scripted LFM2 backbone steps:
  // Step 0 (TEXT): emits token 3 ("Hello").
  // Step 1 (TEXT): emits token 128 (`<|audio_start|>`), switching to AUDIO_OUT.
  // Step 2 (AUDIO_OUT, decoding token 128): hidden_state = {1, 0, 0, 0} ->
  //   Depthformer generates 1 valid 8-codebook frame `[10..17]` and feeds its
  //   `audio_embedding` back into `llm_runner->Decode`.
  // Step 3 (AUDIO_OUT, decoding audio_embedding): hidden_state = {0, 1, 0, 0}
  //   -> Depthformer emits `EOAudio` (`31`) at codebook 0, switching back to
  //   TEXT and feeding `audio_embedding([31]*8)` into `llm_runner->Decode`.
  // Step 4 (TEXT, decoding EOAudio embedding): emits token 4 (" world").
  // Step 5 (TEXT): emits token 7 (`<|im_end|>`), stopping generation.
  auto scripted_llm = std::make_shared<ScriptedLiteRtLmRunner>(
      config.llm_vocab_size, config.llm_hidden_size,
      std::vector<ScriptedLiteRtLmRunner::StepSpec>{
          {.top_token_id = 3, .hidden_state = {0.1f, 0.2f, 0.3f, 0.4f}},
          {.top_token_id = lfm2_audio::kAudioStartTokenId,
           .hidden_state = {0.5f, 0.6f, 0.7f, 0.8f}},
          {.top_token_id = 0, .hidden_state = {1.0f, 0.0f, 0.0f, 0.0f}},
          {.top_token_id = 0, .hidden_state = {0.0f, 1.0f, 0.0f, 0.0f}},
          {.top_token_id = 4, .hidden_state = {0.2f, 0.2f, 0.2f, 0.2f}},
          {.top_token_id = lfm2_audio::kImEndTokenId,
           .hidden_state = {0.0f, 0.0f, 0.0f, 0.0f}},
      });

  // Mock TTS projector (`depth_linear` + `audio_embedding`).
  auto mock_proj = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*mock_proj, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto h_buf, support::CreateTensorBuffer<float>({1, 1, 4}));
            LITERT_ASSIGN_OR_RETURN(
                auto c_buf, support::CreateTensorBuffer<int32_t>({1, 8}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(h_buf));
            bufs.push_back(std::move(c_buf));
            return bufs;
          });
  ON_CALL(*mock_proj, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto df_in, support::CreateTensorBuffer<float>({1, 8, 4}));
            LITERT_ASSIGN_OR_RETURN(
                auto aud_emb, support::CreateTensorBuffer<float>({1, 1, 4}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(df_in));
            bufs.push_back(std::move(aud_emb));
            return bufs;
          });
  ON_CALL(*mock_proj, Run(absl::string_view(""), _, _))
      .WillByDefault([](absl::string_view,
                        absl::Span<const TensorBuffer> inputs,
                        absl::Span<const TensorBuffer> outputs)
                         -> absl::Status {
        std::vector<float> h(4, 0.0f);
        std::vector<int32_t> codes(8, 0);
        LITERT_RETURN_IF_ERROR(const_cast<TensorBuffer&>(inputs[0]).Read<float>(
            absl::MakeSpan(h)));
        LITERT_RETURN_IF_ERROR(
            const_cast<TensorBuffer&>(inputs[1]).Read<int32_t>(
                absl::MakeSpan(codes)));
        std::vector<float> df_in(32, h[0]);
        std::vector<float> aud_emb = {
            static_cast<float>(codes[0]), static_cast<float>(codes[1]),
            static_cast<float>(codes[2]), static_cast<float>(codes[7])};
        LITERT_RETURN_IF_ERROR(const_cast<TensorBuffer&>(outputs[0])
                                   .Write<float>(absl::MakeConstSpan(df_in)));
        LITERT_RETURN_IF_ERROR(const_cast<TensorBuffer&>(outputs[1])
                                   .Write<float>(absl::MakeConstSpan(aud_emb)));
        return absl::OkStatus();
      });

  // Mock 8-codebook Depthformer (`tts_decoder`).
  auto mock_dec = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*mock_dec, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto cond_buf, support::CreateTensorBuffer<float>({1, 1, 4}));
            LITERT_ASSIGN_OR_RETURN(
                auto prev_buf, support::CreateTensorBuffer<int32_t>({1, 1}));
            LITERT_ASSIGN_OR_RETURN(auto step_buf,
                                    support::CreateTensorBuffer<int32_t>({1}));
            LITERT_ASSIGN_OR_RETURN(
                auto kv_buf, support::CreateTensorBuffer<float>({1, 2, 8, 4}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(cond_buf));
            bufs.push_back(std::move(prev_buf));
            bufs.push_back(std::move(step_buf));
            bufs.push_back(std::move(kv_buf));
            return bufs;
          });
  ON_CALL(*mock_dec, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto logits_buf,
                support::CreateTensorBuffer<float>({1, 1, 32}));
            LITERT_ASSIGN_OR_RETURN(
                auto kv_buf, support::CreateTensorBuffer<float>({1, 2, 8, 4}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(logits_buf));
            bufs.push_back(std::move(kv_buf));
            return bufs;
          });
  ON_CALL(*mock_dec, Run(absl::string_view(""), _, _))
      .WillByDefault(
          [](absl::string_view, absl::Span<const TensorBuffer> inputs,
             absl::Span<const TensorBuffer> outputs) -> absl::Status {
            std::vector<float> cond(4, 0.0f);
            int32_t step_idx = 0;
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(inputs[0]).Read<float>(
                    absl::MakeSpan(cond)));
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(inputs[2]).Read<int32_t>(
                    absl::MakeSpan(&step_idx, 1)));
            std::vector<float> logits(32, -5.0f);
            if (cond[0] > 0.5f) {
              // Frame 0: emit code `10 + step_idx`, and also set `logits[31]`
              // high on `step_idx > 0` to verify `Lfm2AudioGenerationStage`
              // masks out `EOAudio` (`31`) on codebooks 1..7!
              logits[10 + step_idx] = 10.0f;
              if (step_idx > 0) {
                logits[31] = 20.0f;
              }
            } else {
              // Frame 1: emit `EOAudio` (`31`) at codebook 0.
              logits[31] = 10.0f;
            }
            std::vector<float> kv(64, 0.0f);
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(outputs[0])
                    .Write<float>(absl::MakeConstSpan(logits)));
            LITERT_RETURN_IF_ERROR(const_cast<TensorBuffer&>(outputs[1])
                                       .Write<float>(absl::MakeConstSpan(kv)));
            return absl::OkStatus();
          });

  // Mock LFM2AudioDetokenizer (`[1, 8, 2] -> [1, 32]`).
  std::vector<int32_t> observed_vocoder_codes;
  auto mock_voc = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*mock_voc, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto in_buf, support::CreateTensorBuffer<int32_t>({1, 8, 2}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(in_buf));
            return bufs;
          });
  ON_CALL(*mock_voc, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto out_buf, support::CreateTensorBuffer<float>({1, 32}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(out_buf));
            return bufs;
          });
  ON_CALL(*mock_voc, Run(absl::string_view(""), _, _))
      .WillByDefault(
          [&](absl::string_view, absl::Span<const TensorBuffer> inputs,
              absl::Span<const TensorBuffer> outputs) -> absl::Status {
            observed_vocoder_codes.resize(16);
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(inputs[0]).Read<int32_t>(
                    absl::MakeSpan(observed_vocoder_codes)));
            std::vector<float> pcm_out(32, 0.25f);
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(outputs[0])
                    .Write<float>(absl::MakeConstSpan(pcm_out)));
            return absl::OkStatus();
          });

  ASSERT_OK_AND_ASSIGN(
      auto gen_stage,
      Lfm2AudioGenerationStage::Create(
          frontend.get(), tokenizer, scripted_llm, std::move(mock_proj),
          std::move(mock_dec), std::move(mock_voc), config));

  // Act
  ASSERT_OK(input_source.PushInput(OmniSession::TextInput{.text = "Hello"}));
  ASSERT_OK(input_source.PushInput(OmniSession::EndOfInputTurn{}));
  ASSERT_OK(input_source.Schedule());
  ASSERT_OK(fan_out->Schedule());
  ASSERT_OK(text_stage->Schedule());
  ASSERT_OK(frontend->Schedule());
  ASSERT_OK(gen_stage->Schedule());

  // Assert:
  // 1. Both audio frames (valid frame `[10..17]` and EOAudio frame `[31]*8`)
  //    fed their `audio_embedding` back into `llm_runner->Decode`!
  ASSERT_EQ(scripted_llm->received_audio_embeddings().size(), 2u);
  EXPECT_THAT(scripted_llm->received_audio_embeddings()[0],
              ElementsAre(10.0f, 11.0f, 12.0f, 17.0f));
  EXPECT_THAT(scripted_llm->received_audio_embeddings()[1],
              ElementsAre(31.0f, 31.0f, 31.0f, 31.0f));

  // 2. Detokenizer received `[1, 8, 2]` with frame 0 = `[10..17]` and frame 1
  //    right-padded with `-1` (and the EOAudio frame excluded).
  ASSERT_EQ(observed_vocoder_codes.size(), 16u);
  for (int k = 0; k < 8; ++k) {
    EXPECT_EQ(observed_vocoder_codes[k * 2 + 0], 10 + k);
    EXPECT_EQ(observed_vocoder_codes[k * 2 + 1], -1);
  }

  // 3. Stage emitted TextOutput("Hello world"), AudioOutput (16 samples for 1
  //    valid frame), and EndOfOutputTurn.
  ASSERT_TRUE(gen_stage->HasOutput());
  ASSERT_OK_AND_ASSIGN(Output out1, gen_stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(out1));
  EXPECT_EQ(std::get<TextOutput>(out1).confirmed_text, "Hello world");

  ASSERT_TRUE(gen_stage->HasOutput());
  ASSERT_OK_AND_ASSIGN(Output out2, gen_stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<AudioOutput>(out2));
  EXPECT_EQ(std::get<AudioOutput>(out2).pcm_samples.size(), 16u);
  EXPECT_EQ(std::get<AudioOutput>(out2).sample_rate_hz, 24000);

  ASSERT_TRUE(gen_stage->HasOutput());
  ASSERT_OK_AND_ASSIGN(Output out3, gen_stage->GetOutput());
  EXPECT_TRUE(std::holds_alternative<EndOfOutputTurn>(out3));

  // 4. Turn-end commit appended `<|im_end|>` (7) and `\n` (5) to KV cache.
  const auto& rx_tokens = scripted_llm->received_text_tokens();
  ASSERT_GE(rx_tokens.size(), 2u);
  EXPECT_EQ(rx_tokens[rx_tokens.size() - 2], lfm2_audio::kImEndTokenId);
  EXPECT_EQ(rx_tokens[rx_tokens.size() - 1], 5);
}

TEST(Lfm2AudioGenerationStageTest,
     RejectsMismatchedSubmodelBufferShapesAndDtypes) {
  Lfm2AudioModelConfig config;
  config.llm_hidden_size = 4;
  config.llm_vocab_size = 256;
  config.tts_hidden_size = 4;
  config.tts_vocab_size = 32;
  config.tts_num_layers = 1;
  config.tts_num_codebooks = 8;
  config.tts_max_cache_len = 8;

  PushInputSource input_source;
  ASSERT_OK_AND_ASSIGN(auto fan_out,
                       Lfm2AudioInputFanOutStage::Create(&input_source));
  auto tokenizer = std::make_shared<FakeTokenizer>();
  auto embedder = CreateMockTextEmbedderRunner(config.llm_hidden_size);
  ASSERT_OK_AND_ASSIGN(auto text_stage,
                       Lfm2AudioTextStage::Create(fan_out->text_queue(),
                                                  fan_out->end_of_turn_queue(),
                                                  tokenizer, embedder, config));
  ASSERT_OK_AND_ASSIGN(
      auto frontend,
      Lfm2AudioFrontendStage::Create(text_stage.get(), /*audio_stage=*/nullptr,
                                     tokenizer, embedder, config));
  auto llm = std::make_shared<ScriptedLiteRtLmRunner>(
      config.llm_vocab_size, config.llm_hidden_size,
      std::vector<ScriptedLiteRtLmRunner::StepSpec>{});

  // 1. tts_projector with wrong dtype on input[0] (Int32 instead of Float32).
  {
    auto bad_proj = std::make_unique<NiceMock<MockLiteRtRunner>>();
    ON_CALL(*bad_proj, CreateInputBuffers(absl::string_view("")))
        .WillByDefault(
            [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
              LITERT_ASSIGN_OR_RETURN(
                  auto h, support::CreateTensorBuffer<int32_t>({1, 1, 4}));
              LITERT_ASSIGN_OR_RETURN(
                  auto c, support::CreateTensorBuffer<int32_t>({1, 8}));
              std::vector<TensorBuffer> bufs;
              bufs.push_back(std::move(h));
              bufs.push_back(std::move(c));
              return bufs;
            });
    ON_CALL(*bad_proj, CreateOutputBuffers(absl::string_view("")))
        .WillByDefault(
            [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
              LITERT_ASSIGN_OR_RETURN(
                  auto df, support::CreateTensorBuffer<float>({1, 8, 4}));
              LITERT_ASSIGN_OR_RETURN(
                  auto emb, support::CreateTensorBuffer<float>({1, 1, 4}));
              std::vector<TensorBuffer> bufs;
              bufs.push_back(std::move(df));
              bufs.push_back(std::move(emb));
              return bufs;
            });
    EXPECT_THAT(Lfm2AudioGenerationStage::Create(
                    frontend.get(), tokenizer, llm, std::move(bad_proj),
                    /*tts_decoder_runner=*/nullptr,
                    /*audio_vocoder_runner=*/nullptr, config),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // 2. tts_decoder with mismatched kv_in vs kv_out shape.
  {
    auto bad_dec = std::make_unique<NiceMock<MockLiteRtRunner>>();
    ON_CALL(*bad_dec, CreateInputBuffers(absl::string_view("")))
        .WillByDefault(
            [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
              LITERT_ASSIGN_OR_RETURN(
                  auto cond, support::CreateTensorBuffer<float>({1, 1, 4}));
              LITERT_ASSIGN_OR_RETURN(
                  auto prev, support::CreateTensorBuffer<int32_t>({1, 1}));
              LITERT_ASSIGN_OR_RETURN(
                  auto step, support::CreateTensorBuffer<int32_t>({1}));
              LITERT_ASSIGN_OR_RETURN(
                  auto kv, support::CreateTensorBuffer<float>({1, 2, 8, 4}));
              std::vector<TensorBuffer> bufs;
              bufs.push_back(std::move(cond));
              bufs.push_back(std::move(prev));
              bufs.push_back(std::move(step));
              bufs.push_back(std::move(kv));
              return bufs;
            });
    ON_CALL(*bad_dec, CreateOutputBuffers(absl::string_view("")))
        .WillByDefault(
            [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
              LITERT_ASSIGN_OR_RETURN(
                  auto logits, support::CreateTensorBuffer<float>({1, 1, 32}));
              LITERT_ASSIGN_OR_RETURN(
                  auto kv, support::CreateTensorBuffer<float>({1, 2, 8, 8}));
              std::vector<TensorBuffer> bufs;
              bufs.push_back(std::move(logits));
              bufs.push_back(std::move(kv));
              return bufs;
            });
    EXPECT_THAT(Lfm2AudioGenerationStage::Create(
                    frontend.get(), tokenizer, llm,
                    /*tts_projector_runner=*/nullptr, std::move(bad_dec),
                    /*audio_vocoder_runner=*/nullptr, config),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // 3. audio_vocoder with Float32 input instead of Int32.
  {
    auto bad_voc = std::make_unique<NiceMock<MockLiteRtRunner>>();
    ON_CALL(*bad_voc, CreateInputBuffers(absl::string_view("")))
        .WillByDefault(
            [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
              LITERT_ASSIGN_OR_RETURN(
                  auto in, support::CreateTensorBuffer<float>({1, 8, 2}));
              std::vector<TensorBuffer> bufs;
              bufs.push_back(std::move(in));
              return bufs;
            });
    ON_CALL(*bad_voc, CreateOutputBuffers(absl::string_view("")))
        .WillByDefault(
            [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
              LITERT_ASSIGN_OR_RETURN(
                  auto out, support::CreateTensorBuffer<float>({1, 32}));
              std::vector<TensorBuffer> bufs;
              bufs.push_back(std::move(out));
              return bufs;
            });
    EXPECT_THAT(Lfm2AudioGenerationStage::Create(
                    frontend.get(), tokenizer, llm,
                    /*tts_projector_runner=*/nullptr,
                    /*tts_decoder_runner=*/nullptr, std::move(bad_voc), config),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST(Lfm2AudioGenerationStageTest,
     EnforcesInterleaveScheduleMaxTtsTokensTextAudioDoneGuardAndNonEosCommit) {
  Lfm2AudioModelConfig config;
  config.llm_hidden_size = 4;
  config.llm_vocab_size = 256;
  config.tts_hidden_size = 4;
  config.tts_vocab_size = 32;  // EOAudio = 31
  config.tts_num_layers = 1;
  config.tts_num_codebooks = 8;
  config.tts_max_cache_len = 8;
  config.vocoder_chunk_frames = 16;
  config.detokenizer_upsample_samples = 4;
  config.generate_audio = true;
  config.generation_mode = Lfm2AudioGenerationMode::kInterleaved;
  config.interleaved_n_text = 6;
  config.interleaved_n_audio = 12;
  config.tts_temperature = 0.0f;
  config.max_new_tokens = 30;
  config.max_tts_tokens = 13;

  PushInputSource input_source;
  ASSERT_OK_AND_ASSIGN(auto fan_out,
                       Lfm2AudioInputFanOutStage::Create(&input_source));
  auto tokenizer = std::make_shared<FakeTokenizer>();
  auto embedder = CreateMockTextEmbedderRunner(config.llm_hidden_size);
  ASSERT_OK_AND_ASSIGN(auto text_stage,
                       Lfm2AudioTextStage::Create(fan_out->text_queue(),
                                                  fan_out->end_of_turn_queue(),
                                                  tokenizer, embedder, config));
  ASSERT_OK_AND_ASSIGN(
      auto frontend,
      Lfm2AudioFrontendStage::Create(text_stage.get(), /*audio_stage=*/nullptr,
                                     tokenizer, embedder, config));

  // Scripted 6/12 interleave + max_tts_tokens (13) forced EOAudio +
  // `text_done && audio_done` guard + non-EOS commit:
  // - Steps 0..5 (6 TEXT): token 3 ("Hello"), auto-switches to AUDIO_OUT at
  //   step 5 (`modality_left == 0`) without `<|audio_start|>`.
  // - Steps 6..17 (12 AUDIO_OUT): emits 12 valid audio frames, auto-switches
  //   back to TEXT at step 17 (`modality_left == 0`) without EOAudio.
  // - Step 18 (TEXT): emits `<|text_end|>` (130), setting `text_done = true`
  //   and switching to AUDIO_OUT.
  // - Step 19 (AUDIO_OUT): frame 13 (`12 < max_tts_tokens=13`).
  // - Step 20 (AUDIO_OUT): `13 >= max_tts_tokens=13` -> forces EOAudio
  //   (`[31]*8`), sets `audio_done = true`, switches back to TEXT.
  // - Step 21 (TEXT): emits non-EOS token 4 (" world") -> `text_done &&
  //   audio_done` guard terminates loop immediately (`stopped_on_eos == false`)
  //   without emitting token 4 into TextOutput, and commits token 4 +
  //   `<|im_end|>\n` into KV cache.
  std::vector<ScriptedLiteRtLmRunner::StepSpec> steps;
  for (int i = 0; i < 6; ++i) {
    steps.push_back({.top_token_id = 3, .hidden_state = {0.1f, 0, 0, 0}});
  }
  for (int i = 0; i < 12; ++i) {
    steps.push_back({.top_token_id = 0, .hidden_state = {1.0f, 0, 0, 0}});
  }
  steps.push_back({.top_token_id = lfm2_audio::kTextEndTokenId,
                   .hidden_state = {0.1f, 0, 0, 0}});
  steps.push_back({.top_token_id = 0, .hidden_state = {1.0f, 0, 0, 0}});
  steps.push_back({.top_token_id = 0, .hidden_state = {1.0f, 0, 0, 0}});
  steps.push_back({.top_token_id = 4, .hidden_state = {0.1f, 0, 0, 0}});
  auto scripted_llm = std::make_shared<ScriptedLiteRtLmRunner>(
      config.llm_vocab_size, config.llm_hidden_size, std::move(steps));

  auto mock_proj = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*mock_proj, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto h, support::CreateTensorBuffer<float>({1, 1, 4}));
            LITERT_ASSIGN_OR_RETURN(
                auto c, support::CreateTensorBuffer<int32_t>({1, 8}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(h));
            bufs.push_back(std::move(c));
            return bufs;
          });
  ON_CALL(*mock_proj, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto df, support::CreateTensorBuffer<float>({1, 8, 4}));
            LITERT_ASSIGN_OR_RETURN(
                auto emb, support::CreateTensorBuffer<float>({1, 1, 4}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(df));
            bufs.push_back(std::move(emb));
            return bufs;
          });
  ON_CALL(*mock_proj, Run(absl::string_view(""), _, _))
      .WillByDefault(
          [](absl::string_view, absl::Span<const TensorBuffer> inputs,
             absl::Span<const TensorBuffer> outputs) -> absl::Status {
            std::vector<int32_t> codes(8, 0);
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(inputs[1]).Read<int32_t>(
                    absl::MakeSpan(codes)));
            std::vector<float> df_in(32, 1.0f);
            std::vector<float> aud_emb(4, static_cast<float>(codes[0]));
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(outputs[0])
                    .Write<float>(absl::MakeConstSpan(df_in)));
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(outputs[1])
                    .Write<float>(absl::MakeConstSpan(aud_emb)));
            return absl::OkStatus();
          });

  auto mock_dec = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*mock_dec, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto cond, support::CreateTensorBuffer<float>({1, 1, 4}));
            LITERT_ASSIGN_OR_RETURN(
                auto prev, support::CreateTensorBuffer<int32_t>({1, 1}));
            LITERT_ASSIGN_OR_RETURN(auto step,
                                    support::CreateTensorBuffer<int32_t>({1}));
            LITERT_ASSIGN_OR_RETURN(
                auto kv, support::CreateTensorBuffer<float>({1, 2, 8, 4}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(cond));
            bufs.push_back(std::move(prev));
            bufs.push_back(std::move(step));
            bufs.push_back(std::move(kv));
            return bufs;
          });
  ON_CALL(*mock_dec, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto logits, support::CreateTensorBuffer<float>({1, 1, 32}));
            LITERT_ASSIGN_OR_RETURN(
                auto kv, support::CreateTensorBuffer<float>({1, 2, 8, 4}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(logits));
            bufs.push_back(std::move(kv));
            return bufs;
          });
  ON_CALL(*mock_dec, Run(absl::string_view(""), _, _))
      .WillByDefault(
          [](absl::string_view, absl::Span<const TensorBuffer> /*inputs*/,
             absl::Span<const TensorBuffer> outputs) -> absl::Status {
            // Always emit valid code 5 (never natural EOAudio), forcing
            // max_tts_tokens to terminate audio generation.
            std::vector<float> logits(32, -5.0f);
            logits[5] = 10.0f;
            std::vector<float> kv(64, 0.0f);
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(outputs[0])
                    .Write<float>(absl::MakeConstSpan(logits)));
            LITERT_RETURN_IF_ERROR(const_cast<TensorBuffer&>(outputs[1])
                                       .Write<float>(absl::MakeConstSpan(kv)));
            return absl::OkStatus();
          });

  auto mock_voc = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*mock_voc, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto in, support::CreateTensorBuffer<int32_t>({1, 8, 16}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(in));
            return bufs;
          });
  ON_CALL(*mock_voc, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto out, support::CreateTensorBuffer<float>({1, 64}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(out));
            return bufs;
          });
  ON_CALL(*mock_voc, Run(absl::string_view(""), _, _))
      .WillByDefault(
          [](absl::string_view, absl::Span<const TensorBuffer> /*inputs*/,
             absl::Span<const TensorBuffer> outputs) -> absl::Status {
            std::vector<float> wav(64, 0.5f);
            LITERT_RETURN_IF_ERROR(const_cast<TensorBuffer&>(outputs[0])
                                       .Write<float>(absl::MakeConstSpan(wav)));
            return absl::OkStatus();
          });

  ASSERT_OK_AND_ASSIGN(
      auto gen_stage,
      Lfm2AudioGenerationStage::Create(
          frontend.get(), tokenizer, scripted_llm, std::move(mock_proj),
          std::move(mock_dec), std::move(mock_voc), config));

  ASSERT_OK(input_source.PushInput(OmniSession::TextInput{.text = "Hello"}));
  ASSERT_OK(input_source.PushInput(OmniSession::EndOfInputTurn{}));
  ASSERT_OK(input_source.Schedule());
  ASSERT_OK(fan_out->Schedule());
  ASSERT_OK(text_stage->Schedule());
  ASSERT_OK(frontend->Schedule());
  ASSERT_OK(gen_stage->Schedule());

  // 1. 13 valid audio frames + 1 forced EOAudio frame fed back to LLM Decode.
  ASSERT_EQ(scripted_llm->received_audio_embeddings().size(), 14u);
  EXPECT_THAT(scripted_llm->received_audio_embeddings()[0],
              ElementsAre(5.0f, 5.0f, 5.0f, 5.0f));
  EXPECT_THAT(scripted_llm->received_audio_embeddings()[13],
              ElementsAre(31.0f, 31.0f, 31.0f, 31.0f));

  // 2. Outputs: 6x "Hello" (token 4 blocked by text_done && audio_done guard),
  //    13 * 4 = 52 audio samples, and EndOfOutputTurn.
  ASSERT_OK_AND_ASSIGN(Output out1, gen_stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(out1));
  EXPECT_EQ(std::get<TextOutput>(out1).confirmed_text,
            "HelloHelloHelloHelloHelloHello");
  ASSERT_OK_AND_ASSIGN(Output out2, gen_stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<AudioOutput>(out2));
  EXPECT_EQ(std::get<AudioOutput>(out2).pcm_samples.size(), 52u);
  ASSERT_OK_AND_ASSIGN(Output out3, gen_stage->GetOutput());
  EXPECT_TRUE(std::holds_alternative<EndOfOutputTurn>(out3));

  // 3. Non-EOS commit pushed trailing token 4 + `<|im_end|>` (7) + `\n` (5).
  const auto& rx_tokens = scripted_llm->received_text_tokens();
  ASSERT_GE(rx_tokens.size(), 3u);
  EXPECT_EQ(rx_tokens[rx_tokens.size() - 3], 4);
  EXPECT_EQ(rx_tokens[rx_tokens.size() - 2], lfm2_audio::kImEndTokenId);
  EXPECT_EQ(rx_tokens[rx_tokens.size() - 1], 5);
}

TEST(Lfm2AudioGenerationStageTest,
     OverlapDiscardChunkedVocoderMatchesOneShotAndEnforcesMaxKvLen) {
  // Arrange: simulate a causal detokenizer whose output at frame `t` depends on
  // `[max(0, t - 17) .. min(T - 1, t + 1)]` (17 left causal LFM2 frames + 1
  // left/right ISTFT overlap frame), and verify that chunked detokenization
  // with `vocoder_chunk_frames = 24` produces bit-for-bit identical output to
  // one-shot detokenization (`vocoder_chunk_frames = 40`) on 32 audio frames.
  constexpr int kNumAudioFrames = 32;
  constexpr int kSamplesPerFrame = 4;

  auto run_with_chunk_frames =
      [&](int chunk_frames,
          int max_kv_len) -> absl::StatusOr<std::vector<float>> {
    Lfm2AudioModelConfig config;
    config.llm_hidden_size = 4;
    config.llm_vocab_size = 256;
    config.max_kv_len = max_kv_len;
    config.tts_hidden_size = 4;
    config.tts_vocab_size = 64;
    config.tts_num_layers = 1;
    config.tts_num_codebooks = 8;
    config.tts_max_cache_len = 8;
    config.vocoder_chunk_frames = chunk_frames;
    config.detokenizer_upsample_samples = kSamplesPerFrame;
    config.generate_audio = true;
    config.generation_mode = Lfm2AudioGenerationMode::kSequential;
    config.system_prompt = "Perform TTS.";
    config.tts_temperature = 0.0f;
    config.max_new_tokens = 64;
    config.max_tts_tokens = 64;

    PushInputSource input_source;
    ABSL_ASSIGN_OR_RETURN(auto fan_out,
                          Lfm2AudioInputFanOutStage::Create(&input_source));
    auto tokenizer = std::make_shared<FakeTokenizer>();
    auto embedder = CreateMockTextEmbedderRunner(config.llm_hidden_size);
    ABSL_ASSIGN_OR_RETURN(auto text_stage, Lfm2AudioTextStage::Create(
                                               fan_out->text_queue(),
                                               fan_out->end_of_turn_queue(),
                                               tokenizer, embedder, config));
    ABSL_ASSIGN_OR_RETURN(
        auto frontend, Lfm2AudioFrontendStage::Create(
                           text_stage.get(), /*audio_stage=*/nullptr, tokenizer,
                           embedder, config));

    // Step 0: emit `<|audio_start|>` (128).
    // Steps 1..32: emit 32 valid audio frames (`hidden[0] = f + 1`).
    // Step 33: emit EOAudio (`hidden[0] = -1.0f`).
    // Step 34: emit `<|im_end|>` (7).
    std::vector<ScriptedLiteRtLmRunner::StepSpec> steps;
    steps.push_back({.top_token_id = lfm2_audio::kAudioStartTokenId,
                     .hidden_state = {0.0f, 0.0f, 0.0f, 0.0f}});
    for (int f = 0; f < kNumAudioFrames; ++f) {
      steps.push_back(
          {.top_token_id = 0,
           .hidden_state = {static_cast<float>(f + 1), 0.0f, 0.0f, 0.0f}});
    }
    steps.push_back(
        {.top_token_id = 0, .hidden_state = {-1.0f, 0.0f, 0.0f, 0.0f}});
    steps.push_back({.top_token_id = lfm2_audio::kImEndTokenId,
                     .hidden_state = {0.0f, 0.0f, 0.0f, 0.0f}});
    auto scripted_llm = std::make_shared<ScriptedLiteRtLmRunner>(
        config.llm_vocab_size, config.llm_hidden_size, std::move(steps));

    auto mock_proj = std::make_unique<NiceMock<MockLiteRtRunner>>();
    ON_CALL(*mock_proj, CreateInputBuffers(absl::string_view("")))
        .WillByDefault(
            [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
              LITERT_ASSIGN_OR_RETURN(
                  auto h_buf, support::CreateTensorBuffer<float>({1, 1, 4}));
              LITERT_ASSIGN_OR_RETURN(
                  auto c_buf, support::CreateTensorBuffer<int32_t>({1, 8}));
              std::vector<TensorBuffer> bufs;
              bufs.push_back(std::move(h_buf));
              bufs.push_back(std::move(c_buf));
              return bufs;
            });
    ON_CALL(*mock_proj, CreateOutputBuffers(absl::string_view("")))
        .WillByDefault(
            [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
              LITERT_ASSIGN_OR_RETURN(
                  auto df_in, support::CreateTensorBuffer<float>({1, 8, 4}));
              LITERT_ASSIGN_OR_RETURN(
                  auto aud_emb, support::CreateTensorBuffer<float>({1, 1, 4}));
              std::vector<TensorBuffer> bufs;
              bufs.push_back(std::move(df_in));
              bufs.push_back(std::move(aud_emb));
              return bufs;
            });
    ON_CALL(*mock_proj, Run(absl::string_view(""), _, _))
        .WillByDefault(
            [](absl::string_view, absl::Span<const TensorBuffer> inputs,
               absl::Span<const TensorBuffer> outputs) -> absl::Status {
              std::vector<float> h(4, 0.0f);
              LITERT_RETURN_IF_ERROR(
                  const_cast<TensorBuffer&>(inputs[0]).Read<float>(
                      absl::MakeSpan(h)));
              std::vector<float> df_in(32, h[0]);
              std::vector<float> aud_emb(4, 0.1f);
              LITERT_RETURN_IF_ERROR(
                  const_cast<TensorBuffer&>(outputs[0])
                      .Write<float>(absl::MakeConstSpan(df_in)));
              LITERT_RETURN_IF_ERROR(
                  const_cast<TensorBuffer&>(outputs[1])
                      .Write<float>(absl::MakeConstSpan(aud_emb)));
              return absl::OkStatus();
            });

    auto mock_dec = std::make_unique<NiceMock<MockLiteRtRunner>>();
    ON_CALL(*mock_dec, CreateInputBuffers(absl::string_view("")))
        .WillByDefault(
            [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
              LITERT_ASSIGN_OR_RETURN(
                  auto cond_buf, support::CreateTensorBuffer<float>({1, 1, 4}));
              LITERT_ASSIGN_OR_RETURN(
                  auto prev_buf, support::CreateTensorBuffer<int32_t>({1, 1}));
              LITERT_ASSIGN_OR_RETURN(
                  auto step_buf, support::CreateTensorBuffer<int32_t>({1}));
              LITERT_ASSIGN_OR_RETURN(
                  auto kv_buf,
                  support::CreateTensorBuffer<float>({1, 2, 8, 4}));
              std::vector<TensorBuffer> bufs;
              bufs.push_back(std::move(cond_buf));
              bufs.push_back(std::move(prev_buf));
              bufs.push_back(std::move(step_buf));
              bufs.push_back(std::move(kv_buf));
              return bufs;
            });
    ON_CALL(*mock_dec, CreateOutputBuffers(absl::string_view("")))
        .WillByDefault([](absl::string_view)
                           -> absl::StatusOr<std::vector<TensorBuffer>> {
          LITERT_ASSIGN_OR_RETURN(
              auto logits_buf, support::CreateTensorBuffer<float>({1, 1, 64}));
          LITERT_ASSIGN_OR_RETURN(
              auto kv_buf, support::CreateTensorBuffer<float>({1, 2, 8, 4}));
          std::vector<TensorBuffer> bufs;
          bufs.push_back(std::move(logits_buf));
          bufs.push_back(std::move(kv_buf));
          return bufs;
        });
    ON_CALL(*mock_dec, Run(absl::string_view(""), _, _))
        .WillByDefault(
            [](absl::string_view, absl::Span<const TensorBuffer> inputs,
               absl::Span<const TensorBuffer> outputs) -> absl::Status {
              std::vector<float> cond(4, 0.0f);
              LITERT_RETURN_IF_ERROR(
                  const_cast<TensorBuffer&>(inputs[0]).Read<float>(
                      absl::MakeSpan(cond)));
              std::vector<float> logits(64, -5.0f);
              if (cond[0] > 0.0f) {
                const int code = static_cast<int>(cond[0]) % 50;
                logits[code] = 10.0f;
              } else {
                logits[63] = 10.0f;  // EOAudio = 63
              }
              std::vector<float> kv(64, 0.0f);
              LITERT_RETURN_IF_ERROR(
                  const_cast<TensorBuffer&>(outputs[0])
                      .Write<float>(absl::MakeConstSpan(logits)));
              LITERT_RETURN_IF_ERROR(
                  const_cast<TensorBuffer&>(outputs[1])
                      .Write<float>(absl::MakeConstSpan(kv)));
              return absl::OkStatus();
            });

    auto mock_voc = std::make_unique<NiceMock<MockLiteRtRunner>>();
    ON_CALL(*mock_voc, CreateInputBuffers(absl::string_view("")))
        .WillByDefault([chunk_frames](absl::string_view)
                           -> absl::StatusOr<std::vector<TensorBuffer>> {
          LITERT_ASSIGN_OR_RETURN(
              auto in_buf,
              support::CreateTensorBuffer<int32_t>({1, 8, chunk_frames}));
          std::vector<TensorBuffer> bufs;
          bufs.push_back(std::move(in_buf));
          return bufs;
        });
    ON_CALL(*mock_voc, CreateOutputBuffers(absl::string_view("")))
        .WillByDefault([chunk_frames](absl::string_view)
                           -> absl::StatusOr<std::vector<TensorBuffer>> {
          LITERT_ASSIGN_OR_RETURN(auto out_buf,
                                  support::CreateTensorBuffer<float>(
                                      {1, chunk_frames * kSamplesPerFrame}));
          std::vector<TensorBuffer> bufs;
          bufs.push_back(std::move(out_buf));
          return bufs;
        });
    ON_CALL(*mock_voc, Run(absl::string_view(""), _, _))
        .WillByDefault(
            [chunk_frames](
                absl::string_view, absl::Span<const TensorBuffer> inputs,
                absl::Span<const TensorBuffer> outputs) -> absl::Status {
              std::vector<int32_t> codes(8 * chunk_frames, -1);
              LITERT_RETURN_IF_ERROR(
                  const_cast<TensorBuffer&>(inputs[0]).Read<int32_t>(
                      absl::MakeSpan(codes)));
              // Causal receptive field of 18 left frames + 1 right ISTFT frame
              // (ignoring -1 padded slots).
              std::vector<float> pcm(chunk_frames * kSamplesPerFrame, 0.0f);
              for (int t = 0; t < chunk_frames; ++t) {
                if (codes[t] < 0) continue;
                float acc = 0.0f;
                for (int d = -18; d <= 1; ++d) {
                  const int idx = t + d;
                  if (idx >= 0 && idx < chunk_frames && codes[idx] >= 0) {
                    acc += static_cast<float>(codes[idx] + 1) *
                           static_cast<float>(20 - std::abs(d)) * 0.01f;
                  }
                }
                for (int s = 0; s < kSamplesPerFrame; ++s) {
                  pcm[t * kSamplesPerFrame + s] =
                      acc + static_cast<float>(s) * 0.001f;
                }
              }
              LITERT_RETURN_IF_ERROR(
                  const_cast<TensorBuffer&>(outputs[0])
                      .Write<float>(absl::MakeConstSpan(pcm)));
              return absl::OkStatus();
            });

    ABSL_ASSIGN_OR_RETURN(
        auto gen_stage,
        Lfm2AudioGenerationStage::Create(
            frontend.get(), tokenizer, scripted_llm, std::move(mock_proj),
            std::move(mock_dec), std::move(mock_voc), config));

    ABSL_RETURN_IF_ERROR(
        input_source.PushInput(OmniSession::TextInput{.text = "Hello"}));
    ABSL_RETURN_IF_ERROR(input_source.PushInput(OmniSession::EndOfInputTurn{}));
    ABSL_RETURN_IF_ERROR(input_source.Schedule());
    ABSL_RETURN_IF_ERROR(fan_out->Schedule());
    ABSL_RETURN_IF_ERROR(text_stage->Schedule());
    ABSL_RETURN_IF_ERROR(frontend->Schedule());
    ABSL_RETURN_IF_ERROR(gen_stage->Schedule());

    while (gen_stage->HasOutput()) {
      ABSL_ASSIGN_OR_RETURN(Output out, gen_stage->GetOutput());
      if (const auto* aud = std::get_if<AudioOutput>(&out)) {
        return aud->pcm_samples;
      }
    }
    return absl::InternalError("Missing AudioOutput.");
  };

  ASSERT_OK_AND_ASSIGN(std::vector<float> wav_oneshot,
                       run_with_chunk_frames(/*chunk_frames=*/40,
                                             /*max_kv_len=*/512));
  ASSERT_OK_AND_ASSIGN(std::vector<float> wav_chunked,
                       run_with_chunk_frames(/*chunk_frames=*/24,
                                             /*max_kv_len=*/512));
  ASSERT_EQ(wav_chunked.size(),
            static_cast<size_t>(kNumAudioFrames * kSamplesPerFrame));
  ASSERT_EQ(wav_oneshot.size(), wav_chunked.size());
  for (size_t i = 0; i < wav_oneshot.size(); ++i) {
    EXPECT_NEAR(wav_chunked[i], wav_oneshot[i], 1e-5f) << "at sample " << i;
  }

  // Verify max_kv_len enforcement returns kResourceExhausted when prompt
  // exceeds max_kv_len.
  EXPECT_THAT(run_with_chunk_frames(/*chunk_frames=*/24, /*max_kv_len=*/4),
              StatusIs(absl::StatusCode::kResourceExhausted));
}

TEST(Lfm2AudioSessionTest, RegistersAllModelNamesInOmniRegistry) {
  constexpr absl::string_view kExpectedNames[] = {
      "lfm2.5-audio-1.5b",
      "LiquidAI/LFM2.5-Audio-1.5B",
      "lfm2_audio",
      "lfm2-audio",
  };
  EXPECT_THAT(OmniRegistry::Get().GetRegisteredModels(),
              testing::IsSupersetOf(kExpectedNames));
  for (absl::string_view name : kExpectedNames) {
    EXPECT_TRUE(OmniRegistry::Get().IsModelRegistered(name)) << name;
    ASSERT_OK_AND_ASSIGN(OmniRegistry::RegisteredModel reg,
                         OmniRegistry::Get().GetRegisteredModel(name));
    EXPECT_EQ(reg.canonical_model_name, "lfm2.5-audio-1.5b");
    EXPECT_EQ(reg.session_config_proto_type_id,
              "litert.lm.proto.Any2AnySessionConfig");
    EXPECT_EQ(reg.metadata_proto_type_id, "litert.lm.proto.Any2AnyMetadata");
  }
}

TEST(Lfm2AudioSessionTest, RegisteredInOmniRegistryAndRunsTextAndAudioTurn) {
  // Arrange
  ASSERT_OK_AND_ASSIGN(std::string bundle_path,
                       WriteLfm2AudioBundle("lfm2_audio_test.litertlm"));

  OmniEngine::Settings settings;
  settings.model_path = bundle_path;
  settings.cache_dir = ":nocache";
  settings.backend = OmniEngine::Settings::Backend::kCpu;
  settings.num_threads = 1;

  lm::proto::Any2AnySessionConfig init_cfg;
  init_cfg.set_generate_audio(false);
  init_cfg.set_max_new_tokens(1);

  // Act
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OmniSessionFactory> base_factory,
                       OmniRegistry::Get().CreateSessionFactory(
                           "LiquidAI/LFM2.5-Audio-1.5B", settings, &init_cfg));
  auto* factory = dynamic_cast<Any2AnySessionFactory*>(base_factory.get());
  ASSERT_NE(factory, nullptr);
  EXPECT_THAT(factory->GetAvailableLanguages(), ElementsAre("en-US"));
  EXPECT_EQ(factory->GetInputSampleRateHz(), 16000);
  EXPECT_EQ(factory->GetDefaultVoice("en-US"), "lfm2_voice");

  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input = input_source.get();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OmniSession> session,
                       factory->Create(std::move(input_source)));

  // Second concurrent session on same factory must fail with
  // kFailedPrecondition.
  EXPECT_THAT(factory->Create(std::make_unique<PushInputSource>()),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  // Push audio metadata + two audio chunks (exercising coalescing and
  // audio_encoder end-to-end) alongside text.
  ASSERT_OK(raw_input->PushInput(
      AudioInputMetadata{.sample_rate_hz = 16000, .num_channels = 1}));
  ASSERT_OK(raw_input->PushInput(
      AudioInput{.pcm_samples = std::vector<float>(800, 0.1f)}));
  ASSERT_OK(raw_input->PushInput(
      AudioInput{.pcm_samples = std::vector<float>(800, 0.2f)}));
  ASSERT_OK(raw_input->PushInput(OmniSession::TextInput{.text = "Hello"}));
  ASSERT_OK(raw_input->PushInput(OmniSession::EndOfInputTurn{}));

  // Assert
  ASSERT_OK_AND_ASSIGN(OmniSession::Output text_out, session->ProcessNext());
  EXPECT_TRUE(std::holds_alternative<TextOutput>(text_out));
  ASSERT_OK_AND_ASSIGN(OmniSession::Output end_out, session->ProcessNext());
  EXPECT_TRUE(std::holds_alternative<OmniSession::EndOfOutputTurn>(end_out));

  // Releasing the active session clears the lease so a subsequent session
  // can be created on the same factory.
  session.reset();
  EXPECT_THAT(factory->Create(std::make_unique<PushInputSource>()), IsOk());
}

TEST(Lfm2AudioSessionTest, DefaultsLanguageAndCustomModelWhenOmitted) {
  lm::proto::Any2AnyMetadata bare_meta;
  bare_meta.set_input_sample_rate_hz(16000);
  bare_meta.set_output_sample_rate_hz(24000);
  ASSERT_OK_AND_ASSIGN(
      std::string bundle_path,
      WriteLfm2AudioBundle("lfm2_audio_defaults_test.litertlm", &bare_meta));

  OmniEngine::Settings settings;
  settings.model_path = bundle_path;
  settings.cache_dir = ":nocache";
  settings.backend = OmniEngine::Settings::Backend::kCpu;
  settings.num_threads = 1;

  lm::proto::Any2AnySessionConfig init_cfg;
  init_cfg.set_generate_audio(false);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OmniSessionFactory> base_factory,
                       OmniRegistry::Get().CreateSessionFactory(
                           "lfm2.5-audio-1.5b", settings, &init_cfg));
  auto* factory = dynamic_cast<Any2AnySessionFactory*>(base_factory.get());
  ASSERT_NE(factory, nullptr);
  EXPECT_THAT(factory->GetAvailableLanguages(), ElementsAre("en-US"));
  ASSERT_TRUE(factory->metadata().any2any_model_type().has_custom_model());
  lm::proto::Lfm2Audio unpacked;
  EXPECT_TRUE(factory->metadata().any2any_model_type().custom_model().UnpackTo(
      &unpacked));
}

TEST(Lfm2AudioSessionTest,
     UnpacksCustomModelProtoEndToEndViaOmniEngineAndMetadataOverrides) {
  ASSERT_OK_AND_ASSIGN(std::string bundle_path,
                       WriteLfm2AudioBundle("lfm2_audio_engine_test.litertlm"));

  OmniEngine::Settings settings;
  settings.model_path = bundle_path;
  settings.cache_dir = ":nocache";
  settings.backend = OmniEngine::Settings::Backend::kCpu;
  settings.num_threads = 1;

  lm::proto::Any2AnySessionConfig init_cfg;
  init_cfg.set_generate_audio(false);
  init_cfg.set_max_new_tokens(1);

  // 1. End-to-end OmniEngine::Create using embedded `custom_model` proto.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OmniEngine> engine,
      OmniEngine::Create("lfm2.5-audio-1.5b", settings, &init_cfg));
  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input = input_source.get();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OmniSession> session,
                       engine->CreateSession(std::move(input_source)));
  // Push audio metadata + two audio chunks (exercising coalescing and
  // audio_encoder end-to-end) alongside text.
  ASSERT_OK(raw_input->PushInput(
      AudioInputMetadata{.sample_rate_hz = 16000, .num_channels = 1}));
  ASSERT_OK(raw_input->PushInput(
      AudioInput{.pcm_samples = std::vector<float>(800, 0.1f)}));
  ASSERT_OK(raw_input->PushInput(
      AudioInput{.pcm_samples = std::vector<float>(800, 0.2f)}));
  ASSERT_OK(raw_input->PushInput(OmniSession::TextInput{.text = "Hello"}));
  ASSERT_OK(raw_input->PushInput(OmniSession::EndOfInputTurn{}));
  ASSERT_OK_AND_ASSIGN(OmniSession::Output text_out, session->ProcessNext());
  EXPECT_TRUE(std::holds_alternative<TextOutput>(text_out));
  ASSERT_OK_AND_ASSIGN(OmniSession::Output end_out, session->ProcessNext());
  EXPECT_TRUE(std::holds_alternative<OmniSession::EndOfOutputTurn>(end_out));

  // 2. Scalar-only metadata override preserves embedded `custom_model`.
  lm::proto::Any2AnyMetadata scalar_override;
  scalar_override.set_input_sample_rate_hz(22050);
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OmniSessionFactory> scalar_base_factory,
      OmniRegistry::Get().CreateSessionFactory("lfm2_audio", settings,
                                               &init_cfg, &scalar_override));
  auto* scalar_factory =
      dynamic_cast<Any2AnySessionFactory*>(scalar_base_factory.get());
  ASSERT_NE(scalar_factory, nullptr);
  EXPECT_EQ(scalar_factory->GetInputSampleRateHz(), 22050);
  EXPECT_EQ(scalar_factory->GetDefaultVoice("en-US"), "lfm2_voice");

  // 3. `custom_model` metadata override replaces embedded `custom_model`.
  lm::proto::Any2AnyMetadata custom_override;
  lm::proto::Lfm2Audio override_lfm2;
  override_lfm2.set_llm_hidden_size(4);
  override_lfm2.set_llm_vocab_size(32);
  override_lfm2.set_generate_audio(false);
  override_lfm2.set_voice("uk_female");
  ASSERT_TRUE(custom_override.mutable_any2any_model_type()
                  ->mutable_custom_model()
                  ->PackFrom(override_lfm2));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OmniSessionFactory> custom_base_factory,
      OmniRegistry::Get().CreateSessionFactory("lfm2-audio", settings,
                                               &init_cfg, &custom_override));
  auto* custom_factory =
      dynamic_cast<Any2AnySessionFactory*>(custom_base_factory.get());
  ASSERT_NE(custom_factory, nullptr);
  EXPECT_EQ(custom_factory->GetDefaultVoice("en-US"), "uk_female");
}

TEST(Lfm2AudioSessionTest,
     RejectsWrongCustomModelProtoTypeAndNonCustomModelType) {
  // 1. Embedded `custom_model` packing an unexpected proto type (`MiniCpmO`).
  lm::proto::Any2AnyMetadata wrong_any_meta;
  lm::proto::MiniCpmO wrong_payload;
  wrong_payload.set_llm_hidden_size(4);
  ASSERT_TRUE(wrong_any_meta.mutable_any2any_model_type()
                  ->mutable_custom_model()
                  ->PackFrom(wrong_payload));
  ASSERT_OK_AND_ASSIGN(
      std::string wrong_any_bundle,
      WriteLfm2AudioBundle("lfm2_audio_wrong_any.litertlm", &wrong_any_meta));

  OmniEngine::Settings settings;
  settings.model_path = wrong_any_bundle;
  settings.cache_dir = ":nocache";
  settings.backend = OmniEngine::Settings::Backend::kCpu;
  settings.num_threads = 1;

  EXPECT_THAT(
      OmniRegistry::Get().CreateSessionFactory("lfm2.5-audio-1.5b", settings),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(OmniEngine::Create("lfm2.5-audio-1.5b", settings),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // 2. Metadata override packing an unexpected proto type into `custom_model`.
  ASSERT_OK_AND_ASSIGN(
      std::string valid_bundle,
      WriteLfm2AudioBundle("lfm2_audio_valid_for_override.litertlm"));
  settings.model_path = valid_bundle;
  EXPECT_THAT(OmniRegistry::Get().CreateSessionFactory(
                  "lfm2.5-audio-1.5b", settings, /*session_config=*/nullptr,
                  &wrong_any_meta),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // 3. Embedded non-custom `any2any_model_type` oneof branch (`minicpm_o`).
  lm::proto::Any2AnyMetadata non_custom_meta;
  non_custom_meta.mutable_any2any_model_type()->mutable_minicpm_o();
  ASSERT_OK_AND_ASSIGN(
      std::string non_custom_bundle,
      WriteLfm2AudioBundle("lfm2_audio_non_custom.litertlm", &non_custom_meta));
  settings.model_path = non_custom_bundle;
  EXPECT_THAT(
      OmniRegistry::Get().CreateSessionFactory("lfm2.5-audio-1.5b", settings),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(Lfm2AudioFactoryTest, CreateLfm2AudioResourcesRejectsInvalidInputs) {
  Lfm2AudioModelConfig config;
  Expected<Environment> env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  auto shared_env = std::make_shared<Environment>(std::move(*env));

  std::shared_ptr<Environment> null_env;
  EXPECT_THAT(CreateLfm2AudioResources(
                  config, "/tmp/nonexistent_lfm2_audio.litertlm", "",
                  lm::Backend::CPU, /*num_threads=*/1, null_env),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      CreateLfm2AudioResources(config, "/tmp/not_a_litertlm_bundle_dir", "",
                               lm::Backend::CPU, /*num_threads=*/1, shared_env),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("LFM2.5-Audio requires a .litertlm model file")));
  EXPECT_THAT(CreateLfm2AudioResources(
                  config, "/tmp/nonexistent_lfm2_audio.litertlm", "",
                  lm::Backend::CPU, /*num_threads=*/1, shared_env),
              testing::Not(IsOk()));
}

TEST(Lfm2AudioFactoryTest,
     CreateLfm2AudioResourcesRetainsEnvDelegatesBackboneAndGatesAudio) {
  Lfm2AudioTestBundleOptions full_bundle_opts;
  full_bundle_opts.include_tts_models = true;
  ASSERT_OK_AND_ASSIGN(std::string full_bundle_path,
                       WriteLfm2AudioBundle("lfm2_audio_factory_full.litertlm",
                                            full_bundle_opts));
  ASSERT_OK_AND_ASSIGN(
      std::string text_only_bundle_path,
      WriteLfm2AudioBundle("lfm2_audio_factory_text_only.litertlm"));

  Expected<Environment> env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  auto shared_env = std::make_shared<Environment>(std::move(*env));

  // 1. With generate_audio=false, TTS/vocoder sections present in the bundle
  //    are skipped while env, text_embedder, audio_encoder, and llm_backbone
  //    (with BackboneLmModelResources delegation) are populated.
  {
    Lfm2AudioModelConfig config;
    config.generate_audio = false;
    ASSERT_OK_AND_ASSIGN(ModelResources resources,
                         CreateLfm2AudioResources(
                             config, full_bundle_path, ":nocache",
                             lm::Backend::CPU, /*num_threads=*/1, shared_env));
    EXPECT_EQ(resources.environment(), shared_env);
    EXPECT_TRUE(resources.HasLmModelResources());
    EXPECT_TRUE(resources.HasCompiledModel("text_embedder"));
    EXPECT_TRUE(resources.HasCompiledModel("audio_encoder"));
    EXPECT_FALSE(resources.HasCompiledModel("tts_projector"));
    EXPECT_FALSE(resources.HasCompiledModel("tts_decoder"));
    EXPECT_FALSE(resources.HasCompiledModel("audio_vocoder"));

    ASSERT_OK_AND_ASSIGN(std::shared_ptr<LiteRtLmRunner> llm_runner,
                         resources.GetLmRunner("llm_backbone"));
    lm::ModelResources* backbone_res = llm_runner->mutable_model_resources();
    ASSERT_NE(backbone_res, nullptr);
    using lm::proto::LlmMetadata;
    EXPECT_TRUE(
        backbone_res->HasTFLiteModel(LlmMetadata::TF_LITE_PREFILL_DECODE));
    EXPECT_TRUE(backbone_res->HasTFLiteModel(LlmMetadata::TF_LITE_EMBEDDER));
    EXPECT_THAT(
        backbone_res->GetTFLiteModel(LlmMetadata::TF_LITE_PREFILL_DECODE),
        IsOk());
    EXPECT_THAT(backbone_res->GetTFLiteModel(LlmMetadata::TF_LITE_EMBEDDER),
                IsOk());
  }

  // 2. With generate_audio=true, all TTS and vocoder sections are compiled.
  {
    Lfm2AudioModelConfig config;
    config.generate_audio = true;
    ASSERT_OK_AND_ASSIGN(ModelResources resources,
                         CreateLfm2AudioResources(
                             config, full_bundle_path, ":nocache",
                             lm::Backend::CPU, /*num_threads=*/1, shared_env));
    EXPECT_EQ(resources.environment(), shared_env);
    EXPECT_TRUE(resources.HasCompiledModel("text_embedder"));
    EXPECT_TRUE(resources.HasCompiledModel("audio_encoder"));
    EXPECT_TRUE(resources.HasCompiledModel("tts_projector"));
    EXPECT_TRUE(resources.HasCompiledModel("tts_decoder"));
    EXPECT_TRUE(resources.HasCompiledModel("audio_vocoder"));
    EXPECT_THAT(resources.GetLmRunner("llm_backbone"), IsOk());
  }

  // 3. With generate_audio=true and missing TTS/vocoder sections, returns
  //    kNotFound.
  {
    Lfm2AudioModelConfig config;
    config.generate_audio = true;
    EXPECT_THAT(CreateLfm2AudioResources(config, text_only_bundle_path,
                                         ":nocache", lm::Backend::CPU,
                                         /*num_threads=*/1, shared_env),
                StatusIs(absl::StatusCode::kNotFound));
  }

  // 4. Pre-populated lm::ModelResources is reused directly without reopening
  //    model_path.
  {
    ASSERT_OK_AND_ASSIGN(std::shared_ptr<lm::ModelResources> lm_resources,
                         CreateLmModelResources(text_only_bundle_path));
    Lfm2AudioModelConfig config;
    config.generate_audio = false;
    ASSERT_OK_AND_ASSIGN(
        ModelResources resources,
        CreateLfm2AudioResources(config, /*model_path=*/"", ":nocache",
                                 lm::Backend::CPU, /*num_threads=*/1,
                                 shared_env, lm_resources));
    EXPECT_EQ(resources.environment(), shared_env);
    EXPECT_EQ(resources.GetLmModelResources(), lm_resources);
    EXPECT_TRUE(resources.HasCompiledModel("text_embedder"));
    EXPECT_THAT(resources.GetLmRunner("llm_backbone"), IsOk());
  }
}

}  // namespace
}  // namespace litert::omni::any2any
