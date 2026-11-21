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

#ifndef MODELS_LFM_LFM2_5_AUDIO_GENERATION_STAGE_H_
#define MODELS_LFM_LFM2_5_AUDIO_GENERATION_STAGE_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "models/lfm/lfm2_5_audio/io_types.h"
#include "models/lfm/lfm2_5_audio/lfm2_audio_model_config.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "omni/any2any/any2any_sampler.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_lm_runner.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::any2any {

// Unified interleaved/sequential generation stage for
// `LiquidAI/LFM2.5-Audio-1.5B` combining:
// 1. LFM2-1.2B hybrid short-conv + GQA LLM backbone (`LiteRtLmRunner`).
// 2. `Lfm2AudioTtsProjectorModule` (`depth_linear` `[2048 -> 8192]` and
//    `audio_embedding(codes + codebook_offsets).sum(0)` `[8 -> 2048]`).
// 3. 8-codebook RQ-Transformer `Depthformer` (`layers=6`, `dim=1024`,
//    `vocab_size=2049`, `EOAudio = vocab_size - 1`).
// 4. `LFM2AudioDetokenizer` (`[1, 8, T] -> [1, T * 1920]` 24 kHz waveform).
//
// During `AUDIO_OUT` generation (triggered when the LFM2 backbone emits
// `<|audio_start|>` = `128`), each LFM2 hidden state is projected through
// `depth_linear` and decoded into 8 codebook tokens by `Depthformer`. If
// codebook 0 emits `EOAudio` (`tts_vocab_size - 1`), all 8 codes are set to
// `EOAudio` and generation switches back to `TEXT` mode; otherwise the 8-code
// frame is appended for detokenization. In both cases, `audio_embedding(codes +
// codebook_offsets).sum(0)` is fed back into `llm_runner->Decode` as the next
// continuous step embedding!
class Lfm2AudioGenerationStage : public SingleThreadedStageWithDeque<Output> {
 public:
  // Creates an `Lfm2AudioGenerationStage`.
  //
  // Args:
  //   frontend: Non-null `Stage<Lfm2AudioFrontendOutput>`.
  //   tokenizer: Non-null tokenizer.
  //   llm_runner: Non-null LFM2 backbone `LiteRtLmRunner`.
  //   tts_projector_runner: Optional runner executing `tf_lite_tts_projector`
  //     (`depth_linear` + `audio_embedding`).
  //   tts_decoder_runner: Optional runner executing `tf_lite_tts_decoder`
  //     (8-codebook RQ-Transformer `Depthformer`).
  //   audio_vocoder_runner: Optional runner executing `tf_lite_audio_vocoder`
  //     (`LFM2AudioDetokenizer`).
  //   config: LFM2.5-Audio model configuration.
  //
  // Returns:
  //   Initialized `Lfm2AudioGenerationStage`, or an error status on invalid
  //   config or incompatible runner tensor shapes.
  static absl::StatusOr<std::unique_ptr<Lfm2AudioGenerationStage>> Create(
      Stage<Lfm2AudioFrontendOutput>* absl_nonnull frontend,
      std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
      std::shared_ptr<LiteRtLmRunner> absl_nonnull llm_runner,
      std::unique_ptr<LiteRtRunner> absl_nullable tts_projector_runner =
          nullptr,
      std::unique_ptr<LiteRtRunner> absl_nullable tts_decoder_runner = nullptr,
      std::unique_ptr<LiteRtRunner> absl_nullable audio_vocoder_runner =
          nullptr,
      const Lfm2AudioModelConfig& config = {});

  ~Lfm2AudioGenerationStage() override;

 protected:
  bool NeedScheduleInternal() const override;
  absl::Status ScheduleInternal() override;
  void ResetInternal() override;

 private:
  struct AudioFrameResult {
    std::vector<int32_t> codes;
    std::vector<float> audio_embedding;
    bool is_eo_audio = false;
  };

  Lfm2AudioGenerationStage(
      Stage<Lfm2AudioFrontendOutput>* absl_nonnull frontend,
      std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
      std::shared_ptr<LiteRtLmRunner> absl_nonnull llm_runner,
      std::unique_ptr<LiteRtRunner> absl_nullable tts_projector_runner,
      std::vector<TensorBuffer> proj_input_buffers,
      std::vector<TensorBuffer> proj_output_buffers,
      std::unique_ptr<LiteRtRunner> absl_nullable tts_decoder_runner,
      std::vector<TensorBuffer> dec_input_buffers,
      std::vector<TensorBuffer> dec_output_buffers,
      std::unique_ptr<LiteRtRunner> absl_nullable audio_vocoder_runner,
      std::vector<TensorBuffer> voc_input_buffers,
      std::vector<TensorBuffer> voc_output_buffers, int voc_chunk_frames,
      int voc_codebooks_dim_index, const Lfm2AudioModelConfig& config);

  absl::StatusOr<AudioFrameResult> GenerateAudioFrame(
      absl::Span<const float> llm_hidden_state, bool force_eo_audio);

  absl::StatusOr<std::vector<float>> SynthesizeWaveform(
      const std::vector<std::vector<int32_t>>& audio_frames);

  Stage<Lfm2AudioFrontendOutput>& frontend_;
  Lfm2AudioModelConfig config_;
  int hidden_dim_;
  std::shared_ptr<support::Tokenizer> tokenizer_;
  std::shared_ptr<LiteRtLmRunner> llm_runner_;

  std::unique_ptr<LiteRtRunner> tts_projector_runner_;
  std::vector<TensorBuffer> proj_input_buffers_;
  std::vector<TensorBuffer> proj_output_buffers_;

  std::unique_ptr<LiteRtRunner> tts_decoder_runner_;
  std::vector<TensorBuffer> dec_input_buffers_;
  std::vector<TensorBuffer> dec_output_buffers_;

  std::unique_ptr<LiteRtRunner> audio_vocoder_runner_;
  std::vector<TensorBuffer> voc_input_buffers_;
  std::vector<TensorBuffer> voc_output_buffers_;
  int voc_chunk_frames_ = 0;
  int voc_codebooks_dim_index_ = 1;

  std::vector<int> turn_end_token_ids_;
  Lfm2AudioGenerationMode resolved_session_mode_ =
      Lfm2AudioGenerationMode::kAuto;
  int kv_tokens_used_ = 0;
  Any2AnySampler text_sampler_;
  Any2AnySampler tts_sampler_;
};

}  // namespace litert::omni::any2any

#endif  // MODELS_LFM_LFM2_5_AUDIO_GENERATION_STAGE_H_
