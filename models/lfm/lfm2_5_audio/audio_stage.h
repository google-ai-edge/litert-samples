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

#ifndef MODELS_LFM_LFM2_5_AUDIO_AUDIO_STAGE_H_
#define MODELS_LFM_LFM2_5_AUDIO_AUDIO_STAGE_H_

#include <memory>
#include <vector>

#include "models/lfm/lfm2_5_audio/io_types.h"
#include "models/lfm/lfm2_5_audio/lfm2_audio_model_config.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "omni/base/fan_out_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/sequenced.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"
#include "omni/utils/audio_resampler.h"

namespace litert::omni::any2any {

// Extracts NeMo `FilterbankFeatures` 128-bin log-mel spectrogram frames from
// 16 kHz mono PCM samples (`preemph = 0.97`, symmetric Hann `win_length = 400`,
// `hop_length = 160`, `n_fft = 512`, power spectrum, Slaney mel basis,
// `log(mel + 2^-24)`, and per-feature Bessel mean/std normalization over valid
// frames `0 .. seq_len - 1`).
//
// Args:
//   pcm_16k: Mono PCM audio samples at 16 kHz.
//   config: LFM2.5-Audio model configuration.
//   out_valid_frames: Output parameter receiving the number of valid mel frames
//     (`pcm_16k.size() / hop_length`, matching NeMo
//     `FilterbankFeatures.get_seq_len`; `0` when `pcm_16k.size() <
//     hop_length`).
//
// Returns:
//   Flattened time-major `[valid_frames, audio_mel_bins]` normalized log-mel
//   spectrogram (empty when `valid_frames == 0`), or an error status on invalid
//   config.
absl::StatusOr<std::vector<float>> ExtractNemoLogMelSpectrogram(
    absl::Span<const float> pcm_16k, const Lfm2AudioModelConfig& config,
    int* absl_nonnull out_valid_frames);

// FastConformer audio input processing stage for `LiquidAI/LFM2.5-Audio-1.5B`:
// - Consumes `Sequenced<AudioInput>` and `Sequenced<AudioInputMetadata>` from
//   `Lfm2AudioInputFanOutStage` in ascending `seqnum` order.
// - Resamples mono PCM audio to 16 kHz via `AudioResampler`.
// - Computes 128-bin NeMo `FilterbankFeatures` log-mel spectrogram features
//   (`preemph=0.97`, symmetric Hann `win_length=400`, `hop_length=160`,
//   `n_fft=512`, Slaney mel filterbank, `log(mel + 2^-24)`, per-feature
//   normalization).
// - Runs the FastConformer encoder + 2-layer `audio_adapter` MLP per chunk and
//   keeps `mel2emb_len(T) = (T + subsampling_factor - 1) / subsampling_factor`
//   continuous embeddings.
// - Outputs `Sequenced<Lfm2AudioEmbeddingSegment>` with
//   `config.audio_embed_placeholder_token_id` (`-100`) per frame (no
//   `<|audio_start|>` / `<|audio_end|>` wrapping around input audio).
class Lfm2AudioStage : public SingleThreadedStageWithDeque<
                           Sequenced<Lfm2AudioEmbeddingSegment>> {
 public:
  // Creates an `Lfm2AudioStage`.
  //
  // Args:
  //   audio_queue: Non-null queue providing `Sequenced<AudioInput>` items.
  //   audio_input_metadata_queue: Non-null queue providing
  //     `Sequenced<AudioInputMetadata>` items.
  //   audio_encoder_runner: Non-null runner executing the FastConformer audio
  //     encoder and `audio_adapter` model.
  //   config: LFM2.5-Audio model configuration.
  //
  // Returns:
  //   A newly initialized `Lfm2AudioStage` on success, or an error status if
  //   configuration parameters or runner buffer dimensions are invalid.
  static absl::StatusOr<std::unique_ptr<Lfm2AudioStage>> Create(
      FanOutOutputQueue<OmniSession::Input, AudioInput>* absl_nonnull
          audio_queue,
      FanOutOutputQueue<OmniSession::Input, AudioInputMetadata>* absl_nonnull
          audio_input_metadata_queue,
      std::unique_ptr<LiteRtRunner> absl_nonnull audio_encoder_runner,
      const Lfm2AudioModelConfig& config = {});

  ~Lfm2AudioStage() override;

 protected:
  bool NeedScheduleInternal() const override;
  absl::Status ScheduleInternal() override;
  void ResetInternal() override;

 private:
  Lfm2AudioStage(
      FanOutOutputQueue<OmniSession::Input, AudioInput>* absl_nonnull
          audio_queue,
      FanOutOutputQueue<OmniSession::Input, AudioInputMetadata>* absl_nonnull
          audio_input_metadata_queue,
      int chunk_frames, bool feature_major_input,
      std::unique_ptr<LiteRtRunner> absl_nonnull audio_encoder_runner,
      std::vector<TensorBuffer> audio_enc_input_buffers,
      std::vector<TensorBuffer> audio_enc_output_buffers,
      const Lfm2AudioModelConfig& config);

  absl::StatusOr<Lfm2AudioEmbeddingSegment> ProcessAudio(
      const AudioInput& audio);

  FanOutOutputQueue<OmniSession::Input, AudioInput>& audio_queue_;
  FanOutOutputQueue<OmniSession::Input, AudioInputMetadata>&
      audio_input_metadata_queue_;
  Lfm2AudioModelConfig config_;
  int hidden_dim_;
  int mel_bins_;
  int chunk_frames_;
  bool feature_major_input_;
  int input_sample_rate_hz_;

  AudioResampler resampler_;
  std::unique_ptr<LiteRtRunner> audio_encoder_runner_;
  std::vector<TensorBuffer> audio_enc_input_buffers_;
  std::vector<TensorBuffer> audio_enc_output_buffers_;
};

}  // namespace litert::omni::any2any

#endif  // MODELS_LFM_LFM2_5_AUDIO_AUDIO_STAGE_H_
