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

#include "models/lfm/lfm2_5_audio/runner/audio_stage.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "models/lfm/lfm2_5_audio/runner/io_types.h"
#include "models/lfm/lfm2_5_audio/runner/lfm2_audio_model_config.h"
#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/cleanup/cleanup.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "omni/base/fan_out_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/sequenced.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"
#include "omni/utils/tensor_utils.h"

namespace litert::omni::any2any {
namespace {

constexpr int kSupportedNfft = 512;
constexpr int kNumFreqBins = kSupportedNfft / 2 + 1;  // 257
constexpr float kPerFeatureNormEps = 1e-5f;

// 512-point radix-2 real-to-complex FFT returning the first 257 bins
// (`0..N/2`), matching `torch.fft.rfft(..., n=512)`.
void RealFft512(const std::array<float, kSupportedNfft>& input,
                std::array<std::complex<float>, kNumFreqBins>& output) {
  std::array<std::complex<double>, kSupportedNfft> buf{};
  for (int i = 0; i < kSupportedNfft; ++i) {
    buf[i] = std::complex<double>(static_cast<double>(input[i]), 0.0);
  }

  for (int i = 1, j = 0; i < kSupportedNfft; ++i) {
    int bit = kSupportedNfft >> 1;
    for (; j & bit; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      std::swap(buf[i], buf[j]);
    }
  }

  const double pi = std::acos(-1.0);
  for (int len = 2; len <= kSupportedNfft; len <<= 1) {
    const double ang = -2.0 * pi / static_cast<double>(len);
    const std::complex<double> wlen(std::cos(ang), std::sin(ang));
    const int half = len >> 1;
    for (int i = 0; i < kSupportedNfft; i += len) {
      std::complex<double> w(1.0, 0.0);
      for (int j = 0; j < half; ++j) {
        const std::complex<double> u = buf[i + j];
        const std::complex<double> v = buf[i + j + half] * w;
        buf[i + j] = u + v;
        buf[i + j + half] = u - v;
        w *= wlen;
      }
    }
  }

  for (int k = 0; k < kNumFreqBins; ++k) {
    output[k] = std::complex<float>(static_cast<float>(buf[k].real()),
                                    static_cast<float>(buf[k].imag()));
  }
}

double HzToSlaneyMel(double hz) {
  constexpr double kFSp = 200.0 / 3.0;
  constexpr double kMinLogHz = 1000.0;
  constexpr double kMinLogMel = kMinLogHz / kFSp;
  const double logstep = std::log(6.4) / 27.0;
  if (hz >= kMinLogHz) {
    return kMinLogMel + std::log(hz / kMinLogHz) / logstep;
  }
  return hz / kFSp;
}

double SlaneyMelToHz(double mel) {
  constexpr double kFSp = 200.0 / 3.0;
  constexpr double kMinLogHz = 1000.0;
  constexpr double kMinLogMel = kMinLogHz / kFSp;
  const double logstep = std::log(6.4) / 27.0;
  if (mel >= kMinLogMel) {
    return kMinLogHz * std::exp(logstep * (mel - kMinLogMel));
  }
  return kFSp * mel;
}

// Computes a Slaney-normalized mel filterbank matrix of shape
// `[n_mels, n_fft / 2 + 1]`, matching `librosa.filters.mel(..., norm="slaney")`
// used by NeMo `FilterbankFeatures`.
std::vector<float> ComputeSlaneyMelFilterbank(int n_mels, int n_fft,
                                              float sample_rate, float f_min,
                                              float f_max) {
  const int n_freqs = n_fft / 2 + 1;
  std::vector<float> weights(static_cast<size_t>(n_mels) * n_freqs, 0.0f);

  std::vector<double> fft_freqs(n_freqs, 0.0);
  for (int k = 0; k < n_freqs; ++k) {
    fft_freqs[k] =
        static_cast<double>(k) * static_cast<double>(sample_rate) / n_fft;
  }

  const double mel_min = HzToSlaneyMel(f_min);
  const double mel_max = HzToSlaneyMel(f_max);
  std::vector<double> f_pts(n_mels + 2, 0.0);
  for (int i = 0; i < n_mels + 2; ++i) {
    const double mel =
        mel_min + (mel_max - mel_min) * static_cast<double>(i) / (n_mels + 1);
    f_pts[i] = SlaneyMelToHz(mel);
  }

  for (int m = 0; m < n_mels; ++m) {
    const double f_left = f_pts[m];
    const double f_center = f_pts[m + 1];
    const double f_right = f_pts[m + 2];
    const double enorm = 2.0 / std::max(1e-12, f_right - f_left);
    float* row = weights.data() + static_cast<size_t>(m) * n_freqs;
    for (int k = 0; k < n_freqs; ++k) {
      const double lower =
          (fft_freqs[k] - f_left) / std::max(1e-12, f_center - f_left);
      const double upper =
          (f_right - fft_freqs[k]) / std::max(1e-12, f_right - f_center);
      const double w = std::max(0.0, std::min(lower, upper));
      row[k] = static_cast<float>(w * enorm);
    }
  }
  return weights;
}

// Builds a symmetric Hann window (`periodic=False`) of length `win_length`
// centered inside a buffer of length `n_fft`, matching `torch.hann_window(400,
// periodic=False)` inside `torch.stft(..., n_fft=512, win_length=400)`.
std::array<float, kSupportedNfft> BuildCenteredSymmetricHannWindow(
    int win_length) {
  std::array<float, kSupportedNfft> window{};
  window.fill(0.0f);
  const int clamped_win = std::clamp(win_length, 1, kSupportedNfft);
  const int pad_left = (kSupportedNfft - clamped_win) / 2;
  if (clamped_win == 1) {
    window[pad_left] = 1.0f;
    return window;
  }
  const double two_pi = 2.0 * std::acos(-1.0);
  const double denom = static_cast<double>(clamped_win - 1);
  for (int i = 0; i < clamped_win; ++i) {
    window[pad_left + i] =
        static_cast<float>(0.5 - 0.5 * std::cos(two_pi * i / denom));
  }
  return window;
}

}  // namespace

// Self-contained NeMo `FilterbankFeatures` log-mel implementation.
// `support::AudioPreprocessorMiniAudio` / `LogMelSpectrogramProcessor` is not
// reused here because it applies per-frame pre-emphasis (`frame[0] = x[0] *
// (1 - alpha)`) after framing without `torch.stft(..., center=True)` symmetric
// `n_fft/2` padding, uses a periodic right-padded Hanning window, and rescales
// quiet audio before STFT; NeMo `FilterbankFeatures` (used by LFM2.5-Audio's
// FastConformer) requires whole-utterance pre-emphasis (`y[0] = x[0]`),
// centered symmetric Hann `win_length=400` in `n_fft=512` with `center=True`
// STFT padding, and utterance-wide Bessel `normalize="per_feature"` over
// `floor(N / hop_length)` valid frames.
absl::StatusOr<std::vector<float>> ExtractNemoLogMelSpectrogram(
    absl::Span<const float> pcm_16k, const Lfm2AudioModelConfig& config,
    int* absl_nonnull out_valid_frames) {
  if (config.audio_mel_bins <= 0 || config.audio_hop_length <= 0 ||
      config.audio_win_length <= 0 ||
      config.audio_win_length > kSupportedNfft ||
      config.audio_n_fft != kSupportedNfft ||
      config.audio_input_sample_rate_hz <= 0) {
    return absl::InvalidArgumentError(
        "Invalid Lfm2AudioModelConfig parameters for NeMo log-mel extraction.");
  }

  // NeMo `FilterbankFeatures.get_seq_len(N) = floor(N / hop_length)`.
  // `torch.stft(..., center=True)` produces `seq_len + 1` frames, normalizes
  // over `0 .. seq_len - 1`, masks index `>= seq_len` to `0.0f`, and passes
  // `seq_len` as `mel_length` to `ConformerEncoder`. Return only the `seq_len`
  // valid frames (empty when `N < hop_length`); `ProcessAudio` zero-pads any
  // trailing frames beyond `actual_frames`.
  const int valid_frames =
      static_cast<int>(pcm_16k.size()) / config.audio_hop_length;
  *out_valid_frames = valid_frames;
  if (valid_frames <= 0) {
    return std::vector<float>();
  }

  // 1. Pre-emphasis filter: y[0] = x[0], y[t] = x[t] - preemph * x[t - 1].
  std::vector<float> preemph_pcm(pcm_16k.size(), 0.0f);
  preemph_pcm[0] = pcm_16k[0];
  for (size_t i = 1; i < pcm_16k.size(); ++i) {
    preemph_pcm[i] = pcm_16k[i] - config.audio_preemph * pcm_16k[i - 1];
  }

  // 2. Center zero-padding (`n_fft // 2 = 256` on both sides).
  const int pad = kSupportedNfft / 2;
  const size_t min_padded_size =
      static_cast<size_t>(valid_frames - 1) * config.audio_hop_length +
      kSupportedNfft;
  std::vector<float> padded(
      std::max<size_t>(min_padded_size, preemph_pcm.size() + 2 * pad), 0.0f);
  absl::c_copy(preemph_pcm, padded.begin() + pad);

  // 3. Windowed STFT power spectrum + Slaney mel filterbank + log(mel + 2^-24).
  const std::array<float, kSupportedNfft> hann_window =
      BuildCenteredSymmetricHannWindow(config.audio_win_length);
  const std::vector<float> mel_basis = ComputeSlaneyMelFilterbank(
      config.audio_mel_bins, kSupportedNfft,
      static_cast<float>(config.audio_input_sample_rate_hz), 0.0f,
      static_cast<float>(config.audio_input_sample_rate_hz) * 0.5f);

  const int n_mels = config.audio_mel_bins;
  const float log_guard = std::max(config.audio_log_zero_guard, 1e-30f);
  std::vector<float> log_mel(static_cast<size_t>(valid_frames) * n_mels, 0.0f);

  std::array<float, kSupportedNfft> windowed_frame{};
  std::array<std::complex<float>, kNumFreqBins> spectrum{};
  std::array<float, kNumFreqBins> power_spec{};

  for (int t = 0; t < valid_frames; ++t) {
    const size_t start = static_cast<size_t>(t) * config.audio_hop_length;
    for (int i = 0; i < kSupportedNfft; ++i) {
      windowed_frame[i] = padded[start + i] * hann_window[i];
    }
    RealFft512(windowed_frame, spectrum);
    for (int k = 0; k < kNumFreqBins; ++k) {
      const float re = spectrum[k].real();
      const float im = spectrum[k].imag();
      power_spec[k] = re * re + im * im;
    }
    float* frame_out = log_mel.data() + static_cast<size_t>(t) * n_mels;
    for (int m = 0; m < n_mels; ++m) {
      const float* filter =
          mel_basis.data() + static_cast<size_t>(m) * kNumFreqBins;
      float mel_energy = 0.0f;
      for (int k = 0; k < kNumFreqBins; ++k) {
        mel_energy += filter[k] * power_spec[k];
      }
      frame_out[m] = std::log(mel_energy + log_guard);
    }
  }

  // 4. NeMo `normalize = "per_feature"` over valid frames `0 .. valid_frames -
  // 1` with Bessel sample std (`denom = valid_frames - 1` when `valid_frames >
  // 1`) and `+ 1e-5f` added to `std`.
  if (valid_frames <= 1) {
    absl::c_fill(log_mel, 0.0f);
    return log_mel;
  }

  for (int m = 0; m < n_mels; ++m) {
    double sum = 0.0;
    for (int t = 0; t < valid_frames; ++t) {
      sum += log_mel[static_cast<size_t>(t) * n_mels + m];
    }
    const double mean = sum / static_cast<double>(valid_frames);
    double sq_sum = 0.0;
    for (int t = 0; t < valid_frames; ++t) {
      const double diff = log_mel[static_cast<size_t>(t) * n_mels + m] - mean;
      sq_sum += diff * diff;
    }
    const double std_val =
        std::sqrt(sq_sum / static_cast<double>(valid_frames - 1));
    const float inv_denom =
        static_cast<float>(1.0 / (std_val + kPerFeatureNormEps));
    for (int t = 0; t < valid_frames; ++t) {
      float& val = log_mel[static_cast<size_t>(t) * n_mels + m];
      val = static_cast<float>((val - mean) * inv_denom);
    }
  }

  return log_mel;
}

Lfm2AudioStage::Lfm2AudioStage(
    FanOutOutputQueue<OmniSession::Input, AudioInput>* absl_nonnull audio_queue,
    FanOutOutputQueue<OmniSession::Input, AudioInputMetadata>* absl_nonnull
        audio_input_metadata_queue,
    int chunk_frames, bool feature_major_input,
    std::unique_ptr<LiteRtRunner> absl_nonnull audio_encoder_runner,
    std::vector<TensorBuffer> audio_enc_input_buffers,
    std::vector<TensorBuffer> audio_enc_output_buffers,
    const Lfm2AudioModelConfig& config)
    : audio_queue_(*audio_queue),
      audio_input_metadata_queue_(*audio_input_metadata_queue),
      config_(config),
      hidden_dim_(config_.llm_hidden_size),
      mel_bins_(config_.audio_mel_bins),
      chunk_frames_(chunk_frames),
      feature_major_input_(feature_major_input),
      input_sample_rate_hz_(config_.audio_input_sample_rate_hz),
      audio_encoder_runner_(std::move(audio_encoder_runner)),
      audio_enc_input_buffers_(std::move(audio_enc_input_buffers)),
      audio_enc_output_buffers_(std::move(audio_enc_output_buffers)) {}

Lfm2AudioStage::~Lfm2AudioStage() = default;

absl::StatusOr<std::unique_ptr<Lfm2AudioStage>> Lfm2AudioStage::Create(
    FanOutOutputQueue<OmniSession::Input, AudioInput>* absl_nonnull audio_queue,
    FanOutOutputQueue<OmniSession::Input, AudioInputMetadata>* absl_nonnull
        audio_input_metadata_queue,
    std::unique_ptr<LiteRtRunner> absl_nonnull audio_encoder_runner,
    const Lfm2AudioModelConfig& config) {
  if (config.llm_hidden_size <= 0 || config.audio_mel_bins <= 0 ||
      config.audio_input_sample_rate_hz <= 0 ||
      config.audio_subsampling_factor <= 0) {
    return absl::InvalidArgumentError(
        "Lfm2AudioStage config dimensions and sample rate must be positive.");
  }

  ABSL_ASSIGN_OR_RETURN(
      auto audio_enc_input_buffers,
      audio_encoder_runner->CreateInputBuffers(/*signature_name=*/""));
  ABSL_ASSIGN_OR_RETURN(
      auto audio_enc_output_buffers,
      audio_encoder_runner->CreateOutputBuffers(/*signature_name=*/""));
  if (audio_enc_input_buffers.empty() || audio_enc_output_buffers.empty()) {
    return absl::InvalidArgumentError(
        "Lfm2AudioStage audio_encoder_runner must have at least 1 input and 1 "
        "output buffer.");
  }

  LITERT_ASSIGN_OR_RETURN(auto in_type,
                          audio_enc_input_buffers[0].TensorType());
  if (in_type.ElementType() != ElementType::Float32) {
    return absl::InvalidArgumentError(
        "Lfm2AudioStage audio_encoder_runner input[0] must be Float32.");
  }
  const auto& in_dims = in_type.Layout().Dimensions();
  LITERT_ASSIGN_OR_RETURN(size_t in_elements, in_type.Layout().NumElements());
  if (in_elements == 0 || in_elements % config.audio_mel_bins != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Lfm2AudioStage audio_encoder_runner input size (", in_elements,
        ") is not divisible by audio_mel_bins (", config.audio_mel_bins, ")."));
  }
  if (audio_enc_input_buffers.size() >= 2) {
    ABSL_RETURN_IF_ERROR(ValidateTensorBuffer(
        audio_enc_input_buffers[1], ElementType::Int32,
        /*expected_elements=*/1, "Lfm2AudioStage audio_encoder mel_length"));
  }

  // `Lfm2ConformerEncoderModule` takes `[1, n_mels (128), T_mel]`
  // (`feature_major_input = true`). Also support `[1, T_mel, n_mels]` if a test
  // runner uses time-major layout.
  bool feature_major_input = true;
  if (in_dims.size() == 3) {
    if (in_dims[1] == config.audio_mel_bins &&
        in_dims[2] != config.audio_mel_bins) {
      feature_major_input = true;
    } else if (in_dims[2] == config.audio_mel_bins &&
               in_dims[1] != config.audio_mel_bins) {
      feature_major_input = false;
    }
  }
  const int chunk_frames =
      static_cast<int>(in_elements / config.audio_mel_bins);

  LITERT_ASSIGN_OR_RETURN(auto out_type,
                          audio_enc_output_buffers[0].TensorType());
  if (out_type.ElementType() != ElementType::Float32) {
    return absl::InvalidArgumentError(
        "Lfm2AudioStage audio_encoder_runner output[0] must be Float32.");
  }
  LITERT_ASSIGN_OR_RETURN(size_t out_elements, out_type.Layout().NumElements());
  if (out_elements == 0 || out_elements % config.llm_hidden_size != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Lfm2AudioStage audio_encoder_runner output size (",
                     out_elements, ") is not divisible by llm_hidden_size (",
                     config.llm_hidden_size, ")."));
  }
  if (audio_enc_output_buffers.size() >= 2) {
    ABSL_RETURN_IF_ERROR(ValidateTensorBuffer(
        audio_enc_output_buffers[1], ElementType::Int32,
        /*expected_elements=*/1, "Lfm2AudioStage audio_encoder output_length"));
  }

  return absl::WrapUnique(
      new Lfm2AudioStage(audio_queue, audio_input_metadata_queue, chunk_frames,
                         feature_major_input, std::move(audio_encoder_runner),
                         std::move(audio_enc_input_buffers),
                         std::move(audio_enc_output_buffers), config));
}

bool Lfm2AudioStage::NeedScheduleInternal() const {
  return !audio_queue_.Empty() || !audio_input_metadata_queue_.Empty();
}

absl::Status Lfm2AudioStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };

  if (!NeedScheduleInternal()) {
    return absl::NotFoundError("No audio items ready to schedule.");
  }

  while (NeedScheduleInternal()) {
    bool take_metadata = false;
    if (!audio_input_metadata_queue_.Empty()) {
      if (audio_queue_.Empty()) {
        take_metadata = true;
      } else {
        ABSL_ASSIGN_OR_RETURN(const Sequenced<AudioInputMetadata>* meta_peek,
                              audio_input_metadata_queue_.PeekFront());
        ABSL_ASSIGN_OR_RETURN(const Sequenced<AudioInput>* audio_peek,
                              audio_queue_.PeekFront());
        take_metadata = meta_peek->seqnum < audio_peek->seqnum;
      }
    }

    if (take_metadata) {
      ABSL_ASSIGN_OR_RETURN(const Sequenced<AudioInputMetadata>* meta,
                            audio_input_metadata_queue_.PeekFront());
      if (meta->item.sample_rate_hz <= 0) {
        return absl::InvalidArgumentError(absl::StrCat(
            "AudioInputMetadata.sample_rate_hz must be positive, got ",
            meta->item.sample_rate_hz));
      }
      if (meta->item.num_channels != 1) {
        return absl::InvalidArgumentError(
            absl::StrCat("Lfm2AudioStage only supports mono audio "
                         "(num_channels == 1), got ",
                         meta->item.num_channels));
      }
      input_sample_rate_hz_ = meta->item.sample_rate_hz;
      resampler_.Reset();
      const uint64_t seqnum = meta->seqnum;
      ABSL_RETURN_IF_ERROR(audio_input_metadata_queue_.PopFront().status());
      PushOutput(Sequenced<Lfm2AudioEmbeddingSegment>{
          .seqnum = seqnum,
          .item = {},
      });
    } else {
      ABSL_ASSIGN_OR_RETURN(const Sequenced<AudioInput>* seq_audio,
                            audio_queue_.PeekFront());
      const uint64_t seqnum = seq_audio->seqnum;
      if (seq_audio->item.pcm_samples.empty()) {
        ABSL_RETURN_IF_ERROR(audio_queue_.PopFront().status());
        PushOutput(Sequenced<Lfm2AudioEmbeddingSegment>{
            .seqnum = seqnum,
            .item = {},
        });
        continue;
      }
      ABSL_ASSIGN_OR_RETURN(Lfm2AudioEmbeddingSegment seg,
                            ProcessAudio(seq_audio->item));
      ABSL_RETURN_IF_ERROR(audio_queue_.PopFront().status());
      PushOutput(Sequenced<Lfm2AudioEmbeddingSegment>{
          .seqnum = seqnum,
          .item = std::move(seg),
      });
    }
  }

  return absl::OkStatus();
}

void Lfm2AudioStage::ResetInternal() {
  SingleThreadedStageWithDeque::ResetInternal();
  resampler_.Reset();
  input_sample_rate_hz_ = config_.audio_input_sample_rate_hz;
}

absl::StatusOr<Lfm2AudioEmbeddingSegment> Lfm2AudioStage::ProcessAudio(
    const AudioInput& audio) {
  ABSL_ASSIGN_OR_RETURN(
      std::vector<float> resampled,
      resampler_.Resample(audio.pcm_samples, input_sample_rate_hz_,
                          config_.audio_input_sample_rate_hz));

  int valid_frames = 0;
  ABSL_ASSIGN_OR_RETURN(
      std::vector<float> log_mel_time_major,
      ExtractNemoLogMelSpectrogram(resampled, config_, &valid_frames));
  if (valid_frames <= 0) {
    return Lfm2AudioEmbeddingSegment{};
  }

  LITERT_ASSIGN_OR_RETURN(auto out_type,
                          audio_enc_output_buffers_[0].TensorType());
  LITERT_ASSIGN_OR_RETURN(size_t out_elements, out_type.Layout().NumElements());
  const int max_chunk_tokens = static_cast<int>(out_elements / hidden_dim_);

  Lfm2AudioEmbeddingSegment seg;
  std::vector<float> chunk_mel(static_cast<size_t>(chunk_frames_) * mel_bins_,
                               0.0f);
  std::vector<float> chunk_out(out_elements, 0.0f);

  for (int start_frame = 0; start_frame < valid_frames;
       start_frame += chunk_frames_) {
    const int actual_frames =
        std::min(chunk_frames_, valid_frames - start_frame);
    absl::c_fill(chunk_mel, 0.0f);

    if (feature_major_input_) {
      // Transpose `[actual_frames, mel_bins_]` -> `[mel_bins_, chunk_frames_]`.
      for (int t = 0; t < actual_frames; ++t) {
        const float* src = log_mel_time_major.data() +
                           static_cast<size_t>(start_frame + t) * mel_bins_;
        for (int m = 0; m < mel_bins_; ++m) {
          chunk_mel[static_cast<size_t>(m) * chunk_frames_ + t] = src[m];
        }
      }
    } else {
      const size_t copy_floats = static_cast<size_t>(actual_frames) * mel_bins_;
      std::copy_n(log_mel_time_major.data() +
                      static_cast<size_t>(start_frame) * mel_bins_,
                  copy_floats, chunk_mel.data());
    }

    LITERT_RETURN_IF_ERROR(audio_enc_input_buffers_[0].Write<float>(
        absl::MakeConstSpan(chunk_mel)));
    if (audio_enc_input_buffers_.size() >= 2) {
      const int32_t len_val = static_cast<int32_t>(actual_frames);
      LITERT_RETURN_IF_ERROR(audio_enc_input_buffers_[1].Write<int32_t>(
          absl::MakeConstSpan(&len_val, 1)));
    }

    ABSL_RETURN_IF_ERROR(audio_encoder_runner_->Run(/*signature_name=*/"",
                                                    audio_enc_input_buffers_,
                                                    audio_enc_output_buffers_));
    LITERT_RETURN_IF_ERROR(
        audio_enc_output_buffers_[0].Read<float>(absl::MakeSpan(chunk_out)));

    int valid_tokens = (actual_frames + config_.audio_subsampling_factor - 1) /
                       config_.audio_subsampling_factor;
    if (audio_enc_output_buffers_.size() >= 2) {
      int32_t reported_tokens = 0;
      if (audio_enc_output_buffers_[1].Read<int32_t>(
              absl::MakeSpan(&reported_tokens, 1))) {
        if (reported_tokens > 0) {
          valid_tokens = reported_tokens;
        }
      }
    }
    valid_tokens = std::clamp(valid_tokens, 1, max_chunk_tokens);

    seg.embeddings.insert(
        seg.embeddings.end(), chunk_out.begin(),
        chunk_out.begin() + static_cast<size_t>(valid_tokens) * hidden_dim_);
    seg.token_ids.insert(seg.token_ids.end(), valid_tokens,
                         config_.audio_embed_placeholder_token_id);
    seg.num_tokens += valid_tokens;
  }

  return seg;
}

}  // namespace litert::omni::any2any
