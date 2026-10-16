# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export kaldi-fbank constants (mel banks, hamming window) as app
assets, and prove the Kotlin-port algorithm (mirrored here in plain
numpy) matches torchaudio.compliance.kaldi.fbank."""
import os
import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torchaudio.compliance.kaldi import get_mel_banks

HERE = os.path.dirname(os.path.abspath(__file__))
SR, NMEL, WIN, HOP, NFFT = 16000, 80, 400, 160, 512
# Reference fixture: any speech clip works. Drop sample_speech.wav next to
# this script (or set SAMPLE_WAV) to regenerate the golden fbank asset.
SAMPLE_WAV = os.path.join(HERE, "sample_speech.wav")

# ---- assets
# mel: [80, 256]
mel, _ = get_mel_banks(NMEL, NFFT, SR, 20.0, 0.0, 100.0, -500.0, 1.0)
# padded to [80, 257]
mel = torch.nn.functional.pad(mel, (0, 1)).numpy()
ham = np.hamming(WIN).astype(np.float32) * 0 + (
    0.54 - 0.46 * np.cos(2 * np.pi * np.arange(WIN) / (WIN - 1))
).astype(np.float32)
mel.astype(np.float32).tofile(os.path.join(HERE, "mel80_257.bin"))
ham.tofile(os.path.join(HERE, "hamming400.bin"))
print("assets: mel", mel.shape, "hamming", ham.shape)

EPS = 1.1920928955078125e-07  # float eps (kaldi log floor)


def fbank_mirror(x):
    """Plain-numpy mirror of the intended Kotlin implementation.

    Args:
        x: 1D float waveform array at 16 kHz in [-1, 1].

    Returns:
        [n_frames, NMEL] float32 log-mel fbank array.
    """
    x = x * 32768.0
    n_frames = 1 + (len(x) - WIN) // HOP
    out = np.zeros((n_frames, NMEL), np.float32)
    for t in range(n_frames):
        fr = x[t * HOP:t * HOP + WIN].astype(np.float64).copy()
        fr -= fr.mean()                                   # remove_dc_offset
        # preemphasis (replicate pad)
        fr = np.concatenate([[fr[0]], fr])
        fr = fr[1:] - 0.97 * fr[:-1]
        fr *= ham
        spec = np.fft.rfft(fr, NFFT)                      # zero-padded to 512
        power = (spec.real ** 2 + spec.imag ** 2)         # [257]
        m = mel @ power
        out[t] = np.log(np.maximum(m, EPS))
    return out


wav, sr = torchaudio.load(SAMPLE_WAV)
wav = torchaudio.functional.resample(
    wav.mean(0, keepdim=True), sr, SR)[0, :80240]
ref = kaldi.fbank(wav[None] * 32768.0, num_mel_bins=NMEL,
                  frame_length=25.0, frame_shift=10.0, dither=0.0,
                  window_type="hamming", use_energy=False,
                  sample_frequency=SR).numpy()
mine = fbank_mirror(wav.numpy())
print("mirror", mine.shape, "ref", ref.shape)
print("max|d|", np.abs(mine - ref).max(),
      " corr", np.corrcoef(mine.ravel(), ref.ravel())[0, 1])
