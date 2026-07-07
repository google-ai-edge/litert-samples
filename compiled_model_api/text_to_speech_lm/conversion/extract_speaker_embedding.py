# Copyright 2026 The AI Edge LiteRT Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Enrolls a voice: reference audio (~3 s) -> 1024-d x-vector .npy.

Runs the Qwen3-TTS speaker encoder (an ECAPA-style x-vector network over a
mel spectrogram) in PyTorch. Enrollment is a one-time, offline step; the
resulting .npy is the `--speaker` input of the sample.

Usage (reference env):
    python extract_speaker_embedding.py reference.wav my_voice.npy
"""

import sys

import librosa
import numpy as np
import torch

from qwen_tts import Qwen3TTSModel


def main() -> None:
    wav_path, out_path = sys.argv[1], sys.argv[2]
    wrapper = Qwen3TTSModel.from_pretrained(
        'Qwen/Qwen3-TTS-12Hz-0.6B-Base', device_map='cpu',
        dtype=torch.float32)
    audio, sample_rate = librosa.load(wav_path, sr=24000, mono=True)
    embedding = wrapper.model.extract_speaker_embedding(
        audio=audio.astype(np.float32), sr=sample_rate)
    np.save(out_path, embedding.detach().numpy().astype(np.float32))
    print(f'saved {out_path}: shape {tuple(embedding.shape)}')


if __name__ == '__main__':
    main()
