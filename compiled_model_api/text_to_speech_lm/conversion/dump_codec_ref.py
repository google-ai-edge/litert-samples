# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
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
"""Dumps codec decoder reference output on fixed random codes.

Usage (reference env):
    python dump_codec_ref.py
"""

import os

import numpy as np
import torch
from huggingface_hub import snapshot_download

from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer


def main() -> None:
    torch.manual_seed(42)
    src = snapshot_download('Qwen/Qwen3-TTS-12Hz-0.6B-Base')
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        f'{src}/speech_tokenizer', dtype=torch.float32, device_map='cpu')
    decoder = tokenizer.model.decoder

    codes = torch.randint(0, 2048, (1, 16, 52))
    with torch.no_grad():
        wav = decoder(codes)

    os.makedirs('ref', exist_ok=True)
    np.savez('ref/codec_equiv_ref.npz', codes=codes.numpy(), wav=wav.numpy())
    print('dumped', codes.shape, '->', wav.shape)


if __name__ == '__main__':
    main()
