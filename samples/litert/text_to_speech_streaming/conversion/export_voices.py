# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Repack the KittenTTS voices.npz into the flat binary the Android app reads.

voices.npz (from the model's Hugging Face repo) maps each of the 8 voice names to a
[400, 256] float32 style table indexed by text length. The app wants one contiguous
little-endian float32 blob [8, 400, 256] in VOICES order (KittenSynthesizer.VOICES).

    python export_voices.py path/to/voices.npz  ->  out/voices.bin
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

# Must match KittenSynthesizer.VOICES exactly.
VOICES = [
    "expr-voice-2-m", "expr-voice-2-f",
    "expr-voice-3-m", "expr-voice-3-f",
    "expr-voice-4-m", "expr-voice-4-f",
    "expr-voice-5-m", "expr-voice-5-f",
]


def main():
    npz = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "models" / "nano-0.8-fp32" / "voices.npz")
    voices = np.load(npz)
    table = np.stack([voices[name] for name in VOICES]).astype("<f4")
    assert table.shape == (len(VOICES), 400, 256), table.shape
    OUT.mkdir(exist_ok=True)
    (OUT / "voices.bin").write_bytes(table.tobytes())
    print(f"[voices] out/voices.bin  {table.nbytes/1e6:.1f} MB  {table.shape}")


if __name__ == "__main__":
    main()
