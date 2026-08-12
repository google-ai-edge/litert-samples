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

"""Speech synthesis via Inflect-Nano-v2 on LiteRT — the fast LiteRT TTS path.

A light streaming TTS running on-LiteRT. The earlier belief that such models
"cannot convert to LiteRT" was wrong: exporting the decoder
with a fixed chunk size (streaming already works on fixed-size chunks) sidesteps
the data-dependent output shapes. Measured on the Pi 5: RTF ~0.11, ~210 ms per
sentence, bit-exact vs the PyTorch reference, and it streams within a sentence.

The text frontend code is Apache-2.0 and calls espeak-ng (GPL-3.0,
loaded in-process) at runtime for phonemization — so arbitrary reply text is
phonemized with no phoneme dictionary needed.

Same interface as the other synthesizers: `speak(text) -> np.ndarray` plus a
`sample_rate` attribute, so it is a drop-in behind the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Bundled package location on the board (models + the Apache-2.0 text
# frontend, which calls espeak-ng/GPL-3.0 in-process at runtime).
DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "assets" / "inflect"
DEFAULT_FRONTEND_DIR = DEFAULT_MODELS_DIR / "frontend"
SAMPLE_RATE = 24000


class InflectSynthesizer:
    def __init__(self, models_dir: Path | None = None,
                 frontend_dir: Path | None = None,
                 precision: str = "fp32", threads: int = 4) -> None:
        # say.py lives inside the package dir; import it from there.
        import sys

        models_dir = Path(models_dir or DEFAULT_MODELS_DIR)
        # Derive the frontend from models_dir when not given explicitly, so an
        # injected models_dir carries its own frontend/ with it (for the default
        # models_dir this equals DEFAULT_FRONTEND_DIR — no behavior change).
        frontend_dir = Path(frontend_dir) if frontend_dir else models_dir / "frontend"
        pkg = str(models_dir)
        if pkg not in sys.path:
            sys.path.insert(0, pkg)
        from say import InflectTTS

        self._tts = InflectTTS(models_dir=str(models_dir),
                               frontend_dir=str(frontend_dir),
                               precision=precision, threads=threads)
        self.sample_rate = SAMPLE_RATE

    def speak(self, text: str) -> np.ndarray:
        """Text -> float32 samples. Empty text yields an empty array, not an error."""
        if not text.strip():
            return np.zeros(0, dtype=np.float32)
        audio = self._tts.say(text)
        return np.asarray(audio, dtype=np.float32)


def load_inflect(models_dir: Path | None = None,
                 frontend_dir: Path | None = None) -> InflectSynthesizer:
    """Build the Inflect synthesizer from the bundled package."""
    return InflectSynthesizer(models_dir=models_dir, frontend_dir=frontend_dir)
