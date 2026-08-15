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

import os
import shutil
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

# Bundled package location on the board: the runtime (say.py) and the Apache-2.0
# text frontend, which calls espeak-ng/GPL-3.0 in-process at runtime. The LiteRT
# weights (*.tflite) are gitignored, not shipped here — they are fetched from
# Hugging Face into this directory on first use (see _ensure_weights), like the
# sample's other models.
DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "assets" / "inflect"
DEFAULT_FRONTEND_DIR = DEFAULT_MODELS_DIR / "frontend"
SAMPLE_RATE = 24000


def _ensure_weights(models_dir: Path, repo: str, precision: str) -> None:
    """Fetch the encoder + decoder LiteRT weights from Hugging Face on first use.

    ``*.tflite`` is gitignored, so the weights are not committed — only the
    runtime (``say.py``) and the espeak frontend are. The two graphs for the
    requested precision are downloaded next to ``say.py`` (where ``InflectTTS``
    reads them by directory), mirroring how the speech-to-text model
    auto-downloads. A file already present is left untouched, so this is a no-op
    after the first run at a given precision (and for a manually populated dir).
    """
    if precision not in ("fp32", "fp16"):
        raise ValueError(f"unknown precision {precision!r}; expected fp32 or fp16")
    suffix = "" if precision == "fp32" else "_fp16"
    for stem in ("inflect_text_encoder", "inflect_decoder"):
        name = f"{stem}{suffix}.tflite"
        dest = models_dir / name
        if dest.exists():
            continue
        try:
            cached = hf_hub_download(repo_id=repo, filename=name)
        except Exception as exc:  # offline, missing repo/file, network error
            raise SystemExit(
                f"could not fetch the Inflect TTS weight {name!r} from Hugging "
                f"Face repo {repo!r}: {type(exc).__name__}: {exc}. Check your "
                f"network, or place the .tflite files in {models_dir} manually "
                "(see assets/inflect/RUN.md).") from exc
        models_dir.mkdir(parents=True, exist_ok=True)
        # Copy into place atomically: a copy interrupted by a crash, a full disk,
        # or a second process would otherwise leave a truncated .tflite that the
        # dest.exists() check trusts forever (every later run then fails at model
        # open). Write a per-process temp file, then os.replace() — an atomic
        # rename on the same filesystem.
        tmp = dest.with_name(f"{name}.{os.getpid()}.tmp")
        try:
            shutil.copyfile(cached, tmp)
            os.replace(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)


class InflectSynthesizer:
    def __init__(self, models_dir: Path | None = None,
                 frontend_dir: Path | None = None,
                 precision: str = "fp32", threads: int = 4,
                 repo: str | None = None) -> None:
        # say.py lives inside the package dir; import it from there.
        import sys

        models_dir = Path(models_dir or DEFAULT_MODELS_DIR)
        # Pull the LiteRT weights from Hugging Face the first time (they are
        # gitignored, so a fresh checkout has only say.py + the frontend).
        if repo:
            _ensure_weights(models_dir, repo, precision)
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
                 frontend_dir: Path | None = None,
                 repo: str | None = None) -> InflectSynthesizer:
    """Build the Inflect synthesizer; fetch the LiteRT weights from ``repo`` on
    first use if they are not already present in ``models_dir``."""
    return InflectSynthesizer(models_dir=models_dir, frontend_dir=frontend_dir,
                              repo=repo)
