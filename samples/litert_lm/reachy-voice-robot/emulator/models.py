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

"""Model registry: every model the pipeline uses, defined once, by name.

Swap a model for a stage by changing the entry here (or the stage's default
name below) — nothing else in the code hard-codes a repo, file, dir, or
endpoint. The loaders in run.py look each model up by name and know how to
fetch it. This is the one place that used to be scattered across run.py and
inflect_tts.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_ASSETS = Path(__file__).resolve().parent.parent / "assets"


@dataclass(frozen=True)
class Model:
    """One model, described by whichever fields its stage needs:

    - Hugging Face download: ``repo`` + ``file``
    - bundled repo asset:    ``dir``
    - served over HTTP (the LLM): ``endpoint`` + ``model``
    """

    repo: str | None = None
    file: str | None = None
    dir: Path | None = None
    endpoint: str | None = None
    model: str | None = None

    def __post_init__(self) -> None:
        # The populated fields must match exactly one kind and nothing else.
        # This turns a typo in the registry below into a loud failure at import
        # time rather than a confusing error deep inside a loader — and unlike a
        # simple "exactly one kind is set" count, it also rejects a stray extra
        # field (e.g. a repo left on a bundled dir, or a dir with no file).
        populated = frozenset(
            name for name in ("repo", "file", "dir", "endpoint", "model")
            if getattr(self, name) is not None
        )
        kinds = {
            frozenset({"repo", "file"}),        # HF download
            frozenset({"dir"}),                 # bundled asset
            frozenset({"endpoint", "model"}),   # served over HTTP
        }
        if populated not in kinds:
            raise ValueError(
                "a Model must populate exactly one kind and nothing else — HF "
                "(repo+file), bundled (dir), or served (endpoint+model); "
                f"got {self!r}")


# Every model, current values, in ONE place. Change a line to swap a model.
MODELS: dict[str, Model] = {
    "yolox-tiny": Model(
        repo="litert-community/yolox-tiny-litert", file="yolox_tiny.tflite"),
    "moonshine-tiny": Model(
        repo="litert-community/moonshine-tiny",
        file="moonshine_tiny_5s_f32.tflite"),
    # Bundled in the repo (assets/inflect), not fetched from HF — the default TTS.
    "inflect-nano-v2": Model(dir=_ASSETS / "inflect"),
    "gemma-4-e2b": Model(
        endpoint="http://127.0.0.1:9379/v1/chat/completions", model="e2b"),
}

# Which model each pipeline stage uses. Swap a stage = change one name here.
DETECTOR = "yolox-tiny"
ASR = "moonshine-tiny"
LLM = "gemma-4-e2b"
# --tts picks a synthesizer by name (see build_synthesizer in run.py).
TTS = {
    "inflect": "inflect-nano-v2",
}


def get(name: str) -> Model:
    """Look a model up by name; raise loudly on an unknown name."""
    try:
        return MODELS[name]
    except KeyError:
        raise KeyError(
            f"unknown model {name!r}; known: {sorted(MODELS)}") from None
