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
    - HF weights into a bundled dir: ``repo`` + ``dir`` — weights fetched from
      HF next to a runtime that ships in the repo (e.g. the Inflect TTS)
    - bundled repo asset:    ``dir`` (a directory), or ``dir`` + ``file`` (one file)
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
        # field (e.g. an endpoint left on a bundled dir, or a bare file with no repo or dir).
        populated = frozenset(
            name for name in ("repo", "file", "dir", "endpoint", "model")
            if getattr(self, name) is not None
        )
        kinds = {
            frozenset({"repo", "file"}),        # HF download
            frozenset({"repo", "dir"}),         # HF weights + bundled runtime dir
            frozenset({"dir"}),                 # bundled asset directory
            frozenset({"dir", "file"}),         # bundled single file
            frozenset({"endpoint", "model"}),   # served over HTTP
        }
        if populated not in kinds:
            raise ValueError(
                "a Model must populate exactly one kind and nothing else — HF "
                "(repo+file), HF weights into a bundled dir (repo+dir), bundled "
                "(dir, or dir+file), or served (endpoint+model); "
                f"got {self!r}")


# Every model, current values, in ONE place. Change a line to swap a model.
MODELS: dict[str, Model] = {
    # Ultralytics YOLO26 (yolo26n), exported to LiteRT once into assets/yolo
    # (git-ignored, like the other weights — see assets/yolo/README.md) rather
    # than fetched from HF. Exported with the raw head (end2end=False) so the
    # graph runs fully on the GPU delegate; postprocessing (decode + NMS) is on
    # the CPU. Swap it for another Ultralytics export by pointing this here.
    "yolo26n": Model(dir=_ASSETS / "yolo", file="yolo26n.tflite"),
    "moonshine-tiny": Model(
        repo="litert-community/moonshine-tiny",
        file="moonshine_tiny_5s_f32.tflite"),
    # Default TTS. The runtime (say.py) and espeak frontend ship in the repo
    # (assets/inflect); the LiteRT weights are fetched from HF on first use (like
    # the other models) because *.tflite is gitignored. Hosted in the LiteRT
    # community org — a LiteRT export of owensong/Inflect-Nano-v2 (Apache-2.0).
    "inflect-nano-v2": Model(repo="litert-community/Inflect-Nano-v2",
                             dir=_ASSETS / "inflect"),
    "gemma-4-e2b": Model(
        endpoint="http://127.0.0.1:9379/v1/chat/completions", model="e2b"),
}

# Which model each pipeline stage uses. Swap a stage = change one name here.
DETECTOR = "yolo26n"
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
