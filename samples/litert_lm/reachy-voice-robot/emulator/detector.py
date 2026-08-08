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

"""Object detector: frame in, list of findings out.

Its role in the pipeline is a cheap, always-on watcher that decides when to
wake the language model. Hence two requirements: stay within the ~220 ms
budget, and don't count frame jitter as an event.

The `yolox-tiny` model's contract was reverse-engineered on the board, and
both of its quirks were non-obvious enough that the first implementation
was silently blind: it stayed within budget and found nothing.

First: **the input is raw pixels [0, 255], not normalized.** On a photo of
a dog, feeding [0,1] input gave an objectness of 0.0003 and zero
detections; on raw values it gave 0.6887 and fourteen. Normalization here
isn't "a different scale" — it fully disables the model.

Second: **the export does not decode boxes.** The output is [1, 3549, 85],
where 3549 = 52² + 26² + 13² — three scales at strides 8, 16, 32. The
cx/cy coordinates lie in the range -0.6..2.5, i.e. they are offsets within
the cell, while width and height are logarithms. The real box is:
(offset + cell) × stride and exp(logarithm) × stride.
"""

from __future__ import annotations

import dataclasses
import functools
from pathlib import Path

import numpy as np

from emulator.litert_runtime import CompiledRunner

STRIDES = (8, 16, 32)


@dataclasses.dataclass(frozen=True)
class Detection:
    label: int
    score: float
    box: tuple[float, float, float, float]
    """Coordinates as fractions of the frame: (x1, y1, x2, y2)."""


def preprocess_frame(frame: np.ndarray,
                     expected_shape: tuple[int, ...]) -> np.ndarray:
    """Convert the frame to what the model expects: raw pixels [0, 255].

    Sources hand frames over in [0, 1] — that's the convenient convention
    most models expect. YOLOX, however, was trained on raw values, and
    dividing by 255 kills it completely: objectness drops from 0.69 to
    0.0003.
    """
    data = np.asarray(frame, dtype=np.float32)
    if data.max() <= 1.0:
        data = data * 255.0
    return data.reshape(expected_shape)


@functools.lru_cache(maxsize=4)
def build_grid(input_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Grid of cells and stride for each output row.

    Cached: the grid depends only on input size, yet without caching it
    would be rebuilt on every frame — and the GPU detect source polls at
    ~4 fps.
    """
    grids = []
    strides = []
    for stride in STRIDES:
        side = input_size // stride
        ys, xs = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
        grids.append(np.stack((xs, ys), axis=2).reshape(-1, 2))
        strides.append(np.full((side * side, 1), stride, dtype=np.float32))
    return (np.concatenate(grids).astype(np.float32),
            np.concatenate(strides))


def decode_yolox_output(raw: np.ndarray, score_threshold: float,
                        input_size: int) -> list[Detection]:
    """Parse the yolox output: [1, N, 85] — box, objectness, 80 classes.

    The export hands back boxes undecoded, so the grid is applied here.
    """
    if raw.size == 0 or raw.shape[1] == 0:
        return []

    predictions = raw[0]
    objectness = predictions[:, 4]
    keep = objectness >= score_threshold
    if not keep.any():
        return []

    grid, strides = build_grid(input_size)
    if grid.shape[0] != predictions.shape[0]:
        raise ValueError(
            f"model output has {predictions.shape[0]} rows, but the grid for "
            f"input {input_size} gives {grid.shape[0]}. Strides {STRIDES} "
            "don't fit this export"
        )

    rows = predictions[keep]
    cell = grid[keep]
    stride = strides[keep]

    centers = (rows[:, :2] + cell) * stride
    # Width and height are on a logarithmic scale.
    sizes = np.exp(np.clip(rows[:, 2:4], -10.0, 10.0)) * stride

    half = sizes / 2.0
    corners = np.concatenate([centers - half, centers + half], axis=1)
    # Convert to fractions of the frame and clip: yolox can predict boxes
    # past the edge, and coordinates outside [0, 1] would break the
    # gaze-direction calculation.
    corners = np.clip(corners / input_size, 0.0, 1.0)

    labels = rows[:, 5:].argmax(axis=1)
    scores = objectness[keep]
    return [
        Detection(label=int(label), score=float(score),
                  box=(float(box[0]), float(box[1]),
                       float(box[2]), float(box[3])))
        for label, score, box in zip(labels, scores, corners)
    ]


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if overlap <= 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - overlap
    return overlap / union if union > 0 else 0.0


def non_max_suppression(detections: list[Detection],
                        iou_threshold: float = 0.45) -> list[Detection]:
    """Keep a single box per object.

    On a photo of a dog, the model produced fourteen rows above threshold —
    that's one dog in fourteen boxes. Without suppression, the prompt sent
    to the language model would read "I see: cat, cat, cat, ...", and it
    would end up responding to nonsense.

    Classes are suppressed independently: a person holding a dog occupy the
    same region of the frame, and both are needed.
    """
    kept: list[Detection] = []
    for candidate in sorted(detections, key=lambda d: d.score, reverse=True):
        if all(_iou(candidate.box, other.box) <= iou_threshold
               for other in kept if other.label == candidate.label):
            kept.append(candidate)
    return kept


def scene_changed(previous: list[Detection],
                  current: list[Detection]) -> bool:
    """Whether the SET of classes in the frame changed.

    We compare sets of classes, not boxes: an object moving doesn't count
    as an event, otherwise the heavy model would wake up on every camera
    jitter and the whole point of the cheap detector would be lost.
    """
    return {d.label for d in previous} != {d.label for d in current}


class Detector:
    def __init__(self, model_path: Path, threads: int = 4,
                 score_threshold: float = 0.3,
                 iou_threshold: float = 0.45) -> None:
        # CompiledModel is the current LiteRT API (Interpreter is deprecated).
        # The detector is single-signature, one input, one output.
        self._runner = CompiledRunner(model_path, threads=threads)
        self._sig = self._runner.only()
        (self._in_name, in_meta), = self._sig.get_input_details().items()
        self._in_shape = in_meta["shape"]
        self._in_dtype = in_meta["dtype"]
        self._threshold = score_threshold
        self._iou = iou_threshold

    def input_shape(self) -> tuple[int, int, int]:
        return tuple(int(x) for x in self._in_shape[1:])   # without the batch dimension

    def detect(self, frame: np.ndarray) -> list[Detection]:
        prepared = preprocess_frame(frame, self._in_shape)
        out = self._sig(**{self._in_name: prepared.astype(self._in_dtype.type)})
        raw = next(iter(out.values()))
        found = decode_yolox_output(raw, self._threshold,
                                    input_size=int(self._in_shape[1]))
        return non_max_suppression(found, self._iou)
