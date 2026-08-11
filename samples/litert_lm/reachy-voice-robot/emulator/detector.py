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
wake the language model. Hence two requirements: stay within the frame
budget, and don't count frame jitter as an event.

The detector is **Ultralytics YOLO26** (`yolo26n`), exported to LiteRT. Two
properties of that export drive the code here, and both were confirmed
against the tensors on a real photo:

First: **the input is normalized [0, 1] in NCHW layout** — a float tensor
[1, 3, H, W]. Sources hand frames over as HWC pixels (either [0, 1], the
convenient convention, or raw [0, 255] straight from a decoded JPEG), so
preprocessing scales to [0, 1] and moves the channel axis first.

Second: **the export uses the raw detection head (`end2end=False`), so the
output is [1, 84, 8400]** — 84 = 4 box coords + 80 COCO class scores, no
objectness, across 8400 anchors (80² + 40² + 20²). YOLO26 is NMS-free by
default, but that end-to-end head lowers to INT64 gather/select ops the
LiteRT GPU delegate rejects (checked with the `gpu-clean-conversion` toolkit);
exporting `end2end=False` lets the GPU service (`demo/gpu_detect.py`) run the
graph fully on the VideoCore VII, with the decode + NMS below on the CPU. This
`Detector` runs the same graph on the CPU cores for the emulator/per-turn
path. The boxes come back already decoded and normalized to [0, 1] (cx, cy, w,
h) — no grid/stride math and no division by the input size: transpose to one
row per anchor, take each anchor's best class, threshold, convert center form
to corners, and suppress duplicates.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from emulator.litert_runtime import CompiledRunner


@dataclasses.dataclass(frozen=True)
class Detection:
    label: int
    score: float
    box: tuple[float, float, float, float]
    """Coordinates as fractions of the frame: (x1, y1, x2, y2)."""


def preprocess_frame(frame: np.ndarray,
                     expected_shape: tuple[int, ...]) -> np.ndarray:
    """Convert an HWC frame to the model's input: normalized [0, 1], NCHW.

    Ultralytics YOLO is trained on [0, 1] pixels and its LiteRT export takes
    NCHW [1, 3, H, W]. Sources hand frames over as HWC — already in [0, 1], or
    as raw [0, 255] when they come straight from a decoded JPEG — so scale to
    [0, 1] when needed, move the channel axis first, and add the batch axis.
    """
    data = np.asarray(frame, dtype=np.float32)
    if data.max() > 1.0:
        data = data / 255.0
    # Validate the incoming HWC frame against the model's H×W before the
    # transpose. A reshape alone catches only a total-element mismatch, so a
    # same-area but wrong-shape frame (e.g. 320×1280 vs 640×640) would reshape
    # into scrambled pixels and detect on garbage — fail loudly instead.
    _, channels, height, width = expected_shape
    if data.shape != (height, width, channels):
        raise ValueError(
            f"frame shape {data.shape} != expected HWC "
            f"{(height, width, channels)}; the source must resize to the "
            "model's input size before detection")
    # HWC -> CHW, add the batch axis.
    return np.transpose(data, (2, 0, 1)).reshape(expected_shape)


def decode_yolo_output(raw: np.ndarray,
                       score_threshold: float) -> list[Detection]:
    """Parse the Ultralytics YOLO detection output: [1, 84, 8400].

    84 = 4 box coords (cx, cy, w, h, already normalized to [0, 1]) + 80 class
    scores, no separate objectness. Anchors are laid out channels-first, so
    transpose to one row per anchor, take each anchor's best class, keep the
    ones over threshold, and convert center form to corner fractions.
    """
    if raw.size == 0:
        return []
    # Guard the layout: this decodes the channels-first raw head [1, 84, N].
    # An Ultralytics export can also emit anchor-major [1, N, 84] (same element
    # count), which the transpose would turn into silent garbage — so fail
    # loudly instead of "staying within budget and finding nothing".
    if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[1] != 84:
        raise ValueError(
            f"detector output shape {raw.shape} is not the expected "
            "[1, 84, N] channels-first raw head — re-check the YOLO26 "
            "end2end=False export (see assets/yolo/README.md)")

    predictions = raw[0].T                     # [8400, 84]
    boxes = predictions[:, :4]
    class_scores = predictions[:, 4:]
    labels = class_scores.argmax(axis=1)
    scores = class_scores.max(axis=1)
    keep = scores >= score_threshold
    if not keep.any():
        return []

    kept_boxes = boxes[keep]
    # The boxes must be normalized to [0, 1]; a pixel-space export (max ~640)
    # would collapse to slivers under the clip below and NMS would keep the
    # wreckage — catch that loudly rather than detecting on garbage.
    if kept_boxes.max() > 2.0:
        raise ValueError(
            "detector boxes look like pixel coordinates, not normalized "
            "[0, 1] — the LiteRT export should return normalized boxes "
            "(see assets/yolo/README.md)")
    cx, cy, w, h = kept_boxes.T
    half_w, half_h = w / 2.0, h / 2.0
    corners = np.stack(
        [cx - half_w, cy - half_h, cx + half_w, cy + half_h], axis=1)
    # Boxes are already normalized to [0, 1]; clip so an edge prediction can't
    # push a coordinate out of frame and break the gaze-direction calculation.
    corners = np.clip(corners, 0.0, 1.0)

    return [
        Detection(label=int(label), score=float(score),
                  box=(float(box[0]), float(box[1]),
                       float(box[2]), float(box[3])))
        for label, score, box in zip(labels[keep], scores[keep], corners)
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

    The raw head (end2end=False) leaves duplicate boxes for postprocessing —
    on a photo of a dog the model produced fourteen rows above threshold, one
    dog in fourteen boxes. Without suppression the prompt sent to the language
    model would read "I see: cat, cat, cat, ...", and it would end up
    responding to nonsense.

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
        # The model takes NCHW [1, 3, H, W]; sources produce HWC frames, so
        # report the (H, W, 3) a source should hand us — not the raw tensor
        # layout — and let preprocess_frame move the channel axis.
        _, channels, height, width = (int(x) for x in self._in_shape)
        return (height, width, channels)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        prepared = preprocess_frame(frame, self._in_shape)
        out = self._sig(**{self._in_name: prepared.astype(self._in_dtype.type)})
        raw = next(iter(out.values()))
        found = decode_yolo_output(raw, self._threshold)
        return non_max_suppression(found, self._iou)
