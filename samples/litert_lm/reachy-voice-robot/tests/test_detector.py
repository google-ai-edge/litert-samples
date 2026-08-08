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

"""Detector — the cheap watcher that decides when to wake the expensive model.

The model itself isn't loaded here: this checks output parsing and event
logic, while inference is verified by a live run on the board. The key
property under test: an object moving does NOT count as an event. The
language model costs ~1.2 s, the detector ~220 ms, and if the former wakes
up on every camera jitter, the whole point of the latter disappears.
"""

from pathlib import Path

import numpy as np
import pytest

from emulator.detector import (
    Detection,
    Detector,
    build_grid,
    decode_yolox_output,
    non_max_suppression,
    preprocess_frame,
    scene_changed,
)


def det(label: int, score: float = 0.9) -> Detection:
    return Detection(label=label, score=score, box=(0.1, 0.1, 0.5, 0.5))


def test_scene_change_on_new_label():
    assert scene_changed([det(0)], [det(0), det(15)]) is True


def test_scene_change_on_disappearance():
    assert scene_changed([det(0), det(15)], [det(0)]) is True


def test_no_change_when_labels_same():
    # An object shifting within the frame is not an event: otherwise the
    # heavy model would wake up on every jitter.
    moved = Detection(label=0, score=0.9, box=(0.4, 0.4, 0.8, 0.8))
    assert scene_changed([det(0)], [moved]) is False


def test_no_change_when_score_wobbles():
    assert scene_changed([det(0, 0.91)], [det(0, 0.72)]) is False


def test_empty_to_empty_is_not_a_change():
    # An empty frame following an empty frame is not an event, otherwise
    # the model would wake up every half second in an empty room.
    assert scene_changed([], []) is False


# --- input preprocessing ---

def test_preprocess_keeps_raw_pixel_range():
    # Measured on the board: [0,1] input gives objectness 0.0003 and ZERO
    # detections, raw [0,255] gives 0.6887 and 14 detections. The model
    # expects raw pixels.
    frame = np.full((416, 416, 3), 0.5, dtype=np.float32)
    out = preprocess_frame(frame, (1, 416, 416, 3))
    assert out.max() > 100, "a normalized frame must be scaled back to [0,255]"


def test_preprocess_leaves_raw_frame_alone():
    frame = np.full((416, 416, 3), 200.0, dtype=np.float32)
    out = preprocess_frame(frame, (1, 416, 416, 3))
    assert out.max() == pytest.approx(200.0), "a raw frame must not be rescaled"


def test_preprocess_adds_batch_dimension():
    frame = np.zeros((416, 416, 3), dtype=np.float32)
    assert preprocess_frame(frame, (1, 416, 416, 3)).shape == (1, 416, 416, 3)


# --- grid ---

def test_grid_covers_all_three_scales():
    # 3549 = 52² + 26² + 13² at strides 8, 16, 32 for a 416 input.
    grid, strides = build_grid(416)
    assert grid.shape == (3549, 2)
    assert strides.shape == (3549, 1)
    assert sorted(set(int(s) for s in strides.ravel())) == [8, 16, 32]


def test_grid_row_count_matches_each_stride():
    _, strides = build_grid(416)
    flat = strides.ravel()
    assert int((flat == 8).sum()) == 52 * 52
    assert int((flat == 16).sum()) == 26 * 26
    assert int((flat == 32).sum()) == 13 * 13


# --- decoding ---

def test_decode_applies_grid_and_stride():
    # Measured: cx lies in the range -0.6..2.5, i.e. it's an offset within
    # the cell, not a pixel. The real center = (offset + cell) * stride.
    raw = np.zeros((1, 3549, 85), dtype=np.float32)
    row = 52 * 52 + 26 * 26 + 5      # sixth cell of the stride-32 scale
    raw[0, row, :4] = [0.0, 0.0, 0.0, 0.0]
    raw[0, row, 4] = 0.9
    raw[0, row, 5] = 1.0
    box = decode_yolox_output(raw, 0.3, 416)[0].box
    # Cell (5, 0) at stride 32 → center (160, 0); exp(0)*32 = 32 width.
    center_x = (box[0] + box[2]) / 2 * 416
    assert center_x == pytest.approx(160.0, abs=1.0)
    assert (box[2] - box[0]) * 416 == pytest.approx(32.0, abs=1.0)


def test_decode_filters_by_score():
    raw = np.zeros((1, 3549, 85), dtype=np.float32)
    raw[0, 0, 4] = 0.95
    raw[0, 0, 5] = 1.0          # class 0
    raw[0, 1, 4] = 0.05         # below threshold
    raw[0, 1, 5] = 1.0
    dets = decode_yolox_output(raw, score_threshold=0.3, input_size=416)
    assert len(dets) == 1
    assert dets[0].label == 0
    assert dets[0].score == pytest.approx(0.95, abs=1e-5)


def test_decode_returns_boxes_inside_frame():
    # Coordinates are fractions of the frame and clipped: a box outside
    # [0,1] would break the gaze direction.
    raw = np.zeros((1, 3549, 85), dtype=np.float32)
    raw[0, 0, :4] = [-5.0, -5.0, 3.0, 3.0]   # deliberately out of bounds
    raw[0, 0, 4] = 0.9
    raw[0, 0, 5] = 1.0
    box = decode_yolox_output(raw, 0.3, 416)[0].box
    assert all(0.0 <= v <= 1.0 for v in box), f"box outside frame: {box}"


def test_decode_picks_highest_class_score():
    raw = np.zeros((1, 3549, 85), dtype=np.float32)
    raw[0, 0, 4] = 0.9
    raw[0, 0, 5] = 0.2          # class 0
    raw[0, 0, 20] = 0.8         # class 15 — should win
    assert decode_yolox_output(raw, 0.3, 416)[0].label == 15


def test_decode_handles_empty_output():
    raw = np.zeros((1, 0, 85), dtype=np.float32)
    assert decode_yolox_output(raw, 0.3, 416) == []


# --- duplicate suppression ---

def test_nms_collapses_overlapping_boxes_of_same_class():
    # On a photo of a dog, the model produced 14 rows above threshold —
    # that's one dog in fourteen boxes. Without suppression, the prompt
    # would end up with "I see: cat, cat, cat, ...".
    a = Detection(label=15, score=0.9, box=(0.1, 0.1, 0.5, 0.5))
    b = Detection(label=15, score=0.7, box=(0.12, 0.12, 0.52, 0.52))
    kept = non_max_suppression([a, b], iou_threshold=0.5)
    assert len(kept) == 1
    assert kept[0].score == 0.9, "the most confident detection must be kept"


def test_nms_keeps_distant_boxes():
    a = Detection(label=15, score=0.9, box=(0.0, 0.0, 0.2, 0.2))
    b = Detection(label=15, score=0.8, box=(0.7, 0.7, 0.9, 0.9))
    assert len(non_max_suppression([a, b], iou_threshold=0.5)) == 2


def test_nms_keeps_different_classes_at_same_place():
    # A person holding a dog occupy the same region — both are needed.
    a = Detection(label=0, score=0.9, box=(0.1, 0.1, 0.5, 0.5))
    b = Detection(label=15, score=0.8, box=(0.1, 0.1, 0.5, 0.5))
    assert len(non_max_suppression([a, b], iou_threshold=0.5)) == 2


def test_nms_handles_empty_input():
    assert non_max_suppression([], iou_threshold=0.5) == []


# --- Detector constructor: the threads parameter must not be dead ---

def test_detector_forwards_threads_to_compiled_runner(monkeypatch):
    captured = {}

    class FakeSig:
        def get_input_details(self):
            return {"in0": {"shape": (1, 416, 416, 3),
                            "dtype": np.dtype(np.float32)}}

    class FakeRunner:
        def __init__(self, model_path, threads=4):
            captured["threads"] = threads

        def only(self):
            return FakeSig()

    monkeypatch.setattr("emulator.detector.CompiledRunner", FakeRunner)
    Detector(Path("fake.tflite"), threads=7)
    assert captured["threads"] == 7, \
        "threads must be forwarded to CompiledRunner, not left unused"
