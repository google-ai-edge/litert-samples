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
language model costs ~1.2 s to the first token, the detector a fraction of
that, and if the former wakes up on every camera jitter, the whole point of
the latter disappears.
"""

from pathlib import Path

import numpy as np
import pytest

from emulator.detector import (
    Detection,
    Detector,
    decode_yolo_output,
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


# --- input preprocessing: Ultralytics YOLO wants normalized [0,1], NCHW ---

def test_preprocess_normalizes_raw_pixels():
    # A decoded JPEG arrives as [0,255]; the model is trained on [0,1].
    frame = np.full((640, 640, 3), 200.0, dtype=np.float32)
    out = preprocess_frame(frame, (1, 3, 640, 640))
    assert out.max() == pytest.approx(200.0 / 255.0), \
        "raw pixels must be scaled to [0,1]"


def test_preprocess_leaves_normalized_frame_alone():
    frame = np.full((640, 640, 3), 0.5, dtype=np.float32)
    out = preprocess_frame(frame, (1, 3, 640, 640))
    assert out.max() == pytest.approx(0.5), "a [0,1] frame must not be rescaled"


def test_preprocess_produces_nchw_batch():
    frame = np.zeros((640, 640, 3), dtype=np.float32)
    assert preprocess_frame(frame, (1, 3, 640, 640)).shape == (1, 3, 640, 640)


def test_preprocess_moves_channels_first():
    # HWC -> CHW: distinct per-channel values must land on the channel axis,
    # not stay interleaved — a transpose bug would swap the colour planes.
    frame = np.zeros((2, 2, 3), dtype=np.float32)
    frame[..., 0], frame[..., 1], frame[..., 2] = 0.1, 0.2, 0.3
    out = preprocess_frame(frame, (1, 3, 2, 2))
    assert out[0, 0].max() == pytest.approx(0.1)
    assert out[0, 1].max() == pytest.approx(0.2)
    assert out[0, 2].max() == pytest.approx(0.3)


def test_preprocess_rejects_wrong_frame_shape():
    # A same-area but wrong-H×W frame would reshape into scrambled pixels; the
    # explicit shape check must reject it loudly rather than detect on garbage.
    frame = np.zeros((480, 640, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        preprocess_frame(frame, (1, 3, 640, 640))


# --- decoding: raw head output is [1, 84, 8400], boxes normalized [0,1] ---

def _anchor(raw: np.ndarray, index: int, cx: float, cy: float,
            w: float, h: float, label: int, score: float) -> None:
    """Fill anchor `index`: channels 0-3 are the box, 4+label is a class."""
    raw[0, 0, index] = cx
    raw[0, 1, index] = cy
    raw[0, 2, index] = w
    raw[0, 3, index] = h
    raw[0, 4 + label, index] = score


def test_decode_converts_center_to_corner_fractions():
    # The raw head hands back boxes already decoded and normalized: (cx, cy, w,
    # h) in [0,1]. No grid, no stride, no division by the input size.
    raw = np.zeros((1, 84, 8400), dtype=np.float32)
    _anchor(raw, 0, cx=0.5, cy=0.5, w=0.2, h=0.4, label=0, score=0.9)
    box = decode_yolo_output(raw, 0.3)[0].box
    assert box == pytest.approx((0.4, 0.3, 0.6, 0.7))


def test_decode_filters_by_score():
    raw = np.zeros((1, 84, 8400), dtype=np.float32)
    _anchor(raw, 0, 0.5, 0.5, 0.2, 0.2, label=0, score=0.95)
    _anchor(raw, 1, 0.5, 0.5, 0.2, 0.2, label=0, score=0.05)  # below threshold
    dets = decode_yolo_output(raw, score_threshold=0.3)
    assert len(dets) == 1
    assert dets[0].label == 0
    assert dets[0].score == pytest.approx(0.95, abs=1e-5)


def test_decode_picks_highest_class_score():
    raw = np.zeros((1, 84, 8400), dtype=np.float32)
    raw[0, 0:4, 0] = [0.5, 0.5, 0.2, 0.2]
    raw[0, 4, 0] = 0.2          # class 0
    raw[0, 4 + 15, 0] = 0.8     # class 15 — should win
    assert decode_yolo_output(raw, 0.3)[0].label == 15


def test_decode_clips_boxes_inside_frame():
    # A box centred near the edge can spill past [0,1]; that would break the
    # gaze-direction calculation, so coordinates are clipped.
    raw = np.zeros((1, 84, 8400), dtype=np.float32)
    _anchor(raw, 0, cx=0.95, cy=0.95, w=0.5, h=0.5, label=0, score=0.9)
    box = decode_yolo_output(raw, 0.3)[0].box
    assert all(0.0 <= v <= 1.0 for v in box), f"box outside frame: {box}"


def test_decode_handles_empty_output():
    raw = np.zeros((1, 84, 0), dtype=np.float32)
    assert decode_yolo_output(raw, 0.3) == []


def test_decode_returns_nothing_when_all_below_threshold():
    raw = np.zeros((1, 84, 8400), dtype=np.float32)   # every class score 0
    assert decode_yolo_output(raw, 0.3) == []


def test_decode_keeps_multiple_detections_aligned():
    # Two anchors at distinct non-zero indices with distinct labels/scores/boxes:
    # proves the masked label/score/box arrays stay aligned (index 0 alone would
    # hide a dropped-mask bug) and that decode returns more than one detection.
    raw = np.zeros((1, 84, 8400), dtype=np.float32)
    _anchor(raw, 100, cx=0.25, cy=0.25, w=0.2, h=0.2, label=7, score=0.8)
    _anchor(raw, 5000, cx=0.75, cy=0.75, w=0.4, h=0.4, label=15, score=0.6)
    got = {(d.label, round(d.score, 3), tuple(round(v, 3) for v in d.box))
           for d in decode_yolo_output(raw, 0.3)}
    assert got == {
        (7, 0.8, (0.15, 0.15, 0.35, 0.35)),
        (15, 0.6, (0.55, 0.55, 0.95, 0.95)),
    }


def test_decode_rejects_transposed_output():
    # An anchor-major [1, 8400, 84] export has the same element count but the
    # wrong layout; decoding it would be silent garbage, so it must raise.
    raw = np.zeros((1, 8400, 84), dtype=np.float32)
    with pytest.raises(ValueError):
        decode_yolo_output(raw, 0.3)


def test_decode_rejects_pixel_space_boxes():
    # Boxes must be normalized [0,1]; a pixel-space export (max ~640) must raise
    # rather than collapse every box to a sliver under the clip.
    raw = np.zeros((1, 84, 8400), dtype=np.float32)
    _anchor(raw, 0, cx=320.0, cy=240.0, w=100.0, h=80.0, label=0, score=0.9)
    with pytest.raises(ValueError):
        decode_yolo_output(raw, 0.3)


# --- duplicate suppression (raw head leaves duplicates for the CPU) ---

def test_nms_collapses_overlapping_boxes_of_same_class():
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


# --- Detector: input-shape reporting and the threads parameter ---

class _FakeSig:
    def __init__(self, shape, output=None):
        self._shape = shape
        self._output = output

    def get_input_details(self):
        return {"args_0": {"shape": self._shape,
                           "dtype": np.dtype(np.float32)}}

    def __call__(self, **inputs):
        return {"out0": self._output}


def _fake_runner(captured, shape=(1, 3, 640, 640), output=None):
    class FakeRunner:
        def __init__(self, model_path, threads=4):
            captured["threads"] = threads

        def only(self):
            return _FakeSig(shape, output)
    return FakeRunner


def test_detector_reports_hwc_input_shape_from_nchw(monkeypatch):
    # The model is NCHW [1,3,H,W]; sources produce HWC frames, so input_shape
    # must report (H, W, 3), not the raw tensor layout.
    monkeypatch.setattr("emulator.detector.CompiledRunner",
                        _fake_runner({}, shape=(1, 3, 640, 640)))
    assert Detector(Path("fake.tflite")).input_shape() == (640, 640, 3)


def test_detector_forwards_threads_to_compiled_runner(monkeypatch):
    captured = {}
    monkeypatch.setattr("emulator.detector.CompiledRunner",
                        _fake_runner(captured))
    Detector(Path("fake.tflite"), threads=7)
    assert captured["threads"] == 7, \
        "threads must be forwarded to CompiledRunner, not left unused"


def test_detector_detect_decodes_and_suppresses(monkeypatch):
    # End-to-end wiring: preprocess (raw NCHW self._in_shape) -> signature call
    # -> decode_yolo_output -> non_max_suppression. Two overlapping same-class
    # anchors in the crafted output must collapse to a single Detection.
    out = np.zeros((1, 84, 8400), dtype=np.float32)
    _anchor(out, 0, cx=0.5, cy=0.5, w=0.4, h=0.4, label=15, score=0.9)
    _anchor(out, 1, cx=0.51, cy=0.51, w=0.4, h=0.4, label=15, score=0.7)  # dup
    monkeypatch.setattr("emulator.detector.CompiledRunner",
                        _fake_runner({}, output=out))
    detector = Detector(Path("fake.tflite"), score_threshold=0.3)
    dets = detector.detect(np.zeros((640, 640, 3), dtype=np.float32))
    assert len(dets) == 1
    assert dets[0].label == 15
    assert dets[0].score == pytest.approx(0.9, abs=1e-5)
