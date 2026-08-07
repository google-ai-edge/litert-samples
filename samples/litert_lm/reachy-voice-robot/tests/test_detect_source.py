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

import logging

import numpy as np

from demo.detect_source import GpuDetectSource


class FakeCamera:
    def __init__(self, frame):
        self._frame = frame

    def latest(self):
        return self._frame


FRAME = np.zeros((8, 8, 3), np.uint8)


def test_latest_returns_frame_and_detections_snapshot():
    source = GpuDetectSource(FakeCamera(FRAME), "http://x/detect")
    dets = [{"label": "cup", "score": 0.8, "box": [0.4, 0.4, 0.6, 0.6]}]
    source._post_detect = lambda frame: dets
    source._cycle()
    frame, out = source.latest()
    assert np.array_equal(frame, FRAME)
    assert out == dets


def test_latest_is_a_snapshot_copy_not_shared_state():
    source = GpuDetectSource(FakeCamera(FRAME), "http://x/detect")
    source._post_detect = lambda frame: []
    source._cycle()
    frame, _ = source.latest()
    frame[:] = 255
    frame2, _ = source.latest()
    assert not np.array_equal(frame2, frame), "latest() returns a copy, not a shared buffer"


def test_no_frame_yet_returns_none_and_empty_list():
    source = GpuDetectSource(FakeCamera(None), "http://x/detect")
    frame, dets = source.latest()
    assert frame is None
    assert dets == []


def test_listener_fires_each_cycle_with_fresh_detections():
    source = GpuDetectSource(FakeCamera(FRAME), "http://x/detect")
    calls = []
    source.add_listener(calls.append)
    responses = iter([[{"label": "person"}], [{"label": "cup"}]])
    source._post_detect = lambda frame: next(responses)
    source._cycle()
    source._cycle()
    assert calls == [[{"label": "person"}], [{"label": "cup"}]]


def test_cycle_skips_when_camera_has_no_frame_yet():
    source = GpuDetectSource(FakeCamera(None), "http://x/detect")
    calls = []
    source.add_listener(calls.append)
    source._cycle()
    assert calls == []


def test_post_detect_failure_is_logged(monkeypatch, caplog):
    source = GpuDetectSource(FakeCamera(FRAME), "http://x/detect")

    def boom(*a, **k):
        raise OSError("gpu down")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with caplog.at_level(logging.WARNING):
        result = source._post_detect(FRAME)
    assert result is None
    assert any("OSError" in r.message and "gpu down" in r.message
                for r in caplog.records)


def test_none_vs_empty_list_distinction_for_healthy():
    """None (failure) bumps the failure counter; [] (empty scene) resets it."""
    source = GpuDetectSource(FakeCamera(FRAME), "http://x/detect")
    source._post_detect = lambda frame: None
    source._cycle()
    source._cycle()
    assert source._consecutive_failures == 2
    source._post_detect = lambda frame: []
    source._cycle()
    assert source._consecutive_failures == 0
    # latest() always hands the consumer a list, never None, either way
    _, dets = source.latest()
    assert dets == []


def test_healthy_flips_false_after_n_consecutive_failures():
    source = GpuDetectSource(FakeCamera(FRAME), "http://x/detect")
    source._post_detect = lambda frame: None
    for _ in range(4):
        source._cycle()
        assert source.healthy() is True
    source._cycle()   # 5th consecutive failure
    assert source.healthy() is False


def test_post_detect_returns_none_when_body_lacks_detections_key(
        monkeypatch, caplog):
    # A 200 whose JSON body has no "detections" key is contract drift (a wrong
    # endpoint or an error-shaped 200), not an empty scene: _post_detect must
    # return None so it counts toward unhealthy(), rather than [] ("saw
    # nothing").
    source = GpuDetectSource(FakeCamera(FRAME), "http://x/detect")

    class FakeResponse:
        def read(self):
            return b'{"ms": 1}'

    monkeypatch.setattr("urllib.request.urlopen",
                        lambda req, timeout=5: FakeResponse())
    with caplog.at_level(logging.WARNING):
        result = source._post_detect(FRAME)
    assert result is None
    assert any("detections" in r.message for r in caplog.records)


def test_healthy_recovers_after_a_success():
    source = GpuDetectSource(FakeCamera(FRAME), "http://x/detect")
    source._post_detect = lambda frame: None
    for _ in range(5):
        source._cycle()
    assert source.healthy() is False
    source._post_detect = lambda frame: [{"label": "cup"}]
    source._cycle()
    assert source.healthy() is True
