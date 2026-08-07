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

"""ReachyPlatform: the demo running entirely on the robot — against a FAKE SDK handle.

No hardware and no real `reachy_mini` install is needed here:
`ReachyPlatform(handle=...)` accepts a ready-made handle (in production, a
real `ReachyMini()`; in tests, the `FakeHandle` below), and
`reachy_mini.utils` (needed only to build the head-pose matrix,
`create_head_pose`) is swapped for a fake module via sys.modules — so the
test doesn't depend on whether the `reachy_mini` package is installed in the
environment at all (unlike `import ReachyMini`, `create_head_pose` is a pure
function, but the import itself still requires the package).
"""

from __future__ import annotations

import sys
import types
import wave

import numpy as np
import pytest

from demo.platform import build_platform
from demo.platform.reachy import ReachyPlatform
from demo.robot_reachy import GESTURE_POSES


class FakeMedia:
    """Fake MediaManager: frame/sample queues + a log of play_sound/close."""

    def __init__(self) -> None:
        self._frames: list = []
        self._samples: list = []
        self.played: list[str] = []
        self.closed = False

    def queue_frame(self, frame) -> None:
        self._frames.append(frame)

    def queue_audio(self, sample) -> None:
        self._samples.append(sample)

    def get_frame(self):
        return self._frames.pop(0) if self._frames else None

    def get_audio_sample(self):
        return self._samples.pop(0) if self._samples else None

    def play_sound(self, path: str) -> None:
        # Read the WAV NOW, during the call: _play_via_handle deletes the file
        # right after play_sound returns (no per-reply temp leak), so capturing
        # its content here is the only chance to inspect it.
        with wave.open(path, "rb") as fh:
            self.played.append({
                "channels": fh.getnchannels(),
                "framerate": fh.getframerate(),
                "nframes": fh.getnframes(),
            })

    def close(self) -> None:
        self.closed = True


class FakeHandle:
    """Fake ReachyMini: media (see above) + a log of goto_target calls for the motors."""

    def __init__(self) -> None:
        self.media = FakeMedia()
        self.goto_calls: list[tuple] = []

    def goto_target(self, head=None, duration=None) -> None:
        self.goto_calls.append((head, duration))


@pytest.fixture(autouse=True)
def fake_reachy_mini_utils(monkeypatch):
    """Swap `reachy_mini.utils.create_head_pose` for a fake in sys.modules.

    `from reachy_mini.utils import create_head_pose` resolves via
    sys.modules BEFORE touching the filesystem — so fake entries in
    sys.modules are enough; the real `reachy_mini` package doesn't need to
    be installed (the test exercises the fake handle, not the hardware).
    """
    calls = []

    def fake_create_head_pose(yaw=0, pitch=0, degrees=True, **kw):
        calls.append({"yaw": yaw, "pitch": pitch, "degrees": degrees})
        return {"yaw": yaw, "pitch": pitch}

    fake_utils = types.ModuleType("reachy_mini.utils")
    fake_utils.create_head_pose = fake_create_head_pose
    fake_pkg = types.ModuleType("reachy_mini")
    fake_pkg.utils = fake_utils
    monkeypatch.setitem(sys.modules, "reachy_mini", fake_pkg)
    monkeypatch.setitem(sys.modules, "reachy_mini.utils", fake_utils)
    return calls


# --- video ---

def test_video_source_reads_frame_from_handle():
    handle = FakeHandle()
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[0, 0] = [1, 2, 3]
    handle.media.queue_frame(frame)

    platform = ReachyPlatform(handle=handle)
    video = platform.video_source()
    # latest() returns a COPY (thread-safety, mirrors CameraStream) — equal in
    # content but a distinct object, so assert equality, not identity.
    assert np.array_equal(video.latest(), frame)

    png = video.latest_png()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_video_source_caches_last_frame_when_get_frame_returns_none():
    handle = FakeHandle()
    frame = np.ones((4, 4, 3), dtype=np.uint8)
    handle.media.queue_frame(frame)

    video = ReachyPlatform(handle=handle).video_source()
    assert np.array_equal(video.latest(), frame)
    # Frame queue is empty — get_frame() returns None, but latest() must
    # hold onto the last known frame instead of falling back to None.
    assert np.array_equal(video.latest(), frame)


def test_video_source_is_none_before_first_frame():
    video = ReachyPlatform(handle=FakeHandle()).video_source()
    assert video.latest() is None
    assert video.latest_png() is None


def test_video_source_alive_reflects_frame_freshness():
    # Parity with CameraStream.alive so run_voice's camera-health print works
    # on the robot too. Before any frame it's not "dead"; a fresh frame keeps
    # it alive; a long gap since the last fresh frame marks it stale.
    handle = FakeHandle()
    video = ReachyPlatform(handle=handle).video_source()
    assert video.alive is True
    handle.media.queue_frame(np.ones((4, 4, 3), dtype=np.uint8))
    video.latest()
    assert video.alive is True
    video._last_fresh -= 100.0  # as if the SDK stopped delivering long ago
    assert video.alive is False


# --- mic ---

def test_mic_chunks_assembled_from_audio_samples():
    handle = FakeHandle()
    # 16kHz, chunk_ms=100 → 1600 samples/chunk; one batch covers 2 chunks at once.
    handle.media.queue_audio(np.ones(3200, dtype=np.float32) * 0.5)

    mic = ReachyPlatform(handle=handle).mic_source()
    chunks = mic.chunks()
    chunk1, level1 = next(chunks)
    chunk2, level2 = next(chunks)

    assert len(chunk1) == len(chunk2) == 1600
    assert chunk1.dtype == np.float32
    assert abs(level1 - 0.5) < 1e-6
    assert abs(level2 - 0.5) < 1e-6


def test_mic_flush_drains_buffered_samples():
    handle = FakeHandle()
    handle.media.queue_audio(np.ones(10, dtype=np.float32))
    handle.media.queue_audio(np.ones(5, dtype=np.float32))

    mic = ReachyPlatform(handle=handle).mic_source()
    mic.flush()

    # Both batches are drained out (until None/empty) — the fake's queue is empty.
    assert handle.media.get_audio_sample() is None


# --- player: writes a WAV and calls play_sound on handle.media ---

def test_make_player_writes_wav_and_calls_play_sound():
    handle = FakeHandle()
    player = ReachyPlatform(handle=handle).make_player(16000)
    player.feed(np.zeros(1600, dtype=np.float32))
    player.close()

    assert len(handle.media.played) == 1
    rec = handle.media.played[0]
    assert rec["channels"] == 1
    assert rec["framerate"] == 16000


def test_make_player_skips_play_sound_when_nothing_fed():
    handle = FakeHandle()
    ReachyPlatform(handle=handle).make_player(16000).close()
    assert handle.media.played == []


def test_play_via_handle_removes_temp_wav_after_playback():
    # No per-reply temp leak: the WAV exists DURING play_sound (so the robot
    # can actually play it) but is gone right after — otherwise a long demo
    # grows the Pi's temp dir without bound.
    import os as _os

    from demo.platform.reachy import _play_via_handle

    seen: dict = {}

    class _Media:
        def play_sound(self, path):
            seen["path"] = path
            seen["existed_during"] = _os.path.exists(path)

    class _Handle:
        media = _Media()

    _play_via_handle(_Handle(), np.zeros(1600, dtype=np.float32), 16000)
    assert seen["existed_during"] is True
    assert not _os.path.exists(seen["path"])


# --- robot: head movement through the same handle ---

def test_robot_look_at_drives_handle_goto_target():
    handle = FakeHandle()
    robot = ReachyPlatform(handle=handle).robot()
    robot.look_at(1.0, 0.0)

    assert len(handle.goto_calls) == 1
    head, duration = handle.goto_calls[0]
    assert duration == 0.5
    assert head == {"yaw": 20.0, "pitch": 0.0}


def test_robot_gesture_plays_each_pose_of_the_sequence():
    handle = FakeHandle()
    robot = ReachyPlatform(handle=handle).robot()
    robot.gesture("shake")

    assert len(handle.goto_calls) == len(GESTURE_POSES["shake"])
    assert all(duration == 0.25 for _, duration in handle.goto_calls)


def test_robot_gesture_ignores_unknown_name():
    handle = FakeHandle()
    ReachyPlatform(handle=handle).robot().gesture("moonwalk")
    assert handle.goto_calls == []


def test_robot_say_plays_via_handle_media():
    handle = FakeHandle()
    robot = ReachyPlatform(handle=handle).robot()
    robot.say(np.zeros(1600, dtype=np.float32), sample_rate=16000)
    assert len(handle.media.played) == 1


def test_robot_say_skips_empty_audio():
    handle = FakeHandle()
    ReachyPlatform(handle=handle).robot().say(
        np.zeros(0, dtype=np.float32), sample_rate=16000)
    assert handle.media.played == []


# --- platform lifecycle ---

def test_close_closes_media():
    handle = FakeHandle()
    ReachyPlatform(handle=handle).close()
    assert handle.media.closed


def test_build_platform_reachy_constructs_reachy_platform(monkeypatch):
    """build_platform("reachy", args) must call ReachyPlatform().

    handle=None by default would spin up a real ReachyMini() — which
    requires the physical robot. So we swap the class itself for a
    lightweight recorder of the call: build_platform does
    `from demo.platform.reachy import ReachyPlatform` as a lazy local import
    inside its body, so patching the module attribute BEFORE the call gets
    picked up by that import.
    """
    import demo.platform.reachy as reachy_mod

    calls = []
    monkeypatch.setattr(reachy_mod, "ReachyPlatform",
                        lambda *a, **k: calls.append((a, k)) or "sentinel")
    result = build_platform("reachy", args=None)
    assert result == "sentinel"
    assert calls == [((), {})]


def test_build_platform_lazy_imports_reachy_module_only_for_reachy_kind():
    """build_platform("mac", ...) must not even import demo.platform.reachy."""
    import argparse

    sys.modules.pop("demo.platform.reachy", None)
    build_platform("mac", argparse.Namespace(video="0", audio="1", no_robot=True))
    assert "demo.platform.reachy" not in sys.modules
