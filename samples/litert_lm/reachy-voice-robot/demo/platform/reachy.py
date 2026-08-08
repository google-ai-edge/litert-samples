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

"""ReachyPlatform: the demo running entirely on the Reachy Mini itself, no Mac.

One `ReachyMini` handle gives us eyes, ears, mouth, and motors — with it the
demo runs entirely on the Pi: previously the Pi computed `gpu_detect`/`serve`
while eyes-ears-mouth lived on the Mac; here the same voice loop drives them
directly on the robot's Pi (the HTTP client/server still exist, just
co-located on 127.0.0.1).

Confirmed SDK API (`reachy_mini.media.media_manager.MediaManager`):
`handle.media.get_frame() -> np.uint8[H,W,3] | None`,
`handle.media.get_audio_sample() -> np.float32[N] | None`,
`handle.media.play_sound(path)`. Physical movement uses the same `handle`
(`handle.goto_target(...)`), just like `demo.robot_reachy.ReachyRobot`, only
on an ALREADY existing handle rather than a new one.

`reachy_mini` is a heavy hardware dependency, imported lazily and ONLY
here (see demo/platform/__init__.py::build_platform). The live hardware
is verified in a separate phase on the Pi — here the class is structurally
sound and covered by tests against a fake handle (tests/test_platform_reachy.py).
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
import wave

import numpy as np

from demo.platform.pcm import frame_to_png
from demo.robot_reachy import GESTURE_POSES, gaze_to_head_pose
from demo.vad import rms

LOG = logging.getLogger(__name__)

# The robot camera is "stale" (alive == False) if the SDK has delivered no
# fresh frame for this long — parity with CameraStream.alive on the Mac.
_STALE_AFTER_S = 5.0


def _write_wav(wave_data: np.ndarray, sample_rate: int) -> str:
    # mkstemp (not the deprecated mktemp): mktemp returns a name with nothing
    # holding it, a predictable-name TOCTOU race; mkstemp creates the file
    # atomically. We only need the path (wave.open reopens it), so close the fd.
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        pcm = (np.clip(wave_data, -1, 1) * 32767).astype(np.int16)
        with wave.open(path, "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(sample_rate)
            fh.writeframes(pcm.tobytes())
    except Exception:
        # Don't leak the temp file if the write fails (e.g. disk full): the
        # caller only removes it after a successful return.
        try:
            os.remove(path)
        except OSError:
            pass
        raise
    return path


def _play_via_handle(handle, audio: np.ndarray, sample_rate: int) -> None:
    """Write a WAV and play it through the robot's speaker (`media.play_sound`).

    The temp WAV is removed after playback (in `finally`, even if play_sound
    raises) — otherwise one file leaks per reply, unbounded disk growth on the
    robot's Pi over a long demo. This mirrors the Mac path (audio_out.play_wav)
    and assumes play_sound BLOCKS until playback finishes, as its name and the
    Mac afplay counterpart imply; confirmed on the real SDK in the Pi hardware
    phase (if it turns out async, cleanup must be deferred instead).
    """
    path = _write_wav(audio, sample_rate)
    try:
        handle.media.play_sound(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


class _ReachyVideoSource:
    """Robot camera frame — a thin wrapper over `handle.media.get_frame()`.

    Unlike `CameraStream` (Mac), there's no need to poll on a background
    thread: the SDK's `get_frame()` already returns the latest available
    frame from the daemon's IPC queue, so `latest()` just asks it directly
    and caches the last non-empty response, in case a frame hasn't landed
    yet by the time we poll.
    """

    def __init__(self, handle) -> None:
        self._handle = handle
        self._latest: np.ndarray | None = None
        self._lock = threading.Lock()
        self._last_fresh: float | None = None

    @property
    def alive(self) -> bool:
        # Parity with CameraStream.alive (Mac) so run_voice's camera-health
        # print works on the robot too: a frozen SDK camera daemon would
        # otherwise feed the last frame forever and the robot would act on a
        # stale scene with confident phantom detections. True until the SDK has
        # delivered no fresh frame for _STALE_AFTER_S.
        with self._lock:
            if self._last_fresh is None:
                return True  # not started yet — absence of a frame isn't death
            return (time.monotonic() - self._last_fresh) < _STALE_AFTER_S

    def latest(self) -> np.ndarray | None:
        # Under `--display web` two threads read the same source: the detect
        # loop and the MJPEG handler. Serialize the SDK call and return a COPY
        # (like CameraStream on the Mac) so one thread can't hand a frame
        # reference to frame_to_jpeg while the other drives the SDK to produce
        # the next one.
        with self._lock:
            frame = self._handle.media.get_frame()
            if frame is not None:
                # Copy at store time too: if the SDK recycles frame buffers in
                # place, keeping its reference would let a later get_frame()
                # mutate our cached frame.
                self._latest = np.asarray(frame).copy()
                self._last_fresh = time.monotonic()
            return None if self._latest is None else self._latest.copy()

    def latest_png(self) -> bytes | None:
        frame = self.latest()
        return frame_to_png(frame) if frame is not None else None

    def close(self) -> None:
        pass  # the shared handle is released by ReachyPlatform.close()


class _ReachyMicSource:
    """Robot microphone — the same chunks (float32, rms) as `MicStream`.

    `get_audio_sample()` returns whatever the audio backend has accumulated
    since the last poll (variable length, can be empty) — we accumulate it
    into a buffer and slice it into fixed-size `chunk_ms` chunks, exactly
    like MicStream slices the ffmpeg stream on the Mac. An empty poll gets a
    short pause before the next one (otherwise the loop would busy-wait and
    burn CPU).
    """

    def __init__(self, handle, sample_rate: int = 16000, chunk_ms: int = 100,
                 poll_interval: float = 0.02) -> None:
        self._handle = handle
        self.sample_rate = sample_rate
        self._chunk_samples = int(sample_rate * chunk_ms / 1000)
        self._poll_interval = poll_interval
        self._buffer = np.zeros(0, dtype=np.float32)

    def chunks(self):
        """Generator of (chunk_float32, rms) pairs — infinite, like MicStream's."""
        while True:
            sample = self._handle.media.get_audio_sample()
            if sample is not None and len(sample) > 0:
                self._buffer = np.concatenate(
                    [self._buffer, np.asarray(sample, dtype=np.float32)])
            while len(self._buffer) >= self._chunk_samples:
                chunk = self._buffer[:self._chunk_samples]
                self._buffer = self._buffer[self._chunk_samples:]
                yield chunk, rms(chunk)
            if sample is None or len(sample) == 0:
                time.sleep(self._poll_interval)

    def flush(self) -> None:
        """Discard the buffer and anything the audio backend has already accumulated (the robot's own voice)."""
        self._buffer = np.zeros(0, dtype=np.float32)
        # Bounded drain: unlike MicStream.flush's non-blocking BlockingIOError
        # exit, this depends on the SDK queue emptying, so cap the loop — audio
        # arriving as fast as we drain it must not spin here forever.
        for _ in range(1000):
            sample = self._handle.media.get_audio_sample()
            if sample is None or len(sample) == 0:
                break

    def close(self) -> None:
        pass  # the shared handle is released by ReachyPlatform.close()


class _PhysicalRobot:
    """Robot on a physical Reachy Mini — the same handle that provides media.

    Mirrors `demo.robot_reachy.ReachyRobot`, but doesn't create its own
    handle — it moves the one that already exists (shared with the
    platform's video/mic/player). `look_at`/`gesture` use the same pure
    `gaze_to_head_pose`/`GESTURE_POSES` as the Mac path — gestures and turns
    are identical on both platforms; only the transport differs
    (`goto_target` on the same handle).
    """

    def __init__(self, handle) -> None:
        self._handle = handle

    def look_at(self, x: float, y: float) -> None:
        from reachy_mini.utils import create_head_pose

        pose = gaze_to_head_pose(x, y)
        self._handle.goto_target(
            head=create_head_pose(yaw=pose["yaw"], pitch=pose["pitch"],
                                  degrees=True),
            duration=0.5)

    def say(self, wave_data: np.ndarray, sample_rate: int) -> None:
        if len(wave_data) == 0:
            return
        _play_via_handle(self._handle, wave_data, sample_rate)

    def gesture(self, name: str) -> None:
        from reachy_mini.utils import create_head_pose

        poses = GESTURE_POSES.get(name)
        if poses is None:
            return
        for yaw, pitch in poses:
            self._handle.goto_target(
                head=create_head_pose(yaw=yaw, pitch=pitch, degrees=True),
                duration=0.25)


class ReachyPlatform:
    """Platform backend that's "its own Pi": one `ReachyMini` handle for
    eyes, ears, mouth, and motors. `handle` can be injected (tests pass a
    fake stub); by default it creates a real `ReachyMini()`. (Note:
    `reachy_mini` is also pulled in on the production path by
    `demo/robot_reachy.py` — `ReachyRobot` for MuJoCo-sim on the Mac; this
    is a second, separate client to the same SDK.)
    """

    def __init__(self, handle=None) -> None:
        if handle is None:
            from reachy_mini import ReachyMini

            handle = ReachyMini()
        self._handle = handle
        self._video: _ReachyVideoSource | None = None
        self._mic: _ReachyMicSource | None = None

    def video_source(self) -> _ReachyVideoSource:
        if self._video is None:
            self._video = _ReachyVideoSource(self._handle)
        return self._video

    def mic_source(self) -> _ReachyMicSource:
        if self._mic is None:
            self._mic = _ReachyMicSource(self._handle)
        return self._mic

    def make_player(self, sample_rate: int):
        from demo.audio_out import StreamPlayer

        return StreamPlayer(
            sample_rate,
            play=lambda audio, sr: _play_via_handle(self._handle, audio, sr))

    def robot(self) -> _PhysicalRobot:
        return _PhysicalRobot(self._handle)

    def close(self) -> None:
        # Release media (camera/mic/speaker) AND the robot handle itself. The
        # same ReachyMini handle also drives the motors, so closing only media
        # would leak the motor/robot connection past shutdown (the sub-sources'
        # close() are no-ops precisely because teardown belongs here). The
        # exact handle-teardown method isn't pinned in the SDK we target, so
        # try the unambiguous ones defensively — verified against the real SDK
        # in the Pi hardware phase.
        media = getattr(self._handle, "media", None)
        media_close = getattr(media, "close", None)
        if media_close is not None:
            try:
                media_close()
            except Exception:  # noqa: BLE001 — one failing teardown must not skip the other
                LOG.exception("ReachyPlatform: media.close() failed")
        for name in ("close", "disconnect"):
            fn = getattr(self._handle, name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:  # noqa: BLE001 — release best-effort; log and move on
                    LOG.exception("ReachyPlatform: handle.%s() failed", name)
                break
