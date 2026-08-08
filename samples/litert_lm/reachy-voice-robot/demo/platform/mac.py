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

"""MacPlatform: Mac camera/mic capture via ffmpeg, afplay, robot.

The I/O device for the voice loop on the Mac: ffmpeg-backed camera and mic
capture, afplay for audio, and the robot handle. Pure transforms (device
parser, PCM converter, frame assembly, PNG) live in demo/platform/pcm.py.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time

import numpy as np

from demo.platform.pcm import (
    frame_to_png,
    parse_avfoundation_devices,
    pcm_bytes_to_float,
    raw_to_frame,
)
from demo.vad import rms

LOG = logging.getLogger(__name__)


def list_devices(timeout: float = 5.0) -> dict:
    proc = subprocess.run(
        ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True, text=True, timeout=timeout)
    return parse_avfoundation_devices(proc.stderr)


def grab_frame(video_index: str = "0", warmup_frames: int = 12,
               timeout: float = 10.0) -> bytes:
    """Grab a frame from the camera, discarding dark warm-up frames.

    The first frame from the camera on macOS is nearly black: avfoundation
    delivers it before auto-exposure kicks in, before the sensor has "woken
    up". A single `-frames:v 1` grabs exactly that black frame, and the
    detector hallucinates objects on it. So we read warmup_frames frames
    (~0.4s at 30 fps) and keep only the last one, which is properly exposed
    by then.

    `timeout` protects the caller (`--enter`, a single snapshot) from a
    hung ffmpeg — without it the process could hang forever if the camera
    is busy.
    """
    n = max(1, warmup_frames)
    proc = subprocess.run(
        ["ffmpeg", "-y", "-f", "avfoundation", "-framerate", "30",
         "-video_size", "1280x720", "-i", video_index,
         "-frames:v", str(n), "-vf", f"select='eq(n\\,{n - 1})'",
         "-vsync", "0", "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True, timeout=timeout)
    if not proc.stdout:
        raise RuntimeError(f"ffmpeg returned no frame: {proc.stderr[-300:]!r}")
    return proc.stdout


def record_audio(seconds: float, audio_index: str = "1",
                 sample_rate: int = 16000, timeout: float | None = None) -> np.ndarray:
    """Record `seconds` seconds from the mic in a single ffmpeg call (`--enter`).

    `timeout` defaults to `seconds` plus a margin for ffmpeg startup/shutdown,
    so a hung process doesn't block the caller forever.
    """
    if timeout is None:
        timeout = seconds + 10.0
    proc = subprocess.run(
        ["ffmpeg", "-y", "-f", "avfoundation", "-i", f":{audio_index}",
         "-t", str(seconds), "-ar", str(sample_rate), "-ac", "1",
         "-f", "s16le", "-"],
        capture_output=True, timeout=timeout)
    if proc.returncode != 0 or not proc.stdout:
        # Mirror grab_frame: in --enter debug mode a crash with the ffmpeg
        # error is more useful than silently returning empty audio — an empty
        # buffer is indistinguishable from silence, so a wrong --audio index
        # or a busy device would look like "you said nothing" instead of a
        # loud, diagnosable failure.
        raise RuntimeError(
            f"ffmpeg captured no audio (exit {proc.returncode}): "
            f"{proc.stderr[-300:]!r}")
    return pcm_bytes_to_float(proc.stdout)


class CameraStream:
    """Continuous stream of camera frames — the camera is always "looking".

    Like MicStream for audio: one ffmpeg process holds the camera, a
    background thread reads frames and keeps the latest one. This solves
    both the black first-frame problem (the camera is always warmed up) and
    device contention (a single holder), and the robot sees continuously
    instead of one snapshot at a time.

    Frames come in as raw rgb24 at a fixed size — so every frame is exactly
    width*height*3 bytes, and the stream is trivial to slice into frames
    without parsing JPEG.

    `alive`/`dead_since`: if ffmpeg dies (or the pipe closes), the reader
    thread exits — without this flag, `latest()` would keep silently
    returning the last saved frame FOREVER, and calling code couldn't tell
    "the camera is alive but the scene isn't changing" from "the camera
    dropped out". `alive` is set to False at exactly the moment the thread
    detects the source has died.
    """

    def __init__(self, video_index: str = "0", width: int = 640,
                 height: int = 480, fps: int = 30) -> None:
        # fps must EXACTLY match the camera's mode. The MacBook offers 15 and
        # 30, but ffmpeg is finicky about floats: "15.0 not supported", and
        # without -framerate it picks 29.97 and also fails. Exactly 30 is the
        # only value that opens.
        self.width = width
        self.height = height
        self._frame_bytes = width * height * 3
        self._latest: np.ndarray | None = None
        self._lock = threading.Lock()
        self.alive = True
        self.dead_since: float | None = None
        self._proc = subprocess.Popen(
            ["ffmpeg", "-f", "avfoundation", "-framerate", str(fps),
             "-video_size", f"{width}x{height}", "-i", video_index,
             "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_exact(self, n: int) -> bytes:
        chunks = []
        got = 0
        while got < n:
            part = self._proc.stdout.read(n - got)
            if not part:
                return b""
            chunks.append(part)
            got += len(part)
        return b"".join(chunks)

    def _read_loop(self) -> None:
        while True:
            raw = self._read_exact(self._frame_bytes)
            frame = raw_to_frame(raw, self.width, self.height)
            if frame is None:
                # ffmpeg died or the pipe closed — no more frames coming.
                # Mark the stream dead so latest() doesn't silently keep
                # handing out a stale frame as if it were live, forever.
                self.alive = False
                self.dead_since = time.monotonic()
                LOG.warning(
                    "CameraStream: reader thread exiting (ffmpeg died or "
                    "pipe closed) — camera marked dead")
                return
            with self._lock:
                self._latest = frame

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def latest_png(self) -> bytes | None:
        frame = self.latest()
        return frame_to_png(frame) if frame is not None else None

    def close(self) -> None:
        self._proc.terminate()
        try:
            self._proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()  # reap the child — avoid a zombie process
        self._thread.join(timeout=2)
        if self._proc.stdout is not None:
            self._proc.stdout.close()


class MicStream:
    """Continuous PCM stream from the mic via ffmpeg.

    Reads in chunk_ms-sized pieces. `flush` discards whatever has accumulated
    (after the robot finishes speaking, so it doesn't hear its own voice).
    """

    def __init__(self, audio_index: str = "1", sample_rate: int = 16000,
                 chunk_ms: int = 100) -> None:
        self.sample_rate = sample_rate
        self._chunk_bytes = int(sample_rate * chunk_ms / 1000) * 2
        self._proc = subprocess.Popen(
            ["ffmpeg", "-f", "avfoundation", "-i", f":{audio_index}",
             "-ar", str(sample_rate), "-ac", "1", "-f", "s16le", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def chunks(self):
        """Generator of (chunk_float32, rms) pairs while the stream is alive.

        An empty read means the stream ended — but for one of two different
        reasons: either the ffmpeg process died on its own (`poll()` is no
        longer None — a real failure, so we log it), or the stream is being
        closed normally from outside. Without the log, both cases would look
        identically silent.
        """
        while True:
            raw = self._proc.stdout.read(self._chunk_bytes)
            if not raw:
                if self._proc.poll() is not None:
                    LOG.warning(
                        "MicStream: ffmpeg process died (exit code %s) — "
                        "mic is dead, not an intentional stop",
                        self._proc.poll())
                return
            chunk = pcm_bytes_to_float(raw)
            yield chunk, rms(chunk)

    def flush(self) -> None:
        """Discard whatever has accumulated in the buffer (e.g. the robot's voice)."""
        fd = self._proc.stdout.fileno()
        os.set_blocking(fd, False)
        try:
            while True:
                data = self._proc.stdout.read(self._chunk_bytes)
                if not data:
                    break
        except BlockingIOError:
            pass
        finally:
            os.set_blocking(fd, True)

    def close(self) -> None:
        self._proc.terminate()
        try:
            self._proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()  # reap the child — avoid a zombie process
        if self._proc.stdout is not None:
            self._proc.stdout.close()  # don't leak the pipe fd (CameraStream does this too)


class MacPlatform:
    """Platform backend for the Mac: ffmpeg camera/mic, afplay, MuJoCo-sim
    or console robot (`--no-robot`). Behavior is preserved as it was in
    run_voice before the platform was pulled out behind a protocol — plus
    added failure observability (camera alive flag, ffmpeg death logging,
    timeout on one-shot calls); this is an addition, not a change to
    happy-path behavior.
    """

    def __init__(self, args) -> None:
        self.args = args
        self._camera: CameraStream | None = None
        self._mic: MicStream | None = None

    def video_source(self) -> CameraStream:
        if self._camera is None:
            self._camera = CameraStream(self.args.video)
        return self._camera

    def mic_source(self) -> MicStream:
        if self._mic is None:
            self._mic = MicStream(self.args.audio, chunk_ms=100)
        return self._mic

    def make_player(self, sample_rate: int):
        # Import and pass _afplay explicitly (rather than relying on
        # StreamPlayer's default): the default binds _afplay at import time,
        # whereas this re-fetches it at call time so a test can monkeypatch it.
        from demo.audio_out import StreamPlayer, _afplay

        return StreamPlayer(sample_rate, play=_afplay)

    def robot(self):
        from demo.robot_reachy import make_robot

        return make_robot(use_robot=not self.args.no_robot)

    def close(self) -> None:
        if self._mic is not None:
            self._mic.close()
        if self._camera is not None:
            self._camera.close()
