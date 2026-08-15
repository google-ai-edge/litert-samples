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

"""Detection source: camera frame -> GPU detect service on the Pi -> boxes.

The screenless core: sends camera frames to the Pi's gpu_detect service
(~4 fps, only boxes come back) and holds a snapshot (latest frame +
detections) behind a lock. It knows nothing about the browser, SSE, or HTTP
serving — those responsibilities moved to demo/display/. The voice loop
always reads detections from here, regardless of whether a display is
attached: `WebDashboard.on_detections` is just one of the listeners
(`add_listener`), and `NullSink.on_detections` is a silent one.
"""
from __future__ import annotations

import io
import json
import logging
import threading
import urllib.request
from typing import Callable

import numpy as np
from PIL import Image

LOG = logging.getLogger(__name__)

# After how many consecutive failed POSTs to gpu_detect the source is
# considered unhealthy (healthy() == False). A single glitch (a brief network
# blip on the Pi) shouldn't trip the signal — what matters is a sustained run
# of failures.
MAX_CONSECUTIVE_FAILURES = 5


def frame_to_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class GpuDetectSource:
    """Sends camera frames to the GPU detect service, holds the latest snapshot.

    `latest()` always returns a list (even after an error — an empty one), so
    consumers (the voice loop, the display) never have to handle None. The
    distinction between "error" (None) and "empty scene" ([]) is tracked only
    internally, for `healthy()`.
    """

    def __init__(self, camera, detect_url: str, fps: int = 4) -> None:
        self.camera = camera
        self.detect_url = detect_url
        self._interval = 1.0 / fps
        self._latest_frame: np.ndarray | None = None
        self._latest_dets: list[dict] = []
        self._lock = threading.Lock()
        self._listeners: list[Callable[[list[dict]], None]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._consecutive_failures = 0

    # — detection —
    def start(self) -> None:
        self._thread = threading.Thread(target=self._detect_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            # Longer than _post_detect's urlopen(timeout=5) so an in-flight
            # POST finishes and the thread is actually reaped, rather than
            # stop() returning while a straggling cycle is still running.
            self._thread.join(timeout=6)

    def latest(self) -> tuple[np.ndarray | None, list[dict]]:
        with self._lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
            return frame, list(self._latest_dets)

    def add_listener(self, fn: Callable[[list[dict]], None]) -> None:
        """fn(detections) is called on every detect cycle with fresh boxes."""
        self._listeners.append(fn)

    def healthy(self) -> bool:
        return self._consecutive_failures < MAX_CONSECUTIVE_FAILURES

    def _post_detect(self, frame: np.ndarray) -> list[dict] | None:
        """POST a frame to gpu_detect. None means a failure (Pi/GPU
        unreachable), [] means the service responded but found nothing
        (empty scene)."""
        try:
            req = urllib.request.Request(
                self.detect_url, data=frame_to_jpeg(frame),
                headers={"Content-Type": "image/jpeg"})
            resp = json.loads(urllib.request.urlopen(req, timeout=5).read())
            dets = resp.get("detections")
            if dets is None:
                # A 200 whose body has no "detections" is contract drift (a wrong
                # endpoint answering, an error-shaped 200) — not an empty scene.
                # Treat it as a failure so it counts toward unhealthy(), rather
                # than masquerading as "the robot sees nothing".
                LOG.warning("gpu_detect 200 response has no 'detections' key")
                return None
            return dets
        except Exception as exc:  # noqa: BLE001 — the GPU may drop out, don't kill the loop
            LOG.warning("gpu_detect POST failed: %s: %s", type(exc).__name__, exc)
            return None

    def _cycle(self) -> None:
        """One loop step: frame -> POST -> snapshot -> listeners.

        Pulled out of `_detect_loop` so tests can drive the steps
        deterministically, without threads or sleeps.
        """
        frame = self.camera.latest()
        if frame is None:
            return
        dets = self._post_detect(frame)
        if dets is None:
            self._consecutive_failures += 1
            dets_out: list[dict] = []
        else:
            self._consecutive_failures = 0
            dets_out = dets
        with self._lock:
            self._latest_frame = frame
            self._latest_dets = dets_out
        for fn in self._listeners:
            try:
                fn(dets_out)
            except Exception as exc:  # noqa: BLE001 — a bad listener must not kill the loop
                LOG.warning("detect listener failed: %s: %s",
                            type(exc).__name__, exc)

    def _detect_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._cycle()
            except Exception as exc:  # noqa: BLE001 — one bad cycle must not kill the loop
                # Log WITH the stack: unlike a POST failure (an expected outage
                # that counts toward healthy()), an exception here is an
                # unexpected bug and needs a traceback to localize.
                LOG.exception("detect cycle failed: %s", type(exc).__name__)
            # wait() (not sleep) so stop() wakes the loop immediately between
            # cycles instead of after a full interval.
            self._stop.wait(self._interval)
