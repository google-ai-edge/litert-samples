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

"""Web dashboard on the Mac: live video + detections + query frame + reply stream.

The presentation half of the demo's live dashboard: serves the MJPEG camera
feed, SSE events, and dashboard.html. Knows nothing about where detections come from —
`on_detections` is just a listener subscribed to GpuDetectSource. Boxes are
drawn by the BROWSER on a canvas over the video — the Mac spends no CPU on
rendering. A single viewer-presenter.
"""
from __future__ import annotations

import base64
import logging
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from demo.detect_source import frame_to_jpeg
from demo.display.events import sse
from demo.http_util import ascii_reason

LOG = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"

# SSE subscriber queues are bounded: if the browser dropped off but events
# keep arriving, we don't buffer forever — we drop the oldest instead
# (drop-oldest), otherwise a slow/dead client's queue would grow unbounded.
SUB_QUEUE_MAXSIZE = 64


class WebDashboard:
    def __init__(self, camera) -> None:
        self.camera = camera
        self._subs: list[queue.Queue] = []
        self._subs_lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None

    def serve(self, host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
        self._server = ThreadingHTTPServer((host, port), make_handler(self))
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self._server

    # — DisplaySink —
    def on_detections(self, detections: list[dict]) -> None:
        self._broadcast(sse("detections", {"boxes": detections}))

    def on_query(self, frame: np.ndarray | None, detections: list[dict]) -> None:
        jpeg_b64 = (base64.b64encode(frame_to_jpeg(frame)).decode()
                    if frame is not None else "")
        self._broadcast(sse("query_frame",
                            {"jpeg_b64": jpeg_b64, "boxes": detections}))

    def on_heard(self, text: str) -> None:
        self._broadcast(sse("heard", {"text": text}))

    def on_reply(self, text: str, done: bool = False) -> None:
        self._broadcast(sse("reply", {"text": text, "done": done}))

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    # — SSE subscriptions —
    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=SUB_QUEUE_MAXSIZE)
        with self._subs_lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._subs_lock:
            if q in self._subs:
                self._subs.remove(q)

    def _broadcast(self, msg: str) -> None:
        with self._subs_lock:
            for q in list(self._subs):
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    # slow subscriber — drop the oldest event and push the
                    # fresh one, don't block the detect loop on its queue
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        q.put_nowait(msg)
                    except queue.Full:
                        pass


def make_handler(dashboard: WebDashboard):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            # Same principle as serve.py/gpu_detect.py: one bad request or
            # missing asset shouldn't crash the presentation server with a
            # traceback to stderr — return a clean 404/500 instead.
            try:
                if self.path == "/" or self.path == "/index.html":
                    self._page()
                elif self.path == "/live.mjpeg":
                    self._mjpeg()
                elif self.path == "/events":
                    self._events()
                else:
                    self.send_error(404)
            except (BrokenPipeError, ConnectionResetError):
                pass  # client dropped off — not a server error (backstop:
                      # _mjpeg/_events already handle this inside their own loops)
            except FileNotFoundError as exc:
                # E.g. dashboard.html missing from disk — that's an
                # environment/deploy fault, not the client's, but still a
                # specific, legible 404 rather than a bare traceback.
                LOG.warning("static asset missing: %s: %s",
                           type(exc).__name__, exc)
                try:
                    self.send_error(404, "asset not found")
                except Exception:
                    pass
            except Exception as exc:  # noqa: BLE001 — presentation must not crash
                # Broad boundary → log WITH the stack so an unexpected bug is
                # localizable, not just a one-line type+message.
                LOG.exception("unhandled exception: %s", type(exc).__name__)
                try:
                    # Raw exception text in the 500 body — fine for a demo only;
                    # ascii_reason so a non-ASCII message can't crash send_error.
                    self.send_error(500, ascii_reason(str(exc)))
                except Exception:
                    pass  # headers may already be gone (inside _mjpeg/_events)

        def _page(self):
            html = (STATIC / "dashboard.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)

        def _mjpeg(self):
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    frame = dashboard.camera.latest()
                    if frame is None:
                        time.sleep(0.05)
                        continue
                    jpeg = frame_to_jpeg(frame)
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n"
                                     .encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    time.sleep(1 / 20)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            q = dashboard.subscribe()
            try:
                while True:
                    self.wfile.write(q.get().encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                dashboard.unsubscribe(q)

        def log_request(self, code="-", size="-"):
            # Stay quiet on ordinary successful responses (2xx: page, MJPEG,
            # SSE connection); 4xx/5xx go to the standard log, a backstop on
            # top of the targeted LOG.warning/error calls above.
            if isinstance(code, int) and 200 <= code < 300:
                return
            super().log_request(code, size)

    return Handler
