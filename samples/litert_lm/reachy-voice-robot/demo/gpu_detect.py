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

"""GPU detection service on the Pi: JPEG frame -> boxes. Isolated from serve.py.

Runs as a separate process in the nightly-LiteRT venv with a GPU env
(VK_ICD_FILENAMES, V3D_WEBGPU_OVERRIDE). The point is to offload the CPU:
detection moves to the VideoCore VII, freeing the CPU cores for the LLM.
Latency is ~385 ms/frame on the Pi's V3D GPU — several times the ~111 ms 4-thread
CPU path, but the win isn't raw speed: it keeps detection off the CPU cores so Gemma
gets them. IMPORTANT: raw GPU inference alone doesn't produce boxes — it still
needs the same postprocessing as the CPU detector (decode + NMS), plus the same
normalized NCHW input. The model is exported with the raw head (end2end=False)
so it runs fully on the GPU; the decode + NMS run here on the CPU.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from ai_edge_litert.compiled_model import CompiledModel, HardwareAccelerator
from PIL import Image

from demo.detections import detections_to_dicts
from demo.http_util import ascii_reason
from emulator import models
from emulator.detector import (decode_yolo_output, non_max_suppression,
                               preprocess_frame)

# decompression-bomb guard: PIL raises DecompressionBombError above ~2x this
# (~20 Mpx); demo frames are <1 Mpx, so a huge PNG is rejected before it can
# balloon in memory.
Image.MAX_IMAGE_PIXELS = 10_000_000

LOG = logging.getLogger(__name__)

INPUT = 640          # yolo26n input; output 8400 = 80²+40²+20² anchors
OUTPUT_SHAPE = (1, 84, 8400)   # 84 = 4 box coords + 80 COCO class scores (raw head)
MAX_BODY = 4 * 1024 * 1024  # overflow guard: /detect request body size limit


class GpuDetector:
    def __init__(self, model_path: str, score_threshold: float = 0.3,
                 iou_threshold: float = 0.45) -> None:
        self._m = CompiledModel.from_file(
            model_path, hardware_accel=HardwareAccelerator.GPU)
        self._ib = self._m.create_input_buffers(0)
        self._ob = self._m.create_output_buffers(0)
        # Fail loudly at boot if the model's real output shape isn't the
        # channels-first raw head we decode. `read()` below imposes OUTPUT_SHAPE
        # on the flat buffer, so a mismatched (e.g. anchor-major) export would
        # otherwise be silently reinterpreted into garbage, per frame.
        actual = tuple(int(x) for x in self._ob[0].get_tensor_details()["shape"])
        if actual != OUTPUT_SHAPE:
            raise SystemExit(
                f"detector output shape {actual} != expected {OUTPUT_SHAPE} — "
                "re-check the YOLO26 end2end=False export (assets/yolo/README.md)")
        self._thr = score_threshold
        self._iou = iou_threshold

    def detect(self, jpeg: bytes) -> list[dict]:
        img = Image.open(io.BytesIO(jpeg)).convert("RGB").resize((INPUT, INPUT))
        frame = np.asarray(img, dtype=np.float32)          # [640,640,3] 0..255
        prepared = preprocess_frame(frame, (1, 3, INPUT, INPUT))  # NCHW, [0,1]
        self._ib[0].write(prepared.astype(np.float32))
        self._m.run_by_index(0, self._ib, self._ob)
        raw = self._ob[0].read(OUTPUT_SHAPE, np.float32)
        found = decode_yolo_output(raw, self._thr)
        return detections_to_dicts(non_max_suppression(found, self._iou))


def make_handler(detector: GpuDetector):
    class Handler(BaseHTTPRequestHandler):
        # Single-threaded HTTPServer (one worker) — the timeout stops a
        # hung/slow client from permanently hogging the only socket.
        timeout = 30

        def do_POST(self):
            if self.path != "/detect":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError as exc:
                # A present-but-non-numeric header (Content-Length: abc) would
                # otherwise raise out of do_POST — an unhandled traceback to
                # stderr and a dropped connection instead of a clean 400.
                LOG.warning("bad Content-Length: %s: %s",
                            type(exc).__name__, exc)
                self.send_error(400, "bad Content-Length")
                return
            if length <= 0 or length > MAX_BODY:
                # Overflow guard: body out of bounds (non-ASCII in the HTTP
                # status line's reason phrase breaks http.server's latin-1
                # encoder — the message itself must be ASCII, comments needn't be)
                self.send_error(413, "request body out of bounds")
                return
            jpeg = self.rfile.read(length)
            try:
                t0 = time.perf_counter()
                dets = detector.detect(jpeg)
                ms = (time.perf_counter() - t0) * 1000
            except Exception as exc:  # noqa: BLE001 — the service must stay up
                # Broad boundary → log WITH the stack (exc_info) so an
                # unexpected bug is localizable, not just a one-line type+message.
                LOG.exception("detect failed: %s", type(exc).__name__)
                # Raw exception text in the 500 body is handy for debugging the
                # demo, not for a prod API (could leak an internal path/trace).
                # ascii_reason: the reason phrase is latin-1-encoded by
                # http.server, so a non-ASCII char in the message must not crash send_error.
                self.send_error(500, ascii_reason(str(exc)))
                return
            body = json.dumps({"detections": dets, "ms": ms}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_request(self, code="-", size="-"):
            # Stay quiet on 2xx (normal demo traffic); 4xx/5xx go to the
            # standard log (log_error -> log_message), a backstop on top of LOG.error above.
            if isinstance(code, int) and 200 <= code < 300:
                return
            super().log_request(code, size)

    return Handler


# Resolve the bundled detector through the catalog (emulator/models.py) — the
# one place that names the model file, so nothing is duplicated here. gpu_detect
# runs as `python -m demo.gpu_detect` from the repo, so the path is on disk.
_detector = models.get(models.DETECTOR)
_MODEL = (_detector.dir / _detector.file
          if _detector.dir is not None and _detector.file is not None else None)


def find_model() -> str:
    if _MODEL is None:
        raise SystemExit(
            f"detector {models.DETECTOR!r} is not a bundled (dir+file) model — "
            "gpu_detect serves a local .tflite")
    if not _MODEL.exists():
        raise SystemExit(
            f"bundled YOLO model not found: {_MODEL} — convert it per "
            "assets/yolo/README.md before running")
    return str(_MODEL)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="GPU detection service (Pi)")
    # Listens on localhost by default; the Mac reaches it via an SSH tunnel
    # (same as serve.py); for direct LAN access, pass --host 0.0.0.0 explicitly
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9600)
    p.add_argument("--model", default=None)
    args = p.parse_args(argv)
    detector = GpuDetector(args.model or find_model())
    server = HTTPServer((args.host, args.port), make_handler(detector))
    print(f"gpu_detect on {args.host}:{args.port}, model loaded (GPU)")
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"  WARNING: bound to non-loopback host {args.host!r} — this "
              f"unauthenticated service is reachable from the network.")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
