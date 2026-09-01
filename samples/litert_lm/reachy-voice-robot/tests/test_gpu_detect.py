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

"""Tests for the pure/hardware-free parts of demo/gpu_detect.py:
find_model() (glob logic) and the HTTP handler via a fake detector (413,
404, 500 branches) — a real GpuDetector needs a GPU and a tflite model, so
make_handler() runs here with a stand-in rather than a real detector (
make_handler itself never checks the object's type, it just calls .detect()).
"""
from __future__ import annotations

import http.client
import json
import threading
from http.server import HTTPServer

import pytest

from demo.gpu_detect import MAX_BODY, find_model, make_handler


# --- find_model(): resolves the model bundled in the repo (assets/yolo) ---

def test_find_model_returns_bundled_model(monkeypatch, tmp_path):
    # Hermetic: the real yolo26n.tflite is git-ignored (converted locally), so
    # don't depend on it being present — point _MODEL at a temp file instead.
    fake = tmp_path / "yolo26n.tflite"
    fake.write_bytes(b"x")
    monkeypatch.setattr("demo.gpu_detect._MODEL", fake)
    assert find_model() == str(fake)


def test_find_model_raises_systemexit_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("demo.gpu_detect._MODEL", tmp_path / "nope.tflite")
    with pytest.raises(SystemExit):
        find_model()


# --- HTTP handler: a fake detector standing in for a real GpuDetector ---

class FakeDetector:
    """Stand-in for GpuDetector — make_handler doesn't check the type, only calls .detect()."""

    def __init__(self, dets=None):
        self._dets = dets if dets is not None else [
            {"label": "cup", "score": 0.8, "box": [0.1, 0.1, 0.3, 0.3]}]

    def detect(self, jpeg: bytes) -> list[dict]:
        return self._dets


class BrokenDetector:
    """Simulates an internal inference failure — checks the mapping to 500."""

    def detect(self, jpeg: bytes) -> list[dict]:
        raise RuntimeError("model exploded")


class _RunningServer:
    """Runs a single-threaded HTTPServer (like gpu_detect.main()) on a
    background thread — tested via real HTTP requests rather than a
    hand-assembled Handler."""

    def __init__(self, detector):
        self.server = HTTPServer(("127.0.0.1", 0), make_handler(detector))
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    @property
    def address(self):
        return self.server.server_address

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def running_server():
    srv = _RunningServer(FakeDetector())
    try:
        yield srv
    finally:
        srv.close()


def test_post_to_unknown_path_returns_404(running_server):
    host, port = running_server.address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    conn.request("POST", "/wrong", body=b"junk")
    resp = conn.getresponse()
    assert resp.status == 404
    resp.read()
    conn.close()


def test_zero_length_body_returns_413(running_server):
    host, port = running_server.address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    conn.request("POST", "/detect", body=b"")
    resp = conn.getresponse()
    assert resp.status == 413
    resp.read()
    conn.close()


def test_oversized_body_returns_413(running_server):
    host, port = running_server.address
    conn = http.client.HTTPConnection(host, port, timeout=10)
    # Content-Length lies about the body size — the server rejects based on
    # the header BEFORE reading the body, so we don't actually need to send MAX_BODY+1 bytes.
    conn.request("POST", "/detect", body=b"",
                headers={"Content-Length": str(MAX_BODY + 1)})
    resp = conn.getresponse()
    assert resp.status == 413
    resp.read()
    conn.close()


def test_valid_jpeg_body_returns_detections(running_server):
    host, port = running_server.address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    jpeg = b"\xff\xd8\xff\xe0fake-jpeg-bytes"
    conn.request("POST", "/detect", body=jpeg,
                 headers={"Content-Type": "image/jpeg"})
    resp = conn.getresponse()
    assert resp.status == 200
    body = json.loads(resp.read())
    assert body["detections"] == [
        {"label": "cup", "score": 0.8, "box": [0.1, 0.1, 0.3, 0.3]}]
    assert "ms" in body
    conn.close()


def test_detector_exception_returns_500_and_is_logged(caplog):
    srv = _RunningServer(BrokenDetector())
    try:
        host, port = srv.address
        conn = http.client.HTTPConnection(host, port, timeout=5)
        with caplog.at_level("ERROR", logger="demo.gpu_detect"):
            conn.request("POST", "/detect", body=b"junk-jpeg-bytes")
            resp = conn.getresponse()
            resp.read()
        conn.close()
        assert resp.status == 500
        assert any("RuntimeError" in record.message
                   for record in caplog.records)
    finally:
        srv.close()
