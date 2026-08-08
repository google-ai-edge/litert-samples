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

import http.client
import json
import time

import numpy as np

from demo.display import NullSink, build_display
from demo.display.events import sse
from demo.display.web import WebDashboard


class FakeCamera:
    def __init__(self, frame):
        self._frame = frame

    def latest(self):
        return self._frame


def drain(q):
    out = []
    while not q.empty():
        out.append(q.get())
    return out


# --- sse() (moved from tests/test_web_events.py) ---

def test_sse_formats_event_and_json():
    out = sse("detections", {"boxes": [{"label": "cup"}]})
    assert out.startswith("event: detections\n")
    assert out.endswith("\n\n")
    line = [l for l in out.splitlines() if l.startswith("data: ")][0]
    assert json.loads(line[len("data: "):]) == {"boxes": [{"label": "cup"}]}


# --- NullSink: all methods are no-ops ---

def test_null_sink_methods_are_all_no_ops():
    sink = NullSink()
    assert sink.on_detections([{"label": "cup"}]) is None
    assert sink.on_query(np.zeros((4, 4, 3), np.uint8), []) is None
    assert sink.on_query(None, []) is None
    assert sink.on_heard("hello") is None
    assert sink.on_reply("hi", False) is None
    assert sink.on_reply("hi there", True) is None
    assert sink.close() is None


# --- WebDashboard: SSE events ---

def test_on_query_broadcasts_query_frame_to_subscribers():
    dash = WebDashboard(FakeCamera(np.zeros((8, 8, 3), np.uint8)))
    q = dash.subscribe()
    dets = [{"label": "cup", "score": 0.8, "box": [0.4, 0.4, 0.6, 0.6]}]
    dash.on_query(np.zeros((8, 8, 3), np.uint8), dets)
    msgs = drain(q)
    assert any(m.startswith("event: query_frame\n") for m in msgs)
    payload = json.loads(msgs[0].splitlines()[1][len("data: "):])
    assert payload["boxes"] == dets
    assert isinstance(payload["jpeg_b64"], str) and payload["jpeg_b64"]


def test_on_query_with_no_frame_sends_empty_jpeg_b64():
    dash = WebDashboard(FakeCamera(None))
    q = dash.subscribe()
    dash.on_query(None, [])
    msgs = drain(q)
    payload = json.loads(msgs[0].splitlines()[1][len("data: "):])
    assert payload["jpeg_b64"] == ""


def test_on_detections_is_the_source_listener():
    """on_detections is what source.add_listener(display.on_detections)
    calls on every detect cycle; it should broadcast a detections event."""
    dash = WebDashboard(FakeCamera(None))
    q = dash.subscribe()
    dets = [{"label": "person", "score": 0.9, "box": [0, 0, 1, 1]}]
    dash.on_detections(dets)
    msgs = drain(q)
    assert any(m.startswith("event: detections\n") for m in msgs)
    payload = json.loads(msgs[0].splitlines()[1][len("data: "):])
    assert payload["boxes"] == dets


def test_on_heard_and_reply_events():
    dash = WebDashboard(FakeCamera(None))
    q = dash.subscribe()
    dash.on_heard("hello")
    dash.on_reply("Hi.", done=False)
    dash.on_reply("Hi there.", done=True)
    msgs = drain(q)
    assert any(m.startswith("event: heard\n") for m in msgs)
    replies = [m for m in msgs if m.startswith("event: reply\n")]
    assert len(replies) == 2


def test_unsubscribe_stops_delivery():
    dash = WebDashboard(FakeCamera(None))
    q = dash.subscribe()
    dash.unsubscribe(q)
    dash.on_heard("hello")
    assert drain(q) == []


def test_subscriber_queue_drops_oldest_when_full():
    """The subscriber queue is bounded — overflow drops the oldest
    message rather than growing without bound or blocking the broadcast."""
    dash = WebDashboard(FakeCamera(None))
    q = dash.subscribe()
    from demo.display.web import SUB_QUEUE_MAXSIZE

    for i in range(SUB_QUEUE_MAXSIZE + 5):
        dash.on_heard(str(i))
    msgs = drain(q)
    assert len(msgs) == SUB_QUEUE_MAXSIZE
    # the oldest events (0..4) were dropped, the freshest ones are still here
    assert '"text": "0"' not in msgs[0]
    assert msgs[-1].strip().endswith(
        f'"text": "{SUB_QUEUE_MAXSIZE + 4}"}}')


# --- WebDashboard: real HTTP — MJPEG multipart + SSE framing ---

def _wait_for_subscriber(dash, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with dash._subs_lock:
            if dash._subs:
                return
        time.sleep(0.01)
    raise AssertionError("subscriber never registered")


def test_mjpeg_endpoint_serves_multipart_boundary():
    dash = WebDashboard(FakeCamera(np.zeros((4, 4, 3), np.uint8)))
    server = dash.serve(host="127.0.0.1", port=0)
    try:
        host, port = server.server_address
        conn = http.client.HTTPConnection(host, port, timeout=5)
        conn.request("GET", "/live.mjpeg")
        resp = conn.getresponse()
        assert resp.status == 200
        assert resp.getheader("Content-Type").startswith(
            "multipart/x-mixed-replace")
        chunk = resp.read(64)
        assert chunk.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n")
        conn.close()
    finally:
        dash.close()


def test_events_endpoint_streams_sse_framed_messages():
    dash = WebDashboard(FakeCamera(None))
    server = dash.serve(host="127.0.0.1", port=0)
    try:
        host, port = server.server_address
        conn = http.client.HTTPConnection(host, port, timeout=5)
        conn.request("GET", "/events")
        resp = conn.getresponse()
        assert resp.status == 200
        assert resp.getheader("Content-Type") == "text/event-stream"
        _wait_for_subscriber(dash)
        dash.on_heard("hello")
        expected = sse("heard", {"text": "hello"}).encode()
        assert resp.read(len(expected)) == expected
        conn.close()
    finally:
        dash.close()


def test_index_page_serves_dashboard_html():
    dash = WebDashboard(FakeCamera(None))
    server = dash.serve(host="127.0.0.1", port=0)
    try:
        host, port = server.server_address
        conn = http.client.HTTPConnection(host, port, timeout=5)
        conn.request("GET", "/")
        resp = conn.getresponse()
        assert resp.status == 200
        assert resp.getheader("Content-Type").startswith("text/html")
        body = resp.read()
        assert b"<html" in body
        conn.close()
    finally:
        dash.close()


def test_unknown_path_returns_404():
    dash = WebDashboard(FakeCamera(None))
    server = dash.serve(host="127.0.0.1", port=0)
    try:
        host, port = server.server_address
        conn = http.client.HTTPConnection(host, port, timeout=5)
        conn.request("GET", "/nope")
        resp = conn.getresponse()
        assert resp.status == 404
        resp.read()
        conn.close()
    finally:
        dash.close()


# --- build_display: factory ---

def test_build_display_none_returns_null_sink():
    display = build_display("none", camera=FakeCamera(None),
                            host="127.0.0.1", port=0)
    assert isinstance(display, NullSink)


def test_build_display_web_returns_started_web_dashboard():
    display = build_display("web", camera=FakeCamera(None),
                            host="127.0.0.1", port=0)
    try:
        assert isinstance(display, WebDashboard)
        assert display._server is not None
    finally:
        display.close()


def test_build_display_rejects_unknown_kind():
    import pytest

    with pytest.raises(ValueError):
        build_display("tui", camera=FakeCamera(None), host="127.0.0.1", port=0)
