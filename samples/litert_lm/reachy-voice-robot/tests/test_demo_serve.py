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
import threading
from http.server import HTTPServer

import numpy as np
import pytest

from demo.contract import decode_response, encode_request
from demo.serve import MAX_BODY, handle_respond, make_handler, png_to_frame


def make_png(width=32, height=24) -> bytes:
    # Minimal valid PNG via numpy+zlib, no external dependencies.
    from PIL import Image
    import io
    img = Image.fromarray(
        (np.random.default_rng(0).random((height, width, 3)) * 255)
        .astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class FakePipeline:
    def __init__(self):
        self.calls = []

    def step(self, frame, pcm, timeline):
        self.calls.append((frame.shape, pcm.shape))
        with timeline.stage("detector"):
            pass
        return {
            "objects": ["person"], "heard": "hello",
            "reply": "I see you.", "phrases": ["I see you."],
            "spoken": ["I see you."], "skipped": [], "speech_s": 1.5,
            "_audio": np.linspace(-0.3, 0.3, 4000, dtype=np.float32),
            "_gaze": (0.2, -0.1),
        }


def test_png_to_frame_resizes_and_normalises():
    frame = png_to_frame(make_png(64, 48), (416, 416, 3))
    assert frame.shape == (416, 416, 3)
    assert frame.dtype == np.float32
    assert 0.0 <= frame.min() and frame.max() <= 1.0


def test_handle_respond_runs_pipeline_and_builds_contract():
    pipe = FakePipeline()
    payload = encode_request(make_png(), np.zeros(80000, np.float32), 16000)
    body = handle_respond(payload, pipe, frame_shape=(416, 416, 3))
    got = decode_response(body)
    assert got.objects == ["person"]
    assert got.heard == "hello"
    assert got.reply == "I see you."
    assert got.gaze == (0.2, -0.1)
    assert len(got.audio) == 4000
    assert "detector" in got.timings
    assert pipe.calls, "pipeline should have been called"


def test_handle_respond_passes_correct_frame_shape():
    pipe = FakePipeline()
    payload = encode_request(make_png(100, 80), np.zeros(80000, np.float32), 16000)
    handle_respond(payload, pipe, frame_shape=(416, 416, 3))
    # The frame must reach the pipeline resized to the detector's input shape.
    assert pipe.calls[0][0] == (416, 416, 3)


class _Recognizer:
    """ASR stub: returns a fixed utterance without touching the real pcm."""
    def __init__(self, heard):
        self._heard = heard

    def transcribe(self, pcm):
        return self._heard


class _ContentLlm:
    """LLM stub: plain speech, no tool call."""
    def reply_stream(self, prompt, system=None, tools=None):
        yield {"type": "content", "text": "Hi there."}


class _GestureLlm:
    """LLM stub: a pure tool_call with no content — a silent gesture, no speech."""
    def __init__(self, gesture="shake"):
        self._gesture = gesture

    def reply_stream(self, prompt, system=None, tools=None):
        yield {"type": "tool_call", "name": "move_head",
               "arguments": {"gesture": self._gesture}}


class _LeakingLlm:
    """LLM stub: a combined turn leaked raw function-call tokens straight into
    content — without stripping they'd go straight into TTS."""
    def reply_stream(self, prompt, system=None, tools=None):
        yield {"type": "content",
               "text": '<|tool_call>call:move_head{gesture:<|"|>shake<|"|>}'
                       " Sure, I'm shaking my head!"}


class FakeStreamPipeline:
    """The detector raises — proves that with detections already provided it is NOT called."""
    class _Det:
        def detect(self, frame):
            raise AssertionError("detector must not be called")
    class _Synth:
        sample_rate = 16000
        def speak(self, text):
            return np.linspace(-0.1, 0.1, 800, dtype=np.float32)

    def __init__(self, llm=None, heard="hello"):
        self.detector = self._Det()
        self.recognizer = _Recognizer(heard)
        self.llm = llm or _ContentLlm()
        self.synthesizer = self._Synth()


def test_stream_respond_uses_provided_detections():
    from demo.serve import stream_respond
    dets = [{"label": "cup", "score": 0.8, "box": [0.4, 0.4, 0.6, 0.6]}]
    payload = encode_request(None, np.zeros(8000, np.float32), 16000,
                             detections=dets)
    events = list(stream_respond(payload, FakeStreamPipeline(),
                                 frame_shape=(416, 416, 3)))
    meta = next(e for e in events if e["type"] == "meta")
    assert meta["objects"] == ["cup"]
    assert meta["detections"] == dets
    assert meta["gaze"] == [0.0, 0.0]


def test_stream_respond_empty_heard_stops_after_meta_without_llm():
    # A false VAD trigger (a cough, background noise) transcribes to nothing.
    # An empty transcript must not cost a full LLM+TTS turn: stream_respond
    # emits meta, then a terminal empty done, and never touches the model.
    from demo.serve import stream_respond

    class _NeverLlm:
        def reply_stream(self, prompt, system=None, tools=None):
            raise AssertionError("LLM must not run on an empty transcript")

    payload = encode_request(None, np.zeros(8000, np.float32), 16000,
                             detections=[])
    pipe = FakeStreamPipeline(llm=_NeverLlm(), heard="   ")
    events = list(stream_respond(payload, pipe, frame_shape=(416, 416, 3)))

    kinds = [e["type"] for e in events]
    assert kinds == ["meta", "done"]
    assert "audio" not in kinds
    assert "gesture" not in kinds
    done = events[-1]
    assert done["reply"] == ""
    assert done["first_sound_ms"] is None


# --- gesture via native function calling, tool-only (silent) ---

def test_stream_respond_emits_gesture_event_and_no_audio_on_tool_call():
    from demo.serve import stream_respond
    payload = encode_request(None, np.zeros(8000, np.float32), 16000,
                             detections=[])
    pipe = FakeStreamPipeline(llm=_GestureLlm("shake"),
                              heard="shake your head please")
    events = list(stream_respond(payload, pipe, frame_shape=(416, 416, 3)))
    kinds = [e["type"] for e in events]
    assert "gesture" in kinds
    assert "audio" not in kinds, "pure tool_call — no content arrived, no speech"
    gesture_event = next(e for e in events if e["type"] == "gesture")
    assert gesture_event["gesture"] == "shake"


def test_stream_respond_emits_audio_and_no_gesture_on_content():
    from demo.serve import stream_respond
    payload = encode_request(None, np.zeros(8000, np.float32), 16000,
                             detections=[])
    pipe = FakeStreamPipeline(heard="what is two plus two?")
    events = list(stream_respond(payload, pipe, frame_shape=(416, 416, 3)))
    kinds = [e["type"] for e in events]
    assert "audio" in kinds
    assert "gesture" not in kinds


def test_stream_respond_skips_gesture_on_negation():
    # Light guard: "don't"/"do not"/"stop"/"no need" in the human's turn
    # blocks the gesture regardless of what the model decided.
    from demo.serve import stream_respond
    payload = encode_request(None, np.zeros(8000, np.float32), 16000,
                             detections=[])
    pipe = FakeStreamPipeline(llm=_GestureLlm("shake"),
                              heard="don't shake your head")
    events = list(stream_respond(payload, pipe, frame_shape=(416, 416, 3)))
    assert not any(e["type"] == "gesture" for e in events)


def test_strip_tool_call_leak_removes_leading_garbage():
    from demo.serve import _strip_tool_call_leak
    text = ('<|tool_call>call:move_head{gesture:<|"|>shake<|"|>} '
            "Yes, I am shaking my head!")
    assert _strip_tool_call_leak(text) == "Yes, I am shaking my head!"


def test_strip_tool_call_leak_is_a_noop_on_clean_text():
    from demo.serve import _strip_tool_call_leak
    assert _strip_tool_call_leak("Hello there!") == "Hello there!"


def test_stream_respond_strips_leaked_tool_call_tokens_from_speech():
    from demo.serve import stream_respond
    payload = encode_request(None, np.zeros(8000, np.float32), 16000,
                             detections=[])
    pipe = FakeStreamPipeline(llm=_LeakingLlm(),
                              heard="shake your head and tell me what you see")
    events = list(stream_respond(payload, pipe, frame_shape=(416, 416, 3)))
    done = next(e for e in events if e["type"] == "done")
    assert "<|tool_call>" not in done["reply"]
    assert "Sure, I'm shaking my head!" in done["reply"]
    for e in events:
        if e["type"] == "audio":
            assert "<|tool_call>" not in e["text"]


def test_negates_gesture_recognises_common_phrasings():
    from demo.serve import _negates_gesture
    assert _negates_gesture("don't shake your head")
    assert _negates_gesture("please stop nodding")
    assert _negates_gesture("no need to look left")
    assert not _negates_gesture("shake your head please")


# --- HTTP level: body-cap, 400-vs-500, validate-before-200 / terminal error ---

class _RunningServer:
    """Runs a single-threaded HTTPServer (as in serve.main()) in a background
    thread — we test through real HTTP requests, not a hand-built Handler."""

    def __init__(self, pipeline, frame_shape=(416, 416, 3)):
        self.server = HTTPServer(("127.0.0.1", 0),
                                 make_handler(pipeline, frame_shape))
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    @property
    def address(self):
        return self.server.server_address

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def test_do_post_rejects_zero_length_body():
    srv = _RunningServer(FakePipeline())
    try:
        host, port = srv.address
        conn = http.client.HTTPConnection(host, port, timeout=5)
        conn.request("POST", "/respond", body=b"")
        resp = conn.getresponse()
        assert resp.status == 413
        resp.read()
        conn.close()
    finally:
        srv.close()


def test_do_post_rejects_body_over_max_body():
    srv = _RunningServer(FakePipeline())
    try:
        host, port = srv.address
        conn = http.client.HTTPConnection(host, port, timeout=10)
        # Content-Length lies about the size — the server rejects based on
        # the header BEFORE reading the body, no need to actually push
        # MAX_BODY+1 bytes.
        conn.request("POST", "/respond", body=b"",
                    headers={"Content-Length": str(MAX_BODY + 1)})
        resp = conn.getresponse()
        assert resp.status == 413
        resp.read()
        conn.close()
    finally:
        srv.close()


def test_respond_maps_malformed_sample_rate_to_400():
    srv = _RunningServer(FakePipeline())
    try:
        host, port = srv.address
        conn = http.client.HTTPConnection(host, port, timeout=5)
        payload = encode_request(make_png(), np.zeros(8000, np.float32), 16000)
        payload["sample_rate"] = ["not", "a", "number"]  # wrong type
        conn.request("POST", "/respond", body=json.dumps(payload).encode())
        resp = conn.getresponse()
        assert resp.status == 400
        resp.read()
        conn.close()
    finally:
        srv.close()


def test_respond_maps_bad_base64_audio_to_400():
    srv = _RunningServer(FakePipeline())
    try:
        host, port = srv.address
        conn = http.client.HTTPConnection(host, port, timeout=5)
        payload = encode_request(make_png(), np.zeros(8000, np.float32), 16000)
        payload["audio_b64"] = "not-valid-base64!!!"
        conn.request("POST", "/respond", body=json.dumps(payload).encode())
        resp = conn.getresponse()
        assert resp.status == 400
        resp.read()
        conn.close()
    finally:
        srv.close()


def test_respond_maps_corrupt_png_to_400():
    srv = _RunningServer(FakePipeline())
    try:
        host, port = srv.address
        conn = http.client.HTTPConnection(host, port, timeout=5)
        payload = encode_request(b"not-a-real-png", np.zeros(8000, np.float32),
                                 16000)
        conn.request("POST", "/respond", body=json.dumps(payload).encode())
        resp = conn.getresponse()
        assert resp.status == 400
        resp.read()
        conn.close()
    finally:
        srv.close()


def test_respond_maps_internal_pipeline_bug_to_500():
    class BrokenPipeline:
        def step(self, frame, pcm, timeline):
            raise RuntimeError("internal bug, not a bad request")

    srv = _RunningServer(BrokenPipeline())
    try:
        host, port = srv.address
        conn = http.client.HTTPConnection(host, port, timeout=5)
        payload = encode_request(make_png(), np.zeros(8000, np.float32), 16000)
        conn.request("POST", "/respond", body=json.dumps(payload).encode())
        resp = conn.getresponse()
        assert resp.status == 500
        resp.read()
        conn.close()
    finally:
        srv.close()


def test_respond_stream_validates_before_sending_200():
    # A broken payload (bad base64) should get a clean 400, NOT a 200 with a
    # truncated/empty NDJSON stream — validation happens before send_response.
    srv = _RunningServer(FakeStreamPipeline())
    try:
        host, port = srv.address
        conn = http.client.HTTPConnection(host, port, timeout=5)
        payload = {"audio_b64": "not-valid-base64!!!", "sample_rate": 16000}
        conn.request("POST", "/respond_stream", body=json.dumps(payload).encode())
        resp = conn.getresponse()
        assert resp.status == 400
        resp.read()
        conn.close()
    finally:
        srv.close()


def test_respond_stream_emits_terminal_error_event_on_mid_stream_failure():
    class _RaisingLlm:
        """Yields the first chunk normally, then fails — simulates a failure
        AFTER the 200 headers and the meta event have already reached the client."""
        def reply_stream(self, prompt, system=None, tools=None):
            yield {"type": "content", "text": "partial"}
            raise RuntimeError("llm blew up mid-stream")

    pipe = FakeStreamPipeline(llm=_RaisingLlm())
    srv = _RunningServer(pipe)
    try:
        host, port = srv.address
        conn = http.client.HTTPConnection(host, port, timeout=5)
        payload = encode_request(None, np.zeros(8000, np.float32), 16000,
                                 detections=[])
        conn.request("POST", "/respond_stream", body=json.dumps(payload).encode())
        resp = conn.getresponse()
        # Validation passed — headers are already 200, not 400.
        assert resp.status == 200
        lines = [l for l in resp.read().decode().splitlines() if l.strip()]
        conn.close()
    finally:
        srv.close()
    events = [json.loads(l) for l in lines]
    assert events[0]["type"] == "meta"
    assert events[-1]["type"] == "error"
    assert "message" in events[-1]
