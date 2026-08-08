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

"""HTTP server on the Pi: one POST, one pipeline pass.

The pipeline is already built (emulator/); the server just accepts a frame
and audio, runs Pipeline.step, and returns the result per the contract. The
Pi does all the compute — that's the point.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import re
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from demo.contract import decode_request, encode_response
from demo.detections import detections_to_dicts, gaze_from_dicts
from demo.http_util import ascii_reason as _ascii_reason
from demo.stream import extract_sentences, flush_tail
from emulator.llm import GESTURE_NAMES
from emulator.pipeline import build_prompt, gaze_target
from emulator.timeline import Timeline

LOG = logging.getLogger(__name__)

# decompression-bomb guard: same cap as gpu_detect.py — unpacking a frame into
# memory shouldn't balloon on a maliciously huge PNG (demo frames are <1 Mpx).
Image.MAX_IMAGE_PIXELS = 10_000_000

# overflow guard: cap on the POST request body. Bigger than MAX_BODY in
# gpu_detect.py (4 MB there — just a raw JPEG frame): here the JSON payload
# carries a PNG frame + audio_b64 (up to ~8 s of speech at 16 kHz float32,
# base64 adds ~1.33x) — 8 MB gives comfortable headroom over real demo
# payloads.
MAX_BODY = 8 * 1024 * 1024


def png_to_frame(frame_png: bytes, shape: tuple[int, int, int]) -> np.ndarray:
    height, width, _ = shape
    img = Image.open(io.BytesIO(frame_png)).convert("RGB").resize(
        (width, height))
    return (np.asarray(img, dtype=np.float32) / 255.0).reshape(shape)


class CollectingRobot:
    """Collects audio and gaze instead of playing them — the server returns them to the client."""

    def __init__(self) -> None:
        self.audio_chunks: list[np.ndarray] = []
        self.sample_rate = 16000
        self.gaze = (0.0, 0.0)

    def look_at(self, x: float, y: float) -> None:
        self.gaze = (float(x), float(y))

    def say(self, wave: np.ndarray, sample_rate: int) -> None:
        self.audio_chunks.append(np.asarray(wave, dtype=np.float32))
        self.sample_rate = sample_rate

    def full_audio(self) -> np.ndarray:
        return (np.concatenate(self.audio_chunks) if self.audio_chunks
                else np.zeros(0, dtype=np.float32))


def handle_respond(payload: dict, pipeline, frame_shape) -> dict:
    frame_png, pcm, _, _ = decode_request(payload)
    frame = png_to_frame(frame_png, frame_shape)
    collector = CollectingRobot()
    # Swapping pipeline.robot for collector is safe ONLY because HTTPServer
    # here is single-threaded (do_POST handles one request at a time) — there
    # can't be concurrent requests against the same pipeline. Under
    # ThreadingHTTPServer this would be a race on a shared attribute. The
    # previous `hasattr(pipeline, "robot")` check was dead code: a real
    # Pipeline always gets robot in its constructor (emulator/pipeline.py,
    # `__init__(self, detector, recognizer, llm, synthesizer, robot)`) — removed.
    pipeline.robot = collector
    timeline = Timeline()
    result = pipeline.step(frame, pcm, timeline)
    audio = result.get("_audio")
    if audio is None:
        audio = collector.full_audio()
    gaze = result.get("_gaze", collector.gaze)
    result = {**result, "gaze": gaze}
    timings = {s.name: s.duration_ms for s in timeline.spans()}
    return encode_response(result=result, audio=audio,
                           sample_rate=collector.sample_rate,
                           timings=timings, total_ms=timeline.total_ms())


# Prompt for streaming: a lively character that answers on point. SHORT on
# purpose — measured that the COMBINED prompt (this system + the user turn)
# going past ~490 chars knocks prefill off the cliff (TTFT 1.2s -> 10.6s,
# flat). The object list now ALWAYS lives in the user prompt (see DECISION in
# emulator/llm.py), and this system prompt also has to mention the move_head
# tool — the budget is tight. Worst case (max heard + the 4 longest
# COCO labels + this system prompt) is checked by the test
# test_system_plus_prompt_under_cliff: worst case ~479 chars < 490, ~11 to
# spare (this streaming prompt is ~475; the offline system prompt is the 479).
STREAM_SYSTEM = (
    'You are Reachy, a friendly desk robot. Talk naturally, not like a '
    'machine. "(You currently see: ...)" is sensor data: mention it only if '
    "asked what you see or about your surroundings, else ignore it. If "
    "asked to move your head, call move_head; else reply in one short "
    "sentence."
)

# Gesture-calling tool: instead of a substring heuristic
# (detect_gesture/GESTURE_HINTS), the model decides for itself via native
# function calling. Verified live against serve 0.15.0 (stream=true + tools
# together): on a pure movement command the runtime emits ONE tool_calls chunk
# with no content (gesture — silently, no speech); on a plain question it's
# just content with no tool_calls.
MOVE_HEAD_TOOL = {
    "type": "function",
    "function": {
        "name": "move_head",
        "description": "Move the robot's head with a gesture.",
        "parameters": {
            "type": "object",
            "properties": {
                "gesture": {"type": "string", "enum": list(GESTURE_NAMES)},
            },
            "required": ["gesture"],
        },
    },
}

# On combined "gesture + question" turns the runtime sometimes puts raw
# internal function-call tokens directly into content instead of clean
# tool_calls (reproduced across several phrasings) — without stripping them
# they'd go straight into TTS. The gesture on such turns may simply not fire
# — that's accepted, but garbage in the speech is not.
_TOOL_CALL_LEAK = re.compile(r"^\s*<\|tool_call\|?>.*?\}\s*", re.DOTALL)


def _strip_tool_call_leak(text: str) -> str:
    """Strip a leaked `<|tool_call>...}` fragment from the start of the text."""
    return _TOOL_CALL_LEAK.sub("", text, count=1)


# Light guard against a gesture on negation: negation isn't always reliably
# honored by the model itself ("please stop nodding" still nodded sometimes)
# — we check the human's turn before acting on a returned gesture, rather
# than relying solely on the model's decision.
_GESTURE_NEGATIONS = ("don't", "do not", "stop", "no need")


def _negates_gesture(heard: str) -> bool:
    lowered = heard.lower()
    return any(word in lowered for word in _GESTURE_NEGATIONS)


def _audio_event(text, audio, sample_rate):
    return {
        "type": "audio",
        "text": text,
        # "<f4": explicit little-endian float32 on the wire (matches
        # demo/contract.py) so it doesn't rely on both ends being LE.
        "audio_b64": base64.b64encode(
            np.asarray(audio, dtype="<f4").tobytes()).decode("ascii"),
        "sample_rate": int(sample_rate),
    }


def stream_respond(payload, pipeline, frame_shape):
    """Streaming pass: detector and ASR, then model streaming with synthesis
    of each finished sentence as it's ready.

    Yields events one at a time: first meta (objects, heard text, gaze), then
    EITHER an audio event per sentence (speech) OR exactly one gesture event
    (gesture — silently, no speech), finally done with
    timings. The robot starts speaking on the first sentence, without
    waiting for the whole reply (saves ~1.8s). The gesture is now decided by
    the model itself via tools=[MOVE_HEAD_TOOL], not a substring heuristic —
    see `LlmClient.reply_stream`.
    """
    t0 = time.perf_counter()
    frame_png, pcm, _, provided = decode_request(payload)
    if provided is not None:
        # Detections were already computed (e.g. the GPU detector on the
        # Mac) — leave the frame alone and don't call the Pi's CPU detector.
        objects = [d["label"] for d in provided]
        detail = provided
        gaze = gaze_from_dicts(provided)
    else:
        frame = png_to_frame(frame_png, frame_shape)
        detections = pipeline.detector.detect(frame)
        gaze = gaze_target(detections)
        # Detailed detections with confidence and box — so we can see WHAT
        # exactly yolox found and how confidently (hallucination diagnostics).
        # Same format that gpu_detect.py returns — reuse the shared converter
        # instead of rebuilding the list here.
        detail = detections_to_dicts(detections)
        objects = [d["label"] for d in detail]
    heard = pipeline.recognizer.transcribe(pcm)
    yield {"type": "meta", "objects": objects, "heard": heard,
           "detections": detail,
           "gaze": [float(gaze[0]), float(gaze[1])],
           "asr_ms": (time.perf_counter() - t0) * 1000}

    if not heard.strip():
        # Empty ASR (a false VAD trigger — a cough, background noise) must not
        # cost a full LLM+TTS turn and a spoken "please repeat" on every blip.
        # Emit an empty done and stop, mirroring Pipeline.step's wake gate.
        yield {"type": "done", "reply": "", "first_sound_ms": None,
               "total_ms": (time.perf_counter() - t0) * 1000}
        return

    prompt = build_prompt(objects=objects, heard=heard)
    # Default of 24000 — same as pipeline.py (`getattr(self.synthesizer,
    # "sample_rate", 24000)`), not 16000: the same fallback constant for the
    # case where synthesizer lacks a sample_rate attribute, rather than two
    # different magic numbers in two places.
    sample_rate = getattr(pipeline.synthesizer, "sample_rate", 24000)
    # Check negation BEFORE the model would have had a chance to call
    # anything — the decision doesn't depend on what it returns.
    gesture_blocked = _negates_gesture(heard)

    def synth(text):
        audio = pipeline.synthesizer.speak(text)
        if len(audio) > 0:
            return _audio_event(text, audio, sample_rate)
        return None

    buffer = ""
    full = ""
    first_sound_ms = None
    for msg in pipeline.llm.reply_stream(prompt, system=STREAM_SYSTEM,
                                         tools=[MOVE_HEAD_TOOL]):
        if msg["type"] == "tool_call":
            if msg["name"] == "move_head":
                gesture = msg["arguments"].get("gesture")
                if gesture in GESTURE_NAMES and not gesture_blocked:
                    yield {"type": "gesture", "gesture": gesture}
            continue
        # Combined "gesture + question" turns sometimes leak raw
        # function-call tokens straight into content (see
        # _strip_tool_call_leak) — clean the buffer and the full reply
        # before synthesis and before returning it in reply.
        buffer = _strip_tool_call_leak(buffer + msg["text"])
        full = _strip_tool_call_leak(full + msg["text"])
        sentences, buffer = extract_sentences(buffer)
        for sentence in sentences:
            event = synth(sentence)
            if event is not None:
                if first_sound_ms is None:
                    first_sound_ms = (time.perf_counter() - t0) * 1000
                yield event
    tail = flush_tail(buffer)
    if tail:
        event = synth(tail)
        if event is not None:
            if first_sound_ms is None:
                first_sound_ms = (time.perf_counter() - t0) * 1000
            yield event

    yield {"type": "done", "reply": full.strip(),
           "first_sound_ms": first_sound_ms,
           "total_ms": (time.perf_counter() - t0) * 1000}


def make_handler(pipeline, frame_shape):
    class Handler(BaseHTTPRequestHandler):
        # Single-threaded HTTPServer (one worker for the whole process) — the
        # timeout keeps a slow/hung client from permanently hogging the one
        # socket.
        timeout = 60

        def do_POST(self):
            # A single bad request shouldn't take down the server: a parse
            # or run error is returned as a status code, not a crash of the
            # whole process.
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError as exc:
                LOG.warning("bad Content-Length: %s: %s",
                           type(exc).__name__, exc)
                self.send_error(400, "bad Content-Length")
                return
            if length <= 0 or length > MAX_BODY:
                LOG.warning("body size out of bounds: %d bytes", length)
                # The reason phrase of the HTTP status line is encoded by
                # http.server as latin-1 — non-ASCII here would break the
                # send with UnicodeEncodeError (that's what the 413 test
                # caught); the message is ASCII, this comment need not be.
                self.send_error(413, "request body out of bounds")
                return
            try:
                payload = json.loads(self.rfile.read(length))
            except json.JSONDecodeError as exc:
                LOG.warning("bad JSON: %s: %s", type(exc).__name__, exc)
                self.send_error(400, "bad JSON")
                return
            try:
                if self.path == "/respond":
                    self._respond(payload)
                elif self.path == "/respond_stream":
                    self._respond_stream(payload)
                else:
                    self.send_error(404)
            except KeyError as exc:
                LOG.warning("missing field: %s: %s", type(exc).__name__, exc)
                self.send_error(400, _ascii_reason(f"missing field {exc}"))
            except (ValueError, TypeError, AttributeError,
                    UnidentifiedImageError) as exc:
                # The client sent broken/mistyped fields (bad base64,
                # corrupt PNG, sample_rate not a number, a non-object JSON
                # payload) — that's a 400, not a 500. We distinguish by
                # exception TYPE rather than where it was raised — an
                # acceptable simplification for this demo endpoint (see also
                # the comment at send_error(500, ...)).
                LOG.warning("bad request field: %s: %s",
                           type(exc).__name__, exc)
                self.send_error(400, _ascii_reason(str(exc)))
            except Exception as exc:  # noqa: BLE001 — the server must survive
                # Broad boundary → log WITH the stack (exc_info) so an
                # unexpected bug is localizable, not just a one-line type+message.
                LOG.exception("unhandled exception: %s", type(exc).__name__)
                try:
                    # A raw exception in the 500 body is handy for debugging
                    # the demo; a real production API shouldn't do this (it
                    # can leak an internal path/traceback).
                    self.send_error(500, _ascii_reason(str(exc)))
                except Exception:
                    pass  # the response may have already partially gone out

        def _respond(self, payload):
            body = handle_respond(payload, pipeline, frame_shape)
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _respond_stream(self, payload):
            # NDJSON: one event per line, flushed immediately, so the Mac
            # gets a sentence and plays it while the next one is computed.
            #
            # Validation happens BEFORE send_response(200): decode_request +
            # an attempt to parse the frame as an image. stream_respond is a
            # generator, its body does NOT execute before the first next() —
            # if we deferred the check until iterating over it, the 200
            # headers would go to the client before a parse error surfaced,
            # and a second send_error() after that is already broken (can't
            # send a status twice). This way a broken payload gets a clean
            # 400 through the shared except in do_POST.
            frame_png, _pcm, _sample_rate, provided = decode_request(payload)
            if provided is None:
                # Neither precomputed detections nor a frame to run the
                # detector on — reject as a clean 400 BEFORE the 200 headers,
                # not as a mid-stream error after them.
                if frame_png is None:
                    raise ValueError("request has neither frame_b64 nor detections")
                png_to_frame(frame_png, frame_shape)

            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            try:
                for event in stream_respond(payload, pipeline, frame_shape):
                    line = (json.dumps(event) + "\n").encode()
                    try:
                        self.wfile.write(line)
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        # The client (dashboard or robot) closed the NDJSON
                        # stream — benign; nothing more can go to a dead socket.
                        # Stop pulling the pipeline and don't log a traceback.
                        # ONLY the write is guarded here: an error from advancing
                        # the pipeline generator (e.g. an upstream LiteRT-LM reset
                        # mid-generation, which raises ConnectionResetError too)
                        # must stay loud and reach the still-connected client, so
                        # it falls through to the broad handler below rather than
                        # being misread as a client disconnect.
                        LOG.info("stream client disconnected mid-response")
                        return
            except Exception as exc:  # noqa: BLE001 — headers are already sent,
                # a second HTTP status can't be sent — instead of a broken
                # response we send a terminal NDJSON error line. This is the broad
                # boundary for anything raised while advancing the pipeline (an
                # upstream model reset, a mid-stream bug): log WITH the stack
                # (exc_info) and surface it to the still-connected client.
                LOG.exception("mid-stream failure: %s", type(exc).__name__)
                error_event = {"type": "error", "message": str(exc)[:200]}
                try:
                    self.wfile.write((json.dumps(error_event) + "\n").encode())
                    self.wfile.flush()
                except Exception:
                    pass  # the connection may have already dropped

        def log_request(self, code="-", size="-"):
            # Don't make noise on ordinary successful demo responses (2xx);
            # 4xx/5xx go through the standard log (log_error -> log_message)
            # — a backstop on top of the targeted LOG.warning/error calls in
            # the except blocks above.
            if isinstance(code, int) and 200 <= code < 300:
                return
            super().log_request(code, size)

    return Handler


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Pi pipeline demo server")
    p.add_argument("--tokenizer", type=Path,
                   default=Path("assets/moonshine_tokenizer.json"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9500)
    p.add_argument("--tts", choices=("inflect",), default="inflect")
    return p.parse_args(argv)


def main(argv=None) -> int:
    from emulator.run import build_pipeline

    args = parse_args(argv)
    pipeline, frame_shape = build_pipeline(args.tokenizer,
                                           skip_llm=False, tts=args.tts)
    server = HTTPServer((args.host, args.port),
                        make_handler(pipeline, frame_shape))
    print(f"demo server on {args.host}:{args.port}, frame {frame_shape}, "
          f"tts {args.tts}")
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        # This endpoint runs the whole pipeline (LLM/ASR/TTS) on client-supplied
        # data with NO auth. Binding to a non-loopback host exposes it to the
        # whole LAN/tailnet — fine for a controlled demo, a footgun otherwise.
        print(f"  WARNING: bound to non-loopback host {args.host!r} — this "
              f"unauthenticated inference server is reachable from the network.")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
