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

import numpy as np

from demo.contract import encode_response
from demo.run_demo import drive_robot, respond_once


class FakeRobot:
    def __init__(self):
        self.calls = []
    def look_at(self, x, y):
        self.calls.append(("look_at", (x, y)))
    def say(self, wave, sample_rate):
        self.calls.append(("say", len(wave)))
    def gesture(self, name):
        self.calls.append(("gesture", name))


class FakePlayer:
    """Stub player: tracks lengths of fed phrases, remembers close."""
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.feeds = []
        self.closed = False
    def feed(self, samples):
        self.feeds.append(len(samples))
    def close(self):
        self.closed = True


def fake_post(endpoint, payload):
    # Simulate the Pi's response per the contract.
    result = {"objects": ["person"], "heard": "hi", "reply": "I see you.",
              "spoken": ["I see you."], "gaze": (0.4, -0.1)}
    return encode_response(result, np.zeros(8000, np.float32), 16000,
                           {"detector": 60.0}, 2400.0)


def test_respond_once_parses_pi_answer():
    resp = respond_once(b"png", np.zeros(80000, np.float32), 16000,
                        endpoint="http://pi:9500/respond", http_post=fake_post)
    assert resp.reply == "I see you."
    assert resp.gaze == (0.4, -0.1)
    assert resp.total_ms == 2400.0


def test_drive_robot_turns_head_then_speaks():
    robot = FakeRobot()
    result = {"objects": [], "heard": "", "reply": "hello", "spoken": ["hello"],
              "gaze": (0.4, -0.1)}
    from demo.contract import decode_response
    resp = decode_response(
        encode_response(result, np.zeros(8000, np.float32), 16000, {}, 1.0))
    drive_robot(robot, resp)
    kinds = [c[0] for c in robot.calls]
    # Turn the head toward the person first, then speak.
    assert kinds == ["look_at", "say"]
    assert robot.calls[0][1] == (0.4, -0.1)


def test_drive_robot_skips_speech_when_no_audio():
    robot = FakeRobot()
    result = {"objects": [], "heard": "", "reply": "", "spoken": [],
              "gaze": (0.0, 0.0)}
    from demo.contract import decode_response
    resp = decode_response(
        encode_response(result, np.zeros(0, np.float32), 16000, {}, 1.0))
    drive_robot(robot, resp)
    # The head can still turn, but we don't call say with empty audio.
    assert ("say", 0) not in robot.calls


# --- streaming mode ---

def test_drive_robot_stream_turns_head_then_speaks_each_phrase():
    import base64
    import numpy as np
    from demo.run_demo import drive_robot_stream

    def audio_event(text):
        a = np.zeros(8000, dtype=np.float32)
        return {"type": "audio", "text": text,
                "audio_b64": base64.b64encode(a.tobytes()).decode(),
                "sample_rate": 16000}

    events = [
        {"type": "meta", "objects": ["person"], "heard": "hi", "gaze": [0.3, -0.1]},
        audio_event("Hello."),
        audio_event("I see you."),
        {"type": "done", "reply": "Hello. I see you.", "first_sound_ms": 1300,
         "total_ms": 3000},
    ]
    robot = FakeRobot()
    player = FakePlayer()
    done = drive_robot_stream(robot, events, make_player=lambda sr: player)
    # The robot only moves (gaze); audio goes to a single player, back to back.
    assert [c[0] for c in robot.calls] == ["look_at"]
    assert robot.calls[0][1] == (0.3, -0.1)
    assert player.feeds == [8000, 8000], "two phrases fed into one player"
    assert player.closed, "player closed at the end of the response"
    assert done["reply"] == "Hello. I see you."


def test_drive_robot_stream_surfaces_mid_stream_error_event():
    # The Pi failed after the 200 headers were already sent, so it couldn't
    # answer with a 4xx/5xx and instead emitted a terminal error event.
    # drive_robot_stream must surface it via on_reply(..., done=True) rather
    # than let the turn go silent.
    from demo.run_demo import drive_robot_stream

    events = [
        {"type": "meta", "objects": [], "heard": "hi", "gaze": [0.0, 0.0]},
        {"type": "error", "message": "model OOM"},
    ]
    replies = []
    done = drive_robot_stream(
        FakeRobot(), events,
        make_player=lambda sr: FakePlayer(),
        on_reply=lambda text, is_done: replies.append((text, is_done)))
    # The error text is forwarded as the terminal (done=True) reply...
    assert ("[error: model OOM]", True) in replies
    # ...and, since no done event followed, the returned dict stays empty.
    assert done == {}


def test_drive_robot_stream_performs_gesture_from_gesture_event():
    # Gesture no longer arrives inside meta (substring heuristic
    # removed) — a separate gesture event from the model's tool_call, with
    # no accompanying sound (gesture is silent).
    import numpy as np
    from demo.run_demo import drive_robot_stream

    events = [
        {"type": "meta", "objects": ["person"], "heard": "shake your head",
         "gaze": [0.0, 0.0]},
        {"type": "gesture", "gesture": "shake"},
        {"type": "done", "reply": ""},
    ]
    robot = FakeRobot()
    player = FakePlayer()
    drive_robot_stream(robot, events, make_player=lambda sr: player)
    # Head toward the person, then the actual gesture; no sound at all.
    assert [c[0] for c in robot.calls] == ["look_at", "gesture"]
    assert ("gesture", "shake") in robot.calls
    assert player.feeds == []


def test_drive_robot_stream_no_gesture_when_absent():
    import base64
    import numpy as np
    from demo.run_demo import drive_robot_stream

    a = np.zeros(8000, dtype=np.float32)
    events = [
        {"type": "meta", "objects": [], "heard": "hi", "gaze": [0.0, 0.0]},
        {"type": "audio", "text": "Hi.",
         "audio_b64": base64.b64encode(a.tobytes()).decode(),
         "sample_rate": 16000},
        {"type": "done", "reply": "Hi."},
    ]
    robot = FakeRobot()
    player = FakePlayer()
    drive_robot_stream(robot, events, make_player=lambda sr: player)
    assert not any(c[0] == "gesture" for c in robot.calls)
    assert player.feeds == [8000]


def test_drive_robot_stream_skips_empty_audio():
    import base64
    import numpy as np
    from demo.run_demo import drive_robot_stream

    events = [
        {"type": "meta", "objects": [], "heard": "", "gaze": [0.0, 0.0]},
        {"type": "audio", "text": "x",
         "audio_b64": base64.b64encode(np.zeros(0, np.float32).tobytes()).decode(),
         "sample_rate": 16000},
        {"type": "done", "reply": "x"},
    ]
    robot = FakeRobot()
    player = FakePlayer()
    drive_robot_stream(robot, events, make_player=lambda sr: player)
    # Empty audio isn't fed to the player and doesn't trigger feed calls.
    assert player.feeds == []


def test_drive_robot_stream_forwards_reply_chunks():
    from demo.run_demo import drive_robot_stream
    import base64, numpy as np

    class DummyRobot:
        def look_at(self, x, y): pass
        def say(self, *a): pass
        def gesture(self, g): pass

    class NullPlayer:
        def __init__(self, sr): pass
        def feed(self, a): pass
        def close(self): pass

    audio_b64 = base64.b64encode(
        np.zeros(400, np.float32).tobytes()).decode()
    events = [
        {"type": "meta", "objects": [], "heard": "hi", "gaze": [0, 0]},
        {"type": "audio", "text": "Hi.", "audio_b64": audio_b64,
         "sample_rate": 16000},
        {"type": "done", "reply": "Hi there.", "first_sound_ms": 100},
    ]
    seen = []
    drive_robot_stream(DummyRobot(), iter(events),
                       on_reply=lambda t, d: seen.append((t, d)),
                       make_player=NullPlayer)
    assert ("Hi.", False) in seen
    assert ("Hi there.", True) in seen


# --- _handle_stream: the voice loop always goes through a DisplaySink ---
# (the source from GpuDetectSource always returns a detections list — the
# "webui vs none" branch was removed from run_voice; _handle_stream accepts
# a DisplaySink of any implementation — here a spy stub, like
# NullSink/WebDashboard.)

class DisplaySpy:
    def __init__(self):
        self.detections = []
        self.queries = []
        self.heard = []
        self.replies = []
        self.closed = False

    def on_detections(self, detections):
        self.detections.append(detections)

    def on_query(self, frame, detections):
        self.queries.append((frame, detections))

    def on_heard(self, text):
        self.heard.append(text)

    def on_reply(self, text, done=False):
        self.replies.append((text, done))

    def close(self):
        self.closed = True


def test_handle_stream_drives_display_spy(monkeypatch):
    import base64

    from demo.run_demo import _handle_stream

    audio_b64 = base64.b64encode(
        np.zeros(400, np.float32).tobytes()).decode()
    events = [
        {"type": "meta", "objects": ["person"], "heard": "hi",
         "gaze": [0.0, 0.0]},
        {"type": "audio", "text": "Hi.", "audio_b64": audio_b64,
         "sample_rate": 16000},
        {"type": "done", "reply": "Hi there.", "first_sound_ms": 100},
    ]
    monkeypatch.setattr("demo.run_demo._http_stream",
                        lambda endpoint, payload: iter(events))
    # The player is a stub: we don't write a real WAV or call afplay in the test.
    monkeypatch.setattr("demo.audio_out.StreamPlayer", FakePlayer)

    display = DisplaySpy()
    robot = FakeRobot()
    dets = [{"label": "cup", "score": 0.9, "box": [0.1, 0.1, 0.2, 0.2]}]
    _handle_stream(b"png", dets, np.zeros(8000, np.float32),
                  "http://pi:9500/respond", robot, display)

    assert display.heard == ["hi"]
    assert display.replies == [("Hi.", False), ("Hi there.", True)]
    # gaze fired — before speech, same as before
    assert robot.calls[0][0] == "look_at"


# --- _FailureGuard: the healthy flag logs the transition, not every call ---

def test_failure_guard_logs_failure_once_not_on_every_repeat(capsys):
    from demo.run_demo import _FailureGuard

    guard = _FailureGuard("robot")

    def boom():
        raise RuntimeError("disconnected")

    guard.call(boom)
    guard.call(boom)
    guard.call(boom)
    out = capsys.readouterr().out
    assert out.count("[robot] skip") == 1, "repeated failures shouldn't spam"
    assert guard.healthy is False


def test_failure_guard_logs_recovery_once(capsys):
    from demo.run_demo import _FailureGuard

    guard = _FailureGuard("robot")
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("disconnected")

    guard.call(flaky)  # first call fails
    assert guard.healthy is False
    guard.call(flaky)  # second call succeeds -> recovered
    assert guard.healthy is True
    out = capsys.readouterr().out
    assert out.count("[robot] skip") == 1
    assert out.count("recovered") == 1


def test_drive_robot_stream_logs_robot_failure_once_across_events(capsys):
    # Within a SINGLE response, several events (meta + gesture + meta) hit a
    # hopelessly failed robot — the "skip" print should happen once, not on
    # every event.
    from demo.run_demo import drive_robot_stream

    class FlakyRobot:
        def look_at(self, x, y):
            raise RuntimeError("disconnected")
        def gesture(self, name):
            raise RuntimeError("disconnected")
        def say(self, *a):
            pass

    events = [
        {"type": "meta", "objects": [], "heard": "hi", "gaze": [0.0, 0.0]},
        {"type": "gesture", "gesture": "shake"},
        {"type": "meta", "objects": [], "heard": "again", "gaze": [0.0, 0.0]},
        {"type": "done", "reply": ""},
    ]
    player = FakePlayer()
    drive_robot_stream(FlakyRobot(), events, make_player=lambda sr: player)
    out = capsys.readouterr().out
    assert out.count("[robot] skip") == 1


def test_robot_guard_healthy_state_persists_across_drive_calls():
    # "per-robot", not per-response: the same guard passed to two separate
    # drive_robot_stream calls (two consecutive responses) shouldn't print
    # again on a repeated failure of the same robot.
    from demo.run_demo import _FailureGuard, drive_robot_stream

    class AlwaysFails:
        def look_at(self, x, y):
            raise RuntimeError("dead")
        def gesture(self, name):
            pass
        def say(self, *a):
            pass

    guard = _FailureGuard("robot")
    robot = AlwaysFails()
    events = [{"type": "meta", "objects": [], "heard": "a", "gaze": [0.0, 0.0]},
             {"type": "done", "reply": ""}]

    drive_robot_stream(robot, events, make_player=lambda sr: FakePlayer(),
                       robot_guard=guard)
    assert guard.healthy is False

    drive_robot_stream(robot, list(events), make_player=lambda sr: FakePlayer(),
                       robot_guard=guard)
    assert guard.healthy is False  # same guard — state wasn't reset


def test_drive_robot_stream_audio_failure_does_not_crash_loop():
    # The audio path is under the same protection as the robot: a broken
    # player shouldn't kill the rest of the response.
    import base64
    from demo.run_demo import drive_robot_stream

    class BrokenPlayer:
        def __init__(self, sr):
            pass
        def feed(self, samples):
            raise RuntimeError("afplay missing")
        def close(self):
            raise RuntimeError("afplay missing")

    a = np.zeros(400, dtype=np.float32)
    audio_b64 = base64.b64encode(a.tobytes()).decode()
    events = [
        {"type": "meta", "objects": [], "heard": "hi", "gaze": [0.0, 0.0]},
        {"type": "audio", "text": "hi", "audio_b64": audio_b64,
         "sample_rate": 16000},
        {"type": "done", "reply": "hi"},
    ]
    done = drive_robot_stream(FakeRobot(), events,
                              make_player=lambda sr: BrokenPlayer(sr))
    assert done["reply"] == "hi"


# --- _handle_stream: `detections is not None`, not truthiness ---

def test_handle_stream_empty_detections_list_does_not_fallback_to_frame(monkeypatch):
    # An empty scene (GPU ran, found nothing) is a valid detections list,
    # not a "GPU unavailable" signal: the frame must not end up in the payload.
    from demo.run_demo import _handle_stream

    captured = {}

    def fake_http_stream(endpoint, payload):
        captured.update(payload)
        return iter([{"type": "done", "reply": ""}])

    monkeypatch.setattr("demo.run_demo._http_stream", fake_http_stream)
    monkeypatch.setattr("demo.audio_out.StreamPlayer", FakePlayer)

    display = DisplaySpy()
    _handle_stream(b"should-be-ignored", [], np.zeros(400, np.float32),
                  "http://pi:9500/respond", FakeRobot(), display)

    assert "frame_b64" not in captured
    assert captured.get("detections") == []


def test_handle_stream_falls_back_to_frame_when_detections_none(monkeypatch):
    from demo.run_demo import _handle_stream

    captured = {}

    def fake_http_stream(endpoint, payload):
        captured.update(payload)
        return iter([{"type": "done", "reply": ""}])

    monkeypatch.setattr("demo.run_demo._http_stream", fake_http_stream)
    monkeypatch.setattr("demo.audio_out.StreamPlayer", FakePlayer)

    display = DisplaySpy()
    frame_png = b"\x89PNG\r\n\x1a\nrest-of-file"
    _handle_stream(frame_png, None, np.zeros(400, np.float32),
                  "http://pi:9500/respond", FakeRobot(), display)

    assert "frame_b64" in captured
    assert "detections" not in captured


def test_handle_stream_sends_explicit_empty_detections_when_neither_available(monkeypatch):
    # A very early utterance: neither frame nor detections exist yet — the
    # payload must not go out without a picture at all; send an explicit
    # empty detections list.
    from demo.run_demo import _handle_stream

    captured = {}

    def fake_http_stream(endpoint, payload):
        captured.update(payload)
        return iter([{"type": "done", "reply": ""}])

    monkeypatch.setattr("demo.run_demo._http_stream", fake_http_stream)
    monkeypatch.setattr("demo.audio_out.StreamPlayer", FakePlayer)

    display = DisplaySpy()
    _handle_stream(None, None, np.zeros(400, np.float32),
                  "http://pi:9500/respond", FakeRobot(), display)

    assert "frame_b64" not in captured
    assert captured.get("detections") == []
