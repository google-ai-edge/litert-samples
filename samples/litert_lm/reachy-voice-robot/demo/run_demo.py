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

"""Voice robot demo running on the Mac.

Two modes. By default — voice: the robot listens continuously, catches an
utterance (VAD), grabs a frame, the Pi does the thinking, the robot replies,
and listens again. With the `--enter` flag — push-to-talk, for debugging.

The Pi does the thinking. This side handles input, calls the server, and
drives the robot. While the robot is talking we don't listen on the
microphone and flush the buffer afterward, otherwise it would trigger on its
own voice.
"""

from __future__ import annotations

import argparse
import http.client
import json
import urllib.request

import time

from demo.contract import decode_response, encode_request
from demo.detect_source import GpuDetectSource
from demo.display import build_display
from demo.platform import build_platform
from demo.platform.mac import grab_frame, record_audio
from demo.platform.pcm import frame_to_png
from demo.robot_reachy import make_robot
from demo.vad import VoiceGate, calibrate_threshold, collect_utterance


def _http_post(endpoint: str, payload: dict) -> dict:
    req = urllib.request.Request(
        endpoint, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


def respond_once(frame_png, audio, sample_rate, endpoint, http_post=_http_post):
    payload = encode_request(frame_png, audio, sample_rate)
    return decode_response(http_post(endpoint, payload))


def _http_stream(endpoint: str, payload: dict):
    """Open a streaming response, yield parsed NDJSON events."""
    req = urllib.request.Request(
        endpoint, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=120)
    for raw in resp:
        line = raw.decode().strip()
        if line:
            yield json.loads(line)


def stream_events(frame_png, audio, sample_rate, endpoint, http_stream=_http_stream):
    payload = encode_request(frame_png, audio, sample_rate)
    return http_stream(endpoint, payload)


def drive_robot(robot, response) -> None:
    """Synchronous (non-streaming) path — only used by --no-stream (for
    comparison with drive_robot_stream). Deliberately WITHOUT
    safe()/_FailureGuard: this is a debug-only branch, not the
    production continuous voice loop, so a loud traceback on robot failure
    is more useful than a silent skip for whoever is debugging via
    --no-stream.
    """
    robot.look_at(*response.gaze)
    if len(response.audio) > 0:
        robot.say(response.audio, response.sample_rate)


class _FailureGuard:
    """Wraps risky calls (robot/audio), logging the transition into
    "unhealthy" and back exactly ONCE, rather than on every event of every
    response. Without this, a failed robot or audio player would print the
    same message on EVERY look_at/gesture/feed of the next response — spam
    instead of signal. The instance lives for the whole voice session
    (created once in run_voice/run_enter and threaded through calls), which
    is why the flag is "per-robot" rather than per-response.

    We deliberately catch broad Exception rather than narrowing the type
    list: both IPC to the MuJoCo sim/real Reachy SDK and afplay/WAV
    recording can fail for reasons no SDK documents exhaustively (dropped
    socket, busy audio device, corrupt data, etc.) — narrowing the types is
    risky: a real failure could slip past the catch and take down the whole
    voice loop, where before it just skipped a frame.
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self.healthy = True

    def call(self, action, *args):
        """Run action(*args) under the guard. Returns its result, or None if
        it raised (so a guarded factory like make_player yields None on
        failure instead of taking down the caller)."""
        try:
            result = action(*args)
        except Exception as exc:  # noqa: BLE001 — see class docstring
            if self.healthy:
                print(f"  [{self.label}] skip ({type(exc).__name__}: {exc})")
            self.healthy = False
            return None
        else:
            if not self.healthy:
                print(f"  [{self.label}] recovered")
            self.healthy = True
            return result


def drive_robot_stream(robot, events, on_meta=None, make_player=None,
                       on_reply=None, robot_guard=None,
                       audio_guard=None) -> dict:
    """Play the response phrase by phrase as it arrives. Returns the done event.

    On meta — only a head turn (gaze). A gesture is a separate `gesture`
    event: the decision is no longer made by a substring heuristic on the
    server side but by the model itself via function calling (move_head, see
    demo/serve.py); the event arrives WITHOUT any accompanying sound (gesture
    is silent, no narration). Audio phrases are fed to a StreamPlayer, which
    buffers them and plays the reply as a single afplay clip when the turn
    completes — this avoids the audible seam a separate afplay-per-phrase would
    leave (an earlier ffplay-from-stdin approach was dropped; see
    demo/audio_out.py). Robot movement is wrapped in try/except — a robot
    failure doesn't kill the dialogue, voice keeps working.

    on_meta(meta) — debug callback: receives the meta event with detections.
    make_player(sample_rate) — player factory (for tests); defaults to StreamPlayer.
    on_reply(text, done) — callback for the web UI: called on every played
    phrase (done=False) and on completion (done=True, full reply).
    """
    import base64

    import numpy as np

    from demo.audio_out import StreamPlayer

    make_player = make_player or StreamPlayer
    robot_guard = robot_guard or _FailureGuard("robot")
    audio_guard = audio_guard or _FailureGuard("audio")

    def safe(action, *args) -> None:
        robot_guard.call(action, *args)

    player = None
    done = {}
    for event in events:
        kind = event.get("type")
        if kind == "meta":
            g = event.get("gaze", [0.0, 0.0])
            safe(robot.look_at, float(g[0]), float(g[1]))
            print(f"  I see:   {event.get('objects') or 'nothing'}")
            print(f"  I heard: {event.get('heard', '')!r}")
            for d in event.get("detections", []):
                print(f"    · {d['label']} {d['score']:.2f}")
            if on_meta is not None:
                on_meta(event)
        elif kind == "gesture":
            # A commanded head gesture ("shake/nod/look left") — actual
            # movement. Decided by the model itself via tool call
            # (demo/serve.py), not a substring heuristic — arrives as its
            # own event, separate from meta, with no accompanying sound
            # (gesture is silent).
            gesture = event.get("gesture")
            print(f"  gesture: {gesture}")
            safe(robot.gesture, gesture)
        elif kind == "audio":
            # "<f4": explicit little-endian float32, matching the server's
            # encode (demo/serve.py) and the contract dtype — not host-native
            # np.float32, so the wire format doesn't depend on both ends being LE.
            audio = np.frombuffer(base64.b64decode(event["audio_b64"]),
                                  dtype="<f4").copy()
            print(f"  saying:  {event.get('text', '')!r}")
            if len(audio) > 0:
                # The audio path is under the same protection as the robot:
                # a broken player — including its CONSTRUCTION (afplay
                # unavailable, WAV not writable) — shouldn't kill the rest of
                # the dialogue; this phrase just silently drops (logged once on
                # the transition to "unhealthy", see _FailureGuard).
                if player is None:
                    player = audio_guard.call(make_player,
                                              event.get("sample_rate", 16000))
                if player is not None:
                    audio_guard.call(player.feed, audio)
            if on_reply is not None:
                on_reply(event.get("text", ""), False)
        elif kind == "error":
            # The Pi failed mid-response: it had already sent the 200 headers,
            # so it couldn't return a 4xx/5xx and instead sent this terminal
            # error event. Surface it — otherwise the turn just goes silent and
            # the operator (watching the Mac) sees nothing while the traceback
            # sits in the Pi's log.
            msg = event.get("message", "")
            print(f"  ! server error mid-response: {msg}")
            if on_reply is not None:
                on_reply(f"[error: {msg}]", True)
        elif kind == "done":
            done = event
            if on_reply is not None:
                on_reply(event.get("reply", ""), True)
    if player is not None:
        audio_guard.call(player.close)
    return done


def _report(response) -> None:
    print(f"  I see:   {response.objects or 'nothing'}")
    print(f"  I heard: {response.heard!r}")
    print(f"  I say:   {response.reply!r}")
    for name, ms in response.timings.items():
        print(f"    {name:<12} {ms:6.0f} ms")
    print(f"  total:   {response.total_ms:.0f} ms")


def _save_debug(save_dir, index, frame_png, meta) -> None:
    """Save the frame and detections for hallucination analysis."""
    import json
    from pathlib import Path

    d = Path(save_dir)
    d.mkdir(parents=True, exist_ok=True)
    stem = d / f"frame_{index:03d}"
    stem.with_suffix(".png").write_bytes(frame_png)
    info = {"heard": meta.get("heard"), "objects": meta.get("objects"),
            "detections": meta.get("detections", [])}
    stem.with_suffix(".json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    saved {stem.name}.png + .json")


def _handle(frame, audio, endpoint, robot, stream=True,
            save_dir=None, index=0, robot_guard=None, audio_guard=None) -> None:
    if stream:
        print("  Pi is thinking (streaming)...")
        events = stream_events(frame, audio, 16000, endpoint + "_stream")
        cb = ((lambda meta: _save_debug(save_dir, index, frame, meta))
              if save_dir else None)
        done = drive_robot_stream(robot, events, on_meta=cb,
                                  robot_guard=robot_guard,
                                  audio_guard=audio_guard)
        if done:
            print(f"  reply:   {done.get('reply', '')!r}")
            if done.get("first_sound_ms"):
                print(f"  first sound: {done['first_sound_ms']:.0f} ms, "
                      f"total: {done.get('total_ms', 0):.0f} ms")
        return
    print("  Pi is thinking...")
    response = respond_once(frame, audio, 16000, endpoint)
    _report(response)
    drive_robot(robot, response)


def _handle_stream(frame_png, detections, audio, endpoint, robot, display,
                   make_player=None, robot_guard=None, audio_guard=None) -> None:
    """Request with ready-made GPU detections. If there are no detections
    (GPU died) — fall back to the old path: send the frame, Pi detects on
    CPU.

    `detections is not None` (not truthiness): an empty scene is a
    valid detector response (GPU ran fine, just found nothing), and it
    shouldn't look like "GPU unavailable" and fall back to the frame. If
    there's NEITHER a frame NOR detections yet (a very early utterance —
    before the first camera frame/first detect cycle), send an explicit
    empty detections list rather than a bare request with no picture at all
    — a payload with "neither frame nor detections" never gets sent to the
    Pi.

    make_player — the platform's response-player factory
    (`platform.make_player`); None leaves the drive_robot_stream default
    (afplay player on the Mac).
    robot_guard/audio_guard — cross-call per-robot "healthy" flags for the
    whole voice session (see _FailureGuard); None creates one-off guards in
    drive_robot_stream (they don't outlive this single response).
    """
    from demo.contract import encode_request

    if detections is not None:
        payload = encode_request(None, audio, 16000, detections=detections)
    elif frame_png is not None:
        payload = encode_request(frame_png, audio, 16000)   # fallback CPU
    else:
        # neither frame nor detections are ready — don't send a bare
        # request with no picture at all; an explicit empty detections list
        # at least tells the server that vision was attempted
        payload = encode_request(None, audio, 16000, detections=[])
    events = _http_stream(endpoint + "_stream", payload)
    done = drive_robot_stream(
        robot, events,
        on_meta=lambda m: display.on_heard(m.get("heard", "")),
        make_player=make_player,
        on_reply=display.on_reply,
        robot_guard=robot_guard,
        audio_guard=audio_guard)
    if done and done.get("first_sound_ms"):
        print(f"  first sound: {done['first_sound_ms']:.0f} ms")


def run_voice(args, endpoint) -> int:
    """Listen and watch continuously, respond to voice.

    Camera and microphone are both open the whole time: the camera is
    always warmed up (no black first frame) and doesn't fight over the
    device between grabs.

    The platform (`--platform mac|reachy`) provides camera/mic/player/robot
    behind a common interface (see demo/platform/) — the loop doesn't know
    whether this is ffmpeg on the Mac or the reachy_mini SDK on the robot
    itself. The detections source (GpuDetectSource) always runs; the
    display is pluggable (`--display web|none`): the loop doesn't branch on
    whether it's present, it just reads `source.latest()` and sends events
    to `display`.
    Constructing platform/source/display happens entirely inside
    try/finally, so an early failure (server bind, camera warmup,
    calibration) doesn't leave platform subprocesses/devices dangling.
    """
    platform = None
    source = None
    display = None
    try:
        platform = build_platform(args.platform, args)
        camera = platform.video_source()
        mic = platform.mic_source()
        robot = platform.robot()
        detect_url = f"http://{args.pi}:{args.gpu_detect_port}/detect"
        source = GpuDetectSource(camera, detect_url)
        display = build_display(args.display, camera=camera,
                                host="127.0.0.1", port=args.web_port)
        source.add_listener(display.on_detections)
        source.start()
        if args.display != "none":
            print(f"  web UI: http://127.0.0.1:{args.web_port}")
        # Wait for the first camera frame before entering the loop. Do NOT
        # fall back to grab_frame on None: it would open a second ffmpeg on
        # the same camera, and both would hang fighting over the device.
        print("  warming up camera...")
        for _ in range(30):
            if camera.latest() is not None:
                break
            time.sleep(0.2)
        if camera.latest() is None:
            print("  WARNING: no camera frame in 6s — is another process "
                  "holding the camera?")
        print("  calibrating background noise (stay quiet ~1s)...")
        threshold = calibrate_threshold(mic.chunks(), seconds=1.0,
                                        multiplier=args.sensitivity)
        print(f"  ready. Speech threshold: {threshold:.3f}. "
              f"Just talk — Ctrl-C to quit.\n")
        # One guard for the whole session (not per response) — repeated
        # robot or audio failures get logged once, not on every phrase.
        robot_guard = _FailureGuard("robot")
        audio_guard = _FailureGuard("audio")
        # Health of the vision path, surfaced once per transition below. The
        # signals (GpuDetectSource.healthy(), CameraStream.alive) exist for
        # exactly this — without reading them a sustained GPU/camera outage is
        # invisible beyond the detect thread's per-cycle warnings.
        detect_healthy = True
        camera_alive = True
        while True:
            gate = VoiceGate(threshold=threshold)
            print("  listening...")
            audio = collect_utterance(mic.chunks(), gate,
                                      max_chunks=int(args.max_utterance * 10))
            if audio is None:
                break
            print(f"  heard speech ({len(audio) / 16000:.1f} s), looking...")
            frame, detections = source.latest()
            healthy = source.healthy()
            if healthy != detect_healthy:
                detect_healthy = healthy
                print("  vision: GPU detect "
                      + ("recovered" if detect_healthy
                         else "unreachable — the Pi will detect on CPU this turn"))
            camera_now = getattr(camera, "alive", True)
            if camera_now != camera_alive:
                camera_alive = camera_now
                print("  camera: " + ("recovered" if camera_alive
                                      else "stopped delivering frames"))
            frame_png = frame_to_png(frame) if frame is not None else None
            display.on_query(frame, detections)
            # When GPU detect is down, send detections=None (not the stale [])
            # so the Pi runs its OWN CPU detector on the frame — the robot keeps
            # seeing during the outage instead of reporting an empty scene.
            stream_detections = detections if healthy else None
            try:
                _handle_stream(frame_png, stream_detections, audio, endpoint,
                               robot, display, make_player=platform.make_player,
                               robot_guard=robot_guard, audio_guard=audio_guard)
            except (OSError, http.client.HTTPException,
                    json.JSONDecodeError) as exc:
                # A transient Pi/network fault must skip this turn, not crash
                # the always-on loop — every other failure path here degrades
                # gracefully, so this one does too. http.client.HTTPException
                # covers a stream truncated mid-body (IncompleteRead) after the
                # 200 headers arrived — an OSError alone would miss that, which
                # is exactly the drop this guard is meant to survive; a malformed
                # NDJSON line raises json.JSONDecodeError.
                print(f"  Pi unreachable — skipping this turn "
                      f"({type(exc).__name__})")
            mic.flush()   # discard the robot's own voice from the buffer
            print()
    except KeyboardInterrupt:
        print("\n  bye")
    finally:
        # Stop the detect source BEFORE closing the display: the source fans
        # out every cycle to display.on_detections, so tearing the display
        # down first could deliver an in-flight cycle to an already-closed
        # sink. Harmless today (the listener guard swallows it), but stopping
        # the producer before its consumer is correct ordering, not just safe.
        if source is not None:
            source.stop()
        if display is not None:
            display.close()
        if platform is not None:
            platform.close()
    return 0


def run_enter(args, endpoint, robot) -> int:
    """Respond on Enter keypress — debug mode.

    grab_frame/record_audio are NOT wrapped in try/except (we
    document rather than guard): these are one-shot ffmpeg calls that
    already have timeout= (see demo/platform/mac.py) — the interactive
    --enter mode is the case where a crash with a traceback is more useful
    than a silent skip (you immediately see what's wrong with the
    camera/mic), unlike the continuous run_voice.
    """
    print("  press Enter to speak, Ctrl-C to quit.")
    robot_guard = _FailureGuard("robot")
    audio_guard = _FailureGuard("audio")
    while True:
        try:
            input("\n[Enter] talk> ")
        except (EOFError, KeyboardInterrupt):
            print("\n  bye")
            return 0
        print("  capturing frame and listening...")
        frame = grab_frame(args.video)
        audio = record_audio(args.record_seconds, args.audio)
        _handle(frame, audio, endpoint, robot, stream=not args.no_stream,
               robot_guard=robot_guard, audio_guard=audio_guard)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Voice robot demo")
    p.add_argument("--pi", default="raspberrypi.local",
                   help="host of the Pi running serve.py + gpu_detect")
    p.add_argument("--port", type=int, default=9500)
    p.add_argument("--video", default="0")
    p.add_argument("--audio", default="1")
    p.add_argument("--record-seconds", type=float, default=5.0,
                   help="recording length in --enter mode")
    p.add_argument("--max-utterance", type=float, default=8.0,
                   help="max utterance length in voice mode, seconds")
    p.add_argument("--sensitivity", type=float, default=3.0,
                   help="how many times the threshold is above background "
                        "noise; lower — more sensitive")
    p.add_argument("--enter", action="store_true",
                   help="trigger on Enter instead of voice")
    p.add_argument("--no-stream", action="store_true",
                   help="wait for the full response instead of streaming by phrase")
    p.add_argument("--save-frames", default=None, metavar="DIR",
                   help="save every frame and yolo detections to a folder")
    p.add_argument("--no-robot", action="store_true",
                   help="no Reachy: turn logged to console, sound via afplay")
    p.add_argument("--web-port", type=int, default=8080,
                   help="web UI port on the Mac")
    p.add_argument("--gpu-detect-port", type=int, default=9600,
                   help="GPU detect service port on the Pi")
    p.add_argument("--display", choices=["web", "none"], default="web",
                   help="voice loop display: web UI or no screen")
    p.add_argument("--platform", choices=["mac", "reachy"], default="mac",
                   help="I/O platform: ffmpeg/afplay on the Mac or the "
                        "reachy_mini SDK directly on the robot")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    endpoint = f"http://{args.pi}:{args.port}/respond"
    print(f"demo ready. Pi: {endpoint}")
    if args.enter:
        # --enter mode is a debug trigger on Enter, always via the Mac's
        # ffmpeg tools (see demo/platform/mac.py), not via --platform:
        # one-shot grab_frame/record_audio aren't part of the Platform
        # protocol (that's about continuous VideoSource/MicSource).
        robot = make_robot(use_robot=not args.no_robot)
        return run_enter(args, endpoint, robot)
    return run_voice(args, endpoint)


if __name__ == "__main__":
    raise SystemExit(main())
