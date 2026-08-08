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

"""Codec for the HTTP contract between the Mac and the Pi.

Pure functions, kept separate from the network: both the server and the
client depend on one shared format, and it can be tested without a Pi.
Audio travels as base64-encoded float32 — JSON can't carry binary, and
shipping a WAV container over HTTP would be overkill.
"""

from __future__ import annotations

import base64
import dataclasses

import numpy as np


# Explicit little-endian dtype (`<f4`), not bare `np.float32`: numpy's native
# float32 follows host byte order, which happens to be LE on most of our
# hosts (Pi, Mac — arm64), but the HTTP contract outlives any given host —
# an explicit byte order makes the codec portable, rather than "works because
# both ends currently happen to be LE".
_AUDIO_DTYPE = "<f4"


def _encode_audio(audio: np.ndarray) -> str:
    return base64.b64encode(
        np.asarray(audio, dtype=_AUDIO_DTYPE).tobytes()).decode("ascii")


def _decode_audio(text: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(text), dtype=_AUDIO_DTYPE).copy()


def encode_request(frame_png: bytes | None, audio: np.ndarray,
                   sample_rate: int, detections: list | None = None) -> dict:
    payload = {
        "audio_b64": _encode_audio(audio),
        "sample_rate": int(sample_rate),
    }
    if frame_png is not None:
        payload["frame_b64"] = base64.b64encode(frame_png).decode("ascii")
    if detections is not None:
        payload["detections"] = detections
    return payload


def decode_request(payload: dict) -> tuple[bytes | None, np.ndarray, int,
                                           list | None]:
    frame_b64 = payload.get("frame_b64")
    frame = base64.b64decode(frame_b64) if frame_b64 else None
    audio = _decode_audio(payload["audio_b64"])
    return frame, audio, int(payload["sample_rate"]), payload.get("detections")


def encode_response(result: dict, audio: np.ndarray, sample_rate: int,
                    timings: dict, total_ms: float) -> dict:
    gaze = tuple(float(v) for v in result.get("gaze", (0.0, 0.0)))
    return {
        "objects": list(result.get("objects", [])),
        "heard": result.get("heard", ""),
        "reply": result.get("reply", ""),
        "spoken": list(result.get("spoken", [])),
        "gaze": [gaze[0], gaze[1]],
        "audio_b64": _encode_audio(audio),
        "sample_rate": int(sample_rate),
        "timings": {k: float(v) for k, v in timings.items()},
        "total_ms": float(total_ms),
    }


# eq=False: this carries an np.ndarray (+ list/dict). The auto-generated __eq__
# would do an elementwise array compare (ValueError: ambiguous truth value), and
# a frozen dataclass with the default eq=True would also build an __hash__ over
# those unhashable fields. With eq=False the dataclass keeps plain object
# identity for __eq__/__hash__ — fine here, since it's a decoded data carrier
# that's never compared or hashed.
@dataclasses.dataclass(frozen=True, eq=False)
class DemoResponse:
    objects: list[str]
    heard: str
    reply: str
    spoken: list[str]
    gaze: tuple[float, float]
    audio: np.ndarray
    sample_rate: int
    timings: dict
    total_ms: float


def decode_response(payload: dict) -> DemoResponse:
    g = payload.get("gaze", [0.0, 0.0])
    return DemoResponse(
        objects=list(payload.get("objects", [])),
        heard=payload.get("heard", ""),
        reply=payload.get("reply", ""),
        spoken=list(payload.get("spoken", [])),
        gaze=(float(g[0]), float(g[1])),
        audio=_decode_audio(payload["audio_b64"]),
        sample_rate=int(payload["sample_rate"]),
        timings=dict(payload.get("timings", {})),
        total_ms=float(payload.get("total_ms", 0.0)),
    )
