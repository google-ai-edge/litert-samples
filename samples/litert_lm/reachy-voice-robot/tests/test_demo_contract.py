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
import pytest

from demo.contract import (
    DemoResponse,
    decode_request,
    decode_response,
    encode_request,
    encode_response,
)


def test_request_roundtrip_preserves_audio():
    audio = np.linspace(-1, 1, 8000, dtype=np.float32)
    payload = encode_request(b"\x89PNG-fake", audio, 16000)
    frame, got_audio, sr, _ = decode_request(payload)
    assert frame == b"\x89PNG-fake"
    assert sr == 16000
    assert np.allclose(got_audio, audio, atol=1e-4)


def test_request_carries_detections_without_frame():
    dets = [{"label": "person", "score": 0.9, "box": [0.1, 0.2, 0.3, 0.4]}]
    payload = encode_request(None, np.zeros(10, np.float32), 16000, detections=dets)
    frame, audio, sr, got = decode_request(payload)
    assert frame is None
    assert got == dets
    assert sr == 16000


def test_request_frame_only_backward_compatible():
    payload = encode_request(b"png", np.zeros(10, np.float32), 16000)
    frame, audio, sr, got = decode_request(payload)
    assert frame == b"png"
    assert got is None


def test_request_payload_is_json_safe():
    # JSON goes out over HTTP — there must be no values outside str/num/list/dict.
    import json
    payload = encode_request(b"png", np.zeros(10, np.float32), 16000)
    json.dumps(payload)  # must not raise


def test_response_roundtrip():
    audio = np.linspace(-0.5, 0.5, 4000, dtype=np.float32)
    result = {"objects": ["person"], "heard": "hi", "reply": "I see you.",
              "spoken": ["I see you."], "gaze": (0.3, -0.2)}
    payload = encode_response(result, audio, 16000,
                              {"detector": 62.0, "tts_call": 147.0}, 2446.0)
    got = decode_response(payload)
    assert isinstance(got, DemoResponse)
    assert got.objects == ["person"]
    assert got.heard == "hi"
    assert got.reply == "I see you."
    assert got.gaze == (0.3, -0.2)
    assert got.total_ms == 2446.0
    assert got.timings["detector"] == 62.0
    assert np.allclose(got.audio, audio, atol=1e-4)


def test_response_handles_empty_audio():
    # The robot may have skipped a phrase — audio is empty but the response is still valid.
    result = {"objects": [], "heard": "", "reply": "", "spoken": [],
              "gaze": (0.0, 0.0)}
    payload = encode_response(result, np.zeros(0, np.float32), 16000, {}, 0.0)
    got = decode_response(payload)
    assert len(got.audio) == 0
    assert got.gaze == (0.0, 0.0)


def test_gaze_is_always_a_two_tuple_of_floats():
    result = {"objects": [], "heard": "", "reply": "", "spoken": [],
              "gaze": [0.1, 0.2]}  # list on the way in
    got = decode_response(encode_response(result, np.zeros(4, np.float32),
                                          16000, {}, 1.0))
    assert isinstance(got.gaze, tuple)
    assert len(got.gaze) == 2
    assert all(isinstance(v, float) for v in got.gaze)
