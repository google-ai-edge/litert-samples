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

from demo.platform.pcm import parse_avfoundation_devices, pcm_bytes_to_float


SAMPLE = """[AVFoundation indev @ 0x1] AVFoundation video devices:
[AVFoundation indev @ 0x1] [0] MacBook Pro Camera
[AVFoundation indev @ 0x1] [1] Desk View Camera
[AVFoundation indev @ 0x1] AVFoundation audio devices:
[AVFoundation indev @ 0x1] [0] iPhone Microphone
[AVFoundation indev @ 0x1] [1] MacBook Pro Microphone
"""


def test_parse_lists_video_and_audio():
    d = parse_avfoundation_devices(SAMPLE)
    assert d["video"]["0"] == "MacBook Pro Camera"
    assert d["audio"]["1"] == "MacBook Pro Microphone"
    assert len(d["video"]) == 2
    assert len(d["audio"]) == 2


def test_parse_handles_no_devices():
    d = parse_avfoundation_devices("no devices here")
    assert d == {"video": {}, "audio": {}}


def test_pcm_s16le_to_float_normalises():
    raw = np.array([32767, -32768, 0], dtype="<i2").tobytes()
    out = pcm_bytes_to_float(raw)
    assert out.dtype == np.float32
    assert 0.99 < out[0] <= 1.0
    assert -1.0 <= out[1] < -0.99
    assert out[2] == 0.0


def test_pcm_handles_odd_byte_count():
    # A truncated last sample shouldn't crash — the trailing byte is dropped.
    raw = np.array([100, 200], dtype="<i2").tobytes() + b"\x01"
    out = pcm_bytes_to_float(raw)
    assert len(out) == 2


# --- continuous camera stream ---

def test_raw_to_frame_reshapes():
    import numpy as np
    from demo.platform.pcm import raw_to_frame
    raw = bytes(range(256)) * (4 * 4 * 3 // 256 + 1)
    frame = raw_to_frame(raw, 4, 4)
    assert frame.shape == (4, 4, 3)
    assert frame.dtype == np.uint8


def test_raw_to_frame_incomplete_is_none():
    from demo.platform.pcm import raw_to_frame
    # Less than a full frame — don't assemble half a frame.
    assert raw_to_frame(b"\x00\x01\x02", 4, 4) is None


def test_frame_to_png_roundtrips():
    import io
    import numpy as np
    from PIL import Image
    from demo.platform.pcm import frame_to_png
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[2, 3] = [255, 128, 0]
    png = frame_to_png(frame)
    back = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
    assert back.shape == (8, 8, 3)
    assert list(back[2, 3]) == [255, 128, 0]
