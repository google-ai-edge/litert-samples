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

"""Pure frame/PCM conversions, shared across all platform backends.

None of these functions touch ffmpeg, a real microphone, or the robot SDK —
so this file is safe to import without ffmpeg or `reachy_mini` installed
(used from mac.py, from reachy.py, and directly in tests).
"""

from __future__ import annotations

import io
import re

import numpy as np

_DEVICE_RE = re.compile(r"\[(\d+)\]\s+(.+)$")


def parse_avfoundation_devices(text: str) -> dict:
    result = {"video": {}, "audio": {}}
    section = None
    for line in text.splitlines():
        low = line.lower()
        if "video devices" in low:
            section = "video"
            continue
        if "audio devices" in low:
            section = "audio"
            continue
        if section is None:
            continue
        m = _DEVICE_RE.search(line)
        if m:
            result[section][m.group(1)] = m.group(2).strip()
    return result


def pcm_bytes_to_float(raw: bytes) -> np.ndarray:
    usable = len(raw) - (len(raw) % 2)
    data = np.frombuffer(raw[:usable], dtype="<i2").astype(np.float32)
    return data / 32768.0


def raw_to_frame(raw: bytes, width: int, height: int) -> np.ndarray | None:
    """Raw rgb24 frame into an [H, W, 3] array. None if there aren't enough bytes for a full frame."""
    need = width * height * 3
    if len(raw) < need:
        return None
    return np.frombuffer(raw[:need], dtype=np.uint8).reshape(height, width, 3)


def frame_to_png(frame: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="PNG")
    return buf.getvalue()
