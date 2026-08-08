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

"""Pure detection conversions: detector objects <-> JSON dicts.

The dict format matches the `detail` field that serve.py already returns:
class name (not int), score, xyxy box in frame fractions. Gaze is computed
from the dicts the same way gaze_target is computed from Detection objects.
"""
from __future__ import annotations

from emulator.detector import Detection
from emulator.pipeline import label_name


def detections_to_dicts(detections: list[Detection]) -> list[dict]:
    return [
        {"label": label_name(d.label), "score": round(float(d.score), 3),
         "box": [round(float(v), 3) for v in d.box]}
        for d in detections
    ]


def gaze_from_dicts(detections: list[dict]) -> tuple[float, float]:
    if not detections:
        return (0.0, 0.0)
    best = max(detections, key=lambda d: d["score"])
    x1, y1, x2, y2 = best["box"]
    return ((x1 + x2) - 1.0, (y1 + y2) - 1.0)
