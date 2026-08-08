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

from emulator.detector import Detection
from demo.detections import detections_to_dicts, gaze_from_dicts


def test_detections_to_dicts_uses_names_and_xyxy():
    got = detections_to_dicts([Detection(label=0, score=0.912,
                                          box=(0.1, 0.2, 0.3, 0.4))])
    assert got == [{"label": "person", "score": 0.912,
                    "box": [0.1, 0.2, 0.3, 0.4]}]


def test_gaze_from_dicts_picks_most_confident_center():
    dets = [{"label": "person", "score": 0.4, "box": [0.0, 0.0, 0.2, 0.2]},
            {"label": "cup", "score": 0.9, "box": [0.4, 0.4, 0.6, 0.6]}]
    # center (0.5, 0.5) -> (0.5+0.5)-1, (0.5+0.5)-1 = (0.0, 0.0)
    assert gaze_from_dicts(dets) == (0.0, 0.0)


def test_gaze_from_dicts_empty_is_forward():
    assert gaze_from_dicts([]) == (0.0, 0.0)
