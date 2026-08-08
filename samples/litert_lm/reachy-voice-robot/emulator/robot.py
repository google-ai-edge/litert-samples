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

"""Robot stub with the Reachy SDK interface.

There's no robot yet, and once there is, this gets swapped for a real
ReachyMini with the same calls. The stub doesn't just log calls: it mirrors
the hardware's constraints so behavior doesn't drift when we move over. The
real robot has a limit on head rotation, so coordinates are clamped here too.
"""

from __future__ import annotations

import numpy as np


class RobotStub:
    def __init__(self) -> None:
        self._events: list[tuple[str, dict]] = []

    def look_at(self, x: float, y: float) -> None:
        self._events.append(("look_at", {
            "x": max(-1.0, min(1.0, float(x))),
            "y": max(-1.0, min(1.0, float(y))),
        }))

    def say(self, wave: np.ndarray, sample_rate: int) -> None:
        self._events.append(("say", {"duration_s": len(wave) / sample_rate}))

    def total_speech_s(self) -> float:
        """Total time the robot has spoken — a metric to check against the budget."""
        return sum(payload["duration_s"]
                   for name, payload in self._events if name == "say")

    def events(self) -> list[tuple[str, dict]]:
        return list(self._events)
