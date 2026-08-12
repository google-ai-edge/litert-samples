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

"""The robot stub must mirror hardware CONSTRAINTS, not just the calls.

If the stub accepts any coordinates while the real robot clamps them,
behavior will drift once we move to hardware — and drift silently.
"""

import numpy as np

from emulator.robot import RobotStub


def test_stub_records_look_at():
    robot = RobotStub()
    robot.look_at(0.3, -0.2)
    assert robot.events() == [("look_at", {"x": 0.3, "y": -0.2})]


def test_stub_records_speech_duration():
    robot = RobotStub()
    robot.say(np.zeros(48000, dtype=np.float32), sample_rate=24000)
    name, payload = robot.events()[0]
    assert name == "say"
    assert payload["duration_s"] == 2.0


def test_stub_clamps_look_at_to_valid_range():
    # The real robot has a limit on head rotation; the stub must behave the
    # same way, or behavior will drift on hardware.
    robot = RobotStub()
    robot.look_at(5.0, -5.0)
    payload = robot.events()[0][1]
    assert -1.0 <= payload["x"] <= 1.0
    assert -1.0 <= payload["y"] <= 1.0


def test_stub_accumulates_events_in_order():
    robot = RobotStub()
    robot.look_at(0.0, 0.0)
    robot.say(np.zeros(24000, dtype=np.float32), sample_rate=24000)
    robot.look_at(0.5, 0.5)
    assert [name for name, _ in robot.events()] == ["look_at", "say", "look_at"]


def test_stub_reports_total_speech_time():
    # Total time the robot has spoken is a budget metric: if synthesis can't
    # keep up, total speech will come out shorter than the total tally.
    robot = RobotStub()
    robot.say(np.zeros(24000, dtype=np.float32), sample_rate=24000)
    robot.say(np.zeros(12000, dtype=np.float32), sample_rate=24000)
    assert robot.total_speech_s() == 1.5


def test_events_are_a_copy_not_the_internal_list():
    robot = RobotStub()
    robot.look_at(0.0, 0.0)
    robot.events().clear()
    assert len(robot.events()) == 1, "internal list must not leak"
