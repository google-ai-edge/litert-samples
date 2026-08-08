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

from demo.robot_reachy import GESTURE_POSES, ConsoleRobot, gaze_to_head_pose


def test_centre_gaze_is_zero_angles():
    pose = gaze_to_head_pose(0.0, 0.0)
    assert abs(pose["yaw"]) < 1e-6
    assert abs(pose["pitch"]) < 1e-6


def test_right_gaze_turns_yaw():
    # x > 0 (object on the right) -> head turns right (yaw != 0).
    pose = gaze_to_head_pose(1.0, 0.0, amplitude_deg=20.0)
    assert pose["yaw"] == 20.0
    assert abs(pose["pitch"]) < 1e-6


def test_gaze_is_clamped():
    # Gaze outside [-1,1] must not produce an angle beyond the amplitude.
    pose = gaze_to_head_pose(5.0, -5.0, amplitude_deg=20.0)
    assert pose["yaw"] == 20.0
    assert pose["pitch"] == -20.0


def test_console_robot_records_and_plays(monkeypatch):
    calls = []
    monkeypatch.setattr("demo.robot_reachy.play_wav",
                        lambda audio, sample_rate: calls.append(sample_rate))
    robot = ConsoleRobot()
    robot.look_at(0.3, -0.2)
    robot.say(np.zeros(16000, dtype=np.float32), sample_rate=16000)
    assert robot.events[-2][0] == "look_at"
    assert robot.events[-1][0] == "say"
    assert calls, "sound should go out through afplay"


def test_console_robot_skips_empty_audio(monkeypatch):
    calls = []
    monkeypatch.setattr("demo.robot_reachy.play_wav",
                        lambda audio, sample_rate: calls.append(sample_rate))
    ConsoleRobot().say(np.zeros(0, dtype=np.float32), sample_rate=16000)
    assert not calls, "empty audio should not be played"


def test_console_robot_records_gesture():
    robot = ConsoleRobot()
    robot.gesture("shake")
    assert robot.events[-1] == ("gesture", {"name": "shake"})


def test_console_robot_ignores_unknown_gesture():
    # An unknown gesture must not crash the demo: record the event, no motion.
    robot = ConsoleRobot()
    robot.gesture("moonwalk")
    assert robot.events[-1] == ("gesture", {"name": "moonwalk"})


def test_gesture_poses_return_head_to_centre():
    # Every gesture ends at (0, 0), otherwise the head stays twisted.
    for name, poses in GESTURE_POSES.items():
        assert poses[-1] == (0, 0), f"{name} does not return the head to center"
