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

"""Demo robot: a real Reachy Mini in simulation, or a console stub.

RobotStub from emulator/ deliberately mirrored the Reachy interface — it pays
off here. gaze_to_head_pose converts the pipeline's gaze into head angles;
ConsoleRobot exists to debug the demo without the SDK installed (--no-robot).
"""

from __future__ import annotations

import numpy as np

from demo.audio_out import play_wav


def gaze_to_head_pose(x: float, y: float,
                      amplitude_deg: float = 20.0) -> dict:
    """Gaze [-1, 1] -> head angles in degrees. Positive x -> positive (right) yaw."""
    cx = max(-1.0, min(1.0, float(x)))
    cy = max(-1.0, min(1.0, float(y)))
    return {"yaw": cx * amplitude_deg, "pitch": cy * amplitude_deg}


# Head gestures as sequences of (yaw, pitch) poses in degrees. Yaw sign matches
# gaze_to_head_pose (positive yaw is right). shake swings yaw, nod swings
# pitch, look_* turns and returns to center. The final pose (0, 0) brings the
# head back to straight so the gesture doesn't leave it twisted.
GESTURE_POSES: dict[str, list[tuple[float, float]]] = {
    "shake": [(-18, 0), (18, 0), (-18, 0), (18, 0), (0, 0)],
    "nod": [(0, 15), (0, -8), (0, 15), (0, 0)],
    "look_left": [(-25, 0), (-25, 0), (0, 0)],
    "look_right": [(25, 0), (25, 0), (0, 0)],
    "look_up": [(0, -20), (0, -20), (0, 0)],
    "look_down": [(0, 20), (0, 20), (0, 0)],
}


class ConsoleRobot:
    """Prints the turn, plays sound via afplay. For --no-robot."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def look_at(self, x: float, y: float) -> None:
        pose = gaze_to_head_pose(x, y)
        self.events.append(("look_at", pose))
        print(f"  [robot] looks: yaw {pose['yaw']:+.0f}°, "
              f"pitch {pose['pitch']:+.0f}°")

    def say(self, wave_data: np.ndarray, sample_rate: int) -> None:
        # Non-default/fallback path: the live voice loop plays audio through
        # StreamPlayer (whole-reply, one clip — see demo/audio_out.py);
        # Robot.say() is only exercised by --no-stream / drive_robot().
        self.events.append(("say", {"n": len(wave_data)}))
        if len(wave_data) == 0:
            return
        play_wav(wave_data, sample_rate)

    def gesture(self, name: str) -> None:
        self.events.append(("gesture", {"name": name}))
        poses = GESTURE_POSES.get(name)
        if poses is None:
            return
        print(f"  [robot] gesture {name}: " +
              " ".join(f"({y:+.0f},{p:+.0f})" for y, p in poses))


class ReachyRobot:
    """A real Reachy Mini in MuJoCo simulation.

    Connects to an ALREADY-RUNNING daemon rather than spawning its own. Two
    "whys", both worked out the hard way on a live run:

    - `spawn_daemon=False`: on macOS the MuJoCo window only opens under
      `mjpython` (`launch_passive` needs the GUI main thread). The plain
      python process running the demo can't open the window itself. So the
      daemon is launched separately under mjpython, and the client attaches
      to it:
        `mjpython -m reachy_mini.daemon.app.main --sim --no-media`

    - `media_backend="no_media"`: by default the client spins up a media
      pipeline (robot camera/audio over GStreamer), which hangs dead in C
      code on the Mac. We don't need the robot's media — the Mac's own webcam
      handles video and afplay handles audio.
    """

    def __init__(self) -> None:
        from reachy_mini import ReachyMini

        self._mini = ReachyMini(spawn_daemon=False, media_backend="no_media")

    def look_at(self, x: float, y: float) -> None:
        from reachy_mini.utils import create_head_pose

        pose = gaze_to_head_pose(x, y)
        self._mini.goto_target(
            head=create_head_pose(yaw=pose["yaw"], pitch=pose["pitch"],
                                  degrees=True),
            duration=0.5)

    def say(self, wave_data: np.ndarray, sample_rate: int) -> None:
        # Non-default/fallback path: same caveat as ConsoleRobot.say() — the
        # live voice loop uses StreamPlayer, this only fires on --no-stream
        # / drive_robot().
        if len(wave_data) == 0:
            return
        # Same playback path as ConsoleRobot; the sim renders the motion, sound plays locally.
        play_wav(wave_data, sample_rate)

    def gesture(self, name: str) -> None:
        """Play a head gesture as a sequence of goto_target calls.

        goto_target blocks for duration, so poses play out one after another —
        in simulation this reads as a swing/turn. Unknown gestures are ignored.
        """
        from reachy_mini.utils import create_head_pose

        poses = GESTURE_POSES.get(name)
        if poses is None:
            return
        for yaw, pitch in poses:
            self._mini.goto_target(
                head=create_head_pose(yaw=yaw, pitch=pitch, degrees=True),
                duration=0.25)


def make_robot(use_robot: bool):
    return ReachyRobot() if use_robot else ConsoleRobot()
