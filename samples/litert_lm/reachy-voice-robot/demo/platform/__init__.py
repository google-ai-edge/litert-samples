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

"""Platform I/O seam: camera, mic, speaker, and robot behind protocols.

The same voice loop (`demo/run_demo.py::run_voice`) runs either on a Mac
(ffmpeg avfoundation + afplay + MuJoCo-sim/console robot) or entirely on a
Reachy Mini itself (SDK `reachy_mini`, a single handle for camera, mic,
speaker, and motors). Platform I/O used to be hardwired into run_demo.py
directly for macOS; now it's hidden behind four protocols and the
`build_platform` factory.

Platform-independent pure logic (RMS, `VoiceGate`, `collect_utterance`,
`calibrate_threshold`) stays in `demo/vad.py`; pure frame and PCM
conversions live in `demo/platform/pcm.py`.
"""
from __future__ import annotations

from typing import Iterator, Protocol

import numpy as np


class VideoSource(Protocol):
    """Camera frame source — what `CameraStream` already did."""

    # True while the source is delivering fresh frames; run_voice reads this to
    # surface a camera outage. Both concrete sources (CameraStream and the
    # robot's _ReachyVideoSource) provide it.
    alive: bool

    def latest(self) -> np.ndarray | None: ...

    def latest_png(self) -> bytes | None: ...

    def close(self) -> None: ...


class MicSource(Protocol):
    """Mic audio source — what `MicStream` already did.

    `chunks()` — a generator of (float32 chunk, its RMS) pairs, same as
    before: this is exactly the contract consumed by
    `collect_utterance`/`calibrate_threshold` in `demo/vad.py`.
    """

    def chunks(self) -> Iterator[tuple[np.ndarray, float]]: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


class Player(Protocol):
    """Reply player — what `StreamPlayer` already did."""

    def feed(self, samples: np.ndarray) -> None: ...

    def close(self) -> None: ...


class Robot(Protocol):
    """Demo robot — what `ConsoleRobot`/`ReachyRobot` already did."""

    def look_at(self, x: float, y: float) -> None: ...

    def say(self, wave_data: np.ndarray, sample_rate: int) -> None: ...

    def gesture(self, name: str) -> None: ...


class Platform(Protocol):
    """Platform backend: bundles camera/mic/player/robot under one node."""

    def video_source(self) -> VideoSource: ...

    def mic_source(self) -> MicSource: ...

    def make_player(self, sample_rate: int) -> Player: ...

    def robot(self) -> Robot: ...

    def close(self) -> None: ...


def build_platform(kind: str, args) -> Platform:
    """Platform factory keyed by the CLI name (`--platform {mac,reachy}`).

    `reachy_mini` is a heavy hardware dependency of the `reachy` backend, so
    it's imported LAZILY inside `demo/platform/reachy.py`, only once that
    backend is actually selected: at module IMPORT time, the Mac (`mac`)
    path doesn't pull it in (verified by a smoke test with the package not
    installed). NB: at runtime, `MacPlatform.robot()` without `--no-robot`
    still creates a `ReachyRobot` (MuJoCo-sim), which does import
    `reachy_mini` — that's a separate, pre-refactor code path.
    """
    if kind == "mac":
        from demo.platform.mac import MacPlatform

        return MacPlatform(args)
    if kind == "reachy":
        from demo.platform.reachy import ReachyPlatform

        return ReachyPlatform()
    raise ValueError(f"unknown platform {kind!r}")
