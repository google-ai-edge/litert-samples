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

"""Display package: the DisplaySink protocol, a no-op stub, and a factory.

The voice loop sends events here without knowing where they end up — a
browser (`WebDashboard`) or nowhere (`NullSink`). With no display the methods
simply do nothing, so the loop never has to branch on whether one is present.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class DisplaySink(Protocol):
    """Presentation interface: the voice loop sends events here."""

    def on_detections(self, detections: list[dict]) -> None:
        """Listener for GpuDetectSource — fresh boxes on every detect cycle."""
        ...

    def on_query(self, frame: np.ndarray | None, detections: list[dict]) -> None:
        """Frame and detections that went into the prompt for this turn."""
        ...

    def on_heard(self, text: str) -> None:
        ...

    def on_reply(self, text: str, done: bool = False) -> None:
        ...

    def close(self) -> None:
        ...


class NullSink:
    """No-op display: without a screen, the loop runs without a single `if display`."""

    def on_detections(self, detections: list[dict]) -> None:
        pass

    def on_query(self, frame: np.ndarray | None, detections: list[dict]) -> None:
        pass

    def on_heard(self, text: str) -> None:
        pass

    def on_reply(self, text: str, done: bool = False) -> None:
        pass

    def close(self) -> None:
        pass


def build_display(kind: str, *, camera, host: str, port: int) -> DisplaySink:
    """Display factory keyed by the CLI name (`--display {web,none}`)."""
    if kind == "none":
        return NullSink()
    if kind == "web":
        from demo.display.web import WebDashboard

        dash = WebDashboard(camera)   # MJPEG reads the camera directly
        dash.serve(host=host, port=port)
        return dash
    raise ValueError(f"unknown display {kind!r}")
