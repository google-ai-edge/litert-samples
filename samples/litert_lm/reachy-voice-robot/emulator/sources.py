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

"""Frame and audio sources.

File-based sources come first, with camera and microphone added later,
because reproducibility matters more than liveness: two runs on the same
data can be compared, while two runs off a live camera cannot.

A note on length: models require an exact input shape (moonshine needs
exactly 80000 samples), so a short file is padded with silence rather than
returning a short buffer. The wrong sample rate, on the other hand, is
rejected rather than silently resampled: a wrong length fails loudly, while
a wrong rate would produce plausible-looking nonsense and push debugging
into the model instead of the loader.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Protocol

import numpy as np


class FrameSource(Protocol):
    def read(self) -> np.ndarray | None:
        """The next frame, or None if the source is exhausted."""


class AudioSource(Protocol):
    def read(self, n_samples: int) -> np.ndarray:
        """Exactly n_samples float32 samples in the [-1, 1] range."""


class StillFrameSource:
    """The same synthetic frame every time. For latency benchmarks."""

    def __init__(self, shape: tuple[int, int, int], seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self._frame = rng.random(shape, dtype=np.float32)

    def read(self) -> np.ndarray:
        return self._frame


class VideoFileFrameSource:
    """Frames from a video file, resized to the required shape.

    Requires opencv-python. Installed separately: it's heavy on the Pi, and
    StillFrameSource is enough for latency benchmarks.
    """

    def __init__(self, path: Path, shape: tuple[int, int, int]) -> None:
        import cv2

        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise ValueError(f"failed to open video: {path}")
        self._shape = shape
        self._cv2 = cv2

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        if not ok:
            return None
        height, width, _ = self._shape
        frame = self._cv2.resize(frame, (width, height))
        frame = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        return frame.astype(np.float32) / 255.0

    def close(self) -> None:
        """Release the OpenCV capture handle (file/device) — otherwise it
        leaks for the lifetime of a long emulator run."""
        self._cap.release()

    def __enter__(self) -> "VideoFileFrameSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class SilenceAudioSource:
    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate

    def read(self, n_samples: int) -> np.ndarray:
        return np.zeros(n_samples, dtype=np.float32)


class WavAudioSource:
    """Mono WAV, normalized to [-1, 1], padded with silence."""

    def __init__(self, path: Path, sample_rate: int = 16000) -> None:
        with wave.open(str(path), "rb") as fh:
            if fh.getframerate() != sample_rate:
                raise ValueError(
                    f"expected {sample_rate} Hz, file has {fh.getframerate()}. "
                    "Silent resampling is not allowed: the model would produce "
                    "plausible-looking nonsense instead of an error"
                )
            if fh.getsampwidth() != 2:
                raise ValueError("expected 16-bit PCM")
            raw = fh.readframes(fh.getnframes())
            channels = fh.getnchannels()

        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)
        self._data = data
        self._pos = 0
        self.sample_rate = sample_rate

    def read(self, n_samples: int) -> np.ndarray:
        chunk = self._data[self._pos:self._pos + n_samples]
        self._pos += n_samples
        if len(chunk) < n_samples:
            chunk = np.pad(chunk, (0, n_samples - len(chunk)))
        return chunk.astype(np.float32)
