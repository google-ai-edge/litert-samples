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

"""Sources must return EXACTLY the shape the model requires.

Two different failure modes get two different treatments. A wrong length
fails loudly, so a short file is padded with silence. A wrong sample rate
would produce plausible-looking nonsense instead of an error, so it's
rejected outright: otherwise debugging would land in the model instead of
the loader.
"""

import wave
from pathlib import Path

import numpy as np
import pytest

from emulator.sources import (
    SilenceAudioSource,
    StillFrameSource,
    VideoFileFrameSource,
    WavAudioSource,
)


def write_wav(path: Path, samples: np.ndarray, rate: int = 16000,
              channels: int = 1) -> None:
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(channels)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(samples.astype(np.int16).tobytes())


def test_still_source_returns_requested_shape():
    src = StillFrameSource(shape=(640, 640, 3))
    frame = src.read()
    assert frame.shape == (640, 640, 3)
    assert frame.dtype == np.float32


def test_still_source_is_deterministic():
    # Two runs must produce identical frames, otherwise benchmarks aren't comparable.
    a = StillFrameSource(shape=(64, 64, 3), seed=7).read()
    b = StillFrameSource(shape=(64, 64, 3), seed=7).read()
    assert np.array_equal(a, b)


# --- VideoFileFrameSource lifecycle (the capture handle must be released) ---

class _FakeCap:
    def __init__(self) -> None:
        self.released = False

    def release(self) -> None:
        self.released = True


def _video_source_over(cap: _FakeCap) -> VideoFileFrameSource:
    # Bypass __init__ (it imports cv2 and opens a real file); we only exercise
    # the close / context-manager path here.
    src = VideoFileFrameSource.__new__(VideoFileFrameSource)
    src._cap = cap
    return src


def test_video_file_source_close_releases_capture():
    cap = _FakeCap()
    _video_source_over(cap).close()
    assert cap.released is True


def test_video_file_source_context_manager_releases_on_exit():
    cap = _FakeCap()
    with _video_source_over(cap) as src:
        assert src is not None
    assert cap.released is True


def test_silence_source_returns_exact_length():
    src = SilenceAudioSource(sample_rate=16000)
    chunk = src.read(8000)
    assert chunk.shape == (8000,)
    assert chunk.dtype == np.float32
    assert np.all(chunk == 0.0)


def test_wav_source_reads_and_pads(tmp_path: Path):
    path = tmp_path / "short.wav"
    write_wav(path, np.sin(np.linspace(0, 20, 1000)) * 16000)

    src = WavAudioSource(path, sample_rate=16000)
    chunk = src.read(4000)
    # The file is shorter than requested — the remainder is padded with
    # silence rather than truncated: the model expects exactly 80000 samples
    # and would fail on a short input.
    assert chunk.shape == (4000,)
    assert np.abs(chunk[:1000]).max() > 0.1
    assert np.all(chunk[1000:] == 0.0)


def test_wav_source_normalises_to_unit_range(tmp_path: Path):
    path = tmp_path / "loud.wav"
    write_wav(path, np.full(500, 32000))
    chunk = WavAudioSource(path).read(500)
    assert 0.9 < chunk.max() <= 1.0, "PCM must normalize to [-1, 1]"


def test_wav_source_rejects_wrong_sample_rate(tmp_path: Path):
    path = tmp_path / "wrong.wav"
    write_wav(path, np.zeros(100), rate=44100)
    # Silent resampling is not allowed: moonshine expects exactly 16 kHz, and
    # at another rate it would produce plausible-looking nonsense instead of
    # an error.
    with pytest.raises(ValueError, match="16000"):
        WavAudioSource(path, sample_rate=16000)


def test_wav_source_advances_position(tmp_path: Path):
    # The source must return DIFFERENT chunks back to back, not the same one
    # every time: otherwise recognition running in a loop would see the same
    # second forever.
    path = tmp_path / "ramp.wav"
    write_wav(path, np.linspace(-30000, 30000, 2000))
    src = WavAudioSource(path)
    first = src.read(1000)
    second = src.read(1000)
    assert not np.array_equal(first, second)
    assert second.mean() > first.mean()


def test_wav_source_mixes_stereo_to_mono(tmp_path: Path):
    path = tmp_path / "stereo.wav"
    # Left channel loud, right channel quiet — the mean should fall between them.
    interleaved = np.empty(400, dtype=np.int16)
    interleaved[0::2] = 20000
    interleaved[1::2] = 0
    write_wav(path, interleaved, channels=2)
    chunk = WavAudioSource(path).read(200)
    assert chunk.shape == (200,)
    assert 0.25 < chunk.mean() < 0.35, "stereo must be mixed down by averaging"
