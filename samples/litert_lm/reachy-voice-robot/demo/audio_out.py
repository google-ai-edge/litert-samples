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

"""Smooth response playback: buffer phrases and play them as ONE clip.

Previously each phrase was played with a separate afplay call — a gap was
audible at the seams (process restart + startup delay). The robot "started
talking, then picked back up the rest".

We tried a seamless ffplay-from-stdin approach, but in some environments it
doesn't reliably open the audio device. afplay is file-based and proven in
this demo, so we buffer the whole response and play a single clip: no
audible seams between phrases at all. For the common one-sentence response
this is exactly one clip, so latency doesn't change.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import wave

import numpy as np

LOG = logging.getLogger(__name__)


def play_wav(audio: np.ndarray, sample_rate: int, player: str = "afplay") -> None:
    """Write float32 audio to a temp WAV and play it (blocks until done).

    Shared helper for every sound-playback path on the Mac: `StreamPlayer`
    (voice loop, `_afplay` below) and `demo/robot_reachy.py` (Robot.say(),
    fallback path) used to duplicate this exact code (write WAV + call
    afplay) — now there's one copy. The temp file is removed in a finally
    block even if the write or playback fails, and we check the player's
    return code — previously a failed afplay (device busy, corrupt WAV)
    went unnoticed.
    """
    # mkstemp, not the deprecated mktemp: it creates the file atomically
    # instead of returning a predictable name nothing holds (a TOCTOU race).
    # We only need the path (wave.open reopens it), so close the fd.
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(path, "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(int(sample_rate))
            fh.writeframes(pcm.tobytes())
        result = subprocess.run([player, path], check=False)
        if result.returncode != 0:
            LOG.warning("%s exited with code %s — audio not played",
                       player, result.returncode)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _afplay(audio: np.ndarray, sample_rate: int) -> None:
    """Write float32 audio to a WAV and play it via afplay (blocks until done)."""
    play_wav(audio, sample_rate)


class StreamPlayer:
    """Buffers response phrases and plays them as one clip on close(), gap-free."""

    def __init__(self, sample_rate: int, play=_afplay) -> None:
        self.sample_rate = int(sample_rate)
        self._play = play
        self._chunks: list[np.ndarray] = []

    def feed(self, samples: np.ndarray) -> None:
        """Add a phrase to the response buffer (empty ones are skipped)."""
        data = np.asarray(samples, dtype=np.float32)
        if data.size:
            self._chunks.append(data)

    def close(self) -> None:
        """Concatenate the buffered phrases and play them as one clip."""
        if not self._chunks:
            return
        audio = np.concatenate(self._chunks)
        self._chunks = []
        self._play(audio, self.sample_rate)
