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

"""Voice detection and continuous microphone listening.

The demo triggers on voice, not a button press: the robot listens the whole
time, catches an utterance, and responds. Two pitfalls, both solved here.

The first is separating speech from silence. A simple energy-based VAD:
compute the loudness (RMS) of short chunks and compare against a threshold.
The threshold isn't fixed — it's calibrated against the room's background
noise at startup, otherwise it wouldn't trigger in a quiet room, or it would
trigger on a fan in a noisy one.

The second is not hearing itself. While the robot is talking, the microphone
picks up its own voice and would trigger on it. So we don't listen during a
reply, and afterward we discard the buffer that accumulated
(`MicSource.flush`, platform-specific implementation — see demo/platform/).

The pure logic (RMS, the `VoiceGate` state machine, utterance assembly) is
separated from the microphone device and is platform-independent: `MicStream`
(Mac, ffmpeg) and `_ReachyMicSource` (Reachy Mini, SDK) live in demo/platform/
and both hand back the same `chunks() -> (chunk_float32, rms)` contract that
is tested here without a microphone.
"""

from __future__ import annotations

import numpy as np


def rms(chunk: np.ndarray) -> float:
    """Root-mean-square loudness of a chunk. An empty chunk is zero."""
    if len(chunk) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))


class VoiceGate:
    """State machine: silence -> speech -> silence.

    Speech starts once loudness stays above the threshold for `onset_chunks`
    chunks in a row (so a single click doesn't trigger it). It ends once
    loudness stays below the threshold for `hangover_chunks` chunks (a pause
    within an utterance shorter than that doesn't cut it off).
    """

    def __init__(self, threshold: float, onset_chunks: int = 2,
                 hangover_chunks: int = 8) -> None:
        self.threshold = threshold
        self.onset_chunks = onset_chunks
        self.hangover_chunks = hangover_chunks
        self.speaking = False
        self._loud_run = 0
        self._quiet_run = 0

    def push(self, chunk_rms: float) -> str | None:
        """Feed in a chunk's loudness. Return 'start', 'end', or None."""
        loud = chunk_rms >= self.threshold
        if not self.speaking:
            self._loud_run = self._loud_run + 1 if loud else 0
            if self._loud_run >= self.onset_chunks:
                self.speaking = True
                self._quiet_run = 0
                return "start"
            return None
        # already speaking — waiting for sustained silence
        self._quiet_run = self._quiet_run + 1 if not loud else 0
        if self._quiet_run >= self.hangover_chunks:
            self.speaking = False
            self._loud_run = 0
            return "end"
        return None


def collect_utterance(chunk_rms_source, gate: VoiceGate,
                      preroll: int = 3, max_chunks: int = 100):
    """Collect a single utterance from a stream of (chunk, its RMS).

    `chunk_rms_source` is an iterator of (chunk, rms) pairs. Returns the
    concatenated utterance audio, or None if the stream ran out. Keeps a
    small `preroll` prebuffer so the start of a word isn't cut off: onset is
    detected with a delay of onset_chunks, and without a prebuffer the first
    ~200 ms would be lost.
    """
    ring: list[np.ndarray] = []
    captured: list[np.ndarray] = []
    for chunk, level in chunk_rms_source:
        event = gate.push(level)
        if not gate.speaking and event != "end":
            # silence before the utterance starts — spin the ring prebuffer
            ring.append(chunk)
            if len(ring) > preroll:
                ring.pop(0)
            continue
        if event == "start":
            captured = list(ring) + [chunk]
            ring = []
            continue
        captured.append(chunk)
        if event == "end" or len(captured) >= max_chunks:
            return np.concatenate(captured) if captured else None
    return np.concatenate(captured) if captured else None


def calibrate_threshold(chunk_rms_source, seconds: float = 1.0,
                        chunk_ms: int = 100, multiplier: float = 3.0,
                        floor: float = 0.01) -> float:
    """Speech threshold from background noise: median silence loudness × multiplier.

    The `floor` lower bound guards against a perfectly silent microphone (a
    median of 0 would give a zero threshold and a permanent trigger).
    """
    n = max(1, int(seconds * 1000 / chunk_ms))
    levels = []
    for _, level in chunk_rms_source:
        levels.append(level)
        if len(levels) >= n:
            break
    baseline = float(np.median(levels)) if levels else 0.0
    return max(baseline * multiplier, floor)
