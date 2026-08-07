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

"""Voice detection logic: no microphone, synthetic chunks only.

Checks what actually drives the robot's behavior: when an utterance starts,
when it ends, and that the start of a word isn't clipped by the prebuffer.
"""

import numpy as np

from demo.vad import VoiceGate, calibrate_threshold, collect_utterance, rms


def test_rms_of_silence_is_zero():
    assert rms(np.zeros(1000, dtype=np.float32)) == 0.0


def test_rms_grows_with_amplitude():
    quiet = rms(np.full(1000, 0.1, dtype=np.float32))
    loud = rms(np.full(1000, 0.5, dtype=np.float32))
    assert loud > quiet
    assert abs(loud - 0.5) < 1e-5


def test_gate_needs_sustained_loudness_to_start():
    gate = VoiceGate(threshold=0.1, onset_chunks=2)
    # A single loud chunk is not speech (a click, a knock).
    assert gate.push(0.5) is None
    assert gate.push(0.0) is None
    assert not gate.speaking


def test_gate_starts_after_onset_chunks():
    gate = VoiceGate(threshold=0.1, onset_chunks=2)
    assert gate.push(0.5) is None
    assert gate.push(0.5) == "start"
    assert gate.speaking


def test_gate_ends_after_sustained_silence():
    gate = VoiceGate(threshold=0.1, onset_chunks=1, hangover_chunks=3)
    gate.push(0.5)  # start
    assert gate.speaking
    assert gate.push(0.0) is None
    assert gate.push(0.0) is None
    assert gate.push(0.0) == "end"
    assert not gate.speaking


def test_gate_short_pause_does_not_end_utterance():
    # A pause between words shorter than hangover must not cut off the utterance.
    gate = VoiceGate(threshold=0.1, onset_chunks=1, hangover_chunks=5)
    gate.push(0.5)  # start
    gate.push(0.0)  # quiet
    gate.push(0.0)  # quiet
    assert gate.push(0.5) is None, "speech resumed — not the end"
    assert gate.speaking
    # the silence counter reset, need 5 quiet chunks in a row again to end
    for _ in range(4):
        assert gate.push(0.0) is None
    assert gate.push(0.0) == "end"


def _source(levels, chunk_len=10):
    """Stream of (chunk, rms) built from a list of loudness levels."""
    for lv in levels:
        yield np.full(chunk_len, lv, dtype=np.float32), lv


def test_collect_utterance_captures_speech_span():
    gate = VoiceGate(threshold=0.1, onset_chunks=1, hangover_chunks=2)
    # silence, speech (3 chunks), silence
    levels = [0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0]
    audio = collect_utterance(_source(levels), gate, preroll=1)
    assert audio is not None
    assert len(audio) > 0


def test_collect_utterance_preroll_keeps_word_onset():
    # The prebuffer must capture a chunk BEFORE onset is detected.
    gate = VoiceGate(threshold=0.1, onset_chunks=2, hangover_chunks=2)
    # two quiet chunks (prebuffer), then loud ones — onset is detected on the
    # 2nd loud chunk, but the first loud chunk and one quiet chunk before it
    # must still land in the recording
    levels = [0.05, 0.05, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0]
    audio = collect_utterance(_source(levels, chunk_len=10), gate, preroll=2)
    assert audio is not None
    # 2 prebuffer chunks + at least 2 loud chunks before the end — at least 4 chunks of 10
    assert len(audio) >= 40


def test_collect_utterance_caps_at_max_chunks():
    gate = VoiceGate(threshold=0.1, onset_chunks=1, hangover_chunks=100)
    levels = [0.5] * 200  # speaks without stopping
    audio = collect_utterance(_source(levels), gate, preroll=0, max_chunks=10)
    assert len(audio) == 100  # 10 chunks of 10 samples


def test_calibrate_threshold_scales_with_noise():
    quiet = calibrate_threshold(_source([0.001] * 10), seconds=1.0,
                                chunk_ms=100, multiplier=3.0, floor=0.0)
    noisy = calibrate_threshold(_source([0.05] * 10), seconds=1.0,
                                chunk_ms=100, multiplier=3.0, floor=0.0)
    assert noisy > quiet
    assert abs(noisy - 0.15) < 1e-4


def test_calibrate_threshold_respects_floor():
    # A perfectly silent microphone must not produce a zero threshold.
    thr = calibrate_threshold(_source([0.0] * 10), seconds=1.0,
                              chunk_ms=100, multiplier=3.0, floor=0.01)
    assert thr == 0.01
