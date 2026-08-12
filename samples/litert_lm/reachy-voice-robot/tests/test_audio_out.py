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

import os
import wave

import numpy as np
import pytest

from demo import audio_out
from demo.audio_out import StreamPlayer


def _recorder():
    calls = []

    def play(audio, sample_rate):
        calls.append((len(audio), sample_rate))

    return calls, play


def test_phrases_play_as_one_clip():
    # Two phrases are buffered and played as one clip — no seams.
    calls, play = _recorder()
    p = StreamPlayer(24000, play=play)
    p.feed(np.zeros(8000, dtype=np.float32))
    p.feed(np.zeros(4000, dtype=np.float32))
    p.close()
    assert calls == [(12000, 24000)], "one clip from the concatenated phrases"


def test_empty_feed_is_ignored():
    calls, play = _recorder()
    p = StreamPlayer(24000, play=play)
    p.feed(np.zeros(0, dtype=np.float32))
    p.close()
    assert calls == [], "empty response is not played"


def test_close_without_feed_is_noop():
    calls, play = _recorder()
    StreamPlayer(24000, play=play).close()
    assert calls == []


def test_sample_rate_kept():
    assert StreamPlayer(22050).sample_rate == 22050


# --- play_wav: shared WAV-write+afplay helper (audio_out.py + robot_reachy.py) ---

class _FakeCompleted:
    """Stub for subprocess.CompletedProcess — only returncode is needed."""

    def __init__(self, returncode: int) -> None:
        self.returncode = returncode


def test_play_wav_logs_warning_on_nonzero_returncode(monkeypatch, caplog):
    monkeypatch.setattr(audio_out.subprocess, "run",
                        lambda *a, **k: _FakeCompleted(returncode=1))
    with caplog.at_level("WARNING", logger="demo.audio_out"):
        audio_out.play_wav(np.zeros(800, dtype=np.float32), 16000)
    assert any("exited with code" in r.message for r in caplog.records)


def test_play_wav_silent_on_success(monkeypatch, caplog):
    monkeypatch.setattr(audio_out.subprocess, "run",
                        lambda *a, **k: _FakeCompleted(returncode=0))
    with caplog.at_level("WARNING", logger="demo.audio_out"):
        audio_out.play_wav(np.zeros(800, dtype=np.float32), 16000)
    assert not caplog.records, "a successful afplay must not log anything"


def test_play_wav_removes_temp_file_after_success(monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["path"] = cmd[1]
        assert os.path.exists(cmd[1]), "WAV must exist while it's being played"
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(audio_out.subprocess, "run", fake_run)
    audio_out.play_wav(np.zeros(800, dtype=np.float32), 16000)
    assert not os.path.exists(seen["path"]), "temp WAV must be removed"


def test_play_wav_removes_temp_file_even_if_player_raises(monkeypatch):
    # try/finally: os.remove must run even if the player itself raised
    # an exception rather than just returning a nonzero code.
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["path"] = cmd[1]
        raise RuntimeError("boom")

    monkeypatch.setattr(audio_out.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError):
        audio_out.play_wav(np.zeros(800, dtype=np.float32), 16000)
    assert not os.path.exists(seen["path"])


def test_play_wav_writes_valid_pcm16_mono_wav(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        with wave.open(cmd[1], "rb") as fh:
            captured["nchannels"] = fh.getnchannels()
            captured["sampwidth"] = fh.getsampwidth()
            captured["framerate"] = fh.getframerate()
            captured["nframes"] = fh.getnframes()
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(audio_out.subprocess, "run", fake_run)
    audio_out.play_wav(np.zeros(1600, dtype=np.float32), 16000)
    assert captured == {"nchannels": 1, "sampwidth": 2, "framerate": 16000,
                        "nframes": 1600}


def test_afplay_delegates_to_shared_play_wav(monkeypatch):
    # _afplay is a thin wrapper around play_wav (the same signature that
    # StreamPlayer injects by default), not a separate WAV-writing copy.
    calls = []
    monkeypatch.setattr(audio_out, "play_wav",
                        lambda audio, sr: calls.append((len(audio), sr)))
    audio_out._afplay(np.zeros(400, dtype=np.float32), 22050)
    assert calls == [(400, 22050)]
