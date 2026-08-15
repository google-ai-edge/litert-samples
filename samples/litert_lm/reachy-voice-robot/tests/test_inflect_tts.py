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

"""InflectSynthesizer.speak on empty text.

An empty phrase (e.g. everything got cut by split_into_phrases) must return
an empty array via the short-circuit path, WITHOUT touching the model: the
`__init__` constructor loads the espeak frontend and LiteRT weights from
disk, which is heavy and not what needs checking here. The instance is
built without `__init__` (object.__new__), so any access to self._tts would
raise AttributeError — if speak() failed to short-circuit on empty text, the
test would catch that as an error rather than passing by accident.
"""

import numpy as np

from emulator.inflect_tts import SAMPLE_RATE, InflectSynthesizer


def _bare_synth() -> InflectSynthesizer:
    """Instance without loading the model — only speak() should ever run here."""
    return object.__new__(InflectSynthesizer)


def test_speak_empty_text_returns_empty_float32_array():
    synth = _bare_synth()
    out = synth.speak("")
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.size == 0


def test_speak_whitespace_only_text_is_also_short_circuited():
    synth = _bare_synth()
    out = synth.speak("   \n\t  ")
    assert out.size == 0


def test_sample_rate_matches_measured_inflect_output():
    assert SAMPLE_RATE == 24000


def test_ensure_weights_downloads_missing_then_skips_present(tmp_path, monkeypatch):
    # Regression guard for the fresh-clone crash: the *.tflite weights are
    # gitignored, so a checkout has none and they must be fetched from HF before
    # the model is opened. Missing weights are downloaded; a warm dir is a no-op.
    import emulator.inflect_tts as it

    fetched = []

    def fake_download(repo_id, filename):
        fetched.append(filename)
        src = tmp_path / f"cache__{filename}"
        src.write_bytes(b"stub")
        return str(src)

    monkeypatch.setattr(it, "hf_hub_download", fake_download)
    dest = tmp_path / "inflect"
    it._ensure_weights(dest, "some/repo", "fp32")
    assert sorted(fetched) == ["inflect_decoder.tflite", "inflect_text_encoder.tflite"]
    assert (dest / "inflect_text_encoder.tflite").exists()
    assert (dest / "inflect_decoder.tflite").exists()

    fetched.clear()
    it._ensure_weights(dest, "some/repo", "fp32")
    assert fetched == []  # already present -> no re-download


def test_init_wires_the_weight_fetch(tmp_path, monkeypatch):
    # The regression that shipped was the call SITE doing nothing, not the helper.
    # Guard against InflectSynthesizer.__init__ ever dropping the `_ensure_weights`
    # call: spy on the helper and stub `say` so __init__ runs without the model.
    import sys
    import types
    import emulator.inflect_tts as it

    calls = []
    monkeypatch.setattr(it, "_ensure_weights",
                        lambda md, repo, prec: calls.append((md, repo, prec)))
    fake_say = types.ModuleType("say")
    fake_say.InflectTTS = lambda **kw: types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "say", fake_say)

    it.InflectSynthesizer(models_dir=tmp_path, repo="some/repo")
    assert calls == [(tmp_path, "some/repo", "fp32")]  # fetch fired with the repo

    calls.clear()
    it.InflectSynthesizer(models_dir=tmp_path, repo=None)
    assert calls == []  # repo=None (manually populated dir) -> no fetch


def test_ensure_weights_fetches_only_the_missing_file(tmp_path, monkeypatch):
    # Each file is checked independently, so a half-populated dir (an interrupted
    # first download) fetches only what is missing, not both.
    import emulator.inflect_tts as it

    fetched = []

    def fake_download(repo_id, filename):
        fetched.append(filename)
        src = tmp_path / f"cache__{filename}"
        src.write_bytes(b"stub")
        return str(src)

    monkeypatch.setattr(it, "hf_hub_download", fake_download)
    dest = tmp_path / "inflect"
    dest.mkdir()
    (dest / "inflect_text_encoder.tflite").write_bytes(b"already here")
    it._ensure_weights(dest, "some/repo", "fp32")
    assert fetched == ["inflect_decoder.tflite"]  # only the missing graph
    assert (dest / "inflect_decoder.tflite").exists()


def test_ensure_weights_fp16_uses_the_fp16_filenames(tmp_path, monkeypatch):
    # The _fp16 suffix must match the names say.py reads (say.py builds the same
    # `{stem}{suffix}.tflite`); a mismatch would fetch names the runtime never loads.
    import emulator.inflect_tts as it

    fetched = []

    def fake_download(repo_id, filename):
        fetched.append(filename)
        src = tmp_path / f"cache__{filename}"
        src.write_bytes(b"stub")
        return str(src)

    monkeypatch.setattr(it, "hf_hub_download", fake_download)
    it._ensure_weights(tmp_path / "inflect", "some/repo", "fp16")
    assert sorted(fetched) == [
        "inflect_decoder_fp16.tflite", "inflect_text_encoder_fp16.tflite"]
