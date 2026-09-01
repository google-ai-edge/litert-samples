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

"""The model catalog: every model by name, current values as constants.

Swapping a model is a one-line change here; these tests pin that the names
resolve and that the stage defaults point at real entries.
"""

from pathlib import Path

import pytest

from emulator import models


def test_model_rejects_illegal_states():
    # A Model must populate exactly one kind, with its co-required fields.
    with pytest.raises(ValueError):
        models.Model()                                    # no kind at all
    with pytest.raises(ValueError):
        models.Model(repo="litert-community/x")           # repo without file
    with pytest.raises(ValueError):
        models.Model(endpoint="http://x")                 # endpoint without model
    with pytest.raises(ValueError):
        models.Model(dir=Path("d"), endpoint="http://x", model="m")  # two kinds
    with pytest.raises(ValueError):
        models.Model(file="f.tflite")                     # bare file, no repo or dir
    with pytest.raises(ValueError):
        models.Model(dir=Path("d"), file="f", repo="r")   # stray repo on a bundled file


def test_model_accepts_both_bundled_kinds():
    # Both bundled kinds must construct: a directory (inflect) and a single
    # file (the detector). A raise here would break the catalog at import.
    models.Model(dir=Path("d"))                     # bundled directory
    models.Model(dir=Path("d"), file="f.tflite")    # bundled single file


def test_get_returns_the_named_model():
    m = models.get("yolo26n")
    assert m.dir is not None and m.file == "yolo26n.tflite"
    assert m.repo is None, "the detector is bundled in the repo, not on HF"


def test_get_unknown_name_raises_and_lists_known_names():
    with pytest.raises(KeyError) as exc:
        models.get("does-not-exist")
    assert "yolo26n" in str(exc.value)


def test_download_resolves_bundled_file_locally(tmp_path, monkeypatch):
    # A bundled model (dir + file) resolves to its local path and must NOT hit
    # Hugging Face — shipping the detector locally is the whole point.
    from emulator import run
    (tmp_path / "yolo26n.tflite").write_bytes(b"x")
    monkeypatch.setattr(run, "hf_hub_download",
                        lambda **k: pytest.fail("HF must not be called for a bundled model"))
    m = models.Model(dir=tmp_path, file="yolo26n.tflite")
    assert run._download(m) == tmp_path / "yolo26n.tflite"


def test_download_missing_bundled_file_raises(tmp_path):
    # A fresh checkout hasn't converted the model yet; the user must get an
    # actionable error, not an opaque LiteRT load failure deeper down.
    from emulator import run
    m = models.Model(dir=tmp_path, file="absent.tflite")
    with pytest.raises(SystemExit):
        run._download(m)


def test_download_hf_model_uses_hugging_face(tmp_path, monkeypatch):
    from emulator import run
    called = {}

    def fake_hf(repo_id=None, filename=None):
        called["repo_id"] = repo_id
        return str(tmp_path / filename)

    monkeypatch.setattr(run, "hf_hub_download", fake_hf)
    m = models.Model(repo="litert-community/x", file="f.tflite")
    assert run._download(m) == tmp_path / "f.tflite"
    assert called["repo_id"] == "litert-community/x"


def test_every_stage_default_resolves():
    # A typo in DETECTOR/ASR/LLM/TTS would break the pipeline at build time;
    # catch it here instead.
    for name in (models.DETECTOR, models.ASR, models.LLM, *models.TTS.values()):
        assert isinstance(models.get(name), models.Model)


def test_llm_is_a_served_endpoint_not_an_hf_download():
    m = models.get(models.LLM)
    assert m.endpoint and m.model
    assert m.repo is None and m.file is None


def test_inflect_fetches_weights_into_a_bundled_dir():
    # Inflect ships its runtime (say.py + frontend) in assets/inflect but fetches
    # the LiteRT weights from HF on first use — repo + dir populated together.
    m = models.get(models.TTS["inflect"])
    assert m.dir is not None and m.dir.name == "inflect"
    assert m.repo is not None


def test_build_synthesizer_selects_the_backend_model_by_name(monkeypatch):
    # The point of the catalog: build_synthesizer resolves the synthesizer by
    # name, so swapping the name swaps the backend with no other code change.
    # It also passes the model's HF repo through so the weights can auto-download.
    from emulator import run

    seen = {}

    def fake_load_inflect(models_dir=None, repo=None):
        seen["models_dir"] = models_dir
        seen["repo"] = repo
        return "INFLECT"

    monkeypatch.setattr("emulator.inflect_tts.load_inflect", fake_load_inflect)
    assert run.build_synthesizer("inflect") == "INFLECT"
    assert seen["models_dir"] == models.get("inflect-nano-v2").dir
    assert seen["repo"] == models.get("inflect-nano-v2").repo
