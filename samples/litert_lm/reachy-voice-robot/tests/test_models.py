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


def test_get_returns_the_named_model():
    m = models.get("yolox-tiny")
    assert m.repo == "litert-community/yolox-tiny-litert"
    assert m.file == "yolox_tiny.tflite"


def test_get_unknown_name_raises_and_lists_known_names():
    with pytest.raises(KeyError) as exc:
        models.get("does-not-exist")
    assert "yolox-tiny" in str(exc.value)


def test_every_stage_default_resolves():
    # A typo in DETECTOR/ASR/LLM/TTS would break the pipeline at build time;
    # catch it here instead.
    for name in (models.DETECTOR, models.ASR, models.LLM, *models.TTS.values()):
        assert isinstance(models.get(name), models.Model)


def test_llm_is_a_served_endpoint_not_an_hf_download():
    m = models.get(models.LLM)
    assert m.endpoint and m.model
    assert m.repo is None and m.file is None


def test_inflect_is_a_bundled_dir():
    m = models.get(models.TTS["inflect"])
    assert m.dir is not None and m.dir.name == "inflect"
    assert m.repo is None


def test_build_synthesizer_selects_the_backend_model_by_name(monkeypatch):
    # The point of the catalog: build_synthesizer resolves the synthesizer by
    # name (no network for the bundled inflect default), so swapping the name
    # swaps the backend with no other code change.
    from emulator import run

    seen = {}

    def fake_load_inflect(models_dir=None):
        seen["models_dir"] = models_dir
        return "INFLECT"

    monkeypatch.setattr("emulator.inflect_tts.load_inflect", fake_load_inflect)
    assert run.build_synthesizer("inflect") == "INFLECT"
    assert seen["models_dir"] == models.get("inflect-nano-v2").dir
