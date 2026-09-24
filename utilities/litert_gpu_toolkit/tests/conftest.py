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
"""Test fixtures: put `utilities/` on the path, detect the GPU, fetch models."""

import os
import sys

import pytest

_UTILITIES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _UTILITIES not in sys.path:
    sys.path.insert(0, _UTILITIES)


@pytest.fixture(scope="session")
def gpu():
    """Skip when the LiteRT GPU accelerator cannot compile a trivial graph."""
    pytest.importorskip("ai_edge_litert")
    from litert_gpu_toolkit.checker import _compile_gpu
    from litert_gpu_toolkit.tests.graphs import GraphBuilder
    import tempfile
    g = GraphBuilder()
    x = g.input("x", [1, 4])
    g.outputs = [g.add(x, g.const("one", [1.0]), "y", [1, 4])]
    path = os.path.join(tempfile.mkdtemp(), "probe.tflite")
    g.build(path)
    try:
        _compile_gpu(path, allow_cpu_fallback=False)
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"no LiteRT GPU accelerator here: {e}")
    return True


@pytest.fixture(scope="session")
def public_model():
    """Download a litert-community file (skips when offline / no hub client)."""
    hub = pytest.importorskip("huggingface_hub")

    def fetch(repo: str, filename: str) -> str:
        try:
            return hub.hf_hub_download(f"litert-community/{repo}", filename)
        except Exception as e:  # pragma: no cover - network dependent
            pytest.skip(f"could not fetch {repo}/{filename}: {e}")

    return fetch
