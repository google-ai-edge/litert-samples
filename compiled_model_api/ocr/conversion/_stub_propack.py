# Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Narrow stub: ONLY shim the broken scipy `_propack` dlopen (macOS zero-fill bug), leaving
scipy.optimize / scipy.signal REAL (librosa/torchlibrosa need them). Import FIRST."""
import sys
import types
import inspect


class _D:
    def __getattr__(self, n):
        return lambda *a, **k: None
    def __call__(self, *a, **k):
        return None


_pp = types.ModuleType("scipy.sparse.linalg._propack")
_pp.__file__ = "<stub:scipy._propack>"
_pp.__spec__ = None
for _nm in ("_spropack", "_dpropack", "_cpropack", "_zpropack"):
    setattr(_pp, _nm, _D())
sys.modules["scipy.sparse.linalg._propack"] = _pp

_orig = inspect.getsourcefile


def _safe(obj):
    try:
        return _orig(obj)
    except (AttributeError, TypeError):
        return None


inspect.getsourcefile = _safe
