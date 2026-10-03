# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
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

"""Shared scipy/_propack + getsourcefile stub for macOS probes.

Import this FIRST (before transformers / litert_torch) in any probe script:

    import _stub  # noqa: F401

Import guards that let the conversion scripts import transformers / litert_torch
on machines where scipy's optional native extensions fail to load.
"""
import inspect
import sys
import types


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

_opt = types.ModuleType("scipy.optimize")
_opt.__file__ = "<stub:scipy.optimize>"
_opt.__spec__ = None
_opt.linear_sum_assignment = lambda *a, **k: None
sys.modules["scipy.optimize"] = _opt

_orig_gsf = inspect.getsourcefile


def _safe_getsourcefile(obj):
    """Wraps inspect.getsourcefile to swallow crashes on stub modules.

    Args:
        obj: Object whose source file is looked up.

    Returns:
        The source file path, or None when the lookup raises.
    """
    try:
        return _orig_gsf(obj)
    except (AttributeError, TypeError):
        nm = getattr(obj, "__name__", repr(obj))
        sys.stderr.write(
            f"[probe] guarded getsourcefile crash on module {nm!r}\n")
        return None


inspect.getsourcefile = _safe_getsourcefile
