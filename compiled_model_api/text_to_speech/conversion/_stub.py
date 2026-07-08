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


class _Dummy:
    """Absorbs any attribute access or call and returns None."""

    def __getattr__(self, name):
        return lambda *args, **kwargs: None

    def __call__(self, *args, **kwargs):
        return None


_propack = types.ModuleType("scipy.sparse.linalg._propack")
_propack.__file__ = "<stub:scipy._propack>"
_propack.__spec__ = None
for _name in ("_spropack", "_dpropack", "_cpropack", "_zpropack"):
    setattr(_propack, _name, _Dummy())
sys.modules["scipy.sparse.linalg._propack"] = _propack

_optimize = types.ModuleType("scipy.optimize")
_optimize.__file__ = "<stub:scipy.optimize>"
_optimize.__spec__ = None
_optimize.linear_sum_assignment = lambda *args, **kwargs: None
sys.modules["scipy.optimize"] = _optimize

_original_getsourcefile = inspect.getsourcefile


def _safe_getsourcefile(obj):
    """Returns the source file like inspect.getsourcefile, but never raises.

    Args:
        obj: Any object accepted by inspect.getsourcefile.

    Returns:
        The source file path, or None when the lookup fails.
    """
    try:
        return _original_getsourcefile(obj)
    except (AttributeError, TypeError):
        name = getattr(obj, "__name__", repr(obj))
        sys.stderr.write(
            f"[probe] guarded getsourcefile crash on module {name!r}\n")
        return None


inspect.getsourcefile = _safe_getsourcefile
