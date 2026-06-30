"""Narrow stub: ONLY shim the broken scipy `_propack` dlopen (macOS zero-fill bug), leaving
scipy.optimize / scipy.signal REAL (librosa/torchlibrosa need them). Import FIRST."""
import sys, types, inspect


class _D:
    def __getattr__(self, n): return lambda *a, **k: None
    def __call__(self, *a, **k): return None


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
