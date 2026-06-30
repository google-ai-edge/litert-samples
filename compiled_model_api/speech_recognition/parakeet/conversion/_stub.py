# scipy _propack broken-binary stub (this env) + import-order safety. Import FIRST.
import sys, types
class _AnyObj:
    def __getattr__(self, k): return (lambda *a, **kw: None)
_p = types.ModuleType("scipy.sparse.linalg._propack")
for fn in ["_spropack","_dpropack","_cpropack","_zpropack"]: setattr(_p, fn, _AnyObj())
sys.modules["scipy.sparse.linalg._propack"] = _p
