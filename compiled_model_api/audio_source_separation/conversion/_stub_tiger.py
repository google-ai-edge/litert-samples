"""Minimal-dep shims so look2hear imports without torch_complex / distutils / full scipy.
Import this FIRST (before look2hear)."""
import sys, types

# torch_complex (only used by layers/stft_tfgn.py, which TIGER never calls)
tc = types.ModuleType("torch_complex")
tct = types.ModuleType("torch_complex.tensor")


class ComplexTensor:  # noqa: D401 - placeholder, never instantiated
    pass


tct.ComplexTensor = ComplexTensor
tc.tensor = tct
sys.modules.setdefault("torch_complex", tc)
sys.modules.setdefault("torch_complex.tensor", tct)

# distutils.version.LooseVersion (removed in py3.12; used by layers/stft.py)
try:
    import distutils.version  # noqa: F401
except ImportError:
    du = types.ModuleType("distutils")
    duv = types.ModuleType("distutils.version")

    class LooseVersion(str):
        def __ge__(self, other):
            return True

        def __lt__(self, other):
            return False

    duv.LooseVersion = LooseVersion
    du.version = duv
    sys.modules.setdefault("distutils", du)
    sys.modules.setdefault("distutils.version", duv)

# stub the whole stft_tfgn submodule (needs torch_complex+typeguard; TIGER never uses it)
stft_tfgn = types.ModuleType("look2hear.layers.stft_tfgn")


class Stft:  # placeholder, never instantiated
    pass


stft_tfgn.Stft = Stft
sys.modules.setdefault("look2hear.layers.stft_tfgn", stft_tfgn)

# scipy _propack shim (same as panns-work; librosa pulls scipy.sparse.linalg)
sys.path.insert(0, __import__("os").path.expanduser("~/Downloads/meeting/panns-work"))
try:
    import _stub_propack  # noqa: F401
except Exception:
    pass
