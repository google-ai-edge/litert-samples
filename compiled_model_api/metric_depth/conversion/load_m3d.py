import sys, types, inspect
_orig = inspect.getsourcefile
def _safe_gsf(o):
    try: return _orig(o)
    except Exception: return None
inspect.getsourcefile = _safe_gsf

_REJECT = {"Config", "DictAction"}   # force hubconf's try-mmcv to fall back to mmengine
class _Stub(types.ModuleType):
    __file__ = "<stub>"; __spec__ = None; __path__ = []
    def __getattr__(s, n):
        if n.startswith("__") or n in _REJECT: raise AttributeError(n)
        return lambda *a, **k: None
for name in ["mmcv", "mmcv.utils", "mmcv.cnn", "mmcv.ops", "mmcv.runner"]:
    sys.modules[name] = _Stub(name)
sys.modules["mmcv.utils"].collect_env = lambda *a, **k: {}
import torch

def load():
    return torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True, trust_repo=True).eval()

if __name__ == "__main__":
    m = load()
    print("loaded:", type(m).__name__, "| params", round(sum(p.numel() for p in m.parameters())/1e6,1), "M")
    for n,c in m.named_children():
        print("  ", n, ":", type(c).__name__, "|", [nn for nn,_ in c.named_children()][:10])
