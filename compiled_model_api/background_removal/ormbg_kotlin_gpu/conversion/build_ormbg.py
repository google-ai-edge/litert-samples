import sys, os, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# defensive: GPU delegate rejects align_corners=True
_orig = F.interpolate
def _patched(*a, **k):
    if k.get("align_corners") is True: k["align_corners"] = False
    return _orig(*a, **k)
F.interpolate = _patched
from ormbg.models.ormbg import ORMBG
net = ORMBG()
sd = torch.load("models/ormbg.pth", map_location="cpu")
net.load_state_dict(sd); net.eval()

class Wrap(nn.Module):
    def __init__(s, n): super().__init__(); s.n = n
    def forward(s, x): return s.n(x)[0][0]   # sigmoid(d1) main mask [1,1,1024,1024]
w = Wrap(net).eval()
dummy = torch.rand(1, 3, 1024, 1024)
with torch.no_grad(): o = w(dummy)
print("out:", tuple(o.shape), "range", float(o.min()), float(o.max()))
np.save("ref_in.npy", dummy.numpy()); np.save("ref_out.npy", o.numpy())
import litert_torch
litert_torch.convert(w, (dummy,)).export("ormbg.tflite")
print("saved %.1f MB" % (os.path.getsize("ormbg.tflite")/1e6))
