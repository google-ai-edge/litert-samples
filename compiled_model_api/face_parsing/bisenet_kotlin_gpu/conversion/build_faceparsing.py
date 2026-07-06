"""Build GPU-compatible BiSeNet face-parsing tflite via litert-torch.
Only patch: align_corners=True -> False (GPU delegate rejects align_corners=True resize)."""
import torch, torch.nn as nn, torch.nn.functional as F, sys, os
from huggingface_hub import hf_hub_download
sys.path.insert(0, os.environ.get("FP_REPO", "fp"))
from model import BiSeNet

# patch 1: align_corners=True -> False (GPU delegate rejects align_corners=True resize)
_orig = F.interpolate
def _patched(*a, **k):
    if k.get("align_corners") is True: k["align_corners"] = False
    return _orig(*a, **k)
F.interpolate = _patched

# patch 2: global avg_pool2d(x, x.size()[2:]) -> mean([2,3]) — the Mali delegate
# rejects AVERAGE_POOL_2D with a full-spatial kernel; a MEAN reduce is supported.
_avg = F.avg_pool2d
def _avg_patched(x, kernel_size, *a, **k):
    ks = tuple(kernel_size) if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
    if ks == tuple(x.shape[-2:]):
        return x.mean(dim=[2, 3], keepdim=True)
    return _avg(x, kernel_size, *a, **k)
F.avg_pool2d = _avg_patched

# patch 3: ResNet maxpool -inf-pad -> explicit 0-pad + unpadded maxpool. The Mali
# delegate rejects the PADV2 (-inf pad) that MaxPool2d(padding=1) lowers to; 0-pad is
# exact here since the maxpool input is post-ReLU (>= 0) so max is unaffected.
class ZeroPadMaxPool(nn.Module):
    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), value=0.0)
        return F.max_pool2d(x, kernel_size=3, stride=2, padding=0)

net = BiSeNet(n_classes=19).eval()
for name, mod in list(net.named_modules()):
    if isinstance(mod, nn.MaxPool2d):
        parent = net
        *path, last = name.split(".")
        for p in path: parent = getattr(parent, p)
        setattr(parent, last, ZeroPadMaxPool())
net.load_state_dict(torch.load(hf_hub_download("AI2lab/face-parsing.PyTorch", "79999_iter.pth"),
                               map_location="cpu", weights_only=True))

class Wrap(nn.Module):
    def __init__(s, n): super().__init__(); s.n = n
    def forward(s, x):
        o = s.n(x)
        return o[0] if isinstance(o, (list, tuple)) else o

dummy = torch.randn(1, 3, 512, 512)
with torch.no_grad(): ref = Wrap(net)(dummy); print("wrap out:", tuple(ref.shape))
import litert_torch
litert_torch.convert(Wrap(net).eval(), (dummy,)).export("faceparsing.tflite")
print("saved %.1f MB" % (os.path.getsize("faceparsing.tflite")/1e6))
