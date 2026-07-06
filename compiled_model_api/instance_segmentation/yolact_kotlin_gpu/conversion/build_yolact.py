import sys, os, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/yolact")
sys.argv=['x']  # yolact modules read argv
torch.cuda.current_device = lambda: 0            # yolact calls this at import (CPU box)
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 2              # -> use_jit=False so FPN is plain nn.Module (traceable)
_load = torch.load
torch.load = lambda *a, **k: _load(*a, **{**k, "map_location": "cpu"})
from data import cfg, set_cfg
set_cfg('yolact_resnet50_config')
from yolact import Yolact
from huggingface_hub import hf_hub_download

pth=hf_hub_download("dbolya/yolact-resnet50","yolact_resnet50_54_800000.pth")
net=Yolact(); net.load_weights(pth); net.eval()
net.detect = lambda pred_outs, *a, **k: pred_outs   # bypass NMS -> raw dict

class ZeroPadMaxPool(nn.Module):
    def forward(self, x):
        x=F.pad(x,(1,1,1,1),value=0.0); return F.max_pool2d(x,3,stride=2,padding=0)
for name,m in list(net.named_modules()):
    if isinstance(m, nn.MaxPool2d):
        p=net; *path,last=name.split('.')
        for q in path: p=getattr(p,q)
        setattr(p,last,ZeroPadMaxPool())

class Wrap(nn.Module):
    def __init__(s,n): super().__init__(); s.n=n
    def forward(s,x):
        d=s.n(x)
        return d['loc'], d['conf'], d['mask'], d['proto']
w=Wrap(net).eval()
dummy=torch.randn(1,3,550,550)
with torch.no_grad():
    o=w(dummy)
print("raw outs:", [tuple(t.shape) for t in o])
np.save("../ref_in.npy", dummy.numpy())
for i,t in enumerate(o): np.save(f"../ref_out{i}.npy", t.numpy())
import litert_torch
litert_torch.convert(w,(dummy,)).export("../yolact.tflite")
print("saved %.1f MB"%(os.path.getsize("../yolact.tflite")/1e6))
