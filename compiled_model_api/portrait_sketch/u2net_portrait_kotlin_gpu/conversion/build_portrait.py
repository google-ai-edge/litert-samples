import sys, os, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
sys.path.insert(0, os.path.expanduser("~/Downloads/meeting/u2net-src"))
_o=F.interpolate; F.interpolate=lambda *a,**k:_o(*a,**{**k,**({'align_corners':False} if k.get('align_corners') is True else {})})
from model.u2net import U2NET
from huggingface_hub import hf_hub_download
net=U2NET(3,1).eval()
sd=torch.load(hf_hub_download("Maxwelltebi/u2net-portrait","u2net_portrait.pth"), map_location="cpu")
sd=sd.get('model_state_dict', sd.get('state_dict', sd)) if isinstance(sd,dict) else sd
sd={k[7:] if k.startswith('module.') else k:v for k,v in sd.items()}
print("load:", net.load_state_dict(sd, strict=False))
class Wrap(nn.Module):
    def __init__(s,n): super().__init__(); s.n=n
    def forward(s,x):
        o=s.n(x)                    # (d0..d6) each sigmoid
        return o[0]                 # [1,1,512,512] portrait map (0..1)
w=Wrap(net).eval()
dummy=torch.rand(1,3,512,512)
with torch.no_grad(): out=w(dummy)
print("out:", tuple(out.shape), "range", round(float(out.min()),3), round(float(out.max()),3))
np.save("ref_in.npy", dummy.numpy()); np.save("ref_out.npy", out.numpy())
import litert_torch
litert_torch.convert(w,(dummy,)).export("portrait.tflite")
print("saved %.1f MB"%(os.path.getsize("portrait.tflite")/1e6))
