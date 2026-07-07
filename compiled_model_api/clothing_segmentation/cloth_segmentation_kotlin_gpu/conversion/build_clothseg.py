import sys, os, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/cloth-segmentation")
_o=F.interpolate
F.interpolate=lambda *a,**k:_o(*a,**{**k,**({'align_corners':False} if k.get('align_corners') is True else {})})
from networks.u2net import U2NET
from huggingface_hub import hf_hub_download
net=U2NET(in_ch=3, out_ch=4).eval()
sd=torch.load(hf_hub_download("tryonlabs/u2net-cloth-segmentation","u2net_cloth_segm.pth"), map_location="cpu")
sd=sd.get('model_state_dict', sd.get('state_dict', sd)) if isinstance(sd,dict) else sd
sd={k[7:] if k.startswith('module.') else k:v for k,v in sd.items()}
print("load:", net.load_state_dict(sd, strict=False))

class Wrap(nn.Module):
    def __init__(s,n): super().__init__(); s.n=n
    def forward(s,x): return s.n(x)[0]   # d0 [1,4,768,768] logits
w=Wrap(net).eval()
dummy=torch.rand(1,3,768,768)
with torch.no_grad(): o=w(dummy)
print("out:", tuple(o.shape))
np.save("ref_in.npy", dummy.numpy()); np.save("ref_out.npy", o.numpy())
import litert_torch
litert_torch.convert(w,(dummy,)).export("clothseg.tflite")
print("saved %.1f MB"%(os.path.getsize("clothseg.tflite")/1e6))
