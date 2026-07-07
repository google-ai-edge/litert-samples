import sys, os, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/TwinLiteNet")
from model.TwinLite import TwinLiteNet as Net

class ZeroStuffConvT2d(nn.Module):
    def __init__(s, ct, Hin, Win):
        super().__init__()
        s.s=ct.stride[0]; s.k=ct.kernel_size[0]; s.p=ct.padding[0]; s.op=ct.output_padding[0]; s.Hin=Hin; s.Win=Win
        w=ct.weight.detach().flip(2).flip(3).permute(1,0,2,3).contiguous()
        s.register_buffer("w", w)
        s.register_buffer("b", ct.bias.detach().clone() if ct.bias is not None else torch.zeros(ct.out_channels))
        mh=np.zeros((Hin*s.s, Win*s.s), np.float32); mh[::s.s, ::s.s]=1.0
        s.register_buffer("mask", torch.from_numpy(mh)[None,None])
    def forward(s, x):
        xn=F.interpolate(x, size=(s.Hin*s.s, s.Win*s.s), mode="nearest")*s.mask
        y=F.conv2d(xn, s.w, bias=s.b, padding=s.k-1)
        olH=(s.Hin-1)*s.s+s.k-2*s.p+s.op; olW=(s.Win-1)*s.s+s.k-2*s.p+s.op
        return y[:,:,s.p:s.p+olH, s.p:s.p+olW]

net=Net(); sd=torch.load("TwinLiteNet/pretrained/best.pth", map_location="cpu")
sd=sd.get('state_dict',sd) if isinstance(sd,dict) else sd
sd={k[7:] if k.startswith('module.') else k:v for k,v in sd.items()}
print("load:", net.load_state_dict(sd, strict=False))
net.eval()
# swap ConvTranspose2d -> ZeroStuffConvT2d (capture input sizes via hooks)
L={}; hk=[]
for n,mo in net.named_modules():
    if isinstance(mo, nn.ConvTranspose2d):
        hk.append(mo.register_forward_pre_hook((lambda nm: (lambda mod,i: L.__setitem__(nm, i[0].shape[-2:])))(n)))
with torch.no_grad(): net(torch.randn(1,3,360,640))
for h in hk: h.remove()
for name,mo in list(net.named_modules()):
    if isinstance(mo, nn.ConvTranspose2d) and name in L:
        par=net; *pth,last=name.split(".")
        for q in pth: par=getattr(par,q)
        hh,ww=L[name]; setattr(par,last,ZeroStuffConvT2d(mo,hh,ww))

class Wrap(nn.Module):
    def __init__(s,n): super().__init__(); s.n=n
    def forward(s,x): da,ll=s.n(x); return da, ll
w=Wrap(net).eval()
dummy=torch.rand(1,3,360,640)
with torch.no_grad(): o=w(dummy)
print("outs:", [tuple(t.shape) for t in o])
np.save("ref_in.npy", dummy.numpy())
for i,t in enumerate(o): np.save(f"ref_out{i}.npy", t.numpy())
import litert_torch
litert_torch.convert(w,(dummy,)).export("twinlite.tflite")
print("saved %.1f MB"%(os.path.getsize("twinlite.tflite")/1e6))
