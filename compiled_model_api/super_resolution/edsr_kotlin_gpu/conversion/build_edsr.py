import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, os
from super_image import EdsrModel

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

def pixelshuffle_to_convt(r, Cout):
    ct=nn.ConvTranspose2d(Cout*r*r, Cout, kernel_size=r, stride=r, bias=False)
    w=torch.zeros(Cout*r*r, Cout, r, r)
    for c in range(Cout):
        for p in range(r):
            for q in range(r):
                w[c*r*r+p*r+q, c, p, q]=1.0
    ct.weight.data=w
    return ct

m = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=4).eval()
# 1) replace PixelShuffle with the equivalent fixed ConvTranspose2d (capture Cout via hook)
info={}; hk=[]
for n,mo in m.named_modules():
    if isinstance(mo, nn.PixelShuffle):
        hk.append(mo.register_forward_pre_hook((lambda nm,r=mo.upscale_factor: (lambda mod,i: info.__setitem__(nm,(r, i[0].shape[1]//(r*r)))))(n)))
with torch.no_grad(): m(torch.rand(1,3,128,128))
for h in hk: h.remove()
for n,(r,cout) in info.items():
    par=m; *pth,last=n.split('.')
    for q in pth: par=getattr(par,q)
    setattr(par,last, pixelshuffle_to_convt(r,cout))
# 2) swap those ConvTranspose2d -> ZeroStuffConvT2d
L={}; hk=[]
for n,mo in m.named_modules():
    if isinstance(mo, nn.ConvTranspose2d):
        hk.append(mo.register_forward_pre_hook((lambda nm: (lambda mod,i: L.__setitem__(nm, i[0].shape[-2:])))(n)))
with torch.no_grad(): m(torch.rand(1,3,128,128))
for h in hk: h.remove()
for n,mo in list(m.named_modules()):
    if isinstance(mo, nn.ConvTranspose2d) and n in L:
        par=m; *pth,last=n.split('.')
        for q in pth: par=getattr(par,q)
        hh,ww=L[n]; setattr(par,last, ZeroStuffConvT2d(mo,hh,ww))

class Wrap(nn.Module):
    def __init__(s,n): super().__init__(); s.n=n
    def forward(s,x): return s.n(x)
w=Wrap(m).eval()
dummy=torch.rand(1,3,128,128)
with torch.no_grad(): o=w(dummy)
print("out:", tuple(o.shape))
np.save("ref_in.npy", dummy.numpy()); np.save("ref_out.npy", o.numpy())
import litert_torch
litert_torch.convert(w,(dummy,)).export("edsr.tflite")
print("saved %.1f MB"%(os.path.getsize("edsr.tflite")/1e6))
