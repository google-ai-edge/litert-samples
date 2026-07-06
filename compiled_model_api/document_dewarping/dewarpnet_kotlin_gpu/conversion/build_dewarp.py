import sys, os, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/dewarp-src")
from models import get_model
from utils import convert_state_dict

class ZeroStuffConvT2d(nn.Module):
    """Exact GPU-clean ConvTranspose2d: nearest-upsample x stride zero-stuff + flipped conv2d + crop."""
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

def swap_convt(net, dummy):
    L={}; hk=[]
    for n,mo in net.named_modules():
        if isinstance(mo, nn.ConvTranspose2d):
            hk.append(mo.register_forward_pre_hook((lambda nm: (lambda mod,i: L.__setitem__(nm, i[0].shape[-2:])))(n)))
    with torch.no_grad(): net(dummy)
    for h in hk: h.remove()
    for name,mo in list(net.named_modules()):
        if isinstance(mo, nn.ConvTranspose2d) and name in L:
            par=net; *pth,last=name.split("."); 
            for q in pth: par=getattr(par,q)
            hh,ww=L[name]; setattr(par,last,ZeroStuffConvT2d(mo,hh,ww))
    return net

wc=get_model('unetnc',3,in_channels=3).eval()
wc.load_state_dict(convert_state_dict(torch.load("dewarp-src/weights/unetnc_doc3d_final.pkl",map_location='cpu')['model_state']))
bm=get_model('dnetccnl',2,in_channels=3).eval()
bm.load_state_dict(convert_state_dict(torch.load("dewarp-src/weights/dnetccnl_doc3d_final.pkl",map_location='cpu')['model_state']))
swap_convt(wc, torch.rand(1,3,256,256))
swap_convt(bm, torch.rand(1,3,128,128))

class DewarpNet(nn.Module):
    def __init__(s,wc,bm): super().__init__(); s.wc=wc; s.bm=bm
    def forward(s,x):
        w=s.wc(x); wc_out=F.relu(w)-F.relu(w-1.0)   # exact clamp(0,1); Mali rejects RELU_0_TO_1 (Hardtanh)
        return s.bm(F.interpolate(wc_out,(128,128),mode='bilinear',align_corners=False))
net=DewarpNet(wc,bm).eval()
dummy=torch.rand(1,3,256,256)
with torch.no_grad(): o=net(dummy)
print("bm out:", tuple(o.shape), "range", round(float(o.min()),3), round(float(o.max()),3))
np.save("ref_in.npy",dummy.numpy()); np.save("ref_out.npy",o.numpy())
import litert_torch
litert_torch.convert(net,(dummy,)).export("dewarp.tflite")
print("saved %.1f MB"%(os.path.getsize("dewarp.tflite")/1e6))
