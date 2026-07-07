import torch, torch.nn as nn, numpy as np, os
from sixdrepnet.model import SixDRepNet
from huggingface_hub import hf_hub_download
net = SixDRepNet(backbone_name='RepVGG-B1g2', backbone_file='', deploy=True, pretrained=False).eval()
sd = torch.load(hf_hub_download("osanseviero/6DRepNet_300W_LP_AFLW2000","model.pth"), map_location="cpu")
sd = sd.get('model_state_dict', sd.get('state_dict', sd)) if isinstance(sd, dict) else sd
sd = {k[7:] if k.startswith('module.') else k: v for k,v in sd.items()}
print("load:", net.load_state_dict(sd, strict=False))

class Wrap(nn.Module):
    def __init__(s, n): super().__init__(); s.n=n
    def forward(s, x):
        x=s.n.layer0(x); x=s.n.layer1(x); x=s.n.layer2(x); x=s.n.layer3(x); x=s.n.layer4(x)
        x=s.n.gap(x); x=torch.flatten(x,1)
        return s.n.linear_reg(x)   # 6D [1,6]
w=Wrap(net).eval()
dummy=torch.randn(1,3,224,224)
with torch.no_grad(): o=w(dummy)
print("6D out:", tuple(o.shape))
np.save("ref_in.npy", dummy.numpy()); np.save("ref_out.npy", o.numpy())
import litert_torch
litert_torch.convert(w,(dummy,)).export("6drepnet.tflite")
print("saved %.1f MB"%(os.path.getsize("6drepnet.tflite")/1e6))
