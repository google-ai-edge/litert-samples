# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Desktop gate for the exported MI-GAN inpainting graph.

Runs the PyTorch reference and the exported .tflite through the LiteRT
CompiledModel API on one demo image + mask and prints their output correlation.
"""
import build_migan as B
import numpy as np
import torch
import os
from PIL import Image, ImageDraw
HERE=os.path.dirname(os.path.abspath(__file__))
R=B.RES
im=Image.open(f"{HERE}/scene.jpg").convert("RGB")
s=min(im.size)
im=im.crop(((im.width-s)//2,(im.height-s)//2,(im.width+s)//2,(im.height+s)//2)).resize((R,R),Image.BILINEAR)
arr=np.asarray(im).astype(np.float32)/255.0
img=(arr-0.5)*2  # [-1,1] HWC
# mask: 1=keep, 0=fill. Remove an elliptical region (lower-center, where a subject often sits)
m=Image.new("L",(R,R),255)
d=ImageDraw.Draw(m)
cx,cy=int(R*0.5),int(R*0.62)
rw,rh=int(R*0.17),int(R*0.30)
d.ellipse([cx-rw,cy-rh,cx+rw,cy+rh],fill=0)
mask=(np.asarray(m).astype(np.float32)/255.0)[...,None]  # HWC 1ch, 1 keep/0 fill
imgc=img.transpose(2,0,1)[None]
maskc=mask.transpose(2,0,1)[None]
x=np.concatenate([maskc-0.5, imgc*maskc],axis=1).astype(np.float32)  # [1,4,R,R]
m_=B.build()
with torch.no_grad(): out=m_(torch.from_numpy(x)).numpy()  # [-1,1]
np.save(f"{HERE}/mg_ref.npy",out)
x.tofile(f"{HERE}/mg_input.bin")
np.save(f"{HERE}/mg_mask.npy",maskc)
np.save(f"{HERE}/mg_img.npy",imgc)
def comp(o): 
    c=imgc*maskc+o*(1-maskc)
    c=((c[0].transpose(1,2,0)*0.5+0.5)*255).clip(0,255).astype(np.uint8)
    return Image.fromarray(c)
comp(out).save(f"{HERE}/mg_torch.png")
masked=((imgc*maskc)[0].transpose(1,2,0)*0.5+0.5)*255
Image.fromarray(masked.clip(0,255).astype(np.uint8)).save(f"{HERE}/mg_masked.png")
im.save(f"{HERE}/mg_in.png")
print(f"input {x.shape}; torch out range [{out.min():.2f},{out.max():.2f}]")
o=B.run_tflite(f"{HERE}/migan_fp16.tflite",x)
print(f"desktop-fp16 vs torch corr {np.corrcoef(o,out.ravel())[0,1]:.6f}")
