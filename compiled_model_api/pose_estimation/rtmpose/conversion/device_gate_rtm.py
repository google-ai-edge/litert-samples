# Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
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

import build_rtm as B  # reuses stubs + Wrap + cfg/ckpt
import numpy as np
import torch
import os
from PIL import Image, ImageDraw
from mmpose.apis import init_model
HERE=os.path.dirname(os.path.abspath(__file__))
MEAN=np.array([123.675,116.28,103.53],np.float32)
STD=np.array([58.395,57.12,57.375],np.float32)
# person crop -> 192x256 (WxH)
im=Image.open(HERE+"/person.jpg").convert("RGB")
tw,th=192,256
ar=tw/th
w,h=im.size
if w/h>ar:
    nw=int(h*ar)
    im=im.crop(((w-nw)//2,0,(w+nw)//2,h))
else:
    nh=int(w/ar)
    im=im.crop((0,(h-nh)//2,w,(h+nh)//2))
im=im.resize((tw,th),Image.BILINEAR)
arr=np.asarray(im).astype(np.float32)
img_np=((arr-MEAN)/STD).transpose(2,0,1)[None].copy()  # [1,3,256,192]
m=init_model(B.cfg,B.ckpt,device="cpu").eval()
wnet=B.Wrap(m).eval()
with torch.no_grad():
    sx,sy=wnet(torch.from_numpy(img_np))
    sx,sy=sx.numpy()[0],sy.numpy()[0]
np.save(HERE+"/rtm_sx.npy",sx)
np.save(HERE+"/rtm_sy.npy",sy)
img_np.tofile(HERE+"/rtm_input.bin")
def decode(sx,sy):
    xb=sx.argmax(1)/2.0
    yb=sy.argmax(1)/2.0  # split=2 -> pixel
    conf=(sx.max(1)+sy.max(1))/2
    return np.stack([xb,yb,conf],1)  # [17,3]
kp=decode(sx,sy)
SK=[(5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16),(5,6),(11,12),(5,11),(6,12),(0,1),(0,2),(1,3),(2,4),(0,5),(0,6)]
def draw(kp,name):
    d=im.copy()
    dr=ImageDraw.Draw(d)
    for a,b in SK:
        if kp[a,2]>0.3 and kp[b,2]>0.3: dr.line([tuple(kp[a,:2]),tuple(kp[b,:2])],fill=(0,255,0),width=3)
    for x,y,c in kp:
        if c>0.3: dr.ellipse([x-3,y-3,x+3,y+3],fill=(255,40,40))
    d.save(name)
draw(kp,HERE+"/rtm_torch.png")
print(f"input {img_np.shape}; torch decoded {int((kp[:,2]>0.3).sum())}/17 kpts conf>0.3")
from ai_edge_litert.interpreter import Interpreter
it=Interpreter(model_path=HERE+"/rtm_fp16.tflite")
it.allocate_tensors()
d=it.get_input_details()[0]
it.set_tensor(d["index"],img_np.astype(d["dtype"]))
it.invoke()
od=it.get_output_details()
outs={tuple(o["shape"]):it.get_tensor(o["index"])[0] for o in od}
ox=outs[(1,17,384)]
oy=outs[(1,17,512)]
print(f"desktop-fp16 vs torch corr x {np.corrcoef(ox.ravel(),sx.ravel())[0,1]:.5f} y {np.corrcoef(oy.ravel(),sy.ravel())[0,1]:.5f}")
