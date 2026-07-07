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

import build_mlsd as B
import numpy as np
import torch
import os
from PIL import Image, ImageDraw
from scipy.ndimage import maximum_filter
HERE=os.path.dirname(os.path.abspath(__file__))
def preprocess(im):
    a=np.asarray(im.resize((512,512),Image.BOX).convert("RGB")).astype(np.float32)
    a=np.concatenate([a, np.ones([512,512,1],np.float32)],axis=-1).transpose(2,0,1)[None]
    return ((a/127.5)-1.0).copy()
def decode(out, score_thr=0.10, dist_thr=20.0, ksize=3, topk=200):
    center=1/(1+np.exp(-out[0]))
    disp=out[1:5]
    mx=maximum_filter(center,size=ksize)
    ys,xs=np.where((center==mx)&(center>score_thr))
    sc=center[ys,xs]
    order=sc.argsort()[::-1][:topk]
    lines=[]
    for i in order:
        y,x=ys[i],xs[i]
        dxs,dys,dxe,dye=disp[:,y,x]
        if np.hypot(dxs-dxe,dys-dye)>dist_thr: lines.append([x+dxs,y+dys,x+dxe,y+dye])
    return (np.array(lines)*2) if lines else np.zeros((0,4))
im=Image.open(HERE+"/building.jpg").convert("RGB")
x=preprocess(im)
m=B.build()
with torch.no_grad(): ref=m(torch.from_numpy(x)).numpy()[0]
np.save(HERE+"/ml_ref.npy",ref)
x.tofile(HERE+"/ml_input.bin")
ln=decode(ref)
print(f"input {x.shape}; torch decoded {len(ln)} lines")
base=im.resize((512,512),Image.BOX)
d=base.copy()
dr=ImageDraw.Draw(d)
for a,b,c,e in ln: dr.line([a,b,c,e],fill=(255,40,40),width=2)
cv=Image.new("RGB",(512*2+8,512),(255,255,255))
cv.paste(base,(0,0))
cv.paste(d,(512+8,0))
cv.save(HERE+"/mlsd_torch_demo.png")
from ai_edge_litert.compiled_model import CompiledModel
model=CompiledModel.from_file(HERE+"/mlsd_fp16.tflite")
ins=model.create_input_buffers(0)
outs=model.create_output_buffers(0)
ins[0].write(np.ascontiguousarray(x,dtype=np.float32))
model.run_by_index(0,ins,outs)
n=model.get_output_buffer_requirements(0,0)["buffer_size"]//np.dtype(np.float32).itemsize
o=outs[0].read(n,np.float32).reshape(ref.shape)
print(f"desktop-fp16 vs torch corr {np.corrcoef(o.ravel(),ref.ravel())[0,1]:.6f}")
