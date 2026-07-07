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

import build_yunet as B
import numpy as np
import torch
import os
import math
from PIL import Image, ImageDraw
HERE=os.path.dirname(os.path.abspath(__file__))
S=B.SIZE
STR=[8,16,32]
im0=Image.open(f"{HERE}/faces.jpg").convert("RGB")
ow,oh=im0.size
im=im0.resize((S,S),Image.BILINEAR)
rgb=np.asarray(im).astype(np.float32)            # RGB 0-255
bgr=rgb[:,:,::-1].copy()                          # -> BGR (to_rgb=False)
x=bgr.transpose(2,0,1)[None].copy()
m=B.build()
with torch.no_grad(): ys=[y.numpy() for y in m(torch.from_numpy(x))]   # 12: cls3,obj3,bbox3,kps3 (torch order)
x.tofile(f"{HERE}/yn_input.bin")
# torch refs grouped
ref={'cls':ys[0:3],'obj':ys[3:6],'bbox':ys[6:9],'kps':ys[9:12]}
def decode(cls,obj,bbox,kps,thr=0.6):
    boxes=[]
    scores=[]
    kpts=[]
    for li,s in enumerate(STR):
        fw=S//s
        n=fw*fw
        c=cls[li][0,:,0]
        o=obj[li][0,:,0]
        bb=bbox[li][0]
        kp=kps[li][0]
        sc=c*o
        for i in range(n):
            if sc[i]<thr: continue
            col=i%fw
            row=i//fw
            px=col*s
            py=row*s
            cx=bb[i,0]*s+px
            cy=bb[i,1]*s+py
            w=math.exp(bb[i,2])*s
            h=math.exp(bb[i,3])*s
            boxes.append([cx-w/2,cy-h/2,cx+w/2,cy+h/2])
            scores.append(float(sc[i]))
            kpts.append([(kp[i,2*j]*s+px,kp[i,2*j+1]*s+py) for j in range(5)])
    return boxes,scores,kpts
def nms(boxes,scores,kpts,iou=0.45):
    idx=sorted(range(len(scores)),key=lambda k:-scores[k])
    keep=[]
    def IOU(a,b):
        x1=max(a[0],b[0])
        y1=max(a[1],b[1])
        x2=min(a[2],b[2])
        y2=min(a[3],b[3])
        inter=max(0,x2-x1)*max(0,y2-y1)
        ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
        return inter/ua if ua>0 else 0
    while idx:
        i=idx.pop(0)
        keep.append(i)
        idx=[j for j in idx if IOU(boxes[i],boxes[j])<iou]
    return [(boxes[i],scores[i],kpts[i]) for i in keep]
def draw(dets,name):
    d=im.copy()
    dr=ImageDraw.Draw(d)
    for box,sc,kp in dets:
        dr.rectangle(box,outline=(0,230,0),width=3)
        for (kx,ky) in kp: dr.ellipse([kx-3,ky-3,kx+3,ky+3],fill=(255,40,40))
    d.save(name)
    return len(dets)
tb,ts,tk=decode(ref['cls'],ref['obj'],ref['bbox'],ref['kps'])
td=nms(tb,ts,tk)
print(f"input {x.shape} (BGR); torch faces {draw(td,f'{HERE}/yn_torch.png')}")
for i,y in enumerate(ys): np.save(f"{HERE}/yn_t{i}.npy",y)
# desktop fp16 + establish tflite output order vs torch
from ai_edge_litert.compiled_model import CompiledModel
model=CompiledModel.from_file(f"{HERE}/yunet_fp16.tflite")
sig=model.get_signature_list()
key=list(sig)[0]
det=model.get_output_tensor_details(key)
ins=model.create_input_buffers(0)
ob=model.create_output_buffers(0)
ins[0].write(np.ascontiguousarray(x,dtype=np.float32))
model.run_by_index(0,ins,ob)
outs=[ob[i].read(int(np.prod(det[n]["shape"])),np.float32).reshape(det[n]["shape"]) for i,n in enumerate(sig[key]["outputs"])]
# map each tflite output to torch ref by shape+corr
order=[]
for o in outs:
    best=-2
    bi=-1
    for ti,t in enumerate(ys):
        if t.shape==o.shape:
            cc=np.corrcoef(o.ravel(),t.ravel())[0,1]
            if cc>best:
                best=cc
                bi=ti
    order.append((bi,best))
print("tflite->torch idx map:",[bi for bi,_ in order],"min corr",round(min(b for _,b in order),4))
np.save(f"{HERE}/yn_order.npy",np.array([bi for bi,_ in order]))
