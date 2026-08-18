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

"""Desktop gate for the exported L2CS gaze graph.

Runs the PyTorch reference and the exported .tflite through the LiteRT
CompiledModel API on one demo face crop and prints the yaw-logit correlation.
"""
import build_gaze as B
import numpy as np
import torch
import os
from PIL import Image, ImageDraw
import math
HERE=os.path.dirname(os.path.abspath(__file__))
S=B.SIZE
MEAN=np.array([0.485,0.456,0.406],np.float32)
STD=np.array([0.229,0.224,0.225],np.float32)
im=Image.open(f"{HERE}/face.jpg").convert("RGB")
s=min(im.size)
im=im.crop(((im.width-s)//2,(im.height-s)//2,
            (im.width+s)//2,(im.height+s)//2)).resize((S,S),Image.BILINEAR)
x=(((np.asarray(im).astype(np.float32)/255-MEAN)/STD)
   .transpose(2,0,1)[None]).copy()
m=B.build()
with torch.no_grad():
    y,p=m(torch.from_numpy(x))
    y,p=y.numpy()[0],p.numpy()[0]
np.save(f"{HERE}/gz_y.npy",y)
np.save(f"{HERE}/gz_p.npy",p)
x.tofile(f"{HERE}/gz_input.bin")
def decode(y,p):
    """Converts 90-bin softmax distributions to yaw/pitch degrees.

    Args:
        y: (90,) softmax yaw distribution.
        p: (90,) softmax pitch distribution.

    Returns:
        A (yaw, pitch) tuple in degrees via the binned expectation.
    """
    idx=np.arange(90)
    yaw=(y*idx).sum()*4-180
    pit=(p*idx).sum()*4-180
    return yaw, pit
def draw(y,p,name):
    """Draws the decoded gaze direction on the demo face crop.

    Args:
        y: (90,) softmax yaw distribution.
        p: (90,) softmax pitch distribution.
        name: Output PNG path.

    Returns:
        The decoded (yaw, pitch) tuple in degrees.
    """
    yaw,pit=decode(y,p)
    yr,pr=math.radians(yaw),math.radians(pit)
    d=im.copy()
    dr=ImageDraw.Draw(d)
    cx,cy=S//2,S//2
    L=S*0.3
    dx=-L*math.sin(yr)*math.cos(pr)
    dy=-L*math.sin(pr)
    dr.line([cx,cy,cx+dx,cy+dy],fill=(255,40,40),width=6)
    dr.ellipse([cx-6,cy-6,cx+6,cy+6],fill=(0,200,0))
    d.save(name)
    return yaw,pit
yaw,pit=draw(y,p,f"{HERE}/gz_torch.png")
im.save(f"{HERE}/gz_face.png")
print(f"input {x.shape}; torch gaze yaw {yaw:.1f} pitch {pit:.1f} deg")
from ai_edge_litert.compiled_model import CompiledModel
model=CompiledModel.from_file(f"{HERE}/gaze_fp16.tflite")
ins=model.create_input_buffers(0)
outs=model.create_output_buffers(0)
ins[0].write(np.ascontiguousarray(x,dtype=np.float32))
model.run_by_index(0,ins,outs)
n=(model.get_output_buffer_requirements(0,0)["buffer_size"]
   //np.dtype(np.float32).itemsize)
o0=outs[0].read(n,np.float32)
print(f"desktop-fp16 vs torch yaw corr {np.corrcoef(o0,y)[0,1]:.6f}")
