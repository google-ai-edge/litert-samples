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

import build_unisal as B
import numpy as np
import torch
import os
from PIL import Image, ImageFilter
HERE=os.path.dirname(os.path.abspath(__file__))
S=B.SIZE
MEAN=np.array([0.485,0.456,0.406],np.float32)
STD=np.array([0.229,0.224,0.225],np.float32)
im=Image.open(f"{HERE}/scene.jpg").convert("RGB")
s=min(im.size)
im=im.crop(((im.width-s)//2,(im.height-s)//2,(im.width+s)//2,(im.height+s)//2)).resize((S,S),Image.BILINEAR)
x=(((np.asarray(im).astype(np.float32)/255-MEAN)/STD).transpose(2,0,1)[None]).copy()
m=B.build()
with torch.no_grad(): ref=m(torch.from_numpy(x)).numpy()[0,0]
np.save(f"{HERE}/us_ref.npy",ref)
x.tofile(f"{HERE}/us_input.bin")
im.save(f"{HERE}/us_in.png")
def overlay(sal,name):
    sm=Image.fromarray((sal*255).clip(0,255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(8))
    s2=np.asarray(sm).astype(np.float32)
    s2=(s2-s2.min())/(s2.max()-s2.min()+1e-6)
    import matplotlib.cm as cm
    heat=(cm.get_cmap("jet")(s2)[:,:,:3]*255).astype(np.uint8)
    out=(0.5*np.asarray(im)+0.5*heat).clip(0,255).astype(np.uint8)
    Image.fromarray(out).save(name)
overlay((ref-ref.min())/(ref.max()-ref.min()+1e-6),f"{HERE}/us_torch.png")
print(f"input {x.shape}; torch saliency range [{ref.min():.2f},{ref.max():.2f}]")
from ai_edge_litert.interpreter import Interpreter
it=Interpreter(model_path=f"{HERE}/unisal_fp16.tflite")
it.allocate_tensors()
d=it.get_input_details()[0]
it.set_tensor(d["index"],x.astype(d["dtype"]))
it.invoke()
o=it.get_tensor(it.get_output_details()[0]["index"]).ravel()
print(f"desktop-fp16 vs torch corr {np.corrcoef(o,ref.ravel())[0,1]:.6f}")
