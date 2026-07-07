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

import build_nafnet as B
import numpy as np
import torch
import os
import glob
from PIL import Image
HERE=os.path.dirname(os.path.abspath(__file__))
HUB=os.path.expanduser("~/.cache/torch/hub/yvanyin_metric3d_main")
p=sorted(glob.glob(HUB+"/data/wild_demo/*.jpg"))[0]
im=Image.open(p).convert("RGB")
s=min(im.size)
im=im.crop(((im.width-s)//2,(im.height-s)//2,(im.width+s)//2,(im.height+s)//2)).resize((B.SIZE,B.SIZE),Image.BILINEAR)
img_np=(np.asarray(im).astype(np.float32)/255.0).transpose(2,0,1)[None].copy()  # [1,3,SIZE,SIZE] in [0,1]
m=B.build()
with torch.no_grad(): ref=m(torch.from_numpy(img_np)).numpy()
np.save(HERE+"/naf_ref.npy",ref)
img_np.tofile(HERE+"/naf_input.bin")
print(f"input {img_np.shape} -> naf_input.bin; ref range [{ref.min():.3f},{ref.max():.3f}]")
from ai_edge_litert.interpreter import Interpreter
it=Interpreter(model_path=HERE+"/nafnet_fp16.tflite")
it.allocate_tensors()
d=it.get_input_details()[0]
it.set_tensor(d["index"],img_np.astype(d["dtype"]))
it.invoke()
o=it.get_tensor(it.get_output_details()[0]["index"])
print(f"desktop-fp16 vs torch corr {np.corrcoef(o.ravel(),ref.ravel())[0,1]:.6f}")
