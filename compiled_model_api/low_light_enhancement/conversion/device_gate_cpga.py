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

"""Desktop gate for the exported CPGA-Net graph.

Runs the PyTorch reference and the exported .tflite through the LiteRT
CompiledModel API on one demo image and prints their output correlation.
"""
import build_cpga as B
import numpy as np
import torch
import os
import sys
from PIL import Image
HERE=os.path.dirname(os.path.abspath(__file__))
S=B.SIZE
p=sys.argv[1] if len(sys.argv)>1 else "low.jpg"
im=Image.open(p).convert("RGB")
s=min(im.size)
im=im.crop(((im.width-s)//2,(im.height-s)//2,(im.width+s)//2,(im.height+s)//2)).resize((S,S),Image.BILINEAR)
x=(np.asarray(im).astype(np.float32)/255.0).transpose(2,0,1)[None].copy()
m=B.build()
with torch.no_grad(): out=m(torch.from_numpy(x)).numpy()
np.save(f"{HERE}/cp_ref.npy",out)
x.tofile(f"{HERE}/cp_input.bin")
im.save(f"{HERE}/cp_in.png")
Image.fromarray((out[0].transpose(1,2,0)*255).clip(0,255).astype(np.uint8)).save(f"{HERE}/cp_torch.png")
print(f"input mean {x.mean():.3f} (dark); torch out mean {out.mean():.3f} (enhanced)")
o=B.run_tflite(f"{HERE}/cpga_fp16.tflite",x)
print(f"desktop-fp16 vs torch corr {np.corrcoef(o,out.ravel())[0,1]:.6f}")
