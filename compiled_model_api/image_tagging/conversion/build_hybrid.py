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

"""Phase 2b: split the Swin encoder for the device-verified hybrid.
  G1 (GPU): image -> feat[1,144,1536]   (patch_embed + stages 0-2)   -- device corr 0.998
  C2 (CPU): feat  -> image_embeds[1,145,512]  (stage 3 + norm + cls + image_proj) -- fp16-fragile on GPU
Run in ~/clipconv:  python build_hybrid.py"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from ram_load import load_ram_plus
from build_swin import patch_gpu_clean, SwinEncoder, corr
WORK = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(WORK, "out")
REF = np.load(os.path.join(WORK, "ref", "ref_demo1.npz"))

class SwinS012(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.ve = model.visual_encoder
    def forward(self, image):
        ve = self.ve
        x = ve.pos_drop(ve.patch_embed(image))
        x = ve.layers[0](x)
        x = ve.layers[1](x)
        x = ve.layers[2](x)
        return x                                       # [1,144,1536]

class Stage3Tail(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.ve = model.visual_encoder
        self.image_proj = model.image_proj
    def forward(self, feat):
        ve = self.ve
        x = ve.layers[3](feat)
        x = ve.norm(x)
        x = torch.cat([x.mean(1, keepdim=True), x], dim=1)   # [1,145,1536]
        return self.image_proj(x)                            # [1,145,512]

def main():
    model = load_ram_plus(384)
    patch_gpu_clean(model)
    g1 = SwinS012(model).eval()
    c2 = Stage3Tail(model).eval()
    enc = SwinEncoder(model).eval()
    image = torch.from_numpy(REF["image"])
    with torch.no_grad():
        feat = g1(image)
        ie_split = c2(feat).numpy()
        ie_full = enc(image).numpy()
    print("split vs full image_embeds corr:", corr(ie_split, ie_full),
          " vs REF:", corr(ie_split, REF["image_embeds"]))
    print("feat[1,144,1536] absmax", float(feat.abs().max()))

    import litert_torch
    sys.path.insert(0, os.path.expanduser("~/Downloads/meeting/cmgan-work"))
    from build_cmgan import opcheck, to_fp16
    # G1 GPU
    a32 = os.path.join(OUT, "ram_swin_s012.tflite")
    litert_torch.convert(g1.eval(), (image,)).export(a32)
    _, cleanA = opcheck(a32, "G1 s012 fp32")
    if cleanA:
        to_fp16(a32, os.path.join(OUT, "ram_swin_s012_fp16.tflite"))
        opcheck(os.path.join(OUT, "ram_swin_s012_fp16.tflite"), "G1 s012 fp16")
    # C2 CPU: keep the GPU-clean patched ops (tanh-GELU/safe-LN). Exact erf-GELU was tried and gained
    # essentially nothing (image_embeds 0.99996 -> 0.99997) while adding a GELU op, so not worth it.
    c32 = os.path.join(OUT, "ram_stage3_tail.tflite")
    litert_torch.convert(c2.eval(), (feat,)).export(c32)
    opcheck(c32, "C2 stage3-tail fp32")
    to_fp16(c32, os.path.join(OUT, "ram_stage3_tail_fp16.tflite"))
    # fixtures: G1 input already swin_input.bin; save feat + image_embeds refs
    feat.numpy().astype("<f4").tofile(os.path.join(OUT, "feat_ref.bin"))
    REF["image_embeds"].astype("<f4").tofile(os.path.join(OUT, "iemb_ref.bin"))
    print("wrote G1 + C2 graphs + feat_ref/iemb_ref")

if __name__ == "__main__":
    main()
