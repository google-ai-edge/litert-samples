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

"""Pixel 8a device-gate fixtures for Metric3D v2, using a REAL image (depth structure).
Writes probe_input.bin (NCHW LE f32), torch_ref.npy (re-authored), orig_ref.npy (original Metric3D).
Prints original-vs-reauthored and desktop-fp16-vs-reauthored corr."""
import load_m3d
import numpy as np
import torch
import os
import glob
import torch.nn.functional as F
from PIL import Image
import build_m3d as B

HERE = os.path.dirname(os.path.abspath(__file__))
HUB = os.path.expanduser("~/.cache/torch/hub/yvanyin_metric3d_main")
imgs = sorted(glob.glob(os.path.join(HUB, "data/wild_demo/*.jpg")))
pick = imgs[0]
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)
im = Image.open(pick).convert("RGB").resize((B.INPUT, B.INPUT), Image.BILINEAR)
arr = np.asarray(im).astype(np.float32)                       # HxWx3, 0-255
arr = (arr - MEAN) / STD
img_np = arr.transpose(2, 0, 1)[None].copy()                  # [1,3,H,W]
img = torch.from_numpy(img_np)
print(f"image {os.path.basename(pick)} -> [1,3,{B.INPUT},{B.INPUT}] norm range [{img_np.min():.2f},{img_np.max():.2f}]")

# ORIGINAL (before any patch)
m = load_m3d.load()
with torch.no_grad():
    orig = m({"input": img})[0].numpy()
np.save(os.path.join(HERE, "orig_ref.npy"), orig)

# re-authored
F.interpolate = B._fixed_interp
B.patch_encoder(m.depth_model.encoder)
B.patch_decoder(m.depth_model.decoder)
wrap = B.M3DWrap(m).eval()
B.finalize_zsct(m.depth_model.decoder, wrap)
with torch.no_grad():
    ref = wrap(img).numpy()
np.save(os.path.join(HERE, "torch_ref.npy"), ref)
img_np.astype(np.float32).tofile(os.path.join(HERE, "probe_input.bin"))
print(f"original depth [{orig.min():.2f},{orig.max():.2f}]m  reauth [{ref.min():.2f},{ref.max():.2f}]m  "
      f"orig-vs-reauth corr {np.corrcoef(orig.ravel(), ref.ravel())[0,1]:.6f}")

# desktop reference run through the LiteRT CompiledModel API (same API as on-device)
o = B.run_tflite(os.path.join(HERE, "m3d_fp16.tflite"), img_np)
print(f"desktop-fp16 vs reauth corr {np.corrcoef(o.ravel(), ref.ravel())[0,1]:.6f}  vs orig {np.corrcoef(o.ravel(), orig.ravel())[0,1]:.6f}")
