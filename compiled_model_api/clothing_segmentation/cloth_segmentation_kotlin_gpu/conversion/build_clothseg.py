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

"""Build GPU-compatible Cloth Segmentation (U2-Net, 4 classes) via litert-torch.

Expects the cloth-segmentation repository checked out next to this script;
the checkpoint is fetched from the tryonlabs HF mirror.

Only patch (defensive): align_corners=True -> False in F.interpolate.

Run: python build_clothseg.py
  # -> clothseg.tflite ([1,3,768,768] -> [1,4,768,768] logits) + ref fixtures
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/cloth-segmentation")

# defensive: GPU delegate rejects align_corners=True
_orig = F.interpolate
def _patched(*a, **k):
    if k.get("align_corners") is True:
        k["align_corners"] = False
    return _orig(*a, **k)
F.interpolate = _patched

from networks.u2net import U2NET


class Wrap(nn.Module):
    """Expose only the finest-scale d0 logits for conversion."""

    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return self.n(x)[0]   # d0 [1,4,768,768] logits


def main():
    net = U2NET(in_ch=3, out_ch=4).eval()
    sd = torch.load(hf_hub_download("tryonlabs/u2net-cloth-segmentation", "u2net_cloth_segm.pth"),
                    map_location="cpu")
    sd = sd.get('model_state_dict', sd.get('state_dict', sd)) if isinstance(sd, dict) else sd
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    print("load:", net.load_state_dict(sd, strict=False))

    w = Wrap(net).eval()
    dummy = torch.rand(1, 3, 768, 768)
    with torch.no_grad():
        o = w(dummy)
    print("out:", tuple(o.shape))
    np.save("ref_in.npy", dummy.numpy())
    np.save("ref_out.npy", o.numpy())
    import litert_torch
    litert_torch.convert(w, (dummy,)).export("clothseg.tflite")
    print("saved %.1f MB" % (os.path.getsize("clothseg.tflite") / 1e6))


if __name__ == "__main__":
    main()
