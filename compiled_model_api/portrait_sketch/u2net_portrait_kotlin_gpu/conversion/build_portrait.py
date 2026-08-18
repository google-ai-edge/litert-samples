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

"""Build GPU-compatible U2-Net Portrait (photo -> pencil-sketch map) via
litert-torch.

Expects the xuebinqin/U-2-Net repository checked out next to this script as
U-2-Net/ (provides model/u2net.py); the portrait checkpoint is fetched from
the Maxwelltebi HF mirror. The map is dark-on-white after the host inverts it
(sketch = 1 - output).

Only patch (defensive): align_corners=True -> False in F.interpolate.

Run: python build_portrait.py
  # -> portrait.tflite ([1,3,512,512] -> [1,1,512,512] map in 0..1)
  #    + ref fixtures
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "U-2-Net"))

# defensive: GPU delegate rejects align_corners=True
_orig = F.interpolate
def _patched(*a, **k):
    """F.interpolate wrapper that forces align_corners=False.

    Args:
        *a: Positional arguments forwarded to F.interpolate.
        **k: Keyword arguments; align_corners=True is rewritten to False.

    Returns:
        The result of the original F.interpolate call.
    """
    if k.get("align_corners") is True:
        k["align_corners"] = False
    return _orig(*a, **k)
F.interpolate = _patched

from model.u2net import U2NET


class Wrap(nn.Module):
    """Expose only the finest-scale d0 map for conversion."""

    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        o = self.n(x)                    # (d0..d6) each sigmoid
        return o[0]                      # [1,1,512,512] portrait map (0..1)


def main():
    """Builds portrait.tflite from the U2-Net portrait checkpoint."""
    net = U2NET(3, 1).eval()
    sd = torch.load(
        hf_hub_download("Maxwelltebi/u2net-portrait", "u2net_portrait.pth"),
        map_location="cpu")
    sd = (sd.get('model_state_dict', sd.get('state_dict', sd))
          if isinstance(sd, dict) else sd)
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    print("load:", net.load_state_dict(sd, strict=False))

    w = Wrap(net).eval()
    dummy = torch.rand(1, 3, 512, 512)
    with torch.no_grad():
        out = w(dummy)
    print("out:", tuple(out.shape), "range",
          round(float(out.min()), 3), round(float(out.max()), 3))
    np.save("ref_in.npy", dummy.numpy())
    np.save("ref_out.npy", out.numpy())
    import litert_torch
    litert_torch.convert(w, (dummy,)).export("portrait.tflite")
    print("saved %.1f MB" % (os.path.getsize("portrait.tflite") / 1e6))


if __name__ == "__main__":
    main()
