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

"""Build GPU-compatible ORMBG (open real-world background removal) via
litert-torch.

Expects the `ormbg` python package next to this script and the checkpoint at
models/ormbg.pth (both from https://huggingface.co/schirrmacher/ormbg).

Only patch (defensive): align_corners=True -> False in F.interpolate.

Run: python build_ormbg.py
  # -> ormbg.tflite ([1,3,1024,1024] -> [1,1,1024,1024] mask)
  #    + ref_in/ref_out.npy
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# defensive: GPU delegate rejects align_corners=True
_orig = F.interpolate
def _patched(*a, **k):
    """Calls F.interpolate with align_corners=True demoted to False.

    Args:
        *a: Positional arguments for F.interpolate.
        **k: Keyword arguments for F.interpolate.

    Returns:
        The result of the original F.interpolate.
    """
    if k.get("align_corners") is True:
        k["align_corners"] = False
    return _orig(*a, **k)
F.interpolate = _patched

from ormbg.models.ormbg import ORMBG


class Wrap(nn.Module):
    """Expose only the sigmoid(d1) main mask output for conversion."""

    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return self.n(x)[0][0]   # sigmoid(d1) main mask [1,1,1024,1024]


def main():
    """Converts ORMBG to ormbg.tflite and saves reference tensors."""
    net = ORMBG()
    sd = torch.load("models/ormbg.pth", map_location="cpu")
    net.load_state_dict(sd)
    net.eval()

    w = Wrap(net).eval()
    dummy = torch.rand(1, 3, 1024, 1024)
    with torch.no_grad():
        o = w(dummy)
    print("out:", tuple(o.shape), "range", float(o.min()), float(o.max()))
    np.save("ref_in.npy", dummy.numpy())
    np.save("ref_out.npy", o.numpy())
    import litert_torch
    litert_torch.convert(w, (dummy,)).export("ormbg.tflite")
    print("saved %.1f MB" % (os.path.getsize("ormbg.tflite") / 1e6))


if __name__ == "__main__":
    main()
