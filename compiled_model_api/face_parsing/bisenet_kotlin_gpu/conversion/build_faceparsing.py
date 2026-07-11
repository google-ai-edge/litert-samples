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

"""Build GPU-compatible BiSeNet face-parsing tflite via litert-torch.

Three GPU re-authoring patches:
  1. align_corners=True -> False (GPU delegate rejects align_corners=True
     resize).
  2. Global avg_pool2d(x, x.size()[2:]) -> mean([2,3]) (Mali rejects the
     full-spatial-kernel AVERAGE_POOL_2D; a MEAN reduce is supported).
  3. ResNet maxpool -inf-pad -> explicit 0-pad + unpadded maxpool (Mali rejects
     the PADV2 (-inf pad) that MaxPool2d(padding=1) lowers to; 0-pad is exact
     here since the maxpool input is post-ReLU (>= 0) so the max is unaffected).

Setup:
    pip install torch litert-torch huggingface_hub
    # or set FP_REPO
    git clone https://github.com/zllrunning/face-parsing.PyTorch.git fp

Run:
    FP_REPO=./fp python build_faceparsing.py
    # -> faceparsing.tflite  ([1,3,512,512] -> [1,19,512,512]
    #    CelebAMask-HQ logits)
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.environ.get("FP_REPO", "fp"))
from model import BiSeNet

# patch 1: align_corners=True -> False (GPU delegate rejects
# align_corners=True resize)
_orig = F.interpolate
def _patched(*a, **k):
    """Forwards to F.interpolate with align_corners=True forced to False.

    Args:
        *a: Positional arguments of F.interpolate.
        **k: Keyword arguments of F.interpolate.

    Returns:
        The interpolated tensor.
    """
    if k.get("align_corners") is True:
        k["align_corners"] = False
    return _orig(*a, **k)
F.interpolate = _patched

# patch 2: global avg_pool2d(x, x.size()[2:]) -> mean([2,3]) — the Mali
# delegate rejects AVERAGE_POOL_2D with a full-spatial kernel; a MEAN
# reduce is supported.
_avg = F.avg_pool2d
def _avg_patched(x, kernel_size, *a, **k):
    """Routes full-spatial global average pools to a MEAN reduce.

    Args:
        x: Input tensor (B, C, H, W).
        kernel_size: Pooling kernel size, int or (kh, kw).
        *a: Remaining positional arguments of F.avg_pool2d.
        **k: Remaining keyword arguments of F.avg_pool2d.

    Returns:
        The pooled tensor.
    """
    ks = (tuple(kernel_size) if isinstance(kernel_size, (tuple, list))
          else (kernel_size, kernel_size))
    if ks == tuple(x.shape[-2:]):
        return x.mean(dim=[2, 3], keepdim=True)
    return _avg(x, kernel_size, *a, **k)
F.avg_pool2d = _avg_patched


class ZeroPadMaxPool(nn.Module):
    """patch 3: explicit 0-pad + unpadded maxpool.

    Replaces the -inf PADV2 that MaxPool2d(padding=1) lowers to.
    """

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), value=0.0)
        return F.max_pool2d(x, kernel_size=3, stride=2, padding=0)


class Wrap(nn.Module):
    """Expose only the main segmentation head for conversion."""

    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        o = self.n(x)
        return o[0] if isinstance(o, (list, tuple)) else o


def main():
    """Patches BiSeNet, loads weights, and exports faceparsing.tflite."""
    net = BiSeNet(n_classes=19).eval()
    for name, mod in list(net.named_modules()):
        if isinstance(mod, nn.MaxPool2d):
            parent = net
            *path, last = name.split(".")
            for p in path:
                parent = getattr(parent, p)
            setattr(parent, last, ZeroPadMaxPool())
    net.load_state_dict(torch.load(
        hf_hub_download("AI2lab/face-parsing.PyTorch", "79999_iter.pth"),
        map_location="cpu", weights_only=True))

    dummy = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        ref = Wrap(net)(dummy)
    print("wrap out:", tuple(ref.shape))
    import litert_torch
    litert_torch.convert(Wrap(net).eval(),
                         (dummy,)).export("faceparsing.tflite")
    print("saved %.1f MB" % (os.path.getsize("faceparsing.tflite") / 1e6))


if __name__ == "__main__":
    main()
