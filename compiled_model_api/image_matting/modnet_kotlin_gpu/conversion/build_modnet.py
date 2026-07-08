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

"""Build GPU-compatible MODNet (trimap-free portrait matting) tflite.

Converted with litert-torch. Two GPU re-authoring patches:
  1. SE block: Linear channel-attention -> 1x1 conv (avoids a NCHW/NHWC
     broadcast-mul mismatch).
  2. IBNorm: InstanceNorm2d -> fp16-safe hierarchical-mean instance norm
     (the stock instance-norm variance overflows fp16 over large spatial
     maps on the Mali GPU delegate, degrading the matte).

Setup:
    pip install torch litert-torch huggingface_hub
    git clone https://github.com/ZHKKKe/MODNet.git    # or set MODNET_REPO

Run:
    MODNET_REPO=./MODNet python build_modnet.py
    # -> modnet.tflite  (26 MB, [1,3,512,512] -> [1,1,512,512] alpha,
    #    0 banned ops)
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.environ.get("MODNET_REPO", "MODNet"))
from src.models.modnet import MODNet, SEBlock, IBNorm


def _hier_mean(t):
    """Exact global spatial mean via a cascade of /2 average-pools.

    Magnitude-safe (each stage averages 4 values) unlike one big SUM that
    overflows fp16 on Mali.

    Args:
        t: Input tensor (B, C, H, W).

    Returns:
        Tensor (B, C, 1, 1) holding the per-channel spatial mean.
    """
    while t.shape[-1] > 1 or t.shape[-2] > 1:
        kh = 2 if t.shape[-2] > 1 else 1
        kw = 2 if t.shape[-1] > 1 else 1
        t = F.avg_pool2d(t, (kh, kw), ceil_mode=True)
    return t


def patch_se(se):
    """Rewrite the SE block's Linear attention as 1x1 convs (GPU-friendly).

    Args:
        se: SEBlock module patched in place.
    """
    lin1, lin2 = se.fc[0], se.fc[2]
    ci, cm, co = lin1.in_features, lin1.out_features, lin2.out_features
    c1 = nn.Conv2d(ci, cm, 1, bias=False)
    c1.weight.data = lin1.weight.data.view(cm, ci, 1, 1)
    c2 = nn.Conv2d(cm, co, 1, bias=False)
    c2.weight.data = lin2.weight.data.view(co, cm, 1, 1)
    se._c1, se._c2 = c1, c2
    se.forward = lambda x: x * torch.sigmoid(se._c2(F.relu(se._c1(se.pool(x)))))


def patch_ibnorm(ib, eps=1e-5):
    """Replace InstanceNorm2d with the fp16-safe hierarchical-mean form.

    Args:
        ib: IBNorm module patched in place.
        eps: Variance epsilon of the instance-norm denominator.
    """
    bc = ib.bnorm_channels
    def fwd(x):
        bn_x = ib.bnorm(x[:, :bc].contiguous())
        ix = x[:, bc:].contiguous()
        mean = _hier_mean(ix)
        dd = ix - mean
        in_x = dd * torch.rsqrt(_hier_mean(dd * dd) + eps)
        return torch.cat((bn_x, in_x), 1)
    ib.forward = fwd


class Wrap(nn.Module):
    """Expose only the final alpha matte output for conversion."""

    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return self.n(x, True)[2]


def main():
    """Loads MODNet weights, applies the GPU patches, exports .tflite."""
    ckpt = hf_hub_download("DavG25/modnet-pretrained-models",
                           "models/modnet_photographic_portrait_matting.ckpt")
    m = MODNet(backbone_pretrained=False).eval()
    sd = {k.replace("module.", ""): v
          for k, v in torch.load(ckpt, map_location="cpu",
                                 weights_only=True).items()}
    m.load_state_dict(sd)
    n_se = n_ib = 0
    for mod in m.modules():
        if isinstance(mod, SEBlock):
            patch_se(mod)
            n_se += 1
        if isinstance(mod, IBNorm):
            patch_ibnorm(mod)
            n_ib += 1
    print(f"patched {n_se} SE blocks, {n_ib} IBNorms")

    import litert_torch
    litert_torch.convert(
        Wrap(m).eval(),
        (torch.randn(1, 3, 512, 512),)).export("modnet.tflite")
    print("saved modnet.tflite (%.1f MB)"
          % (os.path.getsize("modnet.tflite") / 1e6))


if __name__ == "__main__":
    main()
