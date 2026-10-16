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

"""Build GPU-compatible DewarpNet (document dewarping) via litert-torch.

Expects the DewarpNet repository checked out next to this script as dewarp-src/
with the released doc3d weights in dewarp-src/weights/ (unetnc_doc3d_final.pkl,
dnetccnl_doc3d_final.pkl).

Two GPU re-authoring patches:
  1. ZeroStuffConvT2d — exact ConvTranspose2d replacement (Mali rejects
     TRANSPOSE_CONV): nearest-upsample x stride zero-stuff + flipped
     conv2d + crop.
  2. Hardtanh(0,1) -> relu(x) - relu(x-1) (exact clamp; Mali rejects
     RELU_0_TO_1).

The graph emits the backward map; the host applies the grid_sample unwarp.

Run: python build_dewarp.py
  # -> dewarp.tflite ([1,3,256,256] -> [1,2,128,128] backward map)
  #    + ref fixtures
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/dewarp-src")
from models import get_model
from utils import convert_state_dict


class ZeroStuffConvT2d(nn.Module):
    """Exact GPU-clean ConvTranspose2d.

    Nearest-upsample x stride zero-stuff + flipped conv2d + crop.
    """

    def __init__(self, ct, h_in, w_in):
        super().__init__()
        self.s = ct.stride[0]
        self.k = ct.kernel_size[0]
        self.p = ct.padding[0]
        self.op = ct.output_padding[0]
        self.h_in = h_in
        self.w_in = w_in
        w = ct.weight.detach().flip(2).flip(3).permute(1, 0, 2, 3).contiguous()
        self.register_buffer("w", w)
        self.register_buffer(
            "b", ct.bias.detach().clone() if ct.bias is not None
            else torch.zeros(ct.out_channels))
        mh = np.zeros((h_in * self.s, w_in * self.s), np.float32)
        mh[::self.s, ::self.s] = 1.0
        self.register_buffer("mask", torch.from_numpy(mh)[None, None])

    def forward(self, x):
        xn = F.interpolate(
            x, size=(self.h_in * self.s, self.w_in * self.s),
            mode="nearest") * self.mask
        y = F.conv2d(xn, self.w, bias=self.b, padding=self.k - 1)
        out_h = (self.h_in - 1) * self.s + self.k - 2 * self.p + self.op
        out_w = (self.w_in - 1) * self.s + self.k - 2 * self.p + self.op
        return y[:, :, self.p:self.p + out_h, self.p:self.p + out_w]


def swap_convt(net, dummy):
    """Replace every ConvTranspose2d with ZeroStuffConvT2d.

    Input sizes are traced via forward pre-hooks on a dummy pass.

    Args:
        net: Module whose ConvTranspose2d children are replaced in
            place.
        dummy: Example input tensor used to trace per-layer input
            sizes.

    Returns:
        The same net with ConvTranspose2d modules replaced.
    """
    sizes = {}
    hooks = []
    for n, mo in net.named_modules():
        if isinstance(mo, nn.ConvTranspose2d):
            hooks.append(mo.register_forward_pre_hook(
                (lambda nm: (lambda mod, i: sizes.__setitem__(
                    nm, i[0].shape[-2:])))(n)))
    with torch.no_grad():
        net(dummy)
    for h in hooks:
        h.remove()
    for name, mo in list(net.named_modules()):
        if isinstance(mo, nn.ConvTranspose2d) and name in sizes:
            par = net
            *path, last = name.split(".")
            for q in path:
                par = getattr(par, q)
            hh, ww = sizes[name]
            setattr(par, last, ZeroStuffConvT2d(mo, hh, ww))
    return net


class DewarpNet(nn.Module):
    """wc (shape) -> clamp(0,1) -> bm (backward map), as one graph."""

    def __init__(self, wc, bm):
        super().__init__()
        self.wc = wc
        self.bm = bm

    def forward(self, x):
        w = self.wc(x)
        # exact clamp(0,1); Mali rejects RELU_0_TO_1 (Hardtanh)
        wc_out = F.relu(w) - F.relu(w - 1.0)
        return self.bm(F.interpolate(wc_out, (128, 128), mode='bilinear',
                                     align_corners=False))


def main():
    """Converts DewarpNet to dewarp.tflite and saves reference tensors."""
    wc = get_model('unetnc', 3, in_channels=3).eval()
    wc.load_state_dict(convert_state_dict(torch.load(
        "dewarp-src/weights/unetnc_doc3d_final.pkl",
        map_location='cpu')['model_state']))
    bm = get_model('dnetccnl', 2, in_channels=3).eval()
    bm.load_state_dict(convert_state_dict(torch.load(
        "dewarp-src/weights/dnetccnl_doc3d_final.pkl",
        map_location='cpu')['model_state']))
    swap_convt(wc, torch.rand(1, 3, 256, 256))
    swap_convt(bm, torch.rand(1, 3, 128, 128))

    net = DewarpNet(wc, bm).eval()
    dummy = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        o = net(dummy)
    print("bm out:", tuple(o.shape), "range",
          round(float(o.min()), 3), round(float(o.max()), 3))
    np.save("ref_in.npy", dummy.numpy())
    np.save("ref_out.npy", o.numpy())
    import litert_torch
    litert_torch.convert(net, (dummy,)).export("dewarp.tflite")
    print("saved %.1f MB" % (os.path.getsize("dewarp.tflite") / 1e6))


if __name__ == "__main__":
    main()
