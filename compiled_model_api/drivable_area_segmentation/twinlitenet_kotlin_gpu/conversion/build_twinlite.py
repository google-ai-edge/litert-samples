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

"""Build GPU-compatible TwinLiteNet (drivable area + lane segmentation) via litert-torch.

Expects the TwinLiteNet repository checked out next to this script with the
released checkpoint at TwinLiteNet/pretrained/best.pth.

Only patch: ZeroStuffConvT2d — exact ConvTranspose2d replacement (Mali rejects
TRANSPOSE_CONV): nearest-upsample x stride zero-stuff + flipped conv2d + crop.

The two same-shape outputs are emitted IN ORDER (drivable area, lane line);
keep that order when reading the output buffers on device.

Run: python build_twinlite.py
  # -> twinlite.tflite ([1,3,360,640] -> 2x [1,2,360,640]) + ref fixtures
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/TwinLiteNet")
from model.TwinLite import TwinLiteNet as Net


class ZeroStuffConvT2d(nn.Module):
    """Exact GPU-clean ConvTranspose2d: nearest-upsample x stride zero-stuff + flipped conv2d + crop."""

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
            "b", ct.bias.detach().clone() if ct.bias is not None else torch.zeros(ct.out_channels))
        mh = np.zeros((h_in * self.s, w_in * self.s), np.float32)
        mh[::self.s, ::self.s] = 1.0
        self.register_buffer("mask", torch.from_numpy(mh)[None, None])

    def forward(self, x):
        xn = F.interpolate(x, size=(self.h_in * self.s, self.w_in * self.s), mode="nearest") * self.mask
        y = F.conv2d(xn, self.w, bias=self.b, padding=self.k - 1)
        out_h = (self.h_in - 1) * self.s + self.k - 2 * self.p + self.op
        out_w = (self.w_in - 1) * self.s + self.k - 2 * self.p + self.op
        return y[:, :, self.p:self.p + out_h, self.p:self.p + out_w]


class Wrap(nn.Module):
    """Emit the two heads as ordered outputs: (drivable area, lane line)."""

    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        da, ll = self.n(x)
        return da, ll


def main():
    net = Net()
    sd = torch.load("TwinLiteNet/pretrained/best.pth", map_location="cpu")
    sd = sd.get('state_dict', sd) if isinstance(sd, dict) else sd
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    print("load:", net.load_state_dict(sd, strict=False))
    net.eval()

    # swap ConvTranspose2d -> ZeroStuffConvT2d (capture input sizes via hooks)
    sizes = {}
    hooks = []
    for n, mo in net.named_modules():
        if isinstance(mo, nn.ConvTranspose2d):
            hooks.append(mo.register_forward_pre_hook(
                (lambda nm: (lambda mod, i: sizes.__setitem__(nm, i[0].shape[-2:])))(n)))
    with torch.no_grad():
        net(torch.randn(1, 3, 360, 640))
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

    w = Wrap(net).eval()
    dummy = torch.rand(1, 3, 360, 640)
    with torch.no_grad():
        o = w(dummy)
    print("outs:", [tuple(t.shape) for t in o])
    np.save("ref_in.npy", dummy.numpy())
    for i, t in enumerate(o):
        np.save(f"ref_out{i}.npy", t.numpy())
    import litert_torch
    litert_torch.convert(w, (dummy,)).export("twinlite.tflite")
    print("saved %.1f MB" % (os.path.getsize("twinlite.tflite") / 1e6))


if __name__ == "__main__":
    main()
