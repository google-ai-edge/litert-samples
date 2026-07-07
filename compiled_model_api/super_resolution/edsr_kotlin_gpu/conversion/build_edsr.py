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

"""Build GPU-compatible EDSR x4 super resolution via litert-torch.

Two-step PixelShuffle re-authoring (Mali rejects both DEPTH_TO_SPACE and
TRANSPOSE_CONV):
  1. Replace each PixelShuffle(r) with the mathematically identical fixed-weight
     ConvTranspose2d(C*r*r -> C, kernel r, stride r).
  2. Replace every ConvTranspose2d with ZeroStuffConvT2d (nearest-upsample x
     stride zero-stuff + flipped conv2d + crop) — exact and GPU-clean.

Setup: pip install torch litert-torch super-image

Run: python build_edsr.py
  # -> edsr.tflite ([1,3,128,128] -> [1,3,512,512]) + ref fixtures
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from super_image import EdsrModel


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


def pixelshuffle_to_convt(r, c_out):
    """Fixed-weight ConvTranspose2d that exactly reproduces PixelShuffle(r)."""
    ct = nn.ConvTranspose2d(c_out * r * r, c_out, kernel_size=r, stride=r, bias=False)
    w = torch.zeros(c_out * r * r, c_out, r, r)
    for c in range(c_out):
        for p in range(r):
            for q in range(r):
                w[c * r * r + p * r + q, c, p, q] = 1.0
    ct.weight.data = w
    return ct


class Wrap(nn.Module):
    """Plain passthrough wrapper (stable module root for conversion)."""

    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return self.n(x)


def main():
    m = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=4).eval()

    # 1) replace PixelShuffle with the equivalent fixed ConvTranspose2d (capture Cout via hook)
    info = {}
    hooks = []
    for n, mo in m.named_modules():
        if isinstance(mo, nn.PixelShuffle):
            hooks.append(mo.register_forward_pre_hook(
                (lambda nm, r=mo.upscale_factor:
                 (lambda mod, i: info.__setitem__(nm, (r, i[0].shape[1] // (r * r)))))(n)))
    with torch.no_grad():
        m(torch.rand(1, 3, 128, 128))
    for h in hooks:
        h.remove()
    for n, (r, cout) in info.items():
        par = m
        *path, last = n.split('.')
        for q in path:
            par = getattr(par, q)
        setattr(par, last, pixelshuffle_to_convt(r, cout))

    # 2) swap those ConvTranspose2d -> ZeroStuffConvT2d
    sizes = {}
    hooks = []
    for n, mo in m.named_modules():
        if isinstance(mo, nn.ConvTranspose2d):
            hooks.append(mo.register_forward_pre_hook(
                (lambda nm: (lambda mod, i: sizes.__setitem__(nm, i[0].shape[-2:])))(n)))
    with torch.no_grad():
        m(torch.rand(1, 3, 128, 128))
    for h in hooks:
        h.remove()
    for n, mo in list(m.named_modules()):
        if isinstance(mo, nn.ConvTranspose2d) and n in sizes:
            par = m
            *path, last = n.split('.')
            for q in path:
                par = getattr(par, q)
            hh, ww = sizes[n]
            setattr(par, last, ZeroStuffConvT2d(mo, hh, ww))

    w = Wrap(m).eval()
    dummy = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        o = w(dummy)
    print("out:", tuple(o.shape))
    np.save("ref_in.npy", dummy.numpy())
    np.save("ref_out.npy", o.numpy())
    import litert_torch
    litert_torch.convert(w, (dummy,)).export("edsr.tflite")
    print("saved %.1f MB" % (os.path.getsize("edsr.tflite") / 1e6))


if __name__ == "__main__":
    main()
