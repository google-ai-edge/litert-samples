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

"""Converts DehazeFormer-MCT (image dehazing) to a fully GPU-compatible LiteRT model.

Model code and the MIT mixed-dataset checkpoint are fetched from the author's
Hugging Face Space (IDKiro/DehazeFormer_Demo). The basenet runs at a fixed
256x256 and emits 72 per-pixel curve parameters; the curve application to the
full-resolution image is done host-side in the app (the official MCT
grid_sample step).

Exact re-authors (desktop corr vs PyTorch = 1.0000000):
  - reflect pads -> slice+concat (litert-torch lowers reflection_pad2d to
    GATHER_ND, which the GPU delegate rejects; this includes padding_mode=
    'reflect' convs with padding=0, which still route through F.pad)
  - Swin window partition/reverse -> <=4D reshape/permute
  - qkv 5D split -> channel slices -> 4D per-head matmuls
  - relative-position bias -> constant baked from the meta MLP
  - RLN global norm + SKFusion global pool -> hierarchical means (equal-size
    avg_pool windows; a single MEAN over C*H*W elements overflows the Mali
    fp16 accumulator -> NaN), with dd/4 pre-scaling before squaring
  - SKFusion 5D view+softmax -> 4D pairwise softmax
  - Conv2d(1x1)+PixelShuffle(2) -> ConvTranspose2d -> zero-stuff conv
    (per-subpixel conv bias re-added as a constant tiled map)

Run: python build_dehaze.py
  # -> dehazeformer_base.tflite ([1,3,256,256] -> [1,72,256,256])
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

SPACE = "IDKiro/DehazeFormer_Demo"
CODE = hf_hub_download(SPACE, "models/dehazeformer.py", repo_type="space")
CKPT = hf_hub_download(SPACE, "saved_models/dehazeformer.pth", repo_type="space")
sys.path.insert(0, os.path.dirname(CODE))
import dehazeformer as D  # noqa: E402

SIZE = 256
S_RLN = 4.0  # RLN dd absmax measured <= 3.9 -> keeps (dd/S)^2 <= 1 in fp16


def reflect_pad(x, l, r, t, b):
    """Exact 'reflect' padding built from slices+concats (GPU-clean)."""
    W = int(x.shape[3])
    H = int(x.shape[2])
    parts = []
    if l > 0:
        parts.append(torch.cat([x[:, :, :, k:k + 1] for k in range(l, 0, -1)], dim=3))
    parts.append(x)
    if r > 0:
        parts.append(torch.cat([x[:, :, :, W - 2 - k:W - 1 - k] for k in range(r)], dim=3))
    x = torch.cat(parts, dim=3) if len(parts) > 1 else x
    parts = []
    if t > 0:
        parts.append(torch.cat([x[:, :, k:k + 1, :] for k in range(t, 0, -1)], dim=2))
    parts.append(x)
    if b > 0:
        parts.append(torch.cat([x[:, :, H - 2 - k:H - 1 - k, :] for k in range(b)], dim=2))
    return torch.cat(parts, dim=2) if len(parts) > 1 else x


class ReflectConv(nn.Module):
    """Conv2d with padding_mode='reflect' -> explicit slice-pad + valid conv."""

    def __init__(self, conv):
        super().__init__()
        self.p = conv.padding[0]
        self.conv = conv

    def forward(self, x):
        if self.p > 0:
            x = reflect_pad(x, self.p, self.p, self.p, self.p)
        w = self.conv
        return F.conv2d(x, w.weight, w.bias, stride=w.stride, groups=w.groups)


def swap_reflect_convs(model):
    """padding=0 reflect convs still route through F.pad -> swap them all."""
    for name, m in list(model.named_modules()):
        if isinstance(m, nn.Conv2d) and m.padding_mode == 'reflect':
            parent = model
            *path, last = name.split('.')
            for q in path:
                parent = getattr(parent, q)
            setattr(parent, last, ReflectConv(m))


def wp4d(x, ws, H, W):
    """[1,H,W,C] -> [nW, ws*ws, C] window partition, <=4D only."""
    C = x.shape[-1]
    x = x.reshape(H // ws, ws, W // ws, ws * C)
    x = x.permute(0, 2, 1, 3)
    return x.reshape((H // ws) * (W // ws), ws * ws, C)


def wr4d(x, ws, H, W):
    """[nW, ws*ws, C] -> [1,H,W,C] window reverse, <=4D only."""
    C = x.shape[-1]
    x = x.reshape(H // ws, W // ws, ws, ws * C)
    x = x.permute(0, 2, 1, 3)
    return x.reshape(1, H, W, C)


def hier_mean_hw(x):
    """[1,C,H,W] -> [1,C,1,1] mean with no fp16-overflowing partial sum."""
    y = F.avg_pool2d(x, 16)
    return y.mean(dim=(2, 3), keepdim=True)


def hier_mean(x):
    """[1,C,H,W] -> [1,1,1,1] global mean, hierarchical (exact: 16 | H, W)."""
    return hier_mean_hw(x).mean(dim=1, keepdim=True)


def rln_forward(self, x):
    mean = hier_mean(x)
    dd = x - mean
    e = dd * (1.0 / S_RLN)
    v = hier_mean(e * e)
    eps_scaled = self.eps / (S_RLN * S_RLN)
    normalized = e * torch.rsqrt(v + eps_scaled)
    std = S_RLN * torch.sqrt(v + eps_scaled)
    rescale, rebias = self.meta1(std), self.meta2(mean)
    out = normalized * self.weight + self.bias
    return out, rescale, rebias


def window_attn_forward(self, qkv):
    Bn, N, C3 = qkv.shape
    C = C3 // 3
    nH = self.num_heads
    hd = C // nH
    q = qkv[:, :, :C].reshape(Bn, N, nH, hd).permute(0, 2, 1, 3)
    k = qkv[:, :, C:2 * C].reshape(Bn, N, nH, hd).permute(0, 2, 1, 3)
    v = qkv[:, :, 2 * C:].reshape(Bn, N, nH, hd).permute(0, 2, 1, 3)
    attn = (q * self.scale) @ k.transpose(-2, -1)
    attn = attn + self.rpb                       # baked constant [1,nH,N,N]
    attn = torch.softmax(attn, dim=-1)
    x = (attn @ v).permute(0, 2, 1, 3).reshape(Bn, N, C)
    return x


def attention_forward(self, X):
    B, C, H, W = X.shape
    if self.conv_type == 'DWConv' or self.use_attn:
        V = self.V(X)
    if self.use_attn:
        QK = self.QK(X)
        QKV = torch.cat([QK, V], dim=1)
        s = self.shift_size
        ws = self.window_size
        if s > 0:
            QKV = reflect_pad(QKV, s, ws - s, s, ws - s)
        Ht, Wt = int(QKV.shape[2]), int(QKV.shape[3])
        qkv = wp4d(QKV.permute(0, 2, 3, 1), ws, Ht, Wt)
        attn_windows = self.attn(qkv)
        out = wr4d(attn_windows, ws, Ht, Wt)
        out = out[:, s:s + H, s:s + W, :]
        attn_out = out.permute(0, 3, 1, 2)
        if self.conv_type in ['Conv', 'DWConv']:
            conv_out = self.conv(V)
            out = self.proj(conv_out + attn_out)
        else:
            out = self.proj(attn_out)
    else:
        if self.conv_type == 'Conv':
            out = self.conv(X)
        elif self.conv_type == 'DWConv':
            out = self.proj(self.conv(V))
    return out


def skfusion_forward(self, in_feats):
    a, b = in_feats[0], in_feats[1]
    C = a.shape[1]
    feats_sum = a + b
    attn = self.mlp(hier_mean_hw(feats_sum))     # [1,2C,1,1]
    wa = attn[:, :C]
    wb = attn[:, C:]
    m = torch.maximum(wa, wb)
    ea = torch.exp(wa - m)
    eb = torch.exp(wb - m)
    return (a * ea + b * eb) / (ea + eb)


class ZeroStuffConvT2d(nn.Module):
    """ConvTranspose2d as zero-stuffed conv (exact; TRANSPOSE_CONV is rejected)."""

    def __init__(self, weight, bias, stride, kernel, h_in, w_in):
        super().__init__()
        self.s = stride
        self.k = kernel
        self.h_in, self.w_in = h_in, w_in
        w = weight.detach().flip(2).flip(3).permute(1, 0, 2, 3).contiguous()
        self.register_buffer("w", w)
        self.register_buffer("b", bias)
        mask = np.zeros((h_in * stride, w_in * stride), np.float32)
        mask[::stride, ::stride] = 1.0
        self.register_buffer("mask", torch.from_numpy(mask)[None, None])

    def forward(self, x):
        xn = F.interpolate(x, size=(self.h_in * self.s, self.w_in * self.s), mode="nearest") * self.mask
        y = F.conv2d(xn, self.w, bias=self.b, padding=self.k - 1)
        out_h = (self.h_in - 1) * self.s + self.k
        out_w = (self.w_in - 1) * self.s + self.k
        return y[:, :, 0:out_h, 0:out_w]


class FoldedPatchSplit(nn.Module):
    """Conv2d(1x1, dim->out*4) + PixelShuffle(2) -> zero-stuff ConvTranspose (exact).

    The conv bias varies per sub-pixel (dy,dx), which a ConvTranspose bias cannot
    express; it is re-added as a constant [1,out,H,W] tiled map.
    """

    def __init__(self, conv, h_in, w_in):
        super().__init__()
        out4, cin = conv.weight.shape[0], conv.weight.shape[1]
        out = out4 // 4
        wt = torch.zeros(cin, out, 2, 2)
        bias_map = torch.zeros(1, out, 2, 2)
        for o in range(out):
            for dy in range(2):
                for dx in range(2):
                    wt[:, o, dy, dx] = conv.weight[o * 4 + dy * 2 + dx, :, 0, 0]
                    if conv.bias is not None:
                        bias_map[0, o, dy, dx] = conv.bias[o * 4 + dy * 2 + dx]
        self.zs = ZeroStuffConvT2d(wt, torch.zeros(out), 2, 2, h_in, w_in)
        self.register_buffer("bmap", bias_map.repeat(1, 1, h_in, w_in))

    def forward(self, x):
        return self.zs(x) + self.bmap


def bake_rpb(model):
    """Evaluates each WindowAttention's meta MLP once and bakes the bias constant."""
    for m in model.modules():
        if isinstance(m, D.WindowAttention):
            with torch.no_grad():
                rpb = m.meta(m.relative_positions).permute(2, 0, 1).contiguous()
            m.register_buffer("rpb", rpb.unsqueeze(0))


def build():
    net = D.MCT()
    net.load_state_dict(torch.load(CKPT, map_location="cpu")["state_dict"], strict=True)
    net.eval()
    base = net.basenet

    bake_rpb(base)
    D.RLN.forward = rln_forward
    D.WindowAttention.forward = window_attn_forward
    D.Attention.forward = attention_forward
    D.SKFusion.forward = skfusion_forward

    # Fold BEFORE the reflect swap so FoldedPatchSplit sees the raw 1x1 Conv2d.
    base.patch_split1.proj = FoldedPatchSplit(base.patch_split1.proj[0], SIZE // 4, SIZE // 4)
    base.patch_split2.proj = FoldedPatchSplit(base.patch_split2.proj[0], SIZE // 2, SIZE // 2)
    base.patch_unembed.proj = nn.Sequential(base.patch_unembed.proj[0])  # PixelShuffle(1) = no-op
    swap_reflect_convs(base)
    return base


def main():
    base = build()
    dummy = torch.rand(1, 3, SIZE, SIZE) * 2 - 1
    with torch.no_grad():
        out = base(dummy)
    print("out:", tuple(out.shape))
    np.save("ref_in.npy", dummy.numpy())
    np.save("ref_out.npy", out.numpy())
    import litert_torch
    litert_torch.convert(base, (dummy,)).export("dehazeformer_base.tflite")
    print("saved %.1f MB" % (os.path.getsize("dehazeformer_base.tflite") / 1e6))


if __name__ == "__main__":
    main()
