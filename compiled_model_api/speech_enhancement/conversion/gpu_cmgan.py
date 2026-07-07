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

"""GPU-clean re-authoring of CMGAN TSCNet for LiteRT CompiledModel GPU (ML Drift).

Numerically-exact rewrites:
  - torch.stft + power_compress -> in-graph hamming-DFT Conv1d + exp/ln compression
    (POW banned; CPGA-Net recipe), host reflect-pads.
  - phase path CANCELS: mask*mag*cos(angle) == mask*x_r (and sin -> x_i) -> no atan2/cos/sin.
  - conformer folded-batch (b*f, t, c) -> batch-1 4D [1, c, A, n]: LayerNorm -> channel-LN
    per position, Linear -> 1x1 Conv2d, depthwise Conv1d -> (1,k) Conv2d.
  - attention -> per-path 3D BMM [A*h, n, d] with 1/sqrt(d) folded into q; Shaw relative
    position embedding baked to a constant (dist matrix is constant for fixed n) applied as
    a 2D FC + the pad/reshape SKEW realignment (no runtime flip: the constant is pre-flipped,
    no gather, all tensors <= 4D).
  - SPConvTranspose2d 5D view -> [b, r*c, H, W] -> [b, r, c, H*W] -> permute -> [b, c, H, W*r].
  - InstanceNorm2d -> Safe per-channel spatial norm (chained single-axis means, eps 1e-4, C38).
  - BatchNorm1d (eval) -> constant scale/shift. PReLU -> relu(x) - w*relu(-x). GLU -> slices.
  - No dim-1 broadcasts on stacked outputs (C39).
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS_NORM = 1e-4   # fp16-safe (C38); originals are 1e-5, exact-equivalent here (var >> 1e-4)
SAFE = 16.0
MAG2_FLOOR = 1e-6


def prelu_ch(x, w):
    """PReLU with per-channel weight on dim 1 of a 4D tensor."""
    return F.relu(x) - w.view(1, -1, 1, 1) * F.relu(-x)


def chan_ln(x, weight, bias, eps=EPS_NORM):
    """LayerNorm over dim 1 (channels) of [1, C, A, N] (original LN over the last dim of
    (b, n, c) sequences)."""
    m = x.mean(dim=1, keepdim=True)
    d = (x - m) * (1.0 / SAFE)
    v = (d * d).mean(dim=1, keepdim=True)
    xh = (x - m) / torch.sqrt(v * (SAFE * SAFE) + eps)
    return xh * weight.view(1, -1, 1, 1) + bias.view(1, -1, 1, 1)


def inst_norm(x, norm, eps=EPS_NORM):
    """InstanceNorm2d(affine) on [1, C, H, W]: per-channel spatial norm, chained means."""
    m = x.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
    d = (x - m) * (1.0 / SAFE)
    v = (d * d).mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
    xh = (x - m) / torch.sqrt(v * (SAFE * SAFE) + eps)
    return xh * norm.weight.view(1, -1, 1, 1) + norm.bias.view(1, -1, 1, 1)


def lin_as_conv(x, lin):
    """nn.Linear over channels of [1, C, A, N]."""
    return F.conv2d(x, lin.weight.unsqueeze(2).unsqueeze(3), lin.bias)


def dft_weights(n_fft):
    """Hamming-windowed real-DFT bank as Conv1d weights [2*(nfft//2+1), 1, n_fft]."""
    enc = n_fft // 2 + 1
    win = torch.hamming_window(n_fft, periodic=True)
    n = torch.arange(n_fft, dtype=torch.float64)
    k = torch.arange(enc, dtype=torch.float64).unsqueeze(1)
    ang = 2.0 * math.pi * k * n / n_fft
    cos = (torch.cos(ang) * win.double()).float()
    sin = (-torch.sin(ang) * win.double()).float()
    return torch.cat([cos, sin], 0).unsqueeze(1)


class GPUAttention(nn.Module):
    """Attention over the last axis n of [1, C, A, n], per-A, heads folded into the batch."""

    def __init__(self, att, n):
        super().__init__()
        self.att = att
        self.n = n
        self.h = att.heads
        self.d = att.rel_pos_emb.embedding_dim
        # bake Shaw rel-pos for fixed n: needed distances r = i - j in [-(n-1), n-1]
        # index k along P's last axis is chosen so that attn_rel[i, j] = P[i, i + (n-1) - j]:
        # column order = E[dist = k - (n-1)] REVERSED -> pre-flip the constant.
        max_pos = att.max_pos_emb
        r = torch.arange(-(n - 1), n).clamp(-max_pos, max_pos) + max_pos      # [2n-1]
        E = att.rel_pos_emb.weight[r]                                          # [2n-1, d]
        E = torch.flip(E, dims=[0])                                            # pre-flip
        self.register_buffer("relT", E.t().contiguous(), persistent=False)     # [d, 2n-1]

    def skew(self, P):
        """P [B, n, 2n-1] with attn_rel[i, j] = P[i, (n-1) + i - j]  ->  [B, n, n].
        pad right by 1 -> [B, n, 2n] -> flatten -> drop last n -> [B, n, 2n-1] shifted rows;
        classic music-transformer skew, then take the first n columns reversed... implemented
        and VERIFIED against the reference einsum in build_cmgan.py before export."""
        B, n_, w = P.shape                         # w = 2n-1
        P = F.pad(P, (0, 1))                       # [B, n, 2n]
        P = P.reshape(B, n_ * (w + 1))
        P = P[:, n_ - 1:n_ - 1 + n_ * w]           # shift each row by -(n-1) via stride mismatch
        P = P.reshape(B, n_, w)                    # now [i, j] = P_orig[i, (n-1) - i + j]
        return P[:, :, :n_]

    def forward(self, x):
        # x: [1, C, A, n]
        att = self.att
        _, C, A, n = x.shape
        q = lin_as_conv(x, att.to_q)                                   # [1, hd, A, n]
        kv = lin_as_conv(x, att.to_kv)                                 # [1, 2hd, A, n]
        hd = q.shape[1]
        k, v = kv[:, :hd], kv[:, hd:]

        def fold(t):
            # -> [A*h, n, d]
            t = t.squeeze(0).permute(1, 0, 2)                           # [A, hd, n]
            t = t.reshape(A, self.h, self.d, n).permute(0, 1, 3, 2)     # [A, h, n, d]
            return t.reshape(A * self.h, n, self.d)

        scale = self.d ** -0.5
        qf = fold(q) * scale
        kf = fold(k)
        vf = fold(v)
        dots = torch.matmul(qf, kf.transpose(1, 2))                     # [A*h, n, n]
        P = torch.matmul(qf.reshape(A * self.h * n, self.d), self.relT) # 2D FC (const RHS)
        P = P.reshape(A * self.h, n, 2 * n - 1)
        dots = dots + self.skew(P)
        attn = F.softmax(dots, dim=-1)
        out = torch.matmul(attn, vf)                                    # [A*h, n, d]
        out = out.reshape(A, self.h, n, self.d).permute(0, 1, 3, 2)     # [A, h, d, n]
        out = out.reshape(A, self.h * self.d, n).unsqueeze(0).permute(0, 2, 1, 3)  # [1, hd, A, n]
        return lin_as_conv(out, att.to_out)


class GPUConformer(nn.Module):
    """ConformerBlock on [1, C, A, n] (attention/conv over the last axis)."""

    def __init__(self, blk, n):
        super().__init__()
        self.blk = blk
        self.attn = GPUAttention(blk.attn.fn, n)
        # fold eval-mode BatchNorm1d into scale/shift constants
        bn = blk.conv.net[5]
        with torch.no_grad():
            s = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach()
            b = (bn.bias - bn.running_mean * s).detach()
        self.register_buffer("bn_s", s.view(1, -1, 1, 1), persistent=False)
        self.register_buffer("bn_b", b.view(1, -1, 1, 1), persistent=False)

    def ff(self, mod, x):
        # Scale(0.5, PreNorm(dim, FeedForward))
        pre = mod.fn
        y = chan_ln(x, pre.norm.weight, pre.norm.bias)
        net = pre.fn.net
        y = lin_as_conv(y, net[0])
        y = y * torch.sigmoid(y)                    # Swish
        y = lin_as_conv(y, net[3])
        return y * mod.scale

    def conv_module(self, x):
        net = self.blk.conv.net
        y = chan_ln(x, net[0].weight, net[0].bias)
        y = F.conv2d(y, net[2].weight.unsqueeze(2), net[2].bias)        # 1x1 conv (Conv1d k1)
        half = y.shape[1] // 2
        y = y[:, :half] * torch.sigmoid(y[:, half:])                    # GLU dim=1
        dw = net[4]
        y = F.pad(y, (dw.padding[0], dw.padding[1]))                    # same-pad over n
        y = F.conv2d(y, dw.conv.weight.unsqueeze(2), dw.conv.bias, groups=dw.conv.groups)
        y = y * self.bn_s + self.bn_b                                   # BatchNorm1d (eval)
        y = y * torch.sigmoid(y)                                        # Swish
        y = F.conv2d(y, net[7].weight.unsqueeze(2), net[7].bias)
        return y

    def forward(self, x):
        blk = self.blk
        x = self.ff(blk.ff1, x) + x
        pre = blk.attn
        x = self.attn(chan_ln(x, pre.norm.weight, pre.norm.bias)) + x
        x = self.conv_module(x) + x
        x = self.ff(blk.ff2, x) + x
        return chan_ln(x, blk.post_norm.weight, blk.post_norm.bias)


def dilated_dense(dd, x):
    skip = x
    out = x
    for i in range(dd.depth):
        dil = 2 ** i
        pad_len = 2 + (dil - 1) - 1                                  # twidth=2 -> = dil
        out = F.pad(skip, (1, 1, pad_len, 0))                        # (W_l, W_r, H_t, H_b)
        conv = getattr(dd, f"conv{i+1}")
        out = F.conv2d(out, conv.weight, conv.bias, dilation=conv.dilation)
        out = inst_norm(out, getattr(dd, f"norm{i+1}"))
        out = prelu_ch(out, getattr(dd, f"prelu{i+1}").weight)
        skip = torch.cat([out, skip], dim=1)
    return out


def sp_conv(sp, x):
    """SPConvTranspose2d without the 5D view: width-interleave r sub-channel groups."""
    x = F.pad(x, (1, 1, 0, 0))
    out = F.conv2d(x, sp.conv.weight, sp.conv.bias)
    b, rc, H, W = out.shape
    c = rc // sp.r
    out = out.reshape(b, sp.r, c, H * W).permute(0, 2, 3, 1)             # [b, c, H*W, r]
    return out.reshape(b, c, H, W * sp.r)


class GPUCMGAN(nn.Module):
    """padded wav [1, S + n_fft] -> (est_real, est_imag) compressed spectra [1, 1, T, F]."""

    def __init__(self, m, n_samples, n_fft=400, hop=100):
        super().__init__()
        self.m = m
        self.n_fft, self.hop = n_fft, hop
        self.enc = n_fft // 2 + 1
        self.T = n_samples // hop + 1
        self.register_buffer("dft", dft_weights(n_fft), persistent=False)
        self.tscb = nn.ModuleList()
        for b in (m.TSCB_1, m.TSCB_2, m.TSCB_3, m.TSCB_4):
            self.tscb.append(nn.ModuleList([GPUConformer(b.time_conformer, self.T),
                                            GPUConformer(b.freq_conformer, 101)]))

    def dense_encoder(self, x):
        de = self.m.dense_encoder
        y = F.conv2d(x, de.conv_1[0].weight, de.conv_1[0].bias)
        y = inst_norm(y, de.conv_1[1])
        y = prelu_ch(y, de.conv_1[2].weight)
        y = dilated_dense(de.dilated_dense, y)
        y = F.conv2d(y, de.conv_2[0].weight, de.conv_2[0].bias, stride=(1, 2), padding=(0, 1))
        y = inst_norm(y, de.conv_2[1])
        return prelu_ch(y, de.conv_2[2].weight)

    def mask_decoder(self, x):
        md = self.m.mask_decoder
        y = dilated_dense(md.dense_block, x)
        y = sp_conv(md.sub_pixel, y)
        y = F.conv2d(y, md.conv_1.weight, md.conv_1.bias)
        y = prelu_ch(inst_norm(y, md.norm), md.prelu.weight)
        y = F.conv2d(y, md.final_conv.weight, md.final_conv.bias)        # [1, 1, T, F]
        w = md.prelu_out.weight                                           # per-frequency (F)
        return F.relu(y) - w.view(1, 1, 1, -1) * F.relu(-y)

    def complex_decoder(self, x):
        cd = self.m.complex_decoder
        y = dilated_dense(cd.dense_block, x)
        y = sp_conv(cd.sub_pixel, y)
        y = prelu_ch(inst_norm(y, cd.norm), cd.prelu.weight)
        return F.conv2d(y, cd.conv.weight, cd.conv.bias)                  # [1, 2, T, F]

    def forward(self, wav_padded):
        spec = F.conv1d(wav_padded.unsqueeze(1), self.dft, stride=self.hop)  # [1, 2*enc, T]
        r = spec[:, :self.enc].transpose(1, 2).unsqueeze(1)                  # [1, 1, T, F]
        i = spec[:, self.enc:].transpose(1, 2).unsqueeze(1)
        m2 = torch.clamp(r * r + i * i, min=MAG2_FLOOR)
        lg = torch.log(m2)
        mag_c = torch.exp(0.15 * lg)                                         # mag^0.3
        cf = torch.exp(-0.35 * lg)                                           # mag^-0.7
        rc, ic = r * cf, i * cf                                              # compressed r, i
        x = torch.cat([mag_c, rc, ic], dim=1)                                # [1, 3, T, F]

        y = self.dense_encoder(x)                                            # [1, 64, T, 101]
        for time_c, freq_c in self.tscb:
            # time path: tokens = T per frequency -> [1, C, F', T]; TSCB residual AFTER each
            y = y.transpose(2, 3).contiguous()
            y = time_c(y) + y
            # freq path: tokens = F' per frame -> [1, C, T, F']
            y = y.transpose(2, 3).contiguous()
            y = freq_c(y) + y

        mask = self.mask_decoder(y)                                          # [1, 1, T, F]
        cplx = self.complex_decoder(y)                                       # [1, 2, T, F]
        # phase algebra: mask*mag*cos(angle) == mask*rc  (exact; 0 at silence either way)
        est_r = mask * rc + cplx[:, 0:1]
        est_i = mask * ic + cplx[:, 1:2]
        return est_r, est_i
