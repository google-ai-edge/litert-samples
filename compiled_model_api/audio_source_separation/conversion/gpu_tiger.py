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

"""GPU-clean re-authoring of TIGER (look2hear) for LiteRT CompiledModel
GPU (ML Drift).

Numerically-exact rewrites (fp32) of every Mali-hostile construct:
  - torch.stft            -> hann-windowed DFT as a single Conv1d (PANNs
                             recipe); host pre-pads (reflect, win//2) so
                             no in-graph GATHER_ND pad.
  - fold-batch Conv1d     -> the (B*T, N, L) / (B*nband, N, T) folded
                             batches become 4D [1, N, T, L] Conv2d with
                             (1, k) kernels (exact for every conv here).
  - GlobLN on folded batch-> per-position GroupNorm over (C, L) keeping
                             T: chained single-axis means (multi-axis
                             reduce NaNs on Mali) + down-scaled variance
                             (SafeNorm: fp16 sum overflow).
  - adaptive_avg_pool1d   -> constant averaging matrix applied as 2D
                             matmul (FULLY_CONNECTED); exact incl. the
                             non-uniform overlapping windows.
  - F.interpolate nearest -> constant one-hot matrix, same 2D matmul
                             trick; kills Mali off-stride RESIZE_NEAREST
                             misbehaviour.
  - MultiHeadSelfAttention2D batch-cat heads -> per-head batch-1 3D BMM
                             + channel concat (exact).
  - PReLU                 -> relu(x) - w * relu(-x)  (exact; native
                             PRELU lowering emits SELECT).
  - 6D mask-head view     -> static channel slices, all tensors <= 4D
                             (exact).
  - torch.complex/istft   -> graph outputs (real, imag) spectrograms;
                             iSTFT runs host-side.

The wrapper takes the ORIGINAL look2hear TIGER instance and references
its weights; nothing is copied or retrained.
Output = [1, num_sources, enc_dim, T] real + imag.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# fp16-SAFE eps: the originals (GlobLN 1e-8, LN4D 1e-5) UNDERFLOW TO ZERO
# in fp16 (min normal 6.1e-5) -> silent bands give var=0 -> sqrt(0+0)=0 ->
# 0/0 = NaN on the Mali delegate, and the NaNs spread across time via the
# frame path (device bisect 2026-07-02: 8448 NaN at the band bottleneck
# -> 405k by iter 4). 1e-4 is exact-equivalent: silence still normalizes
# to 0, active bands shift by <1e-6 relative.
EPS_GN = 1e-4
EPS_LN4 = 1e-4
SAFE = 32.0     # down-scale factor for SafeNorm variance sums (fp16 headroom)


# ---------------------------------------------------------------- helpers
def prelu(x, w):
    """Exact PReLU as relu(x) - w * relu(-x).

    Args:
        x: Input tensor.
        w: PReLU weight, [C] or scalar tensor; broadcast over channel
            dim=1.

    Returns:
        Tensor of the same shape as x.
    """
    if w.numel() == 1:
        return F.relu(x) - w.reshape(1) * F.relu(-x)
    return F.relu(x) - w.view(1, -1, 1, 1) * F.relu(-x)


def conv1d_as_2d(x, conv, axis_pad):
    """Runs an nn.Conv1d's weights as Conv2d over the LAST axis of x.

    Exact: kernel (1, k), stride (1, s), padding (0, p), same groups.

    Args:
        x: 4D input tensor [1, Cin, A, L].
        conv: nn.Conv1d module whose weights and bias are applied.
        axis_pad: Padding along the last axis.

    Returns:
        4D tensor [1, Cout, A, L'].
    """
    w = conv.weight.unsqueeze(2)  # [Cout, Cin/g, 1, k]
    return F.conv2d(x, w, conv.bias, stride=(1, conv.stride[0]),
                    padding=(0, axis_pad), dilation=(1, conv.dilation[0]),
                    groups=conv.groups)


def glob_ln_per_pos(x, gn, safe=SAFE):
    """GlobLN (GroupNorm(1, C)) per (T) position of a 4D [1, C, T, L].

    Normalizes over (C, L) — matches GlobLN on the folded (B*T, C, L)
    batch. SafeNorm: variance accumulated in a down-scaled domain so the
    fp16 channel-sum cannot overflow.

    Args:
        x: 4D input tensor [1, C, T, L].
        gn: GroupNorm module providing weight and bias.
        safe: Down-scale factor for the variance accumulation.

    Returns:
        Normalized tensor of the same shape as x.
    """
    # E[x] over (C,L)
    m = x.mean(dim=3, keepdim=True).mean(dim=1, keepdim=True)
    d = (x - m) * (1.0 / safe)
    # Var/safe^2
    v = (d * d).mean(dim=3, keepdim=True).mean(dim=1, keepdim=True)
    xh = (x - m) / torch.sqrt(v * (safe * safe) + EPS_GN)
    return xh * gn.weight.view(1, -1, 1, 1) + gn.bias.view(1, -1, 1, 1)


def glob_ln_3d(x, gn, safe=SAFE):
    """GlobLN on a true [1, C, T] tensor (band bottleneck), over (C, T).

    Args:
        x: 3D input tensor [1, C, T].
        gn: GroupNorm module providing weight and bias.
        safe: Down-scale factor for the variance accumulation.

    Returns:
        Normalized tensor of the same shape as x.
    """
    m = x.mean(dim=2, keepdim=True).mean(dim=1, keepdim=True)
    d = (x - m) * (1.0 / safe)
    v = (d * d).mean(dim=2, keepdim=True).mean(dim=1, keepdim=True)
    xh = (x - m) / torch.sqrt(v * (safe * safe) + EPS_GN)
    return xh * gn.weight.view(1, -1, 1) + gn.bias.view(1, -1, 1)


def chan_ln4d(x, ln, safe=SAFE):
    """LayerNormalization4D with n_freqs=1 (dim=(1,)): channel-only LN
    per position.

    Args:
        x: 4D input tensor.
        ln: LayerNormalization4D module providing gamma and beta.
        safe: Down-scale factor for the variance accumulation.

    Returns:
        Normalized tensor of the same shape as x.
    """
    m = x.mean(dim=1, keepdim=True)
    d = (x - m) * (1.0 / safe)
    v = (d * d).mean(dim=1, keepdim=True)
    xh = (x - m) / torch.sqrt(v * (safe * safe) + EPS_LN4)
    return xh * ln.gamma.view(1, -1, 1, 1) + ln.beta.view(1, -1, 1, 1)


def last_axis_mm(x, mat):
    """[1, C, A, L] @ mat[L, L'] -> [1, C, A, L'] via 2D reshape+matmul.

    Lowered as FULLY_CONNECTED with a constant RHS; avoids 4D const-RHS
    BATCH_MATMUL on Mali.

    Args:
        x: 4D input tensor [1, C, A, L].
        mat: Constant matrix [L, L'].

    Returns:
        4D tensor [1, C, A, L'].
    """
    _, C, A, L = x.shape
    y = x.reshape(C * A, L) @ mat
    return y.reshape(1, C, A, mat.shape[1])


def adaptive_pool_matrix(L, out):
    """Exact adaptive_avg_pool1d(L -> out) as a constant matrix.

    Args:
        L: Input length.
        out: Output length.

    Returns:
        [L, out] averaging matrix tensor.
    """
    m = torch.zeros(L, out)
    for i in range(out):
        s = (i * L) // out
        e = -(-((i + 1) * L) // out)  # ceil
        m[s:e, i] = 1.0 / (e - s)
    return m


def nearest_matrix(L_in, L_out):
    """Exact F.interpolate(mode='nearest') as a constant one-hot matrix.

    Args:
        L_in: Input length.
        L_out: Output length.

    Returns:
        [L_in, L_out] one-hot selection matrix tensor.
    """
    m = torch.zeros(L_in, L_out)
    for i in range(L_out):
        m[int(i * L_in / L_out), i] = 1.0
    return m


def stft_conv_weights(win, n_fft):
    """Hann-windowed real-DFT bank as Conv1d weights.

    Matches torch.stft(return_complex):
    X_k = sum_n w[n] x[n] exp(-i 2 pi k n / N).

    Args:
        win: Window length.
        n_fft: FFT size.

    Returns:
        [2*(n_fft//2+1), 1, win] weight tensor (cos rows, then sin rows).
    """
    enc = n_fft // 2 + 1
    hann = torch.hann_window(win, periodic=True)
    n = torch.arange(win, dtype=torch.float64)
    k = torch.arange(enc, dtype=torch.float64).unsqueeze(1)
    ang = 2.0 * math.pi * k * n / n_fft
    cos = (torch.cos(ang) * hann.double()).float()   # real part
    sin = (-torch.sin(ang) * hann.double()).float()  # imag part
    return torch.cat([cos, sin], 0).unsqueeze(1)     # [2*enc, 1, win]


# ------------------------------------------------- re-authored blocks
class GPUUConvBlock(nn.Module):
    """UConvBlock over the LAST axis of [1, N, A, L].

    A is the folded axis kept in-graph.
    """

    def __init__(self, blk, L0):
        super().__init__()
        self.blk = blk
        self.depth = blk.depth
        # per-level lengths along the processed axis
        sizes = [L0]
        for _ in range(1, self.depth):
            # k5 s2 p2 conv output length
            sizes.append((sizes[-1] - 1) // 2 + 1)
        self.sizes = sizes
        Ls = sizes[-1]
        # UNIFORM chain (L_i = Ls * 2^(depth-1-i), e.g. T=1040 ->
        # 520/260/130/65): every adaptive_avg_pool window is
        # exact-uniform and every nearest resize is an exact integer
        # repeat -> use AVERAGE_POOL_2D / on-stride RESIZE_NEAREST
        # instead of the dense matrix-FC route (identical values,
        # ~500 G ops cheaper on the frame axis).
        self.uniform = all(sizes[i] == Ls * (1 << (self.depth - 1 - i))
                           for i in range(self.depth))
        if not self.uniform:
            for i, L in enumerate(sizes):
                self.register_buffer(f"pool_{i}",
                                     adaptive_pool_matrix(L, Ls) if L != Ls
                                     else torch.eye(L), persistent=False)
                self.register_buffer(f"up_{i}", nearest_matrix(Ls, L) if L != Ls
                                     else torch.eye(L), persistent=False)
            # last_layer fusion chain (x_g length -> x_l length):
            # step i=depth-2: x_l=sizes[depth-2], x_g=sizes[depth-3]
            #   (a nearest DOWNsample)
            # step i<depth-2: x_l=sizes[i],       x_g=sizes[i+1]
            self.register_buffer(
                f"fuse_{self.depth-2}",
                nearest_matrix(sizes[self.depth - 3], sizes[self.depth - 2]),
                persistent=False)
            for i in range(self.depth - 3, -1, -1):
                self.register_buffer(f"fuse_{i}",
                                     nearest_matrix(sizes[i + 1], sizes[i]),
                                     persistent=False)

    def resize_to(self, x, L_out):
        """Exact nearest resize between uniform-chain lengths.

        Integer ratios only.
        """
        L_in = x.shape[-1]
        if L_in == L_out:
            return x
        # integer-repeat upsample == F.interpolate nearest (on-stride,
        # Mali-ok)
        if L_out > L_in:
            return F.interpolate(x, size=(x.shape[2], L_out), mode="nearest")
        # nearest downsample by 2 == pick every 2nd == k1 s2 pool
        if L_out * 2 == L_in:
            return F.avg_pool2d(x, kernel_size=(1, 1), stride=(1, 2))
        raise AssertionError(f"unexpected resize {L_in}->{L_out}")

    def cna(self, mod, x, k):
        """ConvNormAct / ConvNorm with per-position GlobLN.

        k = kernel for padding calc.
        """
        y = conv1d_as_2d(x, mod.conv, (k - 1) // 2)
        y = glob_ln_per_pos(y, mod.norm)
        if hasattr(mod, "act"):
            y = prelu(y, mod.act.weight)
        return y

    def nearest(self, x, L_out):
        if x.shape[-1] == L_out:
            return x
        if self.uniform:
            return self.resize_to(x, L_out)
        # non-uniform: look up whichever registered nearest matrix
        # matches (L_in -> L_out)
        for i in range(self.depth):
            m = getattr(self, f"up_{i}", None)
            if (m is not None and m.shape[0] == x.shape[-1]
                    and m.shape[1] == L_out):
                return last_axis_mm(x, m)
        for i in range(self.depth - 1):
            m = getattr(self, f"fuse_{i}", None)
            if (m is not None and m.shape[0] == x.shape[-1]
                    and m.shape[1] == L_out):
                return last_axis_mm(x, m)
        raise AssertionError(f"no nearest matrix {x.shape[-1]}->{L_out}")

    def pool_to_last(self, x, i):
        Ls = self.sizes[-1]
        if x.shape[-1] == Ls:
            return x
        if self.uniform:
            r = x.shape[-1] // Ls
            return F.avg_pool2d(x, kernel_size=(1, r), stride=(1, r))
        return last_axis_mm(x, getattr(self, f"pool_{i}"))

    def inject(self, fus, x_l, x_g, k):
        loc = self.cna(fus.local_embedding, x_l, k)
        g_act = torch.sigmoid(self.cna(fus.global_act, x_g, k))
        out = loc * self.nearest(g_act, x_l.shape[-1])
        if hasattr(fus, "global_embedding"):
            out = out + self.nearest(
                self.cna(fus.global_embedding, x_g, k), x_l.shape[-1])
        return out

    def forward(self, x):
        blk = self.blk
        residual = x
        out1 = conv1d_as_2d(x, blk.proj_1x1.conv, 0)
        out1 = glob_ln_per_pos(out1, blk.proj_1x1.norm)
        out1 = prelu(out1, blk.proj_1x1.act.weight)

        outs = [glob_ln_per_pos(conv1d_as_2d(out1, blk.spp_dw[0].conv, 2),
                                blk.spp_dw[0].norm)]
        for kk in range(1, self.depth):
            outs.append(glob_ln_per_pos(
                conv1d_as_2d(outs[-1], blk.spp_dw[kk].conv, 2),
                blk.spp_dw[kk].norm))

        glob = None
        for i, fea in enumerate(outs):
            p = self.pool_to_last(fea, i)
            glob = p if glob is None else glob + p
        # Mlp globalatt
        mlp = blk.globalatt
        g = glob_ln_per_pos(conv1d_as_2d(glob, mlp.fc1.conv, 0), mlp.fc1.norm)
        g = conv1d_as_2d(g, mlp.dwconv, 2)
        g = F.relu(g)
        g = glob_ln_per_pos(conv1d_as_2d(g, mlp.fc2.conv, 0), mlp.fc2.norm)

        fused = []
        for i in range(self.depth):
            fused.append(self.inject(blk.loc_glo_fus[i], outs[i], g, 1))

        expanded = None
        for i in range(self.depth - 2, -1, -1):
            if i == self.depth - 2:
                expanded = self.inject(blk.last_layer[i], fused[i],
                                       fused[i - 1], 5)
            else:
                expanded = self.inject(blk.last_layer[i], fused[i], expanded, 5)
        return conv1d_as_2d(expanded, blk.res_conv, 0) + residual


class GPUMHSA(nn.Module):
    """MultiHeadSelfAttention2D, dim=4 semantics.

    Input [1, C, F', T'] where attention runs over F' after the internal
    transpose. Re-authored: per-head batch-1 3D BMM.
    """

    def __init__(self, att):
        super().__init__()
        self.att = att

    def qkv(self, mod, x):
        y = F.conv2d(x, mod.conv.weight, mod.conv.bias)
        y = prelu(y, mod.act.weight)
        return chan_ln4d(y, mod.norm)

    def forward(self, x):
        # original dim==4: x.transpose(-2, -1) first
        att = self.att
        x = x.transpose(-2, -1).contiguous()          # [1, C, tokens, emb_axis]
        B, C, Tk, Fx = x.shape
        residual = x
        heads = []
        for h in range(att.n_head):
            q = self.qkv(att.Queries[h], x)           # [1, E, Tk, Fx]
            k = self.qkv(att.Keys[h], x)
            v = self.qkv(att.Values[h], x)            # [1, C/nh, Tk, Fx]
            E = q.shape[1]
            Cv = v.shape[1]
            # fold the 1/sqrt(d) into Q BEFORE the matmul (fp16
            # accumulation headroom)
            q3 = (q.transpose(1, 2).reshape(1, Tk, E * Fx)
                  * (1.0 / math.sqrt(E * Fx)))
            k3 = k.transpose(1, 2).reshape(1, Tk, E * Fx)
            v3 = v.transpose(1, 2).reshape(1, Tk, Cv * Fx)
            attn = torch.matmul(q3, k3.transpose(1, 2))
            attn = F.softmax(attn, dim=2)
            o = torch.matmul(attn, v3)                # [1, Tk, Cv*Fx]
            o = o.reshape(1, Tk, Cv, Fx).transpose(1, 2)  # [1, Cv, Tk, Fx]
            heads.append(o)
        y = torch.cat(heads, dim=1)                   # [1, C, Tk, Fx]
        p = att.attn_concat_proj
        y = F.conv2d(y, p.conv.weight, p.conv.bias)
        y = prelu(y, p.act.weight)
        y = chan_ln4d(y, p.norm)
        y = y + residual
        return y.transpose(-2, -1).contiguous()


class GPUTiger(nn.Module):
    """Full single-TIGER graph: pre-padded wav -> separated spectrograms.

    Outputs (spec_real, spec_imag) of the separated sources,
    [1, K, enc_dim, T] each. iSTFT is host-side.
    """

    def __init__(self, tiger, n_samples):
        super().__init__()
        self.t = tiger
        self.win = tiger.win
        self.hop = tiger.stride
        self.enc = tiger.enc_dim
        self.nband = tiger.nband
        self.bands = list(tiger.band_width)
        self.K = tiger.num_output
        self.N = tiger.feature_dim
        # frames after center-pad (host pads win//2 both sides)
        self.T = n_samples // self.hop + 1
        self.register_buffer("dft", stft_conv_weights(self.win, self.win),
                             persistent=False)

        sep = tiger.separator
        self.iters = sep.iter
        self.freq_u = GPUUConvBlock(sep.freq_path[0], self.nband)
        self.freq_a = GPUMHSA(sep.freq_path[1])
        self.frame_u = GPUUConvBlock(sep.frame_path[0], self.T)
        self.frame_a = GPUMHSA(sep.frame_path[1])

    def band_bottleneck(self, real, imag):
        feats = []
        i0 = 0
        for i, bw in enumerate(self.bands):
            # [1, 2bw, T]
            ri = torch.cat([real[:, i0:i0 + bw], imag[:, i0:i0 + bw]],
                           dim=1)
            gn, cv = self.t.BN[i][0], self.t.BN[i][1]
            y = glob_ln_3d(ri, gn)
            y = F.conv1d(y, cv.weight, cv.bias)
            feats.append(y.unsqueeze(2))              # [1, N, 1, T]
            i0 += bw
        return torch.cat(feats, dim=2)                # [1, N, nband, T]

    def ftp(self, x):
        """freq_time_process on [1, N, nband, T]."""
        sep = self.t.separator
        res1 = x
        y = x.transpose(-2, -1).contiguous()          # [1, N, T, nband]
        y = self.freq_u(y)                            # UConv over band axis
        # MHSA freq: original input (B, N, T, nband), dim=4
        y = self.freq_a(y)
        y = chan_ln4d(y, sep.freq_path[2])            # LN4D over channels
        y = y.transpose(-2, -1).contiguous()          # back to [1, N, nband, T]
        x = y + res1

        res2 = x
        # UConv over T axis (layout already right)
        z = self.frame_u(x)
        # MHSA frame: input (B, N, nband, T), dim=4
        z = self.frame_a(z)
        z = chan_ln4d(z, sep.frame_path[2])
        return z + res2

    def forward(self, wav_padded):
        # wav_padded: [1, n_samples + win]
        # (host reflect-padded win//2 each side)
        # spec: [1, 2*enc, T]
        spec = F.conv1d(wav_padded.unsqueeze(1), self.dft, stride=self.hop)
        real, imag = spec[:, :self.enc], spec[:, self.enc:]

        x = self.band_bottleneck(real, imag)          # [1, N, nband, T]
        sep = self.t.separator
        mixture = x
        for i in range(self.iters):
            if i == 0:
                x = self.ftp(x)
            else:
                c = sep.concat_block[0]
                y = F.conv2d(mixture + x, c.weight, c.bias, groups=c.groups)
                y = prelu(y, sep.concat_block[1].weight)
                x = self.ftp(y)

        # mask heads (6D view -> static slices)
        outs_r, outs_i = [], []
        i0 = 0
        for i, bw in enumerate(self.bands):
            m_pre, m_cv = self.t.mask[i][0], self.t.mask[i][1]
            h = x[:, :, i, :]                          # [1, N, T]
            h = (F.relu(h) - m_pre.weight.view(1, -1, 1) * F.relu(-h)
                 if m_pre.weight.numel() > 1
                 else F.relu(h) - m_pre.weight.reshape(1) * F.relu(-h))
            # o: [1, bw*4*K, T]
            o = F.conv1d(h, m_cv.weight, m_cv.bias, groups=m_cv.groups)
            # channel layout after view(B,2,2,K,BW,T):
            # c = ((m*2+r)*K + k)*BW + b
            def sl(m, r, k):
                base = ((m * 2 + r) * self.K + k) * bw
                return o[:, base:base + bw]            # [1, bw, T]
            # Per-source, SAME-SHAPE [1, bw, T] arithmetic only. The
            # obvious r_in.unsqueeze(1) * mask[1,K,bw,T] dim-1 broadcast
            # MUL is mis-executed by the Mali delegate (device
            # 2026-07-02: all K outputs became source 0's copy), so no
            # dim-1 broadcasts and no .sum over a stacked source dim.
            r_in, i_in = real[:, i0:i0 + bw], imag[:, i0:i0 + bw]
            mrs = [sl(0, 0, k) * torch.sigmoid(sl(1, 0, k))
                   for k in range(self.K)]
            mis = [sl(0, 1, k) * torch.sigmoid(sl(1, 1, k))
                   for k in range(self.K)]
            mr_sum = mrs[0]
            mi_sum = mis[0]
            for k in range(1, self.K):
                mr_sum = mr_sum + mrs[k]
                mi_sum = mi_sum + mis[k]
            mr_off = (mr_sum - 1.0) * (1.0 / self.K)
            mi_off = mi_sum * (1.0 / self.K)
            band_r, band_i = [], []
            for k in range(self.K):
                mr = mrs[k] - mr_off
                mi = mis[k] - mi_off
                # [1, 1, bw, T]
                band_r.append((r_in * mr - i_in * mi).unsqueeze(1))
                band_i.append((r_in * mi + i_in * mr).unsqueeze(1))
            # [1, K, bw, T]
            outs_r.append(torch.cat(band_r, dim=1))
            outs_i.append(torch.cat(band_i, dim=1))
            i0 += bw
        # [1, K, enc, T] x2
        return torch.cat(outs_r, dim=2), torch.cat(outs_i, dim=2)


def host_istft(real, imag, win, hop, length):
    """Reference host-side iSTFT (torch) for parity checks.

    The Kotlin port mirrors this.

    Args:
        real: Real spectrogram [K, enc_dim, T].
        imag: Imaginary spectrogram [K, enc_dim, T].
        win: Window / FFT size.
        hop: Hop length.
        length: Output waveform length in samples.

    Returns:
        Waveform tensor [K, length].
    """
    spec = torch.complex(real, imag)
    return torch.istft(spec, n_fft=win, hop_length=hop,
                       window=torch.hann_window(win), length=length)


def host_pad(wav, win):
    """Reflect-pads win//2 both sides (torch.stft center=True).

    Args:
        wav: Waveform tensor [1, n_samples].
        win: Window size; the pad is win // 2 on each side.

    Returns:
        Padded waveform tensor [1, n_samples + win].
    """
    return F.pad(wav, (win // 2, win // 2), mode="reflect")
