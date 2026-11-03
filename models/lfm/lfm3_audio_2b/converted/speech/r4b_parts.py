# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPU-friendly forms of the LFM3-Audio codec decoder and depth decoder: the speech-output graphs.

The module name `r4b_parts` and the class names reach the exported graphs' tensor names, so they stay as they were
when the published files (litert-community/LFM3-Audio-2B-LiteRT) were exported; with them, a rebuild with the same
toolchain gives the same bytes. The forms re-author r1b_graphs.CodecChunk / DepthStep so that the LiteRT GPU delegate
takes every op and fp16 activations keep their values. Same weights and parameter names (strict loads), fp32 graphs.

  CodecChunkR4b    decode: codes [1,12,F] int32, start [1] int32, k0 v0 k1 v1 k2 v2 [1,2,29,64], c0..c4 [1,6,512]
                   -> frames [1,6F,1280] + the state under the same names + `_out` (r1b_graphs.CodecChunk's names and
                   shapes)
    - RoPE relative to the chunk: the cached keys are kept UNROTATED (k* = keys after k_norm, before RoPE) and every
      call rotates the 29 cached + 6F new keys at positions -29..6F-1 and the queries at 0..6F-1 with constant
      tables. q_m . k_n under RoPE depends on m - n only, so the scores equal CodecChunk's absolute-position scores in
      exact arithmetic; CodecChunk computes cos / sin(position x inv_freq) in the graph from `start` (positions grow
      with the stream: fp16 rounds a position past 2,048). CodecChunk's k* state (rotated keys) and this form's
      (unrotated) are not interchangeable: a stream starts from zeros with one form.
    - the empty-slot term (cache slots before position 0) = relu(x) - relu(x - 1) with x = -(key position - start)
      (= clamp(x, 0, 1) for every real x; torch.clamp lowers to RELU_0_TO_1, which the GPU delegate refuses) times
      -10000 (-1e9 is -inf in fp16, and 0 x -inf = NaN where a slot is not empty).
    - RMSNorm with a per-row scale (rms_norm_scaled): the codec's residual reaches |x| ~ 3,000, whose square is inf in
      fp16.
  DepthStepR4b     r1b_graphs.DepthStep unchanged (GQA already folded, no mask, no index past 2,016);
                   DepthProjR4b / DepthPassR4b split it into the per-frame projection and the per-pass rest.
The torch modules evaluate in fp32. (The published `.litertlm` file's audio encoder section was traced from an
earlier version of this module; in this recipe the audio extension builds it.)
"""
import importlib.util
import math
import pathlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


G = _load("r1b_graphs", HERE / "r1b_graphs.py")

MASK_NEG_CODEC = -10000.0
NORM_FLOOR = 2.0 ** -14


def rms_norm_scaled(x, w, eps):
    """RMSNorm with a per-row scale: x / m, m = max(max|x|, 2^-14), and eps / m^2 (the same value in exact arithmetic:
    rsqrt(mean((x/m)^2) + eps/m^2) * (x/m) = rsqrt(mean(x^2) + eps) * x). The plain form squares x in the activation
    dtype: the codec's residual reaches |x| ~ 3,000 (x^2 9.3e6 at layer 0 on 20 test sequences) = inf in fp16 (max
    65,504). After the scale (x/m)^2 <= 1 and the row mean >= 1/d."""
    m = torch.clamp(torch.amax(torch.abs(x), dim=-1, keepdim=True), min=NORM_FLOOR)
    xs = x / m
    r = math.sqrt(eps) / m                                           # eps / m^2 as (sqrt(eps) / m)^2: m^2 overflows fp16
    return xs * torch.rsqrt(xs.pow(2).mean(-1, keepdim=True) + r * r) * w


class CodecLayerR4b(G.Layer):
    """r1b_graphs' codec Layer (conv layers; sandwich norms) with the scaled RMSNorm."""

    def forward(self, x, **kw):
        h = rms_norm_scaled(x, self.op_norm, self.eps)
        o, c = self.conv(h, kw["conv_state"])
        o = rms_norm_scaled(o, self.post_op_norm, self.eps)
        x = x + o
        f = rms_norm_scaled(self.ffn(rms_norm_scaled(x, self.ffn_norm, self.eps)), self.post_ffn_norm, self.eps)
        return x + f, (c,)


class CodecAttentionR4b(G.Attention):
    """r1b_graphs' codec attention with the keys cached before RoPE; rotation by constant tables (positions relative to the
    chunk start): queries at 0..t-1 (cos_q / sin_q [t, hd]), the 29 cached + t new keys at -29..t-1 (cos_k / sin_k)."""

    def forward(self, h, cos_q, sin_q, cos_k, sin_k, bias, k_past, v_past):
        b, t, _ = h.shape
        q = rms_norm_scaled(F.linear(h, self.q_proj).view(b, t, self.h, self.hd), self.q_norm, self.eps).transpose(1, 2)
        k = rms_norm_scaled(F.linear(h, self.k_proj).view(b, t, self.kv, self.hd), self.k_norm, self.eps).transpose(1, 2)
        v = F.linear(h, self.v_proj).view(b, t, self.kv, self.hd).transpose(1, 2)
        k_raw = torch.cat((k_past, k), dim=2)                      # [b, kv, 29 + t, hd], unrotated
        v_all = torch.cat((v_past, v), dim=2)
        q = q * cos_q + G.rot_half(q) * sin_q
        kr = k_raw * cos_k + G.rot_half(k_raw) * sin_k
        rep = self.h // self.kv
        qg = q.reshape(b, self.kv, rep * t, self.hd)
        att = torch.matmul(qg, kr.transpose(2, 3)) * (self.hd ** -0.5) + bias
        att = torch.softmax(att, dim=-1)
        o = torch.matmul(att, v_all).reshape(b, self.h, t, self.hd).transpose(1, 2)
        gate = torch.sigmoid(F.linear(h, self.gate_proj)).unsqueeze(-1)
        o = (o * gate).reshape(b, t, self.h * self.hd)
        return F.linear(o, self.o_proj), k_raw, v_all


class CodecChunkR4b(G.CodecChunk):
    def __init__(self, sd, frames, window=None):
        super().__init__(sd, frames, window)
        c = G.CODEC_CFG
        for i in range(c["layers"]):
            if i in c["attn_layers"]:
                self.layers[i].attn.__class__ = CodecAttentionR4b
            else:
                self.layers[i].__class__ = CodecLayerR4b
        L = G.CODEC_KV_LEN + self.t
        cq, sq = G.rope_cos_sin(torch.arange(self.t), self.inv_freq)
        ck, sk = G.rope_cos_sin(torch.arange(-G.CODEC_KV_LEN, self.t), self.inv_freq)
        # the tables in float64 then fp32 (exact small-integer positions; CodecChunk computes fp32 position x inv_freq)
        pos_q = torch.arange(self.t, dtype=torch.float64)
        pos_k = torch.arange(-G.CODEC_KV_LEN, self.t, dtype=torch.float64)
        inv = self.inv_freq.double()
        for name, pos in (("q", pos_q), ("k", pos_k)):
            fr = pos[:, None] * inv[None, :]
            emb = torch.cat((fr, fr), dim=-1)
            self.register_buffer(f"cos_{name}", emb.cos().float().contiguous())
            self.register_buffer(f"sin_{name}", emb.sin().float().contiguous())
        rep = c["heads"] // c["kv_heads"]
        static = self.static_bias / -1e9 * MASK_NEG_CODEC          # CodecChunk's [rep * t, L] window mask, 0 / -1e9 -> 0 / -1e4
        assert static.shape == (rep * self.t, L)
        # rank 4: ADD(runtime [L], const [rep t, L]) is a mixed-rank constant operand, which gave wrong values on a GPU
        self.register_buffer("static_bias_r4b", static.reshape(1, 1, rep * self.t, L).contiguous())
        self.register_buffer("krel4", self.krel.reshape(1, 1, 1, L).contiguous())

    def forward(self, codes, start, k0, v0, k1, v1, k2, v2, c0, c1, c2, c3, c4):
        c = G.CODEC_CFG
        lat = F.embedding(torch.clamp(codes, 0, G.CB_SIZE - 1), self.lat_table)
        lat = lat.permute(0, 2, 1, 3).reshape(1, self.f, G.NUM_CB * len(c["fsq_levels"]))
        emb = F.linear(F.linear(lat, self.po_w, self.po_b), self.fq_w, self.fq_b)
        x = torch.cat([emb] * c["frame_stack"], dim=-1).reshape(1, self.t, c["dim"])
        s = start.float().reshape(1, 1, 1, 1)
        xe = -(self.krel4 + s)                                      # > 0 exactly for the slots before position 0
        empty = F.relu(xe) - F.relu(xe - 1.0)                       # = clamp(xe, 0, 1), [1, 1, 1, L]
        bias = self.static_bias_r4b + empty * MASK_NEG_CODEC        # [1, 1, rep t, L] runtime
        kv = [(k0, v0), (k1, v1), (k2, v2)]
        convs = [c0, c1, c2, c3, c4]
        kv_out, conv_out = [], []
        ia = ic = 0
        for i, layer in enumerate(self.layers):
            if i in c["attn_layers"]:
                h = rms_norm_scaled(x, layer.op_norm, layer.eps)
                o, k_raw, v_all = layer.attn(h, self.cos_q, self.sin_q, self.cos_k, self.sin_k, bias,
                                             kv[ia][0], kv[ia][1])
                o = rms_norm_scaled(o, layer.post_op_norm, layer.eps)
                x = x + o
                f = rms_norm_scaled(layer.ffn(rms_norm_scaled(x, layer.ffn_norm, layer.eps)), layer.post_ffn_norm,
                                    layer.eps)
                x = x + f
                kv_out += [k_raw[:, :, -G.CODEC_KV_LEN:], v_all[:, :, -G.CODEC_KV_LEN:]]
                ia += 1
            else:
                x, (cs,) = layer(x, conv_state=convs[ic])
                conv_out.append(cs)
                ic += 1
        x = rms_norm_scaled(x, self.final_norm, self.eps)
        y = F.linear(x, self.lin_w, self.lin_b)
        mag = torch.exp(y[..., :self.nbins])
        ang = y[..., self.nbins:]
        z = torch.cat((mag * torch.cos(ang), mag * torch.sin(ang)), dim=-1)
        frames = F.linear(z, self.basis_t)
        return (frames, *kv_out, *conv_out)


def codec_zero_state_r4b():
    return G.codec_zero_state()


# ---------------------------------------------------------------- depth step
class DepthStepR4b(G.DepthStep):
    """= r1b_graphs.DepthStep (no fp16 overflow site: max |x| 10,557 on 20 test sequences)."""


class DepthProjR4b(nn.Module):
    """depth_proj: hidden [1,2048] -> depth_in [1,12,1024] = depth_linear (DepthStep.project), once per frame."""

    def __init__(self, step):
        super().__init__()
        self.step = step

    def forward(self, hidden):
        return {"depth_in": self.step.project(hidden)}


class DepthPassR4b(nn.Module):
    """depth_pass: depth_in [1,12,1024] (depth_proj's output), codes [1,12] int32, masked [1,12] f32 -> logits
    [1,12,2017] = DepthStep.passes (the split form: the frame's 3 passes reuse one projection; DepthStep recomputes
    the 25.2 M-parameter depth_linear in every pass)."""

    def __init__(self, step):
        super().__init__()
        self.step = step

    def forward(self, depth_in, codes, masked):
        return {"logits": self.step.passes(depth_in, codes, masked)}


def export_depth_split(sd, out_proj, out_pass):
    import litert_torch
    step = DepthStepR4b(sd).eval()
    for p in (out_proj, out_pass):
        pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True)
    litert_torch.signature("depth_proj", DepthProjR4b(step).eval(),
                           sample_kwargs={"hidden": torch.zeros(1, G.LM_HIDDEN)}).convert().export(str(out_proj))
    kw = {"depth_in": torch.zeros(1, G.NUM_CB, G.DEPTH_HIDDEN), "codes": torch.zeros(1, G.NUM_CB, dtype=torch.int32),
          "masked": torch.ones(1, G.NUM_CB)}
    litert_torch.signature("depth_pass", DepthPassR4b(step).eval(), sample_kwargs=kw).convert().export(str(out_pass))
    return out_proj, out_pass


# ---------------------------------------------------------------- export (litert-torch, the same toolchain as r1b_graphs)
class CodecSig(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, codes, start, k0, v0, k1, v1, k2, v2, c0, c1, c2, c3, c4):
        out = self.m(codes, start, k0, v0, k1, v1, k2, v2, c0, c1, c2, c3, c4)
        return dict(zip(["frames"] + [n + "_out" for n in G.CODEC_STATE_NAMES], out))


def export_codec(sd, frames, out_path):
    import litert_torch
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m = CodecSig(CodecChunkR4b(sd, frames)).eval()
    litert_torch.signature("decode", m, sample_kwargs=G.codec_chunk_sample(frames)).convert().export(str(out_path))
    return out_path
