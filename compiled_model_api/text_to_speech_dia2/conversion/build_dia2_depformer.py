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

"""Build the Dia2-1B depformer STEP graph.

Three shared layers, MHA 8/8, RoPE, attention scale 1.0.

Per stage the host computes dep_in = depformer_in[weight_idx] @ hidden
+ audio_embeds[stage][prev_audio]. (Dia2-1B has depformer.text_embedding =
False, so no text embedding is added at stage 0.) The graph runs the 3 shared
layers over the depformer's own KV cache (positions = stage index). The host
then applies
the per-stage logits head. Validated vs Depformer.forward_step. fp32 (CPU).
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, "dia2")

HD = 128
NH = 8
NLAYER = 3
H = 1024
THETA = 10000.0
DMAX = 31
OUT = os.environ.get("DIA2_OUT", "out")


def rms(x, w, eps):
    """Applies RMSNorm.

    Args:
      x: Input tensor.
      w: Per-channel scale.
      eps: Variance epsilon.

    Returns:
      The normalized tensor.
    """
    r = torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps)
    return (x * r) * w


def rope(x, cos, sin):
    """Applies rotary position embedding.

    Args:
      x: Tensor of shape [..., head_dim].
      cos: Cosine table broadcastable to x.
      sin: Sine table broadcastable to x.

    Returns:
      The rotated tensor.
    """
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


class DepStep(nn.Module):
    """The 3-layer depformer stack for one weight-id `wi`."""

    def __init__(self, dp, wi):
        super().__init__()
        self.layers = dp.layers
        self.wi = str(wi)
        self.norm_w = nn.Parameter(dp.norm.weight.data.clone())
        self.eps = dp.norm.eps

    def forward(self, dep_in, cos, sin, mask, pk, pv):
        x = dep_in
        nk, nv = [], []
        for i, layer in enumerate(self.layers):
            a = layer.self_attention
            h = rms(x, layer.pre_norm.weight, layer.pre_norm.eps)
            proj = a.in_proj[self.wi](h).view(1, 1, 3, NH, HD)   # fused QKV
            q = rms(proj[:, :, 0], a.q_norm.weight, a.q_norm.eps)
            k = rms(proj[:, :, 1], a.k_norm.weight, a.k_norm.eps)
            v = proj[:, :, 2]
            q, k = rope(q, cos, sin), rope(k, cos, sin)
            kt, vt = k.transpose(1, 2), v.transpose(1, 2)
            base = i * NH
            k_all = torch.cat([pk[:, base:base + NH], kt], dim=2)
            v_all = torch.cat([pv[:, base:base + NH], vt], dim=2)
            scores = torch.matmul(
                q.transpose(1, 2), k_all.transpose(-1, -2)) + mask
            ctx = torch.matmul(torch.softmax(scores, dim=-1), v_all)
            o = a.out_proj[self.wi](ctx.transpose(1, 2).reshape(1, 1, NH * HD))
            x = x + o
            h2 = rms(x, layer.post_norm.weight, layer.post_norm.eps)
            gate, up = layer.mlp.wi(h2).view(
                1, 1, 2, layer.mlp.hidden).unbind(dim=-2)
            x = x + layer.mlp.wo(F.silu(gate) * up)
            nk.append(kt)
            nv.append(vt)
        return rms(x, self.norm_w, self.eps), torch.cat(nk, 1), torch.cat(nv, 1)


def main():
    """Builds and validates the three depformer step graphs."""
    from dia2.engine import Dia2
    d = Dia2.from_repo("nari-labs/Dia2-1B")
    d.set_device("cpu")
    model = d._ensure_runtime().model
    dp = model.depformer.float().eval()

    stage = 5                      # a stage that exercises weight_idx 0
    wi = str(dp.weights_schedule[stage])
    step = DepStep(dp, dp.weights_schedule[stage]).eval()
    prev = torch.tensor([123])
    thidden = torch.randn(1, 1, H)

    cache = dp.init_cache(1, torch.device("cpu"), DMAX)
    with torch.no_grad():
        logits_ref, _ = dp.forward_step(prev, thidden, stage, cache, None, None)

    # host dep_in
    with torch.no_grad():
        token_emb = dp.audio_embeds[stage](prev[:, None]).float()
        dep_in = dp.depformer_in[wi](thidden.float()) + token_emb
    half = HD // 2
    inv = 1.0 / (THETA ** ((2.0 * torch.arange(half)) / HD))
    ang = stage * inv
    cos = torch.cat([ang.cos(), ang.cos()]).view(1, 1, 1, HD)
    sin = torch.cat([ang.sin(), ang.sin()]).view(1, 1, 1, HD)
    mask = torch.full((1, 1, 1, DMAX + 1), torch.finfo(torch.float32).min)
    # empty cache: attend only the current token (matches forward_step)
    mask[..., DMAX] = 0.0
    pk = torch.zeros(1, NLAYER * NH, DMAX, HD)
    pv = torch.zeros(1, NLAYER * NH, DMAX, HD)
    with torch.no_grad():
        hidden, nk, nv = step(dep_in, cos, sin, mask, pk, pv)
        logits = dp.logits[stage](hidden.float())[
            ..., :dp.audio_vocab_limit].unsqueeze(1)

    def corr(a, b):
        a, b = a.flatten().float(), b.flatten().float()
        return float(((a - a.mean()) * (b - b.mean())).mean()
                     / (a.std() * b.std() + 1e-9))
    print(f"depformer logits corr={corr(logits, logits_ref):.6f} "
          f"argmax match={int(logits.argmax()) == int(logits_ref.argmax())}")

    import litert_torch
    for w in sorted(set(dp.weights_schedule)):
        g = DepStep(dp, w).eval()
        path = f"{OUT}/dia2_depformer_wi{w}_fp32.tflite"
        litert_torch.convert(g, (dep_in, cos, sin, mask, pk, pv)).export(path)
        print(f"exported {path} ({os.path.getsize(path) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
