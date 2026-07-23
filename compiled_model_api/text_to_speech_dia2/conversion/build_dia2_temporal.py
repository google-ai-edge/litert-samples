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

"""Build the Dia2-1B temporal-transformer KV-cache STEP graph.

Qwen3-style attention.

One step: host-summed 34-channel embed [1,1,1024] + host RoPE cos/sin +
packed KV cache -> hidden_norm + action_logits + cb0_logits + new k/v.
Manual matmul+softmax attention (SDPA scale=1.0, q/k-norm, GQA 16Q/8KV via
repeat_interleave), packed [1,L*nkv,Pmax,hd] cache concat-at-tail + additive
mask (no scatter/GATHER). fp32 (deep LM -> CPU). Bit-exact vs forward_step.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, "dia2")

HD = 128
NQ = 16
NKV = 8
NLAYER = 30
H = 1024
THETA = 10000.0
PMAX = 256
OUT = os.environ.get("DIA2_OUT", "out")
os.makedirs(OUT, exist_ok=True)


def safe_rms(x, weight, eps):
    """Applies plain RMSNorm (fp32 on CPU, so no overflow guard).

    Args:
      x: Input tensor.
      weight: Per-channel scale.
      eps: Variance epsilon.

    Returns:
      The normalized tensor.
    """
    r = torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps)
    return (x * r) * weight


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
    rot = torch.cat((-x2, x1), dim=-1)
    return x * cos + rot * sin


class TemporalStep(nn.Module):
    """Reimplements TransformerDecoder.forward_step with packed-KV graph I/O."""

    def __init__(self, t):
        super().__init__()
        self.layers = t.layers
        self.norm_w = nn.Parameter(t.norm.weight.data.clone())
        self.eps = t.layers[0].pre_norm.eps
        self.action_head = t.action_head
        self.cb0_head = t.cb0_head

    def forward(self, x, cos, sin, mask, pk, pv):
        # x [1,1,H]; cos/sin [1,1,1,HD]; mask [1,1,1,PMAX+1];
        # pk/pv [1,L*NKV,PMAX,HD]
        nk_out = []
        nv_out = []
        for i, layer in enumerate(self.layers):
            a = layer.attn
            h = safe_rms(x, layer.pre_norm.weight, layer.pre_norm.eps)
            q = a.q_proj(h).view(1, 1, NQ, HD)
            k = a.k_proj(h).view(1, 1, NKV, HD)
            v = a.v_proj(h).view(1, 1, NKV, HD)
            q = safe_rms(q, a.q_norm.weight, a.q_norm.eps)
            k = safe_rms(k, a.k_norm.weight, a.k_norm.eps)
            q = rope(q, cos, sin)
            k = rope(k, cos, sin)
            # [1,NKV,1,HD] for this token
            kt = k.transpose(1, 2)
            vt = v.transpose(1, 2)
            base = i * NKV
            pkl = pk[:, base:base + NKV]                       # [1,NKV,PMAX,HD]
            pvl = pv[:, base:base + NKV]
            k_all = torch.cat([pkl, kt], dim=2)      # [1,NKV,PMAX+1,HD]
            v_all = torch.cat([pvl, vt], dim=2)
            qh = q.transpose(1, 2)                             # [1,NQ,1,HD]
            # GQA: repeat kv heads grouped, matching sdpa enable_gqa
            rep = NQ // NKV
            k_rep = k_all.repeat_interleave(rep, dim=1)  # [1,NQ,PMAX+1,HD]
            v_rep = v_all.repeat_interleave(rep, dim=1)
            # SDPA scale is 1.0 for this model, not 1/sqrt(head_dim).
            scores = torch.matmul(qh, k_rep.transpose(-1, -2))
            scores = scores + mask
            attn = torch.softmax(scores, dim=-1)
            ctx = torch.matmul(attn, v_rep)                   # [1,NQ,1,HD]
            ctx = ctx.transpose(1, 2).reshape(1, 1, NQ * HD)
            o = a.o_proj(ctx)
            x = x + o
            h2 = safe_rms(x, layer.post_norm.weight, layer.post_norm.eps)
            proj = layer.mlp.wi(h2).view(1, 1, 2, layer.mlp.hidden)
            gate, up = proj.unbind(dim=-2)
            x = x + layer.mlp.wo(F.silu(gate) * up)
            nk_out.append(kt)
            nv_out.append(vt)
        hn = safe_rms(x, self.norm_w, self.eps)
        action = self.action_head(hn)
        cb0 = self.cb0_head(hn)
        nk = torch.cat(nk_out, dim=1)                         # [1,L*NKV,1,HD]
        nv = torch.cat(nv_out, dim=1)
        return hn, action, cb0, nk, nv


def main():
    """Builds and validates the temporal step graph."""
    from dia2.engine import Dia2
    d = Dia2.from_repo("nari-labs/Dia2-1B")
    d.set_device("cpu")
    model = d._ensure_runtime().model
    t = model.transformer.float().eval()
    step = TemporalStep(t).eval()

    # reference: real forward_step, empty cache, one token at pos 0
    torch.manual_seed(0)
    tokens = torch.randint(0, 2000, (1, 34, 1))
    tokens[:, 0, :] = 100
    # second stream padded
    tokens[:, 1, :] = model.config.data.text_pad_token_id
    cache = t.init_cache(1, torch.device("cpu"), PMAX)
    pos = torch.tensor([[0]])
    with torch.no_grad():
        hn_ref, act_ref, cb0_ref, _ = t.forward_step(tokens, pos, cache)

    # host embed for the same tokens
    with torch.no_grad():
        emb = t.text_embed(tokens[:, 0, :], tokens[:, 1, :])
        for idx in range(32):
            emb = emb + t.audio_embeds[idx](tokens[:, idx + 2, :])
        emb = emb.float()
    # host RoPE at pos 0
    half = HD // 2
    frac = (2.0 * torch.arange(half)) / HD
    inv = 1.0 / (THETA ** frac)
    ang = 0.0 * inv
    cos = torch.cat([ang.cos(), ang.cos()]).view(1, 1, 1, HD)
    sin = torch.cat([ang.sin(), ang.sin()]).view(1, 1, 1, HD)
    mask = torch.full((1, 1, 1, PMAX + 1), torch.finfo(torch.float32).min)
    mask[..., PMAX] = 0.0  # only the current token attends (empty cache)
    pk = torch.zeros(1, NLAYER * NKV, PMAX, HD)
    pv = torch.zeros(1, NLAYER * NKV, PMAX, HD)
    with torch.no_grad():
        hn, act, cb0, nk, nv = step(emb, cos, sin, mask, pk, pv)

    def corr(a, b):
        a, b = a.flatten().float(), b.flatten().float()
        return float(((a - a.mean()) * (b - b.mean())).mean()
                     / (a.std() * b.std() + 1e-9))
    print(f"hidden corr={corr(hn, hn_ref):.6f} "
          f"cb0 corr={corr(cb0, cb0_ref):.6f} "
          f"action corr={corr(act, act_ref):.6f}")
    print(f"cb0 argmax match: {int(cb0.argmax()) == int(cb0_ref.argmax())}")

    # --- convert to fp32 tflite (temporal LM -> CPU) ---
    import litert_torch
    path = f"{OUT}/dia2_temporal_fp32.tflite"
    litert_torch.convert(step, (emb, cos, sin, mask, pk, pv)).export(path)
    print(f"exported {path} ({os.path.getsize(path) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
