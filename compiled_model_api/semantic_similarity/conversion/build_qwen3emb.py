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

"""Qwen3-Embedding-0.6B -> LiteRT GPU (fully CompiledModel, fp16).

Single graph, one forward (last-token pooling => no generation, no KV cache):
  input : inputs_embeds [1, L, 1024]   (host does token embedding lookup)
  output: hidden        [1, L, 1024]   (post final RMSNorm; host takes last real
          token, L2-normalizes, optional Matryoshka truncation)

GPU-clean re-authoring of the 28-layer Qwen3 decoder:
  - token embed lookup is GATHER -> done on HOST, embeds fed in
  - GQA (16 Q / 8 KV heads) via [8, rep, L, d] broadcast matmul
    (all <=4D, no gather)
  - Qwen3 per-head q_norm / k_norm = RMSNorm(head_dim)  -> SafeRMS
  - RoPE (theta 1e6) precomputed for fixed L (constants, mul/add only)
  - causal mask = constant additive [L, L]
  - RMSNorm -> SafeRMS (scale-invariant down-scaled reduction, fp16-safe)
  - SwiGLU: down(silu(gate) * up), silu = x*sigmoid(x)  (GPU-clean, no GELU)

Run: python build_qwen3emb.py [--L 128] [--layers 28]
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "Qwen3-Embedding-0.6B")

EPS = 1e-6


def safe_rms(x, weight, eps=EPS):
    """fp16-safe RMSNorm(x)*weight via per-row max-normalization.

    Mali computes fp16 regardless of dtype. The 28-layer residual stream
    grows large, so mean(x^2) overflows fp16 (-> inf -> rsqrt=0 -> whole
    output collapses to 0); dividing each row by its own max first
    bounds x^2 in [0,1] so the sum-of-squares never overflows.
    Mathematically identical to standard RMSNorm: with m=max|x|,
    y = (x/m)*rsqrt(mean((x/m)^2)+eps/m^2)*w.

    Args:
        x: Input tensor; normalized over its last dim.
        weight: RMSNorm weight for the last dim.
        eps: Numerical-stability epsilon.

    Returns:
        RMSNorm(x) * weight, same shape as x.
    """
    # per-row scale, finite > 0
    m = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    xs = x / m
    # in [1/N, 1], fp16-safe
    var = (xs * xs).mean(dim=-1, keepdim=True)
    return xs * torch.rsqrt(var + eps / (m * m)) * weight


def rotate_half(x):
    """RoPE helper: (-x2, x1) where x = cat(x1, x2) over the last dim.

    Args:
        x: Tensor whose last dim is even.

    Returns:
        The rotated tensor, same shape as x.
    """
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


class Qwen3EmbGPU(nn.Module):
    """Fully GPU-clean forward over inputs_embeds -> post-norm hidden states."""

    def __init__(self, base, L, n_layers):
        super().__init__()
        self.model = base
        self.layers = base.layers[:n_layers]
        self.final_norm_w = base.norm.weight
        self.L = L
        cfg = base.config
        self.n_q = cfg.num_attention_heads          # 16
        self.n_kv = cfg.num_key_value_heads          # 8
        self.rep = self.n_q // self.n_kv             # 2
        self.hd = cfg.head_dim                        # 128
        self.scale = self.hd ** -0.5

        # RoPE cos/sin constants for fixed L (pull inv_freq from the
        # model if present)
        hd = self.hd
        rope = getattr(base, "rotary_emb", None)
        if rope is not None and hasattr(rope, "inv_freq"):
            inv_freq = rope.inv_freq.detach().float()
        else:
            rp = (getattr(cfg, "rope_parameters", None)
                  or getattr(cfg, "rope_scaling", None) or {})
            theta = (getattr(cfg, "rope_theta", None)
                     or (rp or {}).get("rope_theta", 1000000.0))
            inv_freq = 1.0 / (theta ** (
                torch.arange(0, hd, 2, dtype=torch.float32) / hd))
        pos = torch.arange(L, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)            # [L, hd/2]
        emb = torch.cat((freqs, freqs), dim=-1)       # [L, hd]
        # [1,L,hd]
        self.register_buffer("cos3", emb.cos()[None], persistent=False)
        self.register_buffer("sin3", emb.sin()[None], persistent=False)

        # causal additive mask [L, L] : 0 on/below diag, big-neg above
        m = torch.full((L, L), -30000.0)
        m = torch.triu(m, diagonal=1)
        self.register_buffer("cmask2", m[None], persistent=False)  # [1,L,L]

    def attn(self, layer, h):
        a = layer.self_attn
        L, hd, nkv, nq, rep = self.L, self.hd, self.n_kv, self.n_q, self.rep
        # q [1,L,16*hd], k/v [1,L,8*hd]
        q = F.linear(h, a.q_proj.weight, getattr(a.q_proj, "bias", None))
        k = F.linear(h, a.k_proj.weight, getattr(a.k_proj, "bias", None))
        v = F.linear(h, a.v_proj.weight, getattr(a.v_proj, "bias", None))

        # q [16,L,hd], k/v [8,L,hd]
        q = q.reshape(L, nq, hd).permute(1, 0, 2)
        k = k.reshape(L, nkv, hd).permute(1, 0, 2)
        v = v.reshape(L, nkv, hd).permute(1, 0, 2)

        # per-head q_norm / k_norm (RMSNorm over hd)
        q = safe_rms(q, a.q_norm.weight)
        k = safe_rms(k, a.k_norm.weight)

        # RoPE (cos/sin [1,L,hd] broadcast over head batch)
        q = q * self.cos3 + rotate_half(q) * self.sin3
        k = k * self.cos3 + rotate_half(k) * self.sin3

        # GQA: repeat each kv head rep times -> [kv0,kv0,kv1,kv1,...]
        # via cat (no BROADCAST_TO); both [16,L,hd]
        k = torch.cat([k.unsqueeze(1)] * rep, dim=1).reshape(nq, L, hd)
        v = torch.cat([v.unsqueeze(1)] * rep, dim=1).reshape(nq, L, hd)

        # [16,L,L]
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # [1,L,L] add
        scores = scores + self.cmask2
        attn = torch.softmax(scores, dim=-1)
        # [16,L,hd]
        ctx = torch.matmul(attn, v)
        # [1,L,16*hd]
        ctx = ctx.permute(1, 0, 2).reshape(1, L, nq * hd)
        return F.linear(ctx, a.o_proj.weight, getattr(a.o_proj, "bias", None))

    def mlp(self, layer, h):
        m = layer.mlp
        g = F.linear(h, m.gate_proj.weight)
        u = F.linear(h, m.up_proj.weight)
        return F.linear(F.silu(g) * u, m.down_proj.weight)

    def forward(self, inputs_embeds):
        # [1,L,1024]
        h = inputs_embeds
        for layer in self.layers:
            h = h + self.attn(layer, safe_rms(h, layer.input_layernorm.weight))
            h = h + self.mlp(
                layer, safe_rms(h, layer.post_attention_layernorm.weight))
        # [1,L,1024]
        return safe_rms(h, self.final_norm_w)


def main():
    """Re-authors, parity-checks, and converts Qwen3-Embedding to fp16."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--out", default=os.path.join(HERE, "qwen3emb_gpu.tflite"))
    args = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer
    print(f"loading {MODEL_DIR} ...", flush=True)
    base = AutoModel.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)

    L = args.L
    net = Qwen3EmbGPU(base, L, args.layers).eval()

    # ---- CPU parity vs stock HF (validates exact re-authoring) ----
    sent = "The capital of France is Paris."
    enc = tok(sent, return_tensors="pt", padding="max_length",
              truncation=True, max_length=L)
    ids, amask = enc.input_ids, enc.attention_mask
    real_len = int(amask.sum())
    with torch.no_grad():
        # host does this
        embeds = base.embed_tokens(ids)
        # [1,L,1024]
        ours = net(embeds)
        ref = base(inputs_embeds=embeds, attention_mask=amask).last_hidden_state
    # pool last REAL token (right-padded => position real_len-1)
    ov = ours[0, real_len - 1]
    rv = ref[0, real_len - 1]
    ov = ov / ov.norm()
    rv = rv / rv.norm()
    corr = torch.dot(ov, rv).item()
    cos_full = F.cosine_similarity(
        ours.reshape(-1), ref.reshape(-1), dim=0).item()
    print(f"[parity] last-token emb cos={corr:.6f}  "
          f"full-tensor cos={cos_full:.6f}", flush=True)
    if corr < 0.999:
        print("!! parity below 0.999 -- re-authoring differs from HF; "
              "fix before convert", flush=True)

    # ---- convert to fp16 tflite ----
    base.to(torch.float32)
    dummy = torch.randn(1, L, base.config.hidden_size)
    print("importing litert_torch (after model load) ...", flush=True)
    import litert_torch
    print("converting ...", flush=True)
    conv = litert_torch.convert(net, (dummy,))
    conv.export(args.out)
    print(f"[export] {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB)",
          flush=True)


if __name__ == "__main__":
    main()
