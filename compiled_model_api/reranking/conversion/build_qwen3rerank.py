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

"""Qwen3-Reranker-0.6B -> LiteRT GPU (fully CompiledModel, fp16).

Single graph, one forward (last-token scoring => no generation, no KV cache):
  input : inputs_embeds [1, L, 1024]   (host does token embedding lookup)
  output: logits        [1, L, 2]      ([no, yes] rows; host takes the last real
          token and softmaxes it into the relevance score P(yes))

GPU-clean re-authoring of the 28-layer Qwen3 decoder:
  - token embed lookup is GATHER -> done on HOST, embeds fed in
  - GQA (16 Q / 8 KV heads) via [8, rep, L, d] broadcast matmul (all <=4D, no gather)
  - Qwen3 per-head q_norm / k_norm = RMSNorm(head_dim)  -> SafeRMS
  - RoPE (theta 1e6) precomputed for fixed L (constants, mul/add only)
  - causal mask = constant additive [L, L]
  - RMSNorm -> SafeRMS (scale-invariant down-scaled reduction, fp16-safe)
  - SwiGLU: down(silu(gate) * up), silu = x*sigmoid(x)  (GPU-clean, no GELU)

Run: python build_qwen3rerank.py [--L 256] [--layers 28]
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "Qwen3-Reranker-0.6B")
NO_ID, YES_ID = 2152, 9693   # Qwen tokenizer ids for "no" / "yes"

EPS = 1e-6


def safe_rms(x, weight, eps=EPS):
    """fp16-safe RMSNorm(x)*weight via per-row max-normalization (Mali computes fp16
    regardless of dtype). The 28-layer residual stream grows large, so mean(x^2) overflows
    fp16 (-> inf -> rsqrt=0 -> whole output collapses to 0); dividing each row by its own
    max first bounds x^2 in [0,1] so the sum-of-squares never overflows. Mathematically
    identical to standard RMSNorm: with m=max|x|, y = (x/m)*rsqrt(mean((x/m)^2)+eps/m^2)*w."""
    m = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)   # per-row scale, finite > 0
    xs = x / m
    var = (xs * xs).mean(dim=-1, keepdim=True)               # in [1/N, 1], fp16-safe
    return xs * torch.rsqrt(var + eps / (m * m)) * weight


def rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


class Qwen3EmbGPU(nn.Module):
    """Fully GPU-clean forward over inputs_embeds -> post-norm hidden states."""

    def __init__(self, base, L, n_layers):
        super().__init__()
        self.model = base
        self.layers = base.layers[:n_layers]
        self.final_norm_w = base.norm.weight
        # baked 2-logit head: rows of the tied embedding for "no"/"yes" -> [2, hidden]
        self.register_buffer(
            "head_w", base.embed_tokens.weight[[NO_ID, YES_ID]].detach().clone(),
            persistent=False)
        self.L = L
        cfg = base.config
        self.n_q = cfg.num_attention_heads          # 16
        self.n_kv = cfg.num_key_value_heads          # 8
        self.rep = self.n_q // self.n_kv             # 2
        self.hd = cfg.head_dim                        # 128
        self.scale = self.hd ** -0.5

        # RoPE cos/sin constants for fixed L (pull inv_freq from the model if present)
        hd = self.hd
        rope = getattr(base, "rotary_emb", None)
        if rope is not None and hasattr(rope, "inv_freq"):
            inv_freq = rope.inv_freq.detach().float()
        else:
            rp = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
            theta = getattr(cfg, "rope_theta", None) or (rp or {}).get("rope_theta", 1000000.0)
            inv_freq = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
        pos = torch.arange(L, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)            # [L, hd/2]
        emb = torch.cat((freqs, freqs), dim=-1)       # [L, hd]
        self.register_buffer("cos3", emb.cos()[None], persistent=False)  # [1,L,hd]
        self.register_buffer("sin3", emb.sin()[None], persistent=False)

        # causal additive mask [L, L] : 0 on/below diag, big-neg above
        m = torch.full((L, L), -30000.0)
        m = torch.triu(m, diagonal=1)
        self.register_buffer("cmask2", m[None], persistent=False)  # [1,L,L]

    def attn(self, layer, h):
        a = layer.self_attn
        L, hd, nkv, nq, rep = self.L, self.hd, self.n_kv, self.n_q, self.rep
        q = F.linear(h, a.q_proj.weight, getattr(a.q_proj, "bias", None))   # [1,L,16*hd]
        k = F.linear(h, a.k_proj.weight, getattr(a.k_proj, "bias", None))   # [1,L,8*hd]
        v = F.linear(h, a.v_proj.weight, getattr(a.v_proj, "bias", None))   # [1,L,8*hd]

        q = q.reshape(L, nq, hd).permute(1, 0, 2)                            # [16,L,hd]
        k = k.reshape(L, nkv, hd).permute(1, 0, 2)                          # [8,L,hd]
        v = v.reshape(L, nkv, hd).permute(1, 0, 2)                          # [8,L,hd]

        # per-head q_norm / k_norm (RMSNorm over hd)
        q = safe_rms(q, a.q_norm.weight)
        k = safe_rms(k, a.k_norm.weight)

        # RoPE (cos/sin [1,L,hd] broadcast over head batch)
        q = q * self.cos3 + rotate_half(q) * self.sin3
        k = k * self.cos3 + rotate_half(k) * self.sin3

        # GQA: repeat each kv head rep times -> [kv0,kv0,kv1,kv1,...] via cat (no BROADCAST_TO)
        k = torch.cat([k.unsqueeze(1)] * rep, dim=1).reshape(nq, L, hd)     # [16,L,hd]
        v = torch.cat([v.unsqueeze(1)] * rep, dim=1).reshape(nq, L, hd)     # [16,L,hd]

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale          # [16,L,L]
        scores = scores + self.cmask2                                       # [1,L,L] add
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v)                                         # [16,L,hd]
        ctx = ctx.permute(1, 0, 2).reshape(1, L, nq * hd)                   # [1,L,16*hd]
        return F.linear(ctx, a.o_proj.weight, getattr(a.o_proj, "bias", None))

    def mlp(self, layer, h):
        m = layer.mlp
        g = F.linear(h, m.gate_proj.weight)
        u = F.linear(h, m.up_proj.weight)
        return F.linear(F.silu(g) * u, m.down_proj.weight)

    def forward(self, inputs_embeds):
        h = inputs_embeds                                                   # [1,L,1024]
        for layer in self.layers:
            h = h + self.attn(layer, safe_rms(h, layer.input_layernorm.weight))
            h = h + self.mlp(layer, safe_rms(h, layer.post_attention_layernorm.weight))
        h = safe_rms(h, self.final_norm_w)                                  # [1,L,1024]
        return F.linear(h, self.head_w)                                    # [1,L,2] = [no,yes] logits


PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
          "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
          "<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
DEFAULT_INSTR = "Given a web search query, retrieve relevant passages that answer the query"


def build_ids(tok, query, doc, L, instruction=DEFAULT_INSTR):
    content = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
    ids = (tok(PREFIX, add_special_tokens=False).input_ids
           + tok(content, add_special_tokens=False).input_ids
           + tok(SUFFIX, add_special_tokens=False).input_ids)
    ids = ids[:L]
    real_len = len(ids)
    ids = ids + [151643] * (L - real_len)          # right-pad with <|endoftext|>
    return torch.tensor([ids]), real_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--out", default=os.path.join(HERE, "qwen3rerank_gpu.tflite"))
    args = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer
    print(f"loading {MODEL_DIR} ...", flush=True)
    base = AutoModel.from_pretrained(MODEL_DIR, torch_dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    L = args.L
    net = Qwen3EmbGPU(base, L, args.layers).eval()

    # ---- CPU parity vs stock HF: rel score P(yes), on relevant + irrelevant pairs ----
    query = "What is the capital of China?"
    pairs = [("The capital of China is Beijing.", "relevant"),
             ("Bananas are a good source of potassium.", "irrelevant")]
    print("[parity] P(yes) ours vs HF (right-pad, pool real_len-1):", flush=True)
    worst = 1.0
    for doc, tag in pairs:
        ids, rl = build_ids(tok, query, doc, L)
        with torch.no_grad():
            embeds = base.embed_tokens(ids)
            ours2 = net(embeds)[0, rl - 1]                                 # [2] = [no,yes]
            refh = base(inputs_embeds=embeds).last_hidden_state[0, rl - 1]  # [1024]
            reflg = refh @ base.embed_tokens.weight[[NO_ID, YES_ID]].t()    # [2] = [no,yes]
        s_ours = torch.log_softmax(ours2, -1)[1].exp().item()
        s_ref = torch.log_softmax(reflg, -1)[1].exp().item()
        print(f"  [{tag:10s}] ours={s_ours:.4f}  HF={s_ref:.4f}  |Δ|={abs(s_ours-s_ref):.5f}", flush=True)
        worst = min(worst, 1 - abs(s_ours - s_ref))
    print(f"[parity] worst score match = {worst:.5f}", flush=True)

    # ---- convert to fp16 tflite ----
    base.to(torch.float32)
    dummy = torch.randn(1, L, base.config.hidden_size)
    print("importing litert_torch (after model load) ...", flush=True)
    import litert_torch
    print("converting ...", flush=True)
    conv = litert_torch.convert(net, (dummy,))
    conv.export(args.out)
    print(f"[export] {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
