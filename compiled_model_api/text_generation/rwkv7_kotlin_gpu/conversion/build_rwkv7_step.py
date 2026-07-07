#!/usr/bin/env python3
"""RWKV-7 "World" 0.1B -> LiteRT CompiledModel GPU per-token step graph.

RWKV-7 is an RNN-style language model: generation feeds one token per step and
carries a small fixed-size recurrent state, so the full forward pass fits a
single static GPU graph — no KV cache growth, no dynamic shapes. All recurrent
state is passed host-side (Mali silently corrupts stateful GPU graphs):

  inputs : x_emb[1,768], att_shift[12,768], ffn_shift[12,768], wkv[144,64,64]
  outputs: logits[1,65536], att_shift', ffn_shift', wkv'

GPU re-authorings (all exact, no approximation):
  * wkv7 recurrence at T=1 becomes plain 4D matmul/elementwise ops.
  * GroupNorm(heads) -> manual per-head mean/var (4D-safe).
  * F.normalize -> x * rsqrt(sum(x^2) + eps).
  * softplus -> relu(z) + log1p(exp(-|z|)) (branch-free; the stock lowering
    emits GREATER+SELECT which the GPU delegate rejects).
  * Token embedding lookup stays host-side (GATHER is not GPU-compatible);
    the graph takes the raw embedding row and applies ln0 itself.
  * torch.export requires .clone()d example inputs (views trip a converter
    assert: "sources must not be empty").

Checkpoint: RWKV-x070-World-0.1B-v2.8 (Apache-2.0), BlinkDL/rwkv-7-world on
Hugging Face. Place rwkv7_0.1b.pth and rwkv_vocab_v20230424.txt next to this
script.

Run:  python build_rwkv7_step.py [stage]        (then validate_rwkv7.py)
      stage in {parity, convert, fp16, assets, all} (default: all)
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(HERE, "rwkv7_0.1b.pth")
VOCAB_PATH = os.path.join(HERE, "rwkv_vocab_v20230424.txt")
FP32_PATH = os.path.join(HERE, "rwkv7_step.tflite")
FP16_PATH = os.path.join(HERE, "rwkv7_step_fp16.tflite")
EMB_PATH = os.path.join(HERE, "rwkv7_emb_fp16.bin")

N_LAYER = 12
N_EMBD = 768
HEAD_DIM = 64
N_HEAD = N_EMBD // HEAD_DIM
VOCAB_SIZE = 65536
GROUP_NORM_EPS = 64e-5
PARITY_PROMPT = "The Eiffel tower is in the city of"


def _lerp(x, prev, mix):
    """Token-shift mix: x + (prev - x) * mix."""
    return x + (prev - x) * mix


def _softplus_stable(z):
    """Branch-free softplus: relu(z) + log1p(exp(-|z|)).

    Exact identity for log(1 + e^z). F.softplus lowers with a threshold
    branch (GREATER + SELECT), which the GPU delegate rejects.
    """
    return torch.relu(z) + torch.log1p(torch.exp(-torch.abs(z)))


class RwkvTokenizer:
    """Greedy longest-match trie tokenizer (official RWKV World vocab)."""

    def __init__(self, vocab_file):
        self.idx2token = {}
        tokens = []
        with open(vocab_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                idx = int(line[: line.index(" ")])
                literal = eval(line[line.index(" ") : line.rindex(" ")])
                token = (
                    literal.encode("utf-8")
                    if isinstance(literal, str)
                    else literal
                )
                tokens.append(token)
                self.idx2token[idx] = token
        self.token2idx = {v: int(k) for k, v in self.idx2token.items()}
        self.table = [[[] for _ in range(256)] for _ in range(256)]
        self.good = [set() for _ in range(256)]
        self.wlen = [0 for _ in range(256)]
        for i in reversed(range(len(tokens))):
            token = tokens[i]
            if len(token) >= 2:
                s0, s1 = int(token[0]), int(token[1])
                self.table[s0][s1] += [token]
                self.wlen[s0] = max(self.wlen[s0], len(token))
                self.good[s0].add(s1)

    def encode(self, text):
        src = text.encode("utf-8")
        ids, i = [], 0
        while i < len(src):
            token = src[i : i + 1]
            if i < len(src) - 1:
                s0, s1 = int(src[i]), int(src[i + 1])
                if s1 in self.good[s0]:
                    window = src[i : i + self.wlen[s0]]
                    try:
                        token = next(
                            filter(window.startswith, self.table[s0][s1])
                        )
                    except StopIteration:
                        pass
            ids.append(self.token2idx[token])
            i += len(token)
        return ids

    def decode(self, ids):
        data = b"".join(self.idx2token[i] for i in ids)
        return data.decode("utf-8", errors="replace")


class Rwkv7Step(nn.Module):
    """One autoregressive RWKV-7 step, GPU re-authored (see module docstring)."""

    def __init__(self, state_dict):
        super().__init__()
        params = {}
        for key, value in state_dict.items():
            tensor = value.float().squeeze() if value.dim() == 3 else value.float()
            params[key] = nn.Parameter(tensor, requires_grad=False)
        self.params = nn.ParameterDict(
            {k.replace(".", "__"): v for k, v in params.items()}
        )

    def _p(self, name):
        return self.params[name.replace(".", "__")]

    def forward(self, x, att_shift, ffn_shift, wkv):
        """One step. x [1,C]; att/ffn_shift [L,C]; wkv [L*H,N,N]."""
        p = self._p
        new_att, new_ffn, new_wkv = [], [], []
        x = F.layer_norm(
            x, (N_EMBD,), p("blocks.0.ln0.weight"), p("blocks.0.ln0.bias")
        )
        v_first = x  # placeholder; assigned the layer-0 value below
        for layer in range(N_LAYER):
            blk = f"blocks.{layer}."
            xn = F.layer_norm(
                x, (N_EMBD,), p(blk + "ln1.weight"), p(blk + "ln1.bias")
            )
            prev = att_shift[layer : layer + 1]
            new_att.append(xn)
            att = blk + "att."
            xr = _lerp(xn, prev, p(att + "x_r"))
            xw = _lerp(xn, prev, p(att + "x_w"))
            xk = _lerp(xn, prev, p(att + "x_k"))
            xv = _lerp(xn, prev, p(att + "x_v"))
            xa = _lerp(xn, prev, p(att + "x_a"))
            xg = _lerp(xn, prev, p(att + "x_g"))

            r = xr @ p(att + "receptance.weight").t()
            w = (
                -_softplus_stable(
                    -(
                        p(att + "w0")
                        + torch.tanh(xw @ p(att + "w1")) @ p(att + "w2")
                    )
                )
                - 0.5
            )
            k = xk @ p(att + "key.weight").t()
            v = xv @ p(att + "value.weight").t()
            if layer == 0:
                v_first = v
            else:
                v = v + (v_first - v) * torch.sigmoid(
                    p(att + "v0") + (xv @ p(att + "v1")) @ p(att + "v2")
                )
            a = torch.sigmoid(
                p(att + "a0") + (xa @ p(att + "a1")) @ p(att + "a2")
            )
            gate = torch.sigmoid(xg @ p(att + "g1")) @ p(att + "g2")

            kk = (k * p(att + "k_k")).view(1, N_HEAD, HEAD_DIM)
            # F.normalize re-authored: x * rsqrt(sum(x^2) + eps).
            kk = kk * torch.rsqrt((kk * kk).sum(-1, keepdim=True) + 1e-12)
            k = k * (1 + (a - 1) * p(att + "k_a"))

            # wkv7 recurrence at T=1: state rows index the v-dim, cols the k-dim.
            state = wkv[layer * N_HEAD : (layer + 1) * N_HEAD]
            decay = torch.exp(-torch.exp(w)).view(N_HEAD, 1, HEAD_DIM)
            in_proj = (-kk).view(N_HEAD, HEAD_DIM, 1)
            out_proj = (kk * a.view(1, N_HEAD, HEAD_DIM)).view(
                N_HEAD, 1, HEAD_DIM
            )
            v_col = v.view(N_HEAD, HEAD_DIM, 1)
            k_row = k.view(N_HEAD, 1, HEAD_DIM)
            state = state * decay + state @ in_proj @ out_proj + v_col @ k_row
            new_wkv.append(state)
            out = (state @ r.view(N_HEAD, HEAD_DIM, 1)).view(1, N_EMBD)

            # GroupNorm over heads re-authored as manual per-head mean/var.
            heads = out.view(1, N_HEAD, HEAD_DIM)
            mean = heads.mean(-1, keepdim=True)
            centered = heads - mean
            var = (centered * centered).mean(-1, keepdim=True)
            heads = centered * torch.rsqrt(var + GROUP_NORM_EPS)
            out = heads.view(1, N_EMBD) * p(att + "ln_x.weight") + p(
                att + "ln_x.bias"
            )

            bonus = (
                (
                    r.view(1, N_HEAD, HEAD_DIM)
                    * k.view(1, N_HEAD, HEAD_DIM)
                    * p(att + "r_k")
                ).sum(-1, keepdim=True)
                * v.view(1, N_HEAD, HEAD_DIM)
            ).view(1, N_EMBD)
            out = out + bonus
            x = x + (out * gate) @ p(att + "output.weight").t()

            xn2 = F.layer_norm(
                x, (N_EMBD,), p(blk + "ln2.weight"), p(blk + "ln2.bias")
            )
            prev2 = ffn_shift[layer : layer + 1]
            new_ffn.append(xn2)
            ffn = blk + "ffn."
            hidden = _lerp(xn2, prev2, p(ffn + "x_k"))
            hidden = torch.relu(hidden @ p(ffn + "key.weight").t()) ** 2
            x = x + hidden @ p(ffn + "value.weight").t()

        x = F.layer_norm(x, (N_EMBD,), p("ln_out.weight"), p("ln_out.bias"))
        logits = x @ p("head.weight").t()
        return (
            logits,
            torch.cat(new_att, 0),
            torch.cat(new_ffn, 0),
            torch.cat(new_wkv, 0),
        )


def _wkv7_reference(r, w, k, v, a, b):
    """Sequential wkv7 scan from the official RWKV-7 demo (fp32 reference)."""
    batch, seq, _ = r.size()
    heads, dim = N_HEAD, HEAD_DIM
    r = r.view(batch, seq, heads, dim)
    k = k.view(batch, seq, heads, dim)
    v = v.view(batch, seq, heads, dim)
    a = a.view(batch, seq, heads, dim)
    b = b.view(batch, seq, heads, dim)
    w = torch.exp(-torch.exp(w.view(batch, seq, heads, dim)))
    out = torch.zeros((batch, seq, heads, dim))
    state = torch.zeros((batch, heads, dim, dim))
    for t in range(seq):
        kk = k[:, t, :].view(batch, heads, 1, dim)
        rr = r[:, t, :].view(batch, heads, dim, 1)
        vv = v[:, t, :].view(batch, heads, dim, 1)
        aa = a[:, t, :].view(batch, heads, dim, 1)
        bb = b[:, t, :].view(batch, heads, 1, dim)
        state = state * w[:, t, :, None, :] + state @ aa @ bb + vv @ kk
        out[:, t, :] = (state @ rr).view(batch, heads, dim)
    return out.view(batch, seq, heads * dim)


def gpt_mode_forward(sd, token_ids):
    """Parallel (GPT-mode) forward from the official demo; parity target.

    Args:
        sd: checkpoint state dict (fp32).
        token_ids: 1-D LongTensor of prompt ids.

    Returns:
        Logits [1, T, VOCAB_SIZE].
    """
    x = sd["emb.weight"][token_ids].unsqueeze(0).float()
    seq = x.shape[1]

    def ln(t, w_name, b_name):
        return F.layer_norm(
            t, (N_EMBD,), sd[w_name].float(), sd[b_name].float()
        )

    x = ln(x, "blocks.0.ln0.weight", "blocks.0.ln0.bias")
    v_first = None
    for layer in range(N_LAYER):
        blk = f"blocks.{layer}."
        att = blk + "att."

        def g(name):
            return sd[att + name].float()

        xn = ln(x, blk + "ln1.weight", blk + "ln1.bias")
        shifted = torch.cat([torch.zeros(1, 1, N_EMBD), xn[:, :-1]], 1) - xn
        xr, xw, xk, xv, xa, xg = [
            xn + shifted * g("x_" + s) for s in "rwkvag"
        ]
        r = xr @ g("receptance.weight").t()
        w = -F.softplus(-(g("w0") + torch.tanh(xw @ g("w1")) @ g("w2"))) - 0.5
        k = xk @ g("key.weight").t()
        v = xv @ g("value.weight").t()
        if layer == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(
                g("v0") + (xv @ g("v1")) @ g("v2")
            )
        a = torch.sigmoid(g("a0") + (xa @ g("a1")) @ g("a2"))
        gate = torch.sigmoid(xg @ g("g1")) @ g("g2")
        kk = k * g("k_k")
        kk = F.normalize(kk.view(1, seq, N_HEAD, -1), dim=-1, p=2.0).view(
            1, seq, N_EMBD
        )
        k = k * (1 + (a - 1) * g("k_a"))
        o = _wkv7_reference(r, w, k, v, -kk, kk * a)
        o = F.group_norm(
            o.view(seq, N_EMBD),
            N_HEAD,
            g("ln_x.weight"),
            g("ln_x.bias"),
            eps=GROUP_NORM_EPS,
        ).view(1, seq, N_EMBD)
        o = o + (
            (r.view(1, seq, N_HEAD, -1) * k.view(1, seq, N_HEAD, -1) * g("r_k"))
            .sum(-1, keepdim=True)
            * v.view(1, seq, N_HEAD, -1)
        ).view(1, seq, N_EMBD)
        x = x + (o * gate) @ g("output.weight").t()
        xn2 = ln(x, blk + "ln2.weight", blk + "ln2.bias")
        shifted2 = torch.cat([torch.zeros(1, 1, N_EMBD), xn2[:, :-1]], 1) - xn2
        hidden = xn2 + shifted2 * sd[blk + "ffn.x_k"].float()
        hidden = torch.relu(hidden @ sd[blk + "ffn.key.weight"].float().t()) ** 2
        x = x + hidden @ sd[blk + "ffn.value.weight"].float().t()
    x = ln(x, "ln_out.weight", "ln_out.bias")
    return x @ sd["head.weight"].float().t()


def _zero_state():
    return (
        torch.zeros(N_LAYER, N_EMBD),
        torch.zeros(N_LAYER, N_EMBD),
        torch.zeros(N_LAYER * N_HEAD, HEAD_DIM, HEAD_DIM),
    )


def stage_parity(sd, tok):
    """Sequential step-mode must reproduce GPT-mode logits on the prompt."""
    ids = tok.encode(PARITY_PROMPT)
    with torch.no_grad():
        ref = gpt_mode_forward(sd, torch.tensor(ids))
        step = Rwkv7Step(sd).eval()
        att, ffn, wkv = _zero_state()
        emb = sd["emb.weight"]
        logits = None
        for t in ids:
            logits, att, ffn, wkv = step(emb[t : t + 1], att, ffn, wkv)
    corr = np.corrcoef(logits[0].numpy(), ref[0, -1].numpy())[0, 1]
    max_diff = (logits[0] - ref[0, -1]).abs().max().item()
    top = tok.decode([int(logits[0].argmax())])
    print(f"parity: corr {corr:.7f}  max|d| {max_diff:.5f}  top {top!r}")
    assert corr > 0.9999, "step-mode does not match GPT-mode"


def stage_convert(sd, tok):
    """litert-torch conversion of the step graph (fp32 flatbuffer)."""
    ids = tok.encode(PARITY_PROMPT)
    emb = sd["emb.weight"]
    att, ffn, wkv = _zero_state()
    # Example inputs must be .clone()d — views break torch.export.
    example = (emb[ids[-1] : ids[-1] + 1].clone().contiguous(), att, ffn, wkv)
    import litert_torch

    litert_torch.convert(Rwkv7Step(sd).eval(), example).export(FP32_PATH)
    print(f"convert: {os.path.getsize(FP32_PATH) / 1e6:.1f} MB -> {FP32_PATH}")


def stage_fp16():
    """fp32 -> fp16 flatbuffer via ai_edge_quantizer FLOAT_CASTING."""
    from ai_edge_quantizer import quantizer, recipe_manager
    from ai_edge_quantizer.recipe import AlgorithmName, qtyping

    rm = recipe_manager.RecipeManager()
    rm.add_quantization_config(
        regex=".*",
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=qtyping.TensorQuantizationConfig(
                num_bits=16, dtype=qtyping.TensorDataType.FLOAT
            ),
            compute_precision=qtyping.ComputePrecision.FLOAT,
        ),
        algorithm_key=AlgorithmName.FLOAT_CASTING,
    )
    if os.path.exists(FP16_PATH):
        os.remove(FP16_PATH)
    qt = quantizer.Quantizer(float_model=FP32_PATH)
    qt.load_quantization_recipe(rm.get_quantization_recipe())
    qt.quantize().export_model(FP16_PATH)
    print(f"fp16: {os.path.getsize(FP16_PATH) / 1e6:.1f} MB -> {FP16_PATH}")


def stage_assets(sd):
    """Exports the fp16 embedding table for host-side row lookup."""
    emb = sd["emb.weight"].float().numpy().astype("<f2")
    emb.tofile(EMB_PATH)
    print(
        f"assets: emb {emb.shape} -> {EMB_PATH}"
        f" ({os.path.getsize(EMB_PATH) / 1e6:.1f} MB)"
    )


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    sd = torch.load(CKPT_PATH, map_location="cpu")
    sd = {k: v.float() for k, v in sd.items()}
    tok = RwkvTokenizer(VOCAB_PATH)
    if stage in ("parity", "all"):
        stage_parity(sd, tok)
    if stage in ("convert", "all"):
        stage_convert(sd, tok)
    if stage in ("fp16", "all"):
        stage_fp16()
    if stage in ("assets", "all"):
        stage_assets(sd)


if __name__ == "__main__":
    main()
