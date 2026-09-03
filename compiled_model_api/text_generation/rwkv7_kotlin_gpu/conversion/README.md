# RWKV-7 World 0.1B → LiteRT conversion

Converts the official Apache-2.0 [RWKV-x070-World-0.1B-v2.8](https://huggingface.co/BlinkDL/rwkv-7-world) checkpoint into a single per-token step graph that runs **entirely on the LiteRT `CompiledModel` GPU delegate**, then validates it token-by-token against the fp32 PyTorch reference.

## Why a step graph

RWKV-7 is an RNN: the transformer-equivalent context is carried in a fixed-size recurrent state (per-layer token-shift vectors + per-head `wkv` matrices), not a growing KV cache. That makes autoregressive generation a **static-shape** problem — one graph, executed once per token — which is exactly what `CompiledModel` GPU wants. All state is passed in/out host-side.

```
in : x_emb[1,768]  att_shift[12,768]  ffn_shift[12,768]  wkv[144,64,64]
out: logits[1,65536]  att_shift'  ffn_shift'  wkv'
```

## GPU re-authorings (all exact — no approximation)

| Original | GPU-safe rewrite |
| :-- | :-- |
| wkv7 recurrence (custom op / 5-D scan) | at T=1: plain 4-D matmul + elementwise |
| `F.softplus` | `relu(z) + log1p(exp(-|z|))` — the stock lowering emits `GREATER`+`SELECT`, which the GPU delegate rejects |
| `GroupNorm(heads)` | manual per-head mean/var (4-D safe) |
| `F.normalize` | `x * rsqrt(sum(x²) + eps)` |
| `emb[token]` lookup | host-side (mmap fp16 table); `GATHER` is not GPU-compatible |

Also: `torch.export` example inputs must be `.clone()`d — passing tensor views trips a converter assert ("sources must not be empty").

## Files

- `build_rwkv7_step.py` — parity (step-mode == GPT-mode), litert-torch conversion, fp16 cast (`ai_edge_quantizer` FLOAT_CASTING), embedding-table export.
- `validate_rwkv7.py` — runs the fp16 tflite through the **CompiledModel API** against fp32 PyTorch: prefill logits corr, then 30 greedy tokens where every tflite pick must be the fp32 argmax or a near-tie (top1–top2 gap < 0.05). fp16 legitimately flips near-tie argmaxes; a flip with a large gap is a real regression.

## Run

```bash
# deps: torch, numpy, litert-torch, ai-edge-quantizer, ai-edge-litert
# inputs: rwkv7_0.1b.pth + rwkv_vocab_v20230424.txt next to the scripts
python build_rwkv7_step.py all     # parity -> convert -> fp16 -> assets
python validate_rwkv7.py           # CompiledModel vs fp32 torch, 30-token greedy
```

Expected: parity corr 1.0000000; validate prefill corr 1.0000000 and 30/30 greedy tokens matching (desktop). On a Pixel 8a the same 30-token run keeps 28/30 identical with 2 near-tie flips (gap ≤ 0.04) — inside the fp16 noise envelope.
