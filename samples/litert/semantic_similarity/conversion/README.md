# Qwen3-Embedding-0.6B → LiteRT CompiledModel GPU (fully-GPU, fp16)

Reproduces the `.tflite` text-embedding graph used by the `text-embedding` sample: a fully GPU-compatible (ML Drift / `CompiledModel` `Accelerator.GPU`) re-authoring of [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) (Apache-2.0), the 2025 SOTA small text-embedding model.

Because sentence embedding uses **last-token pooling of a single forward pass** (no generation, no KV cache), the model is a plain single-graph `.tflite` — not a `.litertlm` — and runs on the same `CompiledModel` GPU path as every other model in this repo.

## Result (device-verified, Pixel 8a / Tensor G3)

| | |
|---|---|
| GPU compile | OK (~47 s, one-time) |
| Inference | **266 ms** per embedding (L=128) |
| Size | 881 MB fp16 (no OOM) |
| fp16 parity vs HF fp32 | last-token cosine **0.9997** (all tokens 0.999–1.000) |
| Retrieval | ranks correct passages first |

## Graph contract

```
input : inputs_embeds  [1, 128, 1024]   (host does the token-embedding lookup)
output: hidden_states  [1, 128, 1024]   (post final RMSNorm)
```

The host tokenizes (Qwen byte-level BPE), looks up token embeddings from the bundled embedding table, feeds `inputs_embeds`, then pools the **last real token** (right-padding + causal attention ⇒ real tokens never attend to pad, so no attention mask is needed), L2-normalizes, and optionally truncates (Matryoshka, 1024→N) — all on the host.

## GPU-clean re-authoring (`build_qwen3emb.py`)

- **Token embedding is GATHER (banned)** → done on the host; embeddings are fed in.
- **GQA (16 Q / 8 KV heads)** → each KV head is `cat`-repeated to 16 heads (`[k0,k0,k1,k1,…]`) then a plain batched matmul. A broadcast matmul instead emits `BROADCAST_TO` ×56 which the Mali delegate rejects. Heads ride the batch dim so every tensor stays ≤4D.
- **Per-head `q_norm` / `k_norm`** = RMSNorm(head_dim) → SafeRMS (below).
- **RoPE** (θ=1e6) precomputed as constant cos/sin for the fixed length (mul/add only). transformers 5.x no longer exposes `config.rope_theta`; pull `model.rotary_emb.inv_freq`.
- **Causal mask** = constant additive `[1,L,L]` (−30000 above the diagonal).
- **SwiGLU**: `down(silu(gate) * up)`, `silu = x*sigmoid(x)` (GPU-clean, no GELU).

### ⭐ max-normalized SafeRMS (the load-bearing device fix)

A 28-layer residual stream grows large, so `mean(x²)` **overflows fp16 (>65504) → rsqrt(inf)=0 → the whole output collapses to 0**. The opposite hazard (tiny/pad rows + eps<6.1e-5) underflows `mean(x²)+eps` to 0 → `rsqrt(0)=inf` → NaN. A single fixed down-scale cannot cover both small and large rows, so each row is normalized by its own max before squaring:

```
m  = max(|x|).clamp_min(1e-4)        # per-row scale, finite > 0
xs = x / m                            # x² now in [0,1] — sum-of-1024 never overflows
y  = xs * rsqrt(mean(xs²) + eps/m²) * w
```

This is mathematically identical to standard RMSNorm (CPU parity 1.000000) and is the general recipe for any deep transformer on Mali fp16.

## Reproduce

```bash
pip install torch transformers ai-edge-litert ai-edge-quantizer litert-torch
hf download Qwen/Qwen3-Embedding-0.6B --local-dir ./Qwen3-Embedding-0.6B

python build_qwen3emb.py --L 128     # CPU parity + qwen3emb_gpu.tflite (fp32)
python check_qwen3emb.py             # op-check (GPU-CLEAN) + cast to qwen3emb_gpu_fp16.tflite
python make_fixture.py               # probe_input.bin + ref_out.npy for a device probe
# device probe with litert-gpu-probe, then:
python compare_probe.py probe_out_0.bin
```

The `.tflite` files are **not committed** (see repo policy); the sample fetches them at build time.
