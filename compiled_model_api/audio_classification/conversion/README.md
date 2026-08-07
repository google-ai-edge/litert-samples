# wav2vec2 KWS → LiteRT conversion

Scripts that produce the two `.tflite` graphs used by the Android sample, from [`superb/wav2vec2-base-superb-ks`](https://huggingface.co/superb/wav2vec2-base-superb-ks), with [litert-torch](https://github.com/google-ai-edge/litert).

## Environment

```bash
pip install litert-torch ai-edge-litert ai-edge-quantizer
pip install "transformers>=5.12" torch numpy
# superb/wav2vec2-base-superb-ks is auto-downloaded from the Hugging Face hub.
# On macOS, `import _stub` first (it guards a scipy/_propack dlopen + an inspect crash).
```

## Run

```bash
python build_w2v2.py all          # single-graph re-authoring + op-check + parity (shows it is op-clean)
python build_w2v2_split.py        # the 2-graph deployment split (frontend + head), fp16
```

`build_w2v2_split.py` emits `w2v2_frontend_fp16.tflite` + `w2v2_head_fp16.tflite`; push both with `../kotlin_cpu_gpu/android/install_to_device.sh`.

## Files

| File | What |
|---|---|
| `build_w2v2.py` | the re-authoring recipe + op-check + parity on the single graph (proves it is GPU-clean). |
| `build_w2v2_split.py` | the 2-graph deployment split (frontend + head) with the incremental baked-weight layer-sum; parity corr 1.0. |
| `_stub.py` | macOS import guards (scipy/_propack, `inspect.getsourcefile`). |

## Re-authoring → GPU-clean (parity corr 1.0)

Every rewrite is numerically equivalent (per-graph tflite-vs-torch corr **1.000000**):

| op | rewrite |
|---|---|
| `nn.GELU` / `GELUActivation` ×20 | tanh-GELU `0.5x(1+tanh(√(2/π)(x+0.044715x³)))` |
| feature-extractor `nn.GroupNorm` (num_groups=channels) | GN4D — reshape `(B,G,C//G,T)`, mean/var over `(2,3)` (kills GATHER_ND) |
| pos-conv (kernel-128 grouped Conv1d) `weight_norm` | fold to a static weight (`remove_parametrizations(..., leave_parametrized=True)`) |
| `create_bidirectional_mask()` | return `None` — it builds an all-valid mask even when `attention_mask=None` (arange/ge/expand → SELECT_V2 + BROADCAST_TO); fixed length, no padding → SDPA full attention |
| `use_weighted_layer_sum` head | accumulate incrementally `acc += w[i]·hᵢ` with **baked** `softmax(layer_weights)` constants |

## Two on-device findings (general)

1. **Whole-graph Mali shader-compile limit.** A graph can be fully op-clean AND have each half compile, yet **fail to compile when fused** (`Failed to compile model`; the delegate reports e.g. "Replacing 923 out of 1008 node(s) … 2 partitions"). The full wav2vec2 graph fails; splitting at the conv-frontend / transformer-encoder boundary makes both halves compile (frontend 134/134 + head 893/893 LITERT_CL). This is a size/complexity ceiling, not a bad op — when a clean graph won't compile, split it.
2. **`use_weighted_layer_sum` on the GPU.** This checkpoint's logits use a softmax-weighted sum of ALL 13 hidden states, not just the last (dropping it flips predictions, corr 0.54 — replicate it exactly). It must be (a) **accumulated incrementally** (`torch.stack` of all 13 keeps every layer output live and splits the partition); and (b) the `softmax(layer_weights)` must be **baked to constants** — the runtime softmax + 13 scalar `w[i]` gathers off a runtime tensor break delegation (3 partitions → compile fail). Baked + incremental → 893/893 LITERT_CL, 1 partition.

The transformer residual peaks at `|x|≈3.2`, so there is no fp16-precision issue — the whole model is fp16-exact on the GPU (no CPU fallback).
