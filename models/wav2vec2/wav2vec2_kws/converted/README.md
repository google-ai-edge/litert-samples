# wav2vec2 Keyword Spotting Conversion

Model recipes, instructions, scripts, and utilities for converting wav2vec2 keyword spotting to LiteRT.

These scripts convert [superb/wav2vec2-base-superb-ks](https://huggingface.co/superb/wav2vec2-base-superb-ks) (Apache-2.0) into the two `.tflite` graphs used for on-device keyword spotting, and numerically verify every step against the PyTorch reference. The published artifacts live at [litert-community/wav2vec2-keyword-spotting](https://huggingface.co/litert-community/wav2vec2-keyword-spotting).

The model classifies 1 s of 16 kHz audio into 12 Speech-Commands labels (yes / no / up / down / left / right / on / off / stop / go / _unknown_ / _silence_). **No FFT anywhere** — the raw waveform goes straight into the 1D-conv feature extractor (no mel step, in or out of the graph), so the whole model rides the GPU delegate.

## Environment

```bash
pip install litert-torch ai-edge-litert ai-edge-quantizer
pip install transformers torch numpy
# superb/wav2vec2-base-superb-ks is auto-downloaded from the Hugging Face hub.
```

Verified with `litert-torch==0.10.0`, `torch==2.12`, `transformers==5.6.2` (originally converted with `transformers==5.12`). `_stub.py` is imported first by both scripts: on macOS machines where scipy's optional native extensions fail to load it installs import guards; where scipy is healthy it does nothing.

## Run

```bash
python build_w2v2.py all          # single-graph re-authoring + op-check + parity (shows it is op-clean)
python build_w2v2_split.py        # the 2-graph deployment split (frontend + head), fp16
```

`build_w2v2_split.py` emits `w2v2_frontend_fp16.tflite` + `w2v2_head_fp16.tflite` — the deployment pair published on Hugging Face.

## Files

| File | What |
|---|---|
| `build_w2v2.py` | the re-authoring recipe + op-check + parity on the single graph (proves it is GPU-clean). |
| `build_w2v2_split.py` | the 2-graph deployment split (frontend + head) with the incremental baked-weight layer-sum; parity corr 1.0. |
| `_stub.py` | macOS import guards (scipy/_propack, `inspect.getsourcefile`), active only when the real scipy fails to import. |

## Re-authoring → GPU-clean

The converted graphs faithfully reproduce the re-authored model (per-graph tflite-vs-torch corr **1.000000**). Every rewrite is exact except tanh-GELU, the standard approximation (re-authored vs original torch: logits corr 0.9998, max abs diff 0.26, argmax unchanged):

| op | rewrite |
|---|---|
| `nn.GELU` / `GELUActivation` ×20 | tanh-GELU `0.5x(1+tanh(√(2/π)(x+0.044715x³)))` |
| feature-extractor `nn.GroupNorm` (num_groups=channels) | GN4D — reshape `(B,G,C//G,T)`, mean/var over `(2,3)` (kills GATHER_ND) |
| pos-conv (kernel-128 grouped Conv1d) `weight_norm` | fold to a static weight (`remove_parametrizations(..., leave_parametrized=True)`) |
| `create_bidirectional_mask()` | return `None` — it builds an all-valid mask even when `attention_mask=None` (arange/ge/expand → SELECT_V2 + BROADCAST_TO); fixed length, no padding → SDPA full attention |
| `use_weighted_layer_sum` head | accumulate incrementally `acc += w[i]·hᵢ` with **baked** `softmax(layer_weights)` constants |

## Verification results

Re-verified with `litert-torch==0.10.0`:

- Re-authoring: re-authored vs original torch logits corr 0.999813 (max abs diff 0.26 — the tanh-GELU approximation across 20 activations; argmax unchanged).
- Single graph: tflite-vs-torch logits corr 1.000000; op-check GPU-clean (no banned ops, no >4D tensors, no FFT family); fp16 variant corr 1.000000, GPU-clean.
- Deployment split: torch full-vs-split corr 1.000000 (max abs diff 0.0); frontend and head tflite-vs-torch corr 1.000000 each, both GPU-clean; fp16 pair emitted.
- On device (Pixel 8a, Tensor G3): frontend `134/134` + head `893/893` nodes on `LITERT_CL` (all-GPU, no CPU fallback), end-to-end ~19 ms for a 1 s clip (**RTF ≈ 0.02**); real-speech validation 10/10 keywords; device-vs-CPU logits corr 0.9995. The transformer residual peaks at `|x|≈3.2`, so the model is fp16-exact on the GPU.

## Two on-device findings (general)

1. **Whole-graph Mali shader-compile limit.** A graph can be fully op-clean AND have each half compile, yet **fail to compile when fused** (`Failed to compile model`; the delegate reports e.g. "Replacing 923 out of 1008 node(s) … 2 partitions"). The full wav2vec2 graph fails; splitting at the conv-frontend / transformer-encoder boundary makes both halves compile (frontend 134/134 + head 893/893 LITERT_CL). This is a size/complexity ceiling, not a bad op — when a clean graph won't compile, split it.
2. **`use_weighted_layer_sum` on the GPU.** This checkpoint's logits use a softmax-weighted sum of ALL 13 hidden states, not just the last (dropping it flips predictions, corr 0.54 — replicate it exactly). It must be (a) **accumulated incrementally** (`acc += w[i]·hᵢ`) — the `torch.stack` of all 13 keeps every layer output live and splits the partition; and (b) the `softmax(layer_weights)` must be **baked to constants** — the runtime softmax + 13 scalar `w[i]` gathers off a runtime tensor break delegation (3 partitions → compile fail). Baked + incremental → 893/893 LITERT_CL, 1 partition.
