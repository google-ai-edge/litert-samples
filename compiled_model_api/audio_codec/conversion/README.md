# Mimi → LiteRT conversion

Scripts that produce the four `.tflite` graphs + the RVQ weight blob used by the Android sample,
from the official [kyutai/mimi](https://huggingface.co/kyutai/mimi) checkpoint, with
[litert-torch](https://github.com/google-ai-edge/litert).

## Environment

```bash
pip install litert-torch ai-edge-litert ai-edge-quantizer
pip install "transformers>=5.12" torch numpy
# kyutai/mimi is auto-downloaded from the Hugging Face hub by transformers.
# On macOS, `import _stub` first (it guards a scipy/_propack dlopen + an inspect crash).
```

## Run

```bash
python build_hybrid_graphs.py        # the 4 deployment graphs (2 GPU conv + 2 CPU transformer), fp16
python mimi_rvq_validate_export.py   # split RVQ: validate vs torch + export mimi_rvq.bin
python build_mimi.py all             # (optional) full op-check + parity + the C33 standalone/tapped graphs
```

`build_hybrid_graphs.py` emits `mimi_{enc_conv,enc_tx,dec_tx,deconly}_fp16.tflite`;
`mimi_rvq_validate_export.py` emits `mimi_rvq.bin`. Push all five to the app with
`../kotlin_cpu_gpu/android/install_to_device.sh`.

## Files

| File | What |
|---|---|
| `build_hybrid_graphs.py` | builds + fp16-quantizes the 4 deployment graphs and validates the full encode→RVQ→decode round-trip (corr 1.0 vs torch). |
| `build_mimi.py` | the full re-authoring recipe + op-check + per-graph parity, **and the C33 test graphs** (`dectrans_standalone` + `decode_tapped`) used to prove on device that the transformer does not collapse-on-fusion. |
| `mimi_rvq_validate_export.py` | the split RVQ (1 semantic + 31 acoustic) reference; validates encode/decode vs torch and exports `mimi_rvq.bin`. |
| `_stub.py` | macOS import guards (scipy/_propack, `inspect.getsourcefile`). |

## Re-authoring → GPU-clean

Mimi rides the `CompiledModel` GPU delegate only after the standard re-authoring. Every rewrite is
**numerically equivalent** (per-graph tflite-vs-torch corr **1.000000**; full round-trip corr 1.0):

| op (banned / Mali-hostile) | rewrite |
|---|---|
| `nn.GELU` (erf → FlexErf) | tanh-GELU `0.5x(1+tanh(√(2/π)(x+0.044715x³)))` (MUL/ADD/TANH, no POW; tanh beats sigmoid here, transformer corr 0.991→0.99999) |
| `MimiRotaryEmbedding` | baked const `cos`/`sin` + `rotate_half` (kills the GATHER_ND position-gather) |
| causal/sliding attention mask | baked const **additive** bias `(1,1,S,S)` — decode IS causal, so bake it, don't drop it (kills CUMSUM/EQUAL/SELECT_V2) |
| `MimiLayerScale` | bake γ into the preceding Linear (`o_proj` / `fc2`) |
| `ConvTranspose1d` (the `upsample` is **depthwise**, groups=512) | grouped-aware `ZeroStuffConvT1d`: reshape weight `(Cin,Cout//G,K)→(Cout,Cin//G,K)`+flip, `F.conv1d(groups=g)` (kills TRANSPOSE_CONV) |
| `MimiConv1d` causal pad | bake a constant `F.pad` — the int64-buffer `.item()` is a dynamic value that fails tracing (jax `ConcretizationError`) |
| `nn.ELU` (→ SELECT) | `relu(x) − relu(1 − exp(min(x,0)))` — SELECT-free, exact, fp16-safe (the SEANet's 13 ELUs were a `SELECT×13` blocker; EXP is GPU-clean) |
| downsample `replicate` pad | edge-slice SLICE+CONCAT — tflite PAD is constant-only; replicate emits GATHER_ND |
| split RVQ (Euclidean argmin + int64 indices) | run on **CPU** (`MimiRvq.kt`) — int64 CAST + EMBEDDING_LOOKUP are Mali-rejected |

These are the same patterns proven on the sibling DAC codec (`ZeroStuffConvT1d`, RVQ→CPU) and the
ASR / Matcha samples (GroupNorm→4D, manual masked attention, host-side embeds).

## On-device finding: GPU convs + CPU transformers (and the C33 result)

This sample uses a **hybrid** placement (convs on GPU, transformers + RVQ on CPU). The reason is a
useful `CompiledModel` data point:

- The SEANet convs are **fp16-exact on Mali** — feeding the decoder-only graph the exact transformer
  output reconstructs audio at **48 dB SNR**.
- The decoder transformer's residual stream reaches **|x| = 27** (L7 damps it back to ~4.4 by
  near-cancellation). The Mali GPU delegate computes fp16 internally, and that cancellation loses
  precision there → full-GPU decode is only ~12 dB SNR on real speech (a synthetic tone hides it; fp32
  and fp16 tflite give the **same** device output because the delegate is fp16 internally either way).
- Crucially, the transformer computes **identically standalone and fused** on device (corr 0.70 either
  way, same activation range). So this is fp16 **precision** in a large-magnitude residual, **not** a
  graph-fusion collapse. `build_mimi.py`'s `c33` stage builds the two graphs (the transformer alone,
  and the full decode with a tap on the transformer output) that prove this on device.

Mimi's transformer is its own LLM-style implementation (RoPE + LayerScale), distinct from the
diffusers `BasicTransformerBlock`. So this result also shows that the transformer-residual-collapse
seen in some diffusion graphs does **not** generalize to it — the two are different failure modes.

The transformers are tiny (8 layers × 512, seq ~50), so running them on CPU is trivial and exact while
the heavy convs stay on the GPU. Pixel 8a: enc_conv `189/189` + deconly `220/220` on `LITERT_CL`,
RTF ≈ 0.35.
