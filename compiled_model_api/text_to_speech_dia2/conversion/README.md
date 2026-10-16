# Dia2-1B → LiteRT conversion

These scripts produce the four `.tflite` graphs, the six baked fp16 tables and the voice prompt used
by the Android app. The graphs are far too big to bundle in the APK (~4.1 GB), so they are pushed to
the device with `../kotlin_cpu_gpu/android/install_to_device.sh`.

## Prerequisites

```bash
pip install torch transformers ai-edge-litert litert-torch numpy
huggingface-cli download nari-labs/Dia2-1B --local-dir Dia2-1B
```

`build_dia2_mimi_decode.py` additionally needs the Mimi conversion helpers (`build_mimi.py`, which
bakes the causal convolutions and re-authors the decoder transformer). Point `MIMI_WORK` at the
directory holding it.

## Build

```bash
export DIA2_OUT=out
python build_dia2_temporal.py           # temporal transformer, packed KV-cache step graph
python build_dia2_depformer.py          # 3 weight-scheduled depformer step graphs
MIMI_WORK=/path/to/mimi python build_dia2_mimi_decode.py
python export_dia2_assets.py            # baked fp16 tables + dia2_constants.json
python bake_prefix.py                   # voice prompt -> ../kotlin_cpu_gpu/.../assets/
```

| Script | Produces | Self-check |
|---|---|---|
| `build_dia2_temporal.py` | `dia2_temporal_fp32.tflite` (3.0 GB) | hidden / cb0 correlation vs `forward_step` |
| `build_dia2_depformer.py` | `dia2_depformer_wi{0,1,2}_fp32.tflite` | per-stage logit correlation |
| `build_dia2_mimi_decode.py` | `dia2_mimi_decode_t256.tflite` | shape + decode parity |
| `export_dia2_assets.py` | `*.f16` tables, `dia2_constants.json` | — |
| `bake_prefix.py` | `dia2_prefix.json` (13 kB) | prints frames / entries / new-word steps |

`dia2_mimi_dequant.tflite` (the RVQ decode step) is produced alongside the Mimi decoder.

## Why the graphs look the way they do

**Everything is a step function.** The KV caches, RoPE tables, the 34-channel embedding sum, the
depformer's input and logit projections and all sampling live on the host. Each graph takes the
packed cache as an input and returns the new key/value slice, which the host appends at the tail —
no `SCATTER` or `GATHER` in the graph, and no dynamic shapes.

**Everything runs on CPU as fp32,** because fp16 collapses these deep stacks on ARM. The GPU delegate
is not the obstacle: the `FULLY_CONNECTED` rejection reported earlier is a LiteRT 2.1.3 bug fixed in
2.1.5, and the depformer's own compile failure was a rank-5 reshape in our fused-QKV authoring (ML
Drift's maximum rank is 4). With last-dimension slicing and a per-head attention mask it delegates
237/237 nodes and matches the CPU reference exactly — but it is no faster there, because a 3-layer
single-token step graph cannot amortise GPU dispatch and readback synchronisation. See the top-level
[`../README.md`](../README.md) for the measured breakdown.

**The Mimi decoder is exported as one 256-frame window.** Its decode path is upsample → *causal*
decoder transformer → SEANet, so its receptive field is unbounded: decoding disjoint windows starts
each one with no history and costs ~13% relative error (correlation 0.991 against a full-sequence
PyTorch decode). One window that spans the whole utterance, with the unused tail zeroed, is exact —
correlation **0.999999**.

**The voice prompt is precomputed.** Dia2 given no voice prefix *samples* the speaker identity, so
the voice changes on every run. Building a prefix normally needs Whisper word timings and a Mimi
*encoder*, both host-only; `bake_prefix.py` does that offline and emits a 13 kB JSON (aligned Mimi
codes, `new_word_steps`, prefix word entries). The device only replays the warm-up, so no Mimi
encoder ships. Word timings come from Dia2's own `GenerationResult.timestamps`, so Whisper is not
required.
