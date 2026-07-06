# MoViNet-A0 — Streaming video action recognition (LiteRT CompiledModel GPU)

Recognises human **actions** across a stream of frames — one frame at a time, with
constant memory — running **fully on the LiteRT `CompiledModel` GPU** delegate.
[MoViNet-A0](https://arxiv.org/abs/2103.11511) streaming variant (Google Research),
trained on **Kinetics-600** (600 action classes).

- **Model:** [litert-community/MoViNet-A0-Stream-LiteRT](https://huggingface.co/litert-community/MoViNet-A0-Stream-LiteRT) · 15 MB
- **Weights:** [Atze00/MoViNet-pytorch](https://github.com/Atze00/MoViNet-pytorch) · Apache-2.0
- **Input:** one RGB frame `[1, 3, 172, 172]` (NCHW, 0..1) per step
- **Output:** `[1, 600]` Kinetics-600 logits + updated recurrent state

## How it works

MoViNet is a causal 3D CNN whose temporal convolutions and global-average-pools each
keep a small buffer of the recent past, so it can be fed one frame at a time and its
prediction sharpens as more frames of the same action arrive. The stock streaming
graph carries that history in **5D** state tensors `[1, T, H, W, C]`, which the GPU
delegate cannot compile (all tensors must be ≤4D). The model is re-authored as a
**single-frame, 4D-only functional forward** (**47 inputs / 28 outputs**) with the
recurrent state threaded through the graph I/O:

| I/O slot        | count | shape           | meaning                              |
|-----------------|-------|-----------------|--------------------------------------|
| `input[0]`      | 1     | `[1,3,172,172]` | current RGB frame (NCHW, 0..1)       |
| `input[1..28]`  | 28    | `[1,C,H,W]`     | temporal-conv stream buffers (11 convs) |
| `input[29..44]` | 16    | `[1,C,1,1]`     | streaming avg-pool running sums      |
| `input[45]`     | 1     | `[1,1,1,1]`     | `inv_count` = 1 / frame number       |
| `input[46]`     | 1     | `[1,1,1,1]`     | constant `1.0` (output decoupler)    |
| `output[0]`     | 1     | `[1,600]`       | Kinetics-600 logits                  |
| `output[1..11]` | 11    | `[1,C,H,W]`     | current per-temporal-conv frame      |
| `output[12..27]`| 16    | `[1,C,1,1]`     | fresh per-frame spatial means        |

The **stream-buffer shift register and pool running-sum accumulation are done
host-side** (see `MoViNet.kt`): the graph consumes the recurrent state but only
emits fresh tensors. This side-steps three quirks of the Mali GPU delegate for
stateful graphs (an input passed through to an output loses its compute-side use; a
`state + tensor` output is zeroed; a conv output that is both consumed and emitted
has its emitted copy corrupted). Every op lowers to a GPU-supported primitive —
**0 tensors of rank > 4, 0 GPU-incompatible ops** — and the converted graph matches
the original PyTorch model bit-for-bit (correlation 0.99999999999).

## Run

```bash
# 1. Get the model (build with ../conversion or download from Hugging Face)
#    then push it into the app's private storage:
cd android
./install_to_device.sh <dir-with-movinet_a0_stream.tflite>

# 2. Build & run
./gradlew :app:installDebug
```

The sample streams a bundled reference clip (13 frames of jumping jacks) through the
model and prints the running top-1 prediction after each frame plus the final top-5 —
a deterministic demonstration of the streaming pipeline (it locks onto *jumping
jacks* within a few frames). Adapt `MainActivity.kt` to feed live camera frames for
a real-time demo.

## Convert

See [`conversion/`](conversion/) — `build_movinet.py` re-authors
`Atze00/MoViNet-pytorch` into the 4D streaming graph and converts it with
litert-torch, verifying parity against the original model.
