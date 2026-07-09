# FLUX.2-klein-4B text-to-image — LiteRT CompiledModel GPU (Kotlin)

Text-to-image with **FLUX.2 [klein] 4B** (Black Forest Labs, Apache-2.0) running end to
end on a phone GPU through the LiteRT `CompiledModel` API. Nothing runs on the CPU: the
Qwen3-4B text encoder, the 4B rectified-flow transformer and the VAE decoder are all
`Accelerator.GPU` graphs.

The upstream model card describes klein as running "on consumer GPUs, with as little as
13 GB VRAM". This sample runs the same weights on a Pixel 8a's Mali-G610, which has no
dedicated VRAM at all.

![generated on a Pixel 8a](docs/pixel8a_generated.png)

*"a red apple on a wooden table, studio lighting" — 4 steps, 256x256, generated on a
Pixel 8a. PSNR 36.8 dB / corr 0.9987 against the fp32 `diffusers` pipeline, in 306 s.*

| | |
|---|---|
| Model | [`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) (Apache-2.0) |
| Graphs | [`litert-community/FLUX.2-klein-4B-LiteRT`](https://huggingface.co/litert-community/FLUX.2-klein-4B-LiteRT) |
| Text encoder | Qwen3-4B, hidden states from layers 9 / 18 / 27 |
| Transformer | 5 double-stream + 20 single-stream blocks, dim 3072, 24 heads |
| Steps | 4 (step-wise distilled — **no classifier-free guidance**) |
| Deploy graphs | 12 x int8, 6.2 GB total, largest 912 MB |

## What this sample demonstrates

**Sequential residency.** `CompiledModel` cannot load a graph over 2 GB (flatbuffer
limit), and a phone's GPU budget is well under that anyway. The model is split into
chunks that are resident one at a time, so the peak footprint is a single ~912 MB graph:

```
  ke_enc0/1/2   Qwen3 layers 1-9 / 10-18 / 19-27          912 MB each
  [host]        interleave the three taps -> [1,512,7680]
  kc_prep       image + context embedders, 3 modulation FCs 166 MB
  kc_double0/1  3 + 2 double-stream blocks                 739 / 492 MB
  [host]        joint = concat(text, image)
  kc_single0-3  5 single-stream blocks each                615 MB each
  kc_final      adaLN-continuous norm + projection          19 MB
  kv_vae        VAE decoder                                 50 MB
```

Chunks of roughly 500 MB - 1 GB are the sweet spot. A 1.8 GB chunk does not merely load
more slowly; it sends the GPU shader compiler into a pathological blowup.

**A distilled sampling loop.** klein's pipeline reports `is_distilled`, so it runs no
classifier-free guidance: one transformer pass per step rather than two, and the update
is a plain flow-matching Euler step, `latents += dsigma[step] * noisePrediction`.

**Host/GPU split.** Tokenization, `embed_tokens`, the causal+padding mask, both rotary
tables, the scheduler sigmas and the two tail permutations are precomputed by
[`conversion/gen_prep_klein.py`](conversion/gen_prep_klein.py) and staged as
little-endian `.bin` files. The tail — unpack by position id, per-channel batch-norm
denormalization, 2x2 unpatchify — is two *pure permutations* around one elementwise op,
so the permutations ship as int32 gather maps recovered with an `arange` probe through
the stock pipeline functions.

## Three GPU-delegate constraints this sample encodes

The first is caught by a desktop op check. The other two are not: the graphs compile,
report zero unsupported operations, and produce wrong numbers.

1. **`BROADCAST_TO` is rejected outright**, and `Tensor.expand` lowers to it.
   Grouped-query attention's `repeat_kv` hits this *and* the 4D tensor limit in one
   expression, so it is rewritten as a `CONCATENATION` of per-KV-head slices — same head
   order, exact.
2. **A broadcast `ADD` whose left operand is a `BATCH_MATMUL` result is silently
   miscomputed.** That is `softmax(q @ kᵀ * scale + mask[1,1,S,S])` — every masked
   attention. There is no error and no NaN: the probabilities still sum to 1 and still
   honour the causal and padding masks, but the logits are wrong and the image comes out
   as structured garbage. The fix is to hand the mask in already expanded to
   `[1, numHeads, S, S]`. Zero extra ops, same formula.

   If you ever hit this, the diagnostic signature is that **token 0 is bit-exact and
   every later token is wrong** — token 0 attends only to key 0, so it is the one row
   insensitive to key mixing. (A broken RoPE looks identical, because RoPE is the
   identity at position 0.) `conversion/probe_enc_layer.py` exports one encoder layer
   with a tap at every stage, which is how this was localized.
3. **Compute must be FP32.** `GpuOptions(precision = FP32)`; the modulated (adaLN) blocks
   overflow fp16 and return NaN.

Two runtime rules round it out, both in [`ChunkRunner`](android/app/src/main/java/com/google/ai/edge/examples/flux2_klein/ChunkRunner.kt):
create one `Environment` and share it (a per-call null environment leaks the OpenCL
context and aborts the process after roughly twenty compiles), and close every
`TensorBuffer` after each run.

## Run it

Download the twelve graphs and the staged host inputs, or produce them yourself:

```bash
cd conversion
python build_klein_enc.py --export        # ke_enc0/1/2
python chunked_export_klein.py --export   # kc_prep, kc_double*, kc_single*, kc_final
python vae_deploy_klein.py --export       # kv_vae
python gen_prep_klein.py                  # ref_fp32.png + klein_bins/*.bin
python gen_verify_klein.py                # the device loop, on the host, tflite-only
```

`gen_verify_klein.py` runs the exact loop the app runs, driven only by the exported
graphs, and reports PSNR against the fp32 `diffusers` reference. If that passes, the
Kotlin is a transcription rather than a redesign.

Then stage and launch:

```bash
cd android
./gradlew :app:installDebug
./install_to_device.sh <graphs_dir> <graphs_dir>/klein_bins
```

The app shows the prompt and a **Generate** button; the prompt is baked into the staged
host inputs, so changing it means re-running `gen_prep_klein.py`.

Each graph is recompiled every time it is loaded — `GpuOptions(serializeProgramCache = true)`
aborts the GPU delegate on this runtime — so shader compilation, not arithmetic,
dominates wall-clock.

## Note on quantization

The graphs are **INTEGER-compute int8** (`full_dynamic_recipe(INT8, CHANNELWISE)`);
weight-only-FLOAT quantization hangs the GPU compile. The desktop int8 path is a
*pessimistic* proxy for the device: the same graphs score 36.4 dB through the host CPU
int8 kernels and 44.1 dB on the GPU delegate.
