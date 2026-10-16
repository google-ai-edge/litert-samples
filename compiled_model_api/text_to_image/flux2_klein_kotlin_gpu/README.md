# FLUX.2-klein-4B text-to-image and image editing — LiteRT CompiledModel GPU (Kotlin)

Text-to-image **and prompt-driven image editing** with **FLUX.2 [klein] 4B** (Black Forest
Labs, Apache-2.0), running end to end on a phone GPU through the LiteRT `CompiledModel`
API. Nothing runs on the CPU: the Qwen3-4B text encoder, the 4B rectified-flow
transformer and both VAE halves are all `Accelerator.GPU` graphs.

The upstream model card describes klein as running "on consumer GPUs, with as little as
13 GB VRAM". This sample runs the same weights on a Pixel 8a's Mali-G715, which has no
dedicated VRAM at all.

![generated on a Pixel 8a](docs/pixel8a_generated.png)

*"a red apple on a wooden table, studio lighting" — 4 steps, 256x256, generated on a
Pixel 8a. PSNR 36.8 dB / corr 0.9987 against the fp32 `diffusers` pipeline, in 306 s.*

![edited on a Pixel 8a](docs/pixel8a_edited.png)

*Editing: source, the fp32 `diffusers` edit, and the same edit on a Pixel 8a. "turn the
apple into a green apple" — 4 steps, 256x256, PSNR 44.3 dB / SSIM 0.9998 against the fp32
pipeline, in 328-369 s.*

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

## How editing works

`Flux2KleinPipeline.__call__` takes `image` as its first argument: klein is natively an
editing model, and text-to-image is the `image=None` case. Editing VAE-encodes the
reference and appends its latent tokens to the noise tokens before every step,

```python
latent_model_input = torch.cat([latents, image_latents], dim=1)
latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)
```

with the reference separated from the noise on the rotary `T` axis (noise `T = 0`, the
i-th reference `T = 10 + 10 i`). Afterwards the pipeline keeps only the noise half:
`noise_pred[:, :latents.size(1)]`.

At 256x256 that grows the image sequence from 256 to 512 tokens, so the joint sequence
the transformer attends over goes 768 to 1024. **The weights do not change**, so the
`kce_*` graphs are the same tensors re-exported at the longer shape and the chunk sizes
are byte-identical; only the activations grow. Measured on a Pixel 8a: peak RSS +2 %,
per-step GPU time 1.6x, shader compile 1.0-1.2x — and since compilation dominates, the
end-to-end cost of editing over generation is about **+7 %**.

The app adds one graph beyond the text-to-image twelve: `kv_vae_enc.tflite`, the VAE
encoder. Its `mode()` chunks the 64-channel moments in half, which lowers to the banned
`SPLIT`; slicing the first 32 channels directly is bit-exact and GPU-clean.

## Run it

Download the graphs and the staged host inputs, or produce them yourself:

```bash
cd conversion
python build_klein_enc.py --export           # ke_enc0/1/2 (shared by both modes)
python chunked_export_klein.py --export      # kc_prep, kc_double*, kc_single*, kc_final
python vae_deploy_klein.py --export          # kv_vae
python gen_prep_klein.py                     # ref_fp32.png + klein_bins/*.bin
python gen_verify_klein.py                   # the device loop, on the host, tflite-only

# image editing
python chunked_export_klein.py --edit --export           # kce_* at the longer sequence
python vae_encode_klein.py --export                      # kv_vae_enc
python gen_prep_klein.py --edit --bins klein_bins_edit   # + patch_perm.bin, edit_source.png
python gen_verify_klein.py --edit --bins klein_bins_edit

# typed prompts (optional): tokenizer + the fp16 embedding table
python export_tokenizer_klein.py --out klein_tokenizer   # 778 MB embedding + vocab/merges
```

`gen_verify_klein.py` runs the exact loop the app runs, driven only by the exported
graphs, and reports PSNR against the fp32 `diffusers` reference. If that passes, the
Kotlin is a transcription rather than a redesign. Add `--device` to run the identical
loop on the phone's GPU instead of the host.

Then stage and launch:

```bash
cd android
./gradlew :app:installDebug
./install_to_device.sh <graphs_dir> <graphs_dir>/klein_bins            # generation only
./install_to_device.sh <graphs_dir> <graphs_dir>/klein_bins \
                       <graphs_dir>/klein_bins_edit                    # + editing
```

If `<graphs_dir>/klein_tokenizer` is present, `install_to_device.sh` stages it too and the
app shows **editable prompt fields** — you type the prompt and it is tokenized and embedded
on device. Otherwise the prompt is the one baked into the staged tensors, and changing it
means re-running `gen_prep_klein.py`.

**How typed prompts work.** The only staged tensors that depend on the words are
`inputs_embeds` and `enc_mask`; the rotary tables, the schedule and the tail permutations
depend on positions, the schedule or the seed. So the app carries a faithful port of the
`Qwen2Tokenizer` (`QwenTokenizer.kt`, fixture-tested byte-for-byte against Python by
`export_tokenizer_klein.py`) and looks the token rows up in a memory-mapped fp16 copy of the
Qwen3 embedding table — a `GATHER` over 151936 rows is not a GPU op, and the gathered row is
the graph's input anyway. Tokenizing and embedding a prompt takes about 1 s.

Sizes: generation needs 6.2 GB of graphs; editing brings it to 10.1 GB; the typed-prompt
embedding table adds 778 MB.

Each graph is recompiled every time it is loaded — `GpuOptions(serializeProgramCache = true)`
aborts the GPU delegate on this runtime — so shader compilation, not arithmetic,
dominates wall-clock.

## Note on quantization

The graphs are **INTEGER-compute int8** (`full_dynamic_recipe(INT8, CHANNELWISE)`);
weight-only-FLOAT quantization hangs the GPU compile. The desktop int8 path is a
*pessimistic* proxy for the device: the same graphs score 36.4 dB through the host CPU
int8 kernels and 44.1 dB on the GPU delegate. The editing loop shows it again — the same
`kce_*` graphs score 39.5 dB on the host CPU and 45.6 dB on the phone GPU.

For editing, report SSIM alongside PSNR. The error is *sparse*: on a moonlit-snow edit
the worst 0.5 % of pixels carry 76 % of the squared error (specular speckles flipping
between near-white and dark blue), which drags PSNR to 27.9 dB while SSIM stays at 0.989
and the image is perceptually identical to the fp32 reference.
