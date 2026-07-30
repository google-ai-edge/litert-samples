# Bonsai Image 4B Text-to-Image - Android

An Android application demonstrating fully on-device text-to-image generation with [Bonsai Image 4B](https://huggingface.co/litert-community/Bonsai-Image-ternary-4B), a ternary-weight diffusion transformer (FLUX.2-klein-4B architecture) converted to LiteRT. The entire pipeline runs on the phone: Qwen3 tokenization (pure Kotlin), text encoding, the FlowMatch-Euler diffusion loop, and VAE decoding — three fixed-shape `.tflite` graphs on CPU via XNNPACK, no server, no torch. Direct port of the iOS app in `../ios` (same golden sets, same seeded noise stream, so the same (prompt, seed, steps) reproduces the same image across platforms).

## Example output

<img src="img/generated_pixel8a_seed7.png" alt="Generated on Pixel 8a: a red panda drinking tea in a bamboo forest, watercolor (seed 7)" width="400">

*Generated fully on a Pixel 8a: "a red panda drinking tea in a bamboo forest, watercolor", seed 7, 4 steps.*

## Features

- **Fully on-device**: prompt in, 512×512 PNG out; airplane mode works.
- **Arbitrary prompts**: byte-level BPE Qwen3 tokenizer in Kotlin, verified token-exact against the Python `transformers` tokenizer on a 26-case golden set (CJK, emoji, NFD, whitespace edges).
- **Cross-platform reproducibility**: the seeded noise generator (SplitMix64 + Box-Muller) is bit-compatible with the iOS app — verified against Swift-generated golden values in the JVM test suite. Measured end-to-end: the same (prompt, seed, steps) produced the same image on Pixel 8a and iPhone 17 Pro (34 dB pixel agreement; the residual is XNNPACK per-CPU kernel accumulation differences amplified through the sampling steps).
- **Memory-safe sequencing**: graphs load and close one at a time, so peak footprint stays near the DiT size instead of the ~4 GiB sum — the difference between running and an LMK kill on 8 GB devices.
- **Progress + cancel**, share via the system sheet, PNGs also land in `Android/data/com.google.ai.edge.samples.imagegeneration/files/outputs/` (adb-pullable).
- **CLI-drivable**: `adb shell am start` intent extras trigger a headless generation (see below).

## Architecture

| Component | File | Description |
|-----------|------|-------------|
| **QwenTokenizer** | `QwenTokenizer.kt` | Byte-level BPE (vocab.json + merges.txt) + structural chat template |
| **BonsaiMath** | `BonsaiMath.kt` | Sigma schedule, position ids, seeded Gaussian noise, latent unpatchify — JVM-tested against recorded pipeline fixtures |
| **BonsaiPipeline** | `BonsaiPipeline.kt` | Sequential three-graph execution over the LiteRT `Interpreter` API (CPU/XNNPACK, mmap'd model files) |
| **MainActivity** | `MainActivity.kt` | Single-screen UI: prompt, steps, seed, progress, image, share |

## Prerequisites & Setup

1. **Assets**: `./prep_assets.sh` copies the tokenizer tables (`vocab.json`, `merges.txt`) and `pipeline_meta.json` from the model download into `app/src/main/assets/` (bundled with the APK, not committed).
2. **Build**: `./gradlew assembleRelease` (release is signed with the debug key for sample convenience) — or open in Android Studio.
3. **Models**: download from [litert-community/Bonsai-Image-ternary-4B](https://huggingface.co/litert-community/Bonsai-Image-ternary-4B) and push the three graphs (3.97 GiB total — the smallest set: int4 DiT + int4 DRQ text encoder):

```bash
adb install app/build/outputs/apk/release/app-release.apk
D=/sdcard/Android/data/com.google.ai.edge.samples.imagegeneration/files
adb push dit_int4b32.tflite textenc_int4.tflite vae_dec_fp32.tflite "$D/"
```

The app lists any missing files on launch.

### Headless run (optional)

```bash
adb shell am start -n com.google.ai.edge.samples.imagegeneration/.MainActivity \
  --ez autorun true --es prompt "a red fox in fresh snow" --el seed 7 --ei steps 4
adb pull "$D/outputs/"   # generated PNGs
```

## How It Works

- **Tokenization**: the graph input is `[<|im_start|>]` + BPE(`"user\n"` + prompt) + a fixed assistant suffix — structurally identical to `transformers.apply_chat_template(..., enable_thinking=False)`. The `"user\n"` prefix must be BPE'd together with the prompt (whitespace merges across the boundary).
- **XNNPACK**: enabled explicitly with the thread count; on the iOS build of this pipeline the delegate proved to be a *correctness* requirement for blockwise int4 (reference kernels were both ~100× slower and numerically wrong), so it is treated as mandatory here too.
- **mmap**: models are opened by file path so the 2.11 GiB DiT is mapped, not copied to the Java heap.
- **Host math**: Euler update, empirical-mu sigma schedule, and BatchNorm-affine + 2×2 unpatchify are JVM-tested against recorded fixtures from the verified Mac pipeline (max error 9.5e-7): device runs never debug math, only the runtime.

## Model Information

| Graph | File | Recipe | Size |
|---|---|---|---|
| DiT (3.88 B, ternary weights) | `dit_int4b32.tflite` | int4 block-32 | 2.11 GiB |
| Text encoder (Qwen3-4B, pruned) | `textenc_int4.tflite` | int4 block-128 DRQ | 1.68 GiB |
| VAE decoder | `vae_dec_fp32.tflite` | fp32 | 0.19 GiB |

Source and conversion details: [model card](https://huggingface.co/litert-community/Bonsai-Image-ternary-4B).

## Performance

| Device | text enc | DiT step | VAE | total (4 steps, 512×512) | outcome |
|---|---|---|---|---|---|
| Pixel 8a (Tensor G3, 8 GB) | 11.7 s | 78–88 s | 69 s | **~7.1 min** | completes; foreground survives, but the OS evicts essentially every background app during the DiT stage |
| iPhone 17 Pro (A19 Pro, 12 GB) — iOS app | 1.9 s | 12–13 s | 3.0 s | ~62 s | comfortable |

Peak footprint is ~2.9 GiB (DiT stage, including XNNPACK weight repacking). On 8 GB devices the run completes as the foreground app but puts the whole system under memory pressure — treat 8 GB as a proof-of-run floor, and 12 GB+ as the practical target.

## Testing

`./gradlew testDebugUnitTest` runs the port-parity suite (skips cleanly if the local golden/fixture artifacts are absent): tokenizer vs the Python golden set, sigma schedule / position ids / Euler+unpatchify vs recorded fixtures, and the noise stream vs Swift-generated values.

