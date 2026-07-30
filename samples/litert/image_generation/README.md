# LiteRT Image Generation (Text-to-Image)

Fully on-device text-to-image generation with [Bonsai Image 4B](https://huggingface.co/litert-community/Bonsai-Image-ternary-4B), a ternary-weight diffusion transformer (FLUX.2-klein-4B architecture) converted to LiteRT. The whole pipeline runs on the phone in three fixed-shape `.tflite` graphs (Qwen3 text encoder, DiT, VAE decoder) on CPU via XNNPACK — no server, no torch. Prompt in, 512×512 PNG out; airplane mode works.

| App | Directory | Stack |
|---|---|---|
| Android | [`android/`](android/) | Kotlin, LiteRT `Interpreter` API |
| iOS | [`ios/`](ios/) | SwiftUI, classic TFLite C API + explicit XNNPACK delegate |

Both apps carry their own ports of the host pipeline — a byte-level-BPE Qwen3 tokenizer (token-exact against the Python `transformers` tokenizer on the golden set in [`testdata/`](testdata/)), the FlowMatch-Euler sampling loop, and the latent unpatchify — and share a bit-identical seeded-noise stream, so the same (prompt, seed, steps) draws the same image on both platforms.

## Performance (smallest set: int4 DiT + int4 DRQ text encoder, 3.97 GiB)

| Device | text encoder | DiT step | VAE | total (4 steps, 512×512) |
|---|---|---|---|---|
| iPhone 17 Pro (12 GB) | 1.9 s | 12–13 s | 3.0 s | ~62 s |
| Pixel 8a (8 GB) | 11.7 s | 78–88 s | 69 s | ~7–8 min |
| Apple M4 Max (reference, Python sample) | 2.6 s | 3.9–4.0 s | 1.4 s | ~20 s |

Peak memory is ~2.9 GiB (DiT stage). 8 GB-RAM phones complete the run but come under system-wide memory pressure — treat 8 GB as the floor and 12 GB+ as the practical target.

## Model

`dit_int4b32.tflite` (2.11 GiB) + `textenc_int4.tflite` (1.68 GiB) + `vae_dec_fp32.tflite` (0.19 GiB), downloaded from the [model card](https://huggingface.co/litert-community/Bonsai-Image-ternary-4B). A Python host reference for the same graphs lives in [`models/bonsai/bonsai_image_4b`](../../../models/bonsai/bonsai_image_4b).
