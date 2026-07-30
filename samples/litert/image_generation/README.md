# LiteRT Image Generation (Text-to-Image)

Fully on-device text-to-image generation with [Bonsai Image 4B](https://huggingface.co/litert-community/Bonsai-Image-ternary-4B), a ternary-weight diffusion transformer (FLUX.2-klein-4B architecture) converted to LiteRT. The whole pipeline runs in three fixed-shape `.tflite` graphs (Qwen3 text encoder, DiT, VAE decoder) on CPU via XNNPACK — no server, no torch. Prompt in, 512×512 PNG out.

## Where things live

- **Python host sample** (tokenize + FlowMatch-Euler loop + unpatchify, ~150 lines): [`models/bonsai/bonsai_image_4b`](../../../models/bonsai/bonsai_image_4b) in this repo.
- **Demo apps (iOS SwiftUI + Android Kotlin)**: [hf-to-litertlm/bonsai_image_work/device](https://github.com/john-rocky/hf-to-litertlm/tree/main/bonsai_image_work/device) — single-screen apps with on-device Qwen3 tokenization (byte-level BPE in Swift/Kotlin, token-exact against the Python tokenizer), seed control, progress, cancel, and share. A polished iOS/macOS demo app built on the LiteRT Swift package is planned for this repo.
- **Model files**: [model card](https://huggingface.co/litert-community/Bonsai-Image-ternary-4B) (int4 DiT 2.11 GiB + int4-DRQ text encoder 1.68 GiB + fp32 VAE 0.19 GiB).

## Measured (smallest set, 512×512, 4 steps)

| Device | DiT step | total |
|---|---|---|
| Apple M4 Max (Python sample) | 3.9–4.0 s | ~20 s |
| iPhone 17 Pro | 12–13 s | ~62 s |

Peak memory is ~2.9 GiB — the three graphs load sequentially and are freed between stages. Details, Android numbers, and the cross-platform seeded-reproducibility notes are in the demo repo's READMEs.
