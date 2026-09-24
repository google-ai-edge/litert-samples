# Blind Face Restoration with LiteRT — GFPGAN (on-device)

An Android sample that runs [GFPGAN v1.4](https://github.com/TencentARC/GFPGAN) blind face restoration (Apache-2.0) on device with the LiteRT `CompiledModel` API, **fully on the GPU**. It reconstructs a degraded / low-quality face using a StyleGAN2 generative facial prior. Pick a photo, and the app detects the face, aligns it, and shows the original and the restored face side by side.

```
photo →[YuNet detect+5 landmarks, GPU]→ FFHQ-align to 512 →[GFPGAN, GPU]→ restored face [1,3,512,512]
```

## Models

| Stage | In → Out | Delegate (Pixel 8a) |
| :-- | :-- | :--: |
| YuNet face detect | image[1,3,640,640] BGR → box + 5 landmarks | **GPU** |
| FFHQ align (Kotlin) | 5 landmarks → 512×512 similarity warp | CPU |
| GFPGAN v1.4 | face[1,3,512,512] NCHW [-1,1] → restored[1,3,512,512] [-1,1] | **GPU** |

Both models load with `CompiledModel.create(...)` on `Accelerator.GPU`. fp16, converted with [litert-torch](https://github.com/google-ai-edge/litert) — fp16 tflite-vs-torch corr **1.000000**. On a Pixel 8a (Tensor G3): GFPGAN **551/551** nodes on `LITERT_CL` (1 partition, no CPU fallback), ~1.2 s/face; YuNet 0.3 MB, ~4 ms. The fp16 device output matches the desktop fp32 result.

## Why it's GPU-clean — the one real re-authoring

GFPGAN's StyleGAN2 `ModulatedConv2d` originally builds a **5D** weight `(b,c_out,c_in,k,k)` at runtime from the style vector and convolves with that runtime filter — both GPU-incompatible (>4D; a GPU `CONV_2D` needs a constant filter). It is rewritten to an exact 4D form: the modulation becomes an input channel-scale + a constant `CONV_2D`, and the demodulation becomes a constant `(c_out×c_in)` matmul + `RSQRT`. Because the style reaches `|s|~1000`, the demod sum overflows fp16 on Mali, so the style is normalized by its per-image max before squaring (the scale cancels exactly). Everything else is already GPU-friendly (`RESIZE_BILINEAR` upsampling, no GroupNorm, deterministic noise). See [`conversion/`](conversion/).

## Alignment (required for quality)

GFPGAN's StyleGAN prior mangles the mouth on off-template crops, so the app aligns first: YuNet gives 5 landmarks, then a least-squares similarity transform warps the face to facexlib's 512 template ([`FaceAligner.kt`](kotlin_cpu_gpu/android/app/src/main/java/com/google/ai/edge/examples/face_restoration/FaceAligner.kt)). It falls back to a center-square crop if no face is found.

## Run

1. Get the models: build `gfpgan_fp16.tflite` with [`conversion/build_gfpgan.py`](conversion/build_gfpgan.py) (or from Hugging Face [`litert-community/GFPGAN-v1.4-LiteRT`](https://huggingface.co/litert-community/GFPGAN-v1.4-LiteRT)), plus `yunet_fp16.tflite`.
2. Push them into the app's private storage (too large to bundle):
   ```bash
   cd kotlin_cpu_gpu/android
   ./install_to_device.sh <dir-with-the-tflites>
   ```
3. Build & install the app, pick a face photo, and compare the original with the restored result.
