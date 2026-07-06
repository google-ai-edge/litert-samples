# Image Inpainting with LiteRT — MI-GAN (on-device, fully-GPU)

An Android sample that runs **MI-GAN** ([Picsart-AI-Research/MI-GAN](https://github.com/Picsart-AI-Research/MI-GAN), ICCV 2023, MIT) end-to-end on device with the LiteRT `CompiledModel` API — a "magic eraser": paint over an object and it is removed and inpainted. A mobile-designed StyleGAN-style generator (separable convs, nearest-upsample, **no norm**). The app lets you finger-paint a mask and erase on a bundled image and any picked image.

## Model

| Graph | In → Out | Delegate (Pixel 8a) |
| :-- | :-- | :--: |
| MI-GAN | in[1,4,512,512] = concat(mask-0.5, rgb·mask) → out[1,3,512,512] ([-1,1]) | **GPU** |

fp16, converted with [litert-torch](https://github.com/google-ai-edge/litert) — tflite-vs-torch corr **1.0**, device-vs-torch corr **0.99998**. On a Pixel 8a (Tensor G3): **509 / 509** nodes on `LITERT_CL` (full GPU), **~6 ms**, **16.3 MB** fp16.

## Re-authoring (litert-torch) — none needed

The MI-GAN **inference** generator is already GPU-friendly: depthwise-separable `Conv2d`, `nn.Upsample(nearest)`
+ a fixed FIR-filter grouped conv (**no transposed conv**), leaky-ReLU with clamp (→ `MAXIMUM`/`MINIMUM`), and
**no normalization**. See [`conversion/`](conversion/).

## I/O

**Input** (4 ch): `concat(mask − 0.5, rgb · mask)` — rgb ∈ [-1,1] (`pixel/127.5 − 1`); mask = **1 keep, 0 erase**. **Output** (3 ch): generated image in [-1,1]; composite as `rgb·mask + out·(1−mask)` so only the erased region changes. See `MiganInpainter.kt`.

## Run

1. Build the tflite with `conversion/build_migan.py`, or get it from [litert-community/MI-GAN-512-Places2-LiteRT](https://huggingface.co/litert-community/MI-GAN-512-Places2-LiteRT).
2. Build/install the app and push the model:
   ```bash
   cd kotlin_cpu_gpu/android
   ./gradlew :app:installDebug
   ./install_to_device.sh <dir-with-migan_fp16.tflite>
   ```
3. Launch, paint over an object, tap **Erase**. (The first launch fails with "Model not found" until the model is pushed.)

**Preprocessing**: center-crop, resize 512×512.

Upstream: [Picsart-AI-Research/MI-GAN](https://github.com/Picsart-AI-Research/MI-GAN) (MIT).
