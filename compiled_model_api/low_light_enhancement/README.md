# Low-Light Enhancement with LiteRT — CPGA-Net (on-device, fully-GPU)

An Android sample that runs **CPGA-Net** ([Shyandram/CPGA-Net-Pytorch](https://github.com/Shyandram/CPGA-Net-Pytorch),
IJPRAI, MIT) end-to-end on device with the LiteRT `CompiledModel` API — **brightens dark photos** via Channel
Prior + Gamma Correction. At **0.025 M params / 0.1 MB fp16** it is one of the smallest deep models you can run.
The app shows the enhanced image; press-and-hold to compare with the original.

## Model

| Graph | In → Out | Delegate (Pixel 8a) |
| :-- | :-- | :--: |
| CPGA-Net | image[1,3,256,256] (RGB [0,1]) → enhanced[1,3,256,256] | **GPU** |

fp16, converted with [litert-torch](https://github.com/google-ai-edge/litert) — tflite-vs-torch corr **1.0**,
device-vs-torch corr **0.99999**. On a Pixel 8a (Tensor G3): **135 / 135** nodes on `LITERT_CL` (full GPU),
**~2 ms**, **0.1 MB** fp16.

## Re-authoring (litert-torch) — three numerically-exact fixes

1. **Gamma correction `x^γ` → `exp(γ·log x)`** (avoids the banned `POW`; exact with the base clamped to [1e-9,1]).
2. **CBAM / gamma global pools** → `mean(3).mean(2)` and `F.max_pool2d(x,(H,W))`.
3. The dark/bright **channel prior** (`max`/`min` over RGB) stays as `REDUCE_MAX`/`REDUCE_MIN`.

See [`conversion/`](conversion/).

## Run

1. Build the tflite with `conversion/build_cpga.py`, or get it from
   [litert-community/CPGA-Net-LowLight-LiteRT](https://huggingface.co/litert-community/CPGA-Net-LowLight-LiteRT).
2. Build/install the app and push the model:
   ```bash
   cd kotlin_cpu_gpu/android
   ./gradlew :app:installDebug
   ./install_to_device.sh <dir-with-cpga_fp16.tflite>
   ```
3. Launch. (The first launch fails with "Model not found" until the model is pushed.)

**Preprocessing**: center-crop, resize 256×256, RGB scaled to [0,1], NCHW.

Upstream: [Shyandram/CPGA-Net-Pytorch](https://github.com/Shyandram/CPGA-Net-Pytorch) (MIT).
