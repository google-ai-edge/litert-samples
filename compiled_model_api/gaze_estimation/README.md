# Gaze Estimation with LiteRT — L2CS-Net (on-device, fully-GPU)

An Android sample that runs **L2CS-Net** ([Ahmednull/L2CS-Net](https://github.com/Ahmednull/L2CS-Net), MIT) end-to-end on device with the LiteRT `CompiledModel` API. L2CS estimates **where a (centered) face is looking** — yaw and pitch — for attention, AR, and accessibility. ResNet50 backbone trained on Gaze360. The app draws a gaze-direction arrow on a bundled image and any image picked from the gallery.

## Model

| Graph | In → Out | Delegate (Pixel 8a) |
| :-- | :-- | :--: |
| L2CS-Net | face[1,3,448,448] → yaw[1,90], pitch[1,90] | **GPU** |

fp16, converted with [litert-torch](https://github.com/google-ai-edge/litert) — tflite-vs-torch corr **1.0**, device-vs-torch corr **0.9999** (gaze angle within ~0.1°). On a Pixel 8a (Tensor G3): **139 / 139** nodes on `LITERT_CL` (full GPU residency), **~3 ms**, **47.9 MB** fp16.

## Re-authoring (litert-torch, two ResNet fixes)

Pure CNN (ResNet50 + 2 FC heads). 1) The stem `MaxPool2d(3,s2,p1)` → zero-pad + valid max-pool (PyTorch's `-inf`-pad lowers to a `PADV2` the Mali delegate won't delegate; since the pool follows a ReLU, a 0-pad is exactly equivalent → `PAD`). 2) The global `AdaptiveAvgPool2d(1)` → `mean(3).mean(2)`. The angle-bin softmax is baked in. See [`conversion/`](conversion/).

## Output & decode

90 angle bins span [-180, 180]° at 4° each; the gaze angle is the softmax expectation `Σ p_i·i · 4 − 180`. The decode and the gaze-direction arrow run in the app (`GazeEstimator.kt`).

## Run

1. Build the tflite with `conversion/build_gaze.py`, or get it from [litert-community/L2CS-Gaze360-LiteRT](https://huggingface.co/litert-community/L2CS-Gaze360-LiteRT).
2. Build/install the app and push the model:
   ```bash
   cd kotlin_cpu_gpu/android
   ./gradlew :app:installDebug
   ./install_to_device.sh <dir-with-gaze_fp16.tflite>
   ```
3. Launch. (The first launch fails with "Model not found" until the model is pushed.)

**Preprocessing**: center-crop to a (centered) face, resize 448×448, /255, ImageNet mean/std, NCHW.

Upstream: [Ahmednull/L2CS-Net](https://github.com/Ahmednull/L2CS-Net) (MIT); dataset [Gaze360](http://gaze360.csail.mit.edu/).
