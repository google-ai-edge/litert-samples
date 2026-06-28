# Visual Saliency with LiteRT — UniSal (on-device, fully-GPU)

An Android sample that runs **UniSal** ([rdroste/unisal](https://github.com/rdroste/unisal), Apache-2.0)
end-to-end on device with the LiteRT `CompiledModel` API. UniSal predicts a heatmap of **where humans look** in
an image. MobileNetV2 encoder + bilinear decoder, **3.71 M params**. The app overlays a jet saliency heatmap on a
bundled image and any picked image.

## Model

| Graph | In → Out | Delegate (Pixel 8a) |
| :-- | :-- | :--: |
| UniSal | image[1,3,256,256] → saliency[1,1,256,256] | **GPU** |

fp16, converted with [litert-torch](https://github.com/google-ai-edge/litert) — tflite-vs-torch corr **1.0**,
device-vs-torch corr **0.9998**. On a Pixel 8a (Tensor G3): **158 / 158** nodes on `LITERT_CL` (full GPU),
**~3 ms**, **6.5 MB** fp16.

## Re-authoring (litert-torch) — three numerically-exact fixes

1. The MobileNetV2 strided subsample `x[..., ::2, ::2]` → `F.avg_pool2d(x, 1, 2)` (same pixels, avoids `GATHER_ND`).
2. The 16 Gaussian prior maps **baked** to constants (size-only; avoids `GATHER_ND`/`BROADCAST_TO`).
3. The 41×41 Gaussian-smoothing `replicate`-pad → 0-pad (the smoothing is kept — it suppresses border artifacts).

For static images the Bypass-RNN path is used + the SALICON domain pinned. See [`conversion/`](conversion/).

## Run

1. Build the tflite with `conversion/build_unisal.py`, or get it from
   [litert-community/UniSal-Saliency-LiteRT](https://huggingface.co/litert-community/UniSal-Saliency-LiteRT).
2. Build/install the app and push the model:
   ```bash
   cd kotlin_cpu_gpu/android
   ./gradlew :app:installDebug
   ./install_to_device.sh <dir-with-unisal_fp16.tflite>
   ```
3. Launch. (The first launch fails with "Model not found" until the model is pushed.)

**Preprocessing**: center-crop, resize 256×256, /255, ImageNet mean/std, NCHW. The app min-max normalizes the
saliency and overlays a jet heatmap.

Upstream: [rdroste/unisal](https://github.com/rdroste/unisal) (Apache-2.0).
