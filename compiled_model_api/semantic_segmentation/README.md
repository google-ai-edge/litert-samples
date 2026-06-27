# Semantic Segmentation with LiteRT — LR-ASPP MobileNetV3 (on-device, fully-GPU)

An Android sample that runs **Lite R-ASPP** with a MobileNetV3-Large backbone (torchvision
`lraspp_mobilenet_v3_large`, COCO-VOC 21 classes) end-to-end on device with the LiteRT `CompiledModel` API.
It labels every pixel as one of 21 PASCAL-VOC classes (person, dog, car, chair, …) — general-scene semantic
segmentation. The app segments a bundled image at launch and any image picked from the gallery, overlaying
the class colormap. (Distinct from the `image_segmentation` sample, which is selfie/body-part segmentation.)

## Model

| Graph | In → Out | Delegate (Pixel 8a) |
| :-- | :-- | :--: |
| LR-ASPP MobileNetV3 | image[1,3,512,512] → logits[1,512,512,21] | **GPU** |

fp16, converted with [litert-torch](https://github.com/google-ai-edge/litert) — tflite-vs-torch corr
**1.0**, device-vs-torch corr **0.99998** / argmax agreement **99.85%**. On a Pixel 8a (Tensor G3): **242 /
242** nodes on `LITERT_CL` (full GPU residency), **~5 ms** @ 512×512, **6.7 MB** fp16. The per-pixel argmax
runs on the CPU (trivial).

## Re-authoring (litert-torch, parity corr 1.0)

Pure CNN — a single re-authoring: the MobileNetV3 Squeeze-Excite blocks and the R-ASPP scale branch use
`AdaptiveAvgPool2d(1)` (global average pool), each replaced with `mean(3).mean(2)` (two single-axis means;
a single multi-axis pool is mis-computed on the Mali delegate). Everything else is already GPU-clean
(`Hardswish`/`Hardsigmoid` → native `HARD_SWISH`, `align_corners=False`). See [`conversion/`](conversion/).

## Run

1. Build the tflite with `conversion/build_lraspp.py` (weights auto-downloaded by torchvision), or get it from
   [mlboydaisuke/LRASPP-MobileNetV3-LiteRT](https://huggingface.co/mlboydaisuke/LRASPP-MobileNetV3-LiteRT).
2. Build/install the app, then push the model into its private storage:
   ```bash
   cd kotlin_cpu_gpu/android
   ./gradlew :app:installDebug
   ./install_to_device.sh <dir-with-lraspp_fp16.tflite>
   ```
3. Launch the app. (The first launch fails with "Model not found" until the model is pushed.)

## Preprocessing

Center-crop to square, resize to 512×512, divide by 255, ImageNet normalize
`[0.485,0.456,0.406]`/`[0.229,0.224,0.225]`, NCHW planar. Output is 21-class logits; argmax per pixel.

Upstream: [torchvision](https://github.com/pytorch/vision) `lraspp_mobilenet_v3_large` (BSD-3-Clause).
