# Face Detection with LiteRT — YuNet (on-device, fully-GPU)

An Android sample that runs **YuNet** ([ShiqiYu/libfacedetection](https://github.com/ShiqiYu/libfacedetection), BSD-3-Clause) end-to-end on device with the LiteRT `CompiledModel` API. YuNet is a tiny, fast **face detector** (faces + 5 landmarks). At **0.076 M params / 0.3 MB fp16** it is one of the smallest deep models you can run. The app detects faces in a bundled image and any image picked from the gallery, drawing boxes and landmarks.

## Model

| Graph | In → Out | Delegate (Pixel 8a) |
| :-- | :-- | :--: |
| YuNet | image[1,3,640,640] (BGR,0-255) → 12 × (cls/obj/bbox/kps per stride) | **GPU** |

fp16, converted with [litert-torch](https://github.com/google-ai-edge/litert) — tflite-vs-torch corr **1.0**, device-vs-torch corr **0.9999**. On a Pixel 8a (Tensor G3): **146 / 146** nodes on `LITERT_CL` (full GPU), **~4 ms**, **0.3 MB** fp16.

## Re-authoring (litert-torch) — none needed

Pure CNN (depthwise-separable `ConvDPUnit`) + a **nearest-upsample** neck (no transposed conv) + non-padded `MaxPool2d` (no `PADV2`). The head's per-stride `permute/reshape/sigmoid` is baked in → 12 decode-ready outputs. See [`conversion/`](conversion/).

## Output & decode

Anchor-free, priors at each stride with offset 0 (`px=col·s, py=row·s`): score = `cls·obj`; box = center + `exp(wh)·s` → (x1,y1,x2,y2); 5 landmarks `kps·s+prior`; then NMS (IoU 0.45). Runs in `FaceDetector.kt`.

## Run

1. Build the tflite with `conversion/build_yunet.py`, or get it from [litert-community/YuNet-Face-LiteRT](https://huggingface.co/litert-community/YuNet-Face-LiteRT).
2. Build/install the app and push the model:
   ```bash
   cd kotlin_cpu_gpu/android
   ./gradlew :app:installDebug
   ./install_to_device.sh <dir-with-yunet_fp16.tflite>
   ```
3. Launch. (The first launch fails with "Model not found" until the model is pushed.)

**Preprocessing**: letterbox to 640×640, **BGR, 0-255 (no normalization)**, NCHW.

Upstream: [ShiqiYu/libfacedetection](https://github.com/ShiqiYu/libfacedetection) (BSD-3-Clause).
