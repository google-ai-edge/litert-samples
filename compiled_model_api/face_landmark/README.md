# Face Landmark (alignment) with LiteRT — RTMPose-Face WFLW (on-device, fully-GPU)

An Android sample that runs **RTMPose-Face** ([mmpose](https://github.com/open-mmlab/mmpose), Apache-2.0, trained
on **WFLW**) end-to-end on device with the LiteRT `CompiledModel` API. It estimates **98 dense facial landmarks**
(contour, eyebrows, eyes, nose, mouth, pupils) — the dense complement to a 5-point detector (detect → align).
The app draws the face mesh on a bundled image and any picked image.

## Model

| Graph | In → Out | Delegate (Pixel 8a) |
| :-- | :-- | :--: |
| RTMPose-m | face[1,3,256,256] → simcc_x[1,98,512], simcc_y[1,98,512] | **GPU** |

fp16, converted with [litert-torch](https://github.com/google-ai-edge/litert) — tflite-vs-torch corr **1.0**,
device-vs-torch SimCC corr **0.9995**. On a Pixel 8a (Tensor G3): **333 / 333** nodes on `LITERT_CL` (full GPU),
**~4 ms**, **33.6 MB** fp16.

## Re-authoring (litert-torch) — the RTMPose recipe, unchanged

Same model family as the human-pose RTMPose; only the config/checkpoint change to WFLW. Two on-device-only Mali
fixes transfer without modification: **`ScaleNorm` → SafeRMSNorm** (the RMS-norm channel `Σx²` overflows fp16 on
Mali → scale down by 64 before squaring) and **GAU `act@act` BMM → broadcast-reduce**. See [`conversion/`](conversion/).

## Output & decode

output[0] = simcc_x, output[1] = simcc_y; each landmark = `argmax` over its 1D SimCC (bins = pixels × 2). The
98-point face mesh is drawn in `MainActivity.kt`.

## Run

1. Build the tflite with `conversion/build_rtm_face.py`, or get it from
   [litert-community/RTMPose-Face-WFLW-LiteRT](https://huggingface.co/litert-community/RTMPose-Face-WFLW-LiteRT).
2. Build/install the app and push the model:
   ```bash
   cd kotlin_cpu_gpu/android
   ./gradlew :app:installDebug
   ./install_to_device.sh <dir-with-rtm_face_fp16.tflite>
   ```
3. Launch. (The first launch fails with "Model not found" until the model is pushed.)

**Preprocessing**: center-crop to a face, resize 256×256, mmpose mean/std (RGB, 0-255), NCHW.

Upstream: [open-mmlab/mmpose](https://github.com/open-mmlab/mmpose) (Apache-2.0); dataset
[WFLW](https://wywu.github.io/projects/LAB/WFLW.html).
