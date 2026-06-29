# LiteRT Object Detection Sample (YOLOX)

<p align="center">
  <img src="https://huggingface.co/litert-community/yolox-s-litert/resolve/main/demo.png" width="300" alt="YOLOX real-time object detection on LiteRT GPU">
</p>

This directory contains an Android **real-time object detection** sample demonstrating the LiteRT
(Google's runtime for TensorFlow Lite) **Compiled Model API** on CPU and GPU. It runs
**[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Megvii, Apache-2.0)** live on the camera and
on gallery images, drawing labelled COCO bounding boxes.

YOLOX is a permissively-licensed (**Apache-2.0**) alternative to the AGPL/GPL YOLO family, and a pure
CNN — so it re-authors cleanly to a **GPU-native** `.tflite` with
[`litert-torch`](https://github.com/google-ai-edge/ai-edge-torch).

## Overview

Four model sizes are selectable in the app, all converted the same way and downloaded from
[`litert-community`](https://huggingface.co/litert-community) at build time:

| Model | Input | FP16 size | COCO AP | Best on Pixel 8a GPU |
|-------|-------|-----------|---------|----------------------|
| YOLOX-Nano | 416 | 2.2 MB | 25.8 | ~2.6 ms |
| YOLOX-Tiny | 416 | 10.4 MB | 32.8 | ~1.2 ms |
| **YOLOX-S** (default) | 640 | 18.2 MB | 40.5 | ~1.7 ms |
| YOLOX-M | 640 | 51.0 MB | 46.9 | ~2.5 ms |

The models are **GPU-clean** (full `LITERT_CL` residency — no GATHER_ND / TopK / Cast, no >4-D
tensors). The single conversion trick is folding YOLOX's `Focus` stem (stride-2 space-to-depth, which
otherwise lowers to GPU-rejected `GATHER_ND`) into one numerically-exact 6×6 stride-2 conv — see
[`conversion/`](conversion). On device the GPU output matches the CPU/PyTorch reference (correlation
≥ 0.999); residency **and** correctness both hold.

The model graph outputs **raw detection heads**; the grid + stride decode and per-class NMS run in the
app ([`ObjectDetectorHelper.kt`](android/app/src/main/java/com/google/aiedge/examples/objectdetection/ObjectDetectorHelper.kt)),
which keeps meshgrid/exp/gather out of the graph.

## Features

- Real-time **camera** detection + **gallery** image/video detection
- **CPU / GPU** delegate toggle (Compiled Model API)
- **4 selectable model sizes** (Nano / Tiny / S / M)
- Adjustable score threshold; 80 COCO classes with colored boxes
- Letterbox preprocessing + grid/stride decode + per-class NMS in Kotlin

## Build & run

Open `android/` in Android Studio (or `./gradlew :app:installDebug`). The four YOLOX `.tflite` models
are fetched from Hugging Face into `assets/` by `download_model.gradle` on the first build; grant the
camera permission on launch. Default is **YOLOX-S on GPU**.

## Models & conversion

- Models: [`litert-community/yolox-{nano,tiny,s,m}-litert`](https://huggingface.co/litert-community)
  (Apache-2.0).
- Reproduce / modify the conversion: [`conversion/`](conversion) (`build_yolox.py`,
  `validate_decode.py`).

## License

The sample is Apache-2.0. The YOLOX models are Apache-2.0 (Megvii), trained on COCO 2017.
