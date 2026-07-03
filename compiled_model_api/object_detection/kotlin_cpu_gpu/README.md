# LiteRT Object Detection Sample (YOLOX + RF-DETR)

<p align="center">
  <img src="https://huggingface.co/litert-community/yolox-s-litert/resolve/main/demo.png" width="300" alt="Real-time object detection on LiteRT GPU">
</p>

This directory contains an Android **real-time object detection** sample demonstrating the LiteRT
(Google's runtime for TensorFlow Lite) **Compiled Model API** on CPU and GPU. It runs live on the
camera and on gallery images, drawing labelled COCO bounding boxes, with **two selectable detector
families**:

- **[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)** (Megvii, Apache-2.0) — a
  permissively-licensed pure-CNN alternative to the AGPL/GPL YOLO family, in four sizes.
- **[RF-DETR Nano](https://github.com/roboflow/rf-detr)** (Roboflow, Apache-2.0) — a two-stage
  **transformer** DETR (windowed DINOv2-S backbone + deformable-attention decoder), running fully on
  the GPU as a two-graph split.

Both are re-authored to **GPU-native** `.tflite` with
[`litert-torch`](https://github.com/google-ai-edge/ai-edge-torch) and downloaded from
[`litert-community`](https://huggingface.co/litert-community) at build time.

## Overview

Five models are selectable in the app:

| Model | Input | FP16 size | COCO AP | Best on Pixel 8a GPU |
|-------|-------|-----------|---------|----------------------|
| YOLOX-Nano | 416 | 2.2 MB | 25.8 | ~2.6 ms |
| YOLOX-Tiny | 416 | 10.4 MB | 32.8 | ~1.2 ms |
| **YOLOX-S** (default) | 640 | 18.2 MB | 40.5 | ~1.7 ms |
| YOLOX-M | 640 | 51.0 MB | 46.9 | ~2.5 ms |
| RF-DETR Nano (2 graphs) | 384 | 54 MB | 48.4 | ~110 ms/frame end-to-end (~9 fps) |

### YOLOX (single graph)

The models are **GPU-clean** (full `LITERT_CL` residency — no GATHER_ND / TopK / Cast, no >4-D
tensors). The single conversion trick is folding YOLOX's `Focus` stem (stride-2 space-to-depth, which
otherwise lowers to GPU-rejected `GATHER_ND`) into one numerically-exact 6×6 stride-2 conv — see
[`conversion/`](conversion). On device the GPU output matches the CPU/PyTorch reference (correlation
≥ 0.999); residency **and** correctness both hold.

The model graph outputs **raw detection heads**; the grid + stride decode and per-class NMS run in the
app ([`ObjectDetectorHelper.kt`](android/app/src/main/java/com/google/aiedge/examples/objectdetection/ObjectDetectorHelper.kt)),
which keeps meshgrid/exp/gather out of the graph.

### RF-DETR Nano (two-graph transformer split)

Transformer detectors don't ride the GPU API off-the-shelf: the two-stage query selection is
`TOPK`/`GATHER` (no GPU op), deformable attention's `grid_sample` lowers to `GATHER_ND`, and windowed
attention produces 5-D/6-D tensors. Here each of those is re-authored or split out, so the whole
detector runs on the GPU as two graphs with a tiny host step between them (the standard
two-stage-DETR edge split):

```
image →[GPU Graph A]→ enc_class, enc_coord, memory        (backbone + encoder + proposal heads)
      →[host: top-300 by max class score → gather coords]→ refpoint_ts
      →[GPU Graph B (memory, refpoint_ts)]→ boxes, logits  (two-stage combine + decoder + heads)
      →[host: sigmoid + threshold + cxcywh→xyxy + per-class NMS]→ detections
```

On a Pixel 8a both graphs are fully GPU-resident (Graph A `1381/1381`, Graph B `404/404` nodes on
`LITERT_CL`) and the live camera runs at **~9 fps** — a transformer detector entirely on the GPU. On
a real image the device chain reproduces the PyTorch detections at IoU 0.98–0.99. See
[`conversion/rf_detr/`](conversion/rf_detr) for the re-authoring (windowed attention ≤4-D, deformable
`grid_sample` → tent-matmul, baked sine pos-embed) and the fp16-safe LayerNorm fixes the Mali delegate
needs. RF-DETR works best with the score threshold around **0.45**.

## Features

- Real-time **camera** detection + **gallery** image/video detection
- **CPU / GPU** delegate toggle (Compiled Model API)
- **5 selectable models**: YOLOX Nano / Tiny / S / M and RF-DETR Nano
- Adjustable score threshold; 80 COCO classes with colored boxes
- All pre/post-processing in Kotlin: letterbox + grid/stride decode (YOLOX), square resize +
  topk/gather between graphs + DETR decode (RF-DETR), per-class NMS

## Build & run

Open `android/` in Android Studio (or `./gradlew :app:installDebug`). The model files are fetched
from Hugging Face into `assets/` by `download_model.gradle` on the first build; grant the camera
permission on launch. Default is **YOLOX-S on GPU**.

## Models & conversion

- YOLOX: [`litert-community/yolox-{nano,tiny,s,m}-litert`](https://huggingface.co/litert-community)
  (Apache-2.0). Reproduce / modify: [`conversion/`](conversion) (`build_yolox.py`,
  `validate_decode.py`).
- RF-DETR Nano:
  [`litert-community/RF-DETR-Nano-LiteRT`](https://huggingface.co/litert-community/RF-DETR-Nano-LiteRT)
  (Apache-2.0). Reproduce / modify: [`conversion/rf_detr/`](conversion/rf_detr)
  (`build_rfdetr_split.py`).

## License

The sample is Apache-2.0. The YOLOX models are Apache-2.0 (Megvii) and the RF-DETR Nano model is
Apache-2.0 (Roboflow), both trained on COCO 2017.
