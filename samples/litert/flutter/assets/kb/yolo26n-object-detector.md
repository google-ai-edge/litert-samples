---
title: YOLO26n object detector for LiteRT
source: tool/prune_yolo26n_head.py ; tool/models.lock ; tool/fetch_models.sh ; lib/data/services/detector/detector_codec.dart ; lib/data/services/detector/detector_engine.dart ; lib/domain/models/detector_spec.dart ; lib/config/live_camera_config.dart
license: Apache-2.0
---

# YOLO26n object detector for LiteRT

This document describes the object detector that the LiteRT Demos app runs on the live camera: what YOLO26n is, the input size and preprocessing it expects, the raw output and how the app decodes it, why the app removes the model's built-in selection head, how the detector is held to a strict GPU fp32 configuration, the latency measured on a Mac, and where the model file comes from.

## What the YOLO26n detector is

YOLO26n is the nano size of Ultralytics YOLO26, a real-time object detector trained on COCO. It recognises the 80 COCO object classes, such as person, cat, cup, laptop and cell phone. The app's class list uses the Darknet spellings of the class names that ship with the converted model, for example `motorbike`, `aeroplane`, `sofa`, `pottedplant`, `diningtable` and `tvmonitor`.

The weights come from Arm's LiteRT conversion of YOLO26n, published on Hugging Face as `Arm/yolo26n-fp16-litert` (file `yolo26n_conv2d_f16_weights.tflite`). The model is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0), and so is the file the app derives from it. An app that distributes the file must make its corresponding source available under AGPL-compatible terms, including the script that derives the file.

YOLO26 is trained with one-to-one label assignment, so it predicts at most one box per object and needs no non-maximum suppression (NMS).

## Input size of the YOLO26 nano detector

The YOLO26n detector takes one image of 640 × 640 pixels. The input tensor has shape `[1, 3, 640, 640]`: batch 1, three colour planes, 640 rows and 640 columns, in NCHW layout (channels first). The values are float32.

The app prepares each camera frame like this:

- Rotate the frame upright first, so boxes come out upright.
- Letterbox it into the 640 × 640 square: scale by `min(640 / width, 640 / height)` so the whole frame fits, centre it, and fill the bars with the grey value 114 (114 / 255 ≈ 0.447 after normalisation), the padding Ultralytics uses.
- Write the pixels as separate R, G and B planes, in RGB order, not BGR.
- Divide every value by 255, so each channel lies in [0, 1]. There is no mean or standard-deviation normalisation.

A 640 × 480 camera frame fits at scale 1.0, with an 80-pixel bar above and below. Feeding 0–255 values instead of 0–1 saturates every score at 1.0, and stretching instead of letterboxing loses detections, so both are bugs rather than tuning choices.

## The raw output tensor and top-k decoding in Dart

The file the app runs has a single output of shape `[1, 8400, 84]`, float32. Each of the 8,400 rows is one anchor position (an 80 × 80, a 40 × 40 and a 20 × 20 grid). A row holds four box coordinates `x1, y1, x2, y2` — corners in 640 × 640 input pixels, with the grid strides already applied — followed by 80 per-class scores that have already been through a sigmoid.

The app decodes this in Dart. It keeps every (anchor, class) pair whose score is at least 0.25, sorts them by score, keeps the best 100, and maps each box back from the letterboxed square to the upright frame: subtract the padding, divide by the scale, and clamp to the frame, because the coordinates can overshoot the edge slightly. It applies no NMS. Boxes drawn on the live view need a score of at least 0.35.

On the CPU this decode gives the same detections as the selection head it replaces. It takes about 0.75 ms per frame in AOT-compiled Dart on an Apple M4 Pro.

## Why the app cuts off the model's selection head

Arm's file ends in an in-graph selection head that turns the 8,400 candidates into a fixed list of 300 detections, shaped `[1, 300, 6]`. That head uses TopK and GatherND and works on int64 tensors, and the LiteRT GPU delegate supports neither. The file also declares its ADD operator at version 4 because of the int64 head, and the Metal GPU delegate in the LiteRT build that `flutter_litert` uses on macOS accepts ADD only up to version 2.

As a result the original file never runs fully on the GPU. Compiled for the GPU alone, it fails to build. Compiled for GPU plus CPU, it builds, but only 54 of its 461 operations go to the GPU on macOS, and it runs at CPU speed with no error — a silent fallback.

The app therefore derives its own file, `yolo26n_fp16_rawhead.tflite`, with `tool/prune_yolo26n_head.py`:

- It keeps operations 0 to 410 — the backbone and the Detect head — unchanged, with the same weights.
- It makes the `[1, 8400, 84]` tensor the model's only output and drops the selection head after it.
- It sets the shared ADD opcode to version 1, after checking that every remaining ADD is float32, which makes the change exact.

Every remaining operation is float32 and supported by the GPU delegate, so a GPU-only build either takes the whole graph or fails outright. The top-k selection the head used to do now runs in Dart.

## Strict GPU fp32 and the full-acceleration check

The detector runs through `flutter_litert`'s `CompiledModel` with exactly the accelerator that was asked for — the GPU alone, or the CPU alone when the user chooses it — never a policy that retries on the CPU. Loading performs these checks in order, and any failure is an error shown in the app rather than a quiet switch to another backend:

1. The file must have the raw-head file's exact size, which tells it apart from Arm's original.
2. The model is built with the requested accelerator at fp32 precision. A GPU that rejects the graph is an error.
3. On the GPU, the compiled model must report the accelerator set that was requested, no fallback, and `isFullyAccelerated` true — the whole graph on the GPU, nothing left on the CPU.
4. The input and output byte sizes must match `[1, 3, 640, 640]` and `[1, 8400, 84]` float32.
5. One run on a test input is compared against a plain CPU reference, within 1% of the output range.
6. One warm-up run on an all-padding frame.

The precision is fp32 because fp16 failed the comparison: on the GPU at fp16 the output deviated by 1.47% of its range, above the 1% tolerance, and boxes moved by up to 0.84 px. At fp32 the largest deviation was about 0.002 on an output range of about 864, and the detections matched the CPU reference within 0.1 px.

## Measured detector latency on an Apple M4 Pro

These figures were measured on an Apple M4 Pro Mac under macOS, as median and 90th percentile over repeated runs after warm-up:

| Configuration | Median | p90 |
| --- | --- | --- |
| Raw-head file, `flutter_litert`, strict GPU fp32, including the input and output copies | 4.11 ms | 4.79 ms |
| Raw-head file, `flutter_litert`, CPU | 22.8 ms | 25.8 ms |
| Arm's original file, `flutter_litert`, GPU plus CPU (54 of 461 operations on the GPU) | 21.6 ms | — |
| Raw-head file, Python LiteRT 2.2.0, strict GPU fp32 | 3.47 ms | — |
| Whole frame in Dart (gather, run, decode; JIT) | 9.3 ms | 11.0 ms |

The GPU makes the raw-head file about 5.5 times faster than the CPU on this machine. Creating the compiled model for the GPU takes about 296 ms the first time in a process and about 50 ms after that. The live camera feeds the detector at most 15 frames per second, and a frame that arrives while the detector is busy is dropped rather than queued, so the detector always works on the newest frame.

## Where the YOLO26n model file comes from

The model file is not checked into the repository. `tool/fetch_models.sh` builds it before the first build, as recorded in `tool/models.lock`:

1. It downloads Arm's `yolo26n_conv2d_f16_weights.tflite` from `Arm/yolo26n-fp16-litert` at a pinned commit and checks its size and SHA-256.
2. It creates a Python virtual environment with the pinned `ai-edge-litert` and `flatbuffers` wheels.
3. It runs `tool/prune_yolo26n_head.py` on the download.
4. It checks that the result is byte for byte the expected file: 10,361,332 bytes with SHA-256 `5ddd5eebad18587d56500a30b0995c07c9e1a241640750f568e4a974f66ed80a`.

The derived file is placed in `assets/models/`, bundled into the app as a Flutter asset and loaded from memory, so the app downloads nothing at run time. The app's licence screen and `assets/models/NOTICE.md` carry the AGPL-3.0 notice.
