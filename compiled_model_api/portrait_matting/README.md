# LiteRT Portrait Matting Sample

This directory contains an Android portrait-matting sample demonstrating how to use LiteRT (Google's
runtime for TensorFlow Lite) with the Compiled Model API. It runs
[MODNet](https://github.com/ZHKKKe/MODNet) to predict a soft alpha matte of a person from a single
RGB image — no trimap, no green screen — and composites the foreground onto a transparent background.

## Overview

MODNet (Ke et al., AAAI 2022) is a pure CNN (a MobileNetV2 backbone with a lightweight conv decoder)
that predicts a single-channel **soft alpha matte** in `[0, 1]`, capturing hair-level detail at the
subject boundary. The foreground is composited using that matte as the alpha channel to produce the
cut-out shown in the app.

The model is converted with [`litert-torch`](https://github.com/google-ai-edge/ai-edge-torch) and runs
entirely on the Compiled Model API's CPU or GPU delegate — every op is GPU-native, so no part of the
graph falls back to the CPU.

## Available Implementations

### 1. kotlin_cpu_gpu

A standard implementation utilizing the Compiled Model API with support for CPU and GPU delegates.

**Features:**
-   **Portrait matting**: Produces a transparent cut-out of the person with a soft alpha matte.
-   **Real-time inference**: Mattes the camera feed or a gallery image (Camera and Gallery tabs).
-   **Hardware acceleration**: Switch between CPU and GPU delegates at runtime.
-   **Fully GPU-native CNN**: The entire MODNet graph runs on the Compiled Model GPU delegate.
-   **Jetpack Compose**: Modern Android UI toolkit.

### Screenshots

| Live portrait matting |
| :---: |
| ![Portrait Matting](https://huggingface.co/mlboydaisuke/modnet-litert/resolve/main/app_screenshot.jpg) |

## Technical Details

### Model Architecture
-   **Task**: Trimap-free portrait matting.
-   **Model**: MODNet (`modnet_512_fp16.tflite`), converted with `litert-torch` and float16-quantized.
-   **Input**: `[1, 512, 512, 3]` NHWC float32, RGB in `[0, 1]`. The `[0,1] → [-1,1]` normalization
    (the official `Normalize(0.5, 0.5)`) is baked into the graph, so the app feeds a plain `[0, 1]`
    image.
-   **Output**: `[1, 512, 512, 1]` alpha matte in `[0, 1]` (`1` = person, `0` = background). The matte
    is upscaled to the source size and used as the alpha channel of the cut-out.
-   **Format**: TensorFlow Lite (`.tflite`), float16 weights (~13.4 MB).

### GPU compatibility notes
MODNet is a pure CNN (Conv2d / IBNorm / ReLU / SE block / bilinear-upsample / concat / sigmoid) with no
attention, no PixelShuffle, and no `grid_sample`, so all intermediate tensors stay 4-D. Converting with
`litert-torch` is clean apart from two trace-friendly rewrites of the SE block (same math, same weights,
no converter fork): the global pool is emitted as a spatial-mean reduce rather than `AdaptiveAvgPool2d`
(whose NHWC relayout introduced a `GATHER_ND`, a banned GPU op), and the channel scale is a broadcast
multiply rather than `expand_as`. The result compiles to **515 / 515 nodes on the LiteRT GPU delegate
(LITERT_CL)** — full GPU residency, a single partition, no CPU fallback (~4.3 ms/frame on a Pixel 8a).
The float16 weights add `DEQUANTIZE` ops that the GPU delegate consumes natively.

### Key Dependencies
-   **LiteRT** (`com.google.ai.edge.litert:litert:2.1.5`)
-   **Jetpack Compose** (UI)
-   **CameraX** (Camera feed)

### Architecture Components
-   **`PortraitMattingHelper`**: Initializes the Compiled Model (CPU/GPU), preprocesses the image,
    runs MODNet, and composites the foreground onto transparency.
-   **`MainActivity`**: Setup of the main screen and UI components.
-   **`CameraScreen` / `GalleryScreen`**: Composables for the camera preview and gallery input.
-   **`MainViewModel`**: Manages UI state and communicates between the UI and the Helper.

## Getting Started

1.  Open `kotlin_cpu_gpu/android` in Android Studio.
2.  The MODNet model (`modnet_512_fp16.tflite`) is downloaded into `app/src/main/assets/` at build time
    by `download_model.gradle`.
3.  Build and run the application on an Android device.
4.  Point the camera at a person (or pick a portrait from the Gallery tab).
5.  Observe the matted cut-out with the background removed.
6.  Use the settings sheet to switch between CPU and GPU acceleration.

## Model conversion & license

MODNet is licensed under **Apache-2.0** (© the MODNet authors,
[ZHKKKe/MODNet](https://github.com/ZHKKKe/MODNet)). The `.tflite` is a format conversion of the official
`modnet_photographic_portrait_matting` weights, produced with `litert-torch`; all credit to the original
authors.
