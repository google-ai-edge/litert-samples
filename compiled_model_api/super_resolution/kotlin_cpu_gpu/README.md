# LiteRT Super Resolution Sample (Real-ESRGAN)

<p align="center"> <img src="https://huggingface.co/litert-community/real-esrgan-x4v3-litert/resolve/main/samples/sample.png" width="320" alt="Real-ESRGAN x4 super-resolution on LiteRT GPU"> </p>

Android **×4 super-resolution** sample on the LiteRT (Google's TensorFlow Lite runtime) **Compiled Model API**, CPU/GPU. Pick an image and drag the divider to compare **bicubic vs Real-ESRGAN**. Model: **[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) realesr-general-x4v3** (SRVGGNetCompact, BSD-3) — a pure CNN re-authored **GPU-native** with `litert_torch`.

## Overview

- Model `litert-community/real-esrgan-x4v3-litert` (downloaded at build time), FP16, **3.5 MB**.
- Input 128×128 → output 512×512 (×4). Larger images are tiled.
- **GPU-clean** (full `LITERT_CL` residency): the stock conversion's PReLU (→GREATER/SELECT) and PixelShuffle (→>4-D) are re-authored to a relu-form PReLU and a ZeroStuffConvT pixel-shuffle — see [`conversion/`](conversion). On a Pixel 8a the graph runs entirely on the GPU delegate (**211/211 nodes, ~1 ms**) and the GPU output matches the CPU/PyTorch reference (corr ≈ 0.995).

## Features
- Gallery image pick; **before/after compare slider** (bicubic vs Real-ESRGAN)
- **CPU / GPU** delegate toggle (Compiled Model API); inference-time readout

## Build & run
Open `android/` in Android Studio (or `./gradlew :app:installDebug`). The model is fetched from Hugging Face into `assets/` by `download_model.gradle` on first build. Default: GPU.

## Models & conversion
- Model: [`litert-community/real-esrgan-x4v3-litert`](https://huggingface.co/litert-community/real-esrgan-x4v3-litert) (BSD-3).
- Reproduce/modify: [`conversion/`](conversion) (`convert_realesrgan.py`).

## License
Sample Apache-2.0. The Real-ESRGAN model is BSD-3-Clause; trained on public SR datasets.
