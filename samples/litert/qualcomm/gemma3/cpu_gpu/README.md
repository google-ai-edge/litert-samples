# Gemma 3 On-Device Sample (CPU/GPU)

This sample demonstrates how to run the **Gemma 3 1B IT** model on Android using the LiteRT-LM framework, optimized for **CPU** and **GPU** (OpenCL) backends.

## Features
- **Strictly Non-NPU**: Uses GPU (OpenCL) and CPU backends for maximum compatibility.
- **Multimodal Ready (Text-only UI)**: Engineered with LiteRT-LM for high-performance inference.
- **Dynamic Initialization**: Automatic fallback from GPU to CPU if needed.

## Model Setup
1. Download the `gemma3-1b-it-int4.litertlm` model from [Hugging Face](https://huggingface.co/litert-community/Gemma3-1B-IT).
2. Push the model to your device's internal storage:
   ```bash
   adb shell mkdir -p /sdcard/Download
   adb push gemma3-1b-it-int4.litertlm /sdcard/Download/
   ```
   Or push directly to the app's files directory:
   ```bash
   adb push gemma3-1b-it-int4.litertlm /data/local/tmp/
   adb shell run-as com.example.gemma3_on_device cp /data/local/tmp/gemma3-1b-it-int4.litertlm files/
   ```

## Backends
- **GPU (LITERT_CL)**: Uses OpenCL acceleration.
- **CPU**: Standard LiteRT CPU execution.

## Getting Started
1. Open this project in Android Studio.
2. Build and run on a device with OpenCL support (e.g., Samsung S25 Ultra).
3. The app will automatically detect the model in `/sdcard/Download` or its local files directory.
