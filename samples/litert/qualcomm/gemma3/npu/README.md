# Gemma 3 On-Device Sample (Qualcomm NPU)

This sample demonstrates how to run the **Gemma 3 1B IT** model on Android using the LiteRT-LM framework, optimized for the **Qualcomm NPU (QNN)** backend.

## Features
- **NPU Accelerated**: Optimized for Snapdragon 8 Elite (SM8750) and other QNN-compatible chipsets.
- **Sub-Second Initialization**: Loads and warms up the NPU engine in <700ms.
- **Text-Only Sanitized**: Streamlined UI and engine logic for text interaction.

## Model Setup
1. Download the **NPU-optimized** `Gemma3-1B-IT_q4_ekv1280_sm8750.litertlm` model from [Hugging Face](https://huggingface.co/litert-community/Gemma3-1B-IT).
2. Push the model to your device:
   ```bash
   adb shell mkdir -p /sdcard/Download
   adb push Gemma3-1B-IT_q4_ekv1280_sm8750.litertlm /sdcard/Download/
   ```

## Backends
- **NPU (Qualcomm QNN)**: Primary backend using the QNN HTP delegate.
- **GPU (LITERT_CL)**: Fallback backend.
- **CPU**: Secondary fallback.

## Getting Started
1. Open this project in Android Studio.
2. Build and run on a Qualcomm Snapdragon device (S25 Ultra recommended).
3. Ensure the `.litertlm` file matches the hardware version (e.g., SM8750).
