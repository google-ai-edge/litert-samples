# Gemma 4 E2B Multimodal On-Device Chat Sample (CPU / GPU)

An Android multimodal chat sample demonstrating on-device **Gemma 4 E2B IT** inference (text, image, and 16 kHz PCM audio) using **Google LiteRT-LM** on **GPU** and **CPU** backends.

## Features
- **Multimodal On-Device Inference**: Supports text prompts, gallery image attachments, and live 16 kHz mono WAV audio recording completely offline.
- **GPU & CPU Fallback**: Automatically initializes text and vision executors on GPU with fallback to CPU.
- **Flexible Model Provisioning**: Download directly in-app from Hugging Face (with optional HF token authentication) or push via `adb` to scoped app storage.

## Quick Start

### 1. Build & Install
```bash
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

### 2. Provision the Model
Download [`gemma-4-E2B-it.litertlm`](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/blob/main/gemma-4-E2B-it.litertlm) from Hugging Face and push it to the app's scoped external files directory:
```bash
adb shell mkdir -p /sdcard/Android/data/com.example.gemma_on_device/files
adb push gemma-4-E2B-it.litertlm \
  /sdcard/Android/data/com.example.gemma_on_device/files/
```
Or launch the app and tap **Download Now** in the setup prompt.

## License
Apache License 2.0
