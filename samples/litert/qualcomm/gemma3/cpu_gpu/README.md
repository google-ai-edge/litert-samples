# Gemma 3 1B On-Device Chat Sample (CPU / GPU)

An Android chat sample demonstrating on-device **Gemma 3 1B IT** inference using **Google LiteRT-LM** on **GPU (OpenCL)** and **CPU** backends.

## Features
- **GPU & CPU Execution**: Prioritizes GPU acceleration with automatic CPU fallback across Android devices without requiring vendor NPU libraries.
- **Flexible Model Provisioning**: Download directly in-app from Hugging Face (with optional HF token configuration) or push via `adb` to scoped app storage.
- **Live Telemetry**: Displays active backend, initialization time, Time-to-First-Token (`TTFT`), and decode throughput (`tokens/s`).

## Quick Start

### 1. Build & Install
```bash
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

### 2. Provision the Model
Download [`gemma3-1b-it-int4.litertlm`](https://huggingface.co/litert-community/Gemma3-1B-IT/blob/main/gemma3-1b-it-int4.litertlm) from Hugging Face and push it to the app's scoped external files directory:
```bash
adb shell mkdir -p /sdcard/Android/data/com.example.gemma3_on_device/files
adb push gemma3-1b-it-int4.litertlm \
  /sdcard/Android/data/com.example.gemma3_on_device/files/
```
Or launch the app and tap **Download Now** in the setup prompt.

## License
Apache License 2.0
