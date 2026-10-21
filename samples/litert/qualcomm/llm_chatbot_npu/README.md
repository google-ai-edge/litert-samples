# Multimodal LLM Chatbot on Qualcomm NPU (Gemma 4 E2B & FastVLM 0.5B)

An Android chat sample powered by **Google LiteRT-LM** and the **Qualcomm AI Runtime (QNN HTP)** backend, running **Gemma 4 E2B** (text, image, and 16 kHz PCM audio) and **FastVLM 0.5B** (text + image) completely on-device on Snapdragon 8 Elite (`SM8750`) hardware.

## Features
- **Qualcomm Hexagon NPU Acceleration**: Runs pre-compiled `.litertlm` packages with NPU-accelerated text & vision execution and automatic `NPU -> GPU -> CPU` fallback.
- **Multimodal Chat**: Supports text prompts, gallery image attachments, and live 16 kHz mono WAV voice recording (for Gemma 4 E2B).
- **Resilient Conversation Management**: Preserves system instructions, automatically recovers from NPU static KV-cache context limits without corrupting turn alternation, and includes one-tap session reset.
- **Live Telemetry**: Displays active backend, engine load latency, Time-to-First-Token (`TTFT`), and decode throughput (`tokens/s`).

## Prerequisites
- **Android Studio** Ladybug or newer (`compileSdk = 36`, JDK 17)
- **Qualcomm Snapdragon 8 Elite (`SM8750`)** device (e.g., Samsung Galaxy S25 series) with USB debugging enabled
- `adb` installed on your workstation

## Quick Start

### 1. Download Pre-Compiled SM8750 `.litertlm` Models
Download one or both pre-compiled Qualcomm `SM8750` models from Hugging Face:
- **Gemma 4 E2B IT (Text + Image + Audio)**:
  [`gemma-4-E2B-it_qualcomm_sm8750.litertlm`](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/blob/main/gemma-4-E2B-it_qualcomm_sm8750.litertlm)
- **FastVLM 0.5B (Text + Image)**:
  [`FastVLM-0.5B.qualcomm.sm8750.litertlm`](https://huggingface.co/litert-community/FastVLM-0.5B/blob/main/FastVLM-0.5B.qualcomm.sm8750.litertlm)

### 2. Build, Install, Push Model, and Launch
Use the helper script:
```bash
./run.sh --model /path/to/gemma-4-E2B-it_qualcomm_sm8750.litertlm
```

Or run the steps manually:
```bash
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk

adb shell mkdir -p /sdcard/Android/data/com.example.qnn_litertlm_gemma/files
adb push gemma-4-E2B-it_qualcomm_sm8750.litertlm \
  /sdcard/Android/data/com.example.qnn_litertlm_gemma/files/
adb push FastVLM-0.5B.qualcomm.sm8750.litertlm \
  /sdcard/Android/data/com.example.qnn_litertlm_gemma/files/
```
> **Note**: Both `gemma-4-E2B-it_qualcomm_sm8750.litertlm` and legacy `model.litertlm` filenames are detected automatically in `/sdcard/Android/data/com.example.qnn_litertlm_gemma/files/`.

### 3. Updating Qualcomm QNN Runtime Libraries (Optional)
Prebuilt SM8750 (`v79`) QNN runtime libraries are included under `app/src/main/jniLibs/arm64-v8a/`. To refresh them or target another Hexagon DSP architecture (`69`, `73`, `75`, `79`, `81`):
```bash
./fetch_qualcomm_library.sh --hexagon-version 79
# Or from an existing local QAIRT SDK installation:
./fetch_qualcomm_library.sh --qairt-dir /path/to/qairt/2.44.0.260225 --hexagon-version 79
```

See [NPU_COMPILATION_GUIDE.md](NPU_COMPILATION_GUIDE.md) for details on `.litertlm` pre-compiled NPU packages vs. standalone `.tflite` AOT compilation with `compile_npu.py`.

## License
Apache License 2.0
