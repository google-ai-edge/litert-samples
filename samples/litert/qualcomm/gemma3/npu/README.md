# Gemma 3 1B On-Device Chat Sample (Qualcomm NPU)

An Android chat sample demonstrating on-device **Gemma 3 1B IT** inference using **Google LiteRT-LM** accelerated on the **Qualcomm Hexagon NPU (QNN HTP)** backend.

## Features
- **Qualcomm NPU Accelerated**: Sub-second model initialization and high-throughput token generation on Snapdragon 8 Elite (`SM8750`) hardware with automatic `NPU -> GPU -> CPU` fallback.
- **Scoped-Storage Ready Provisioning**: Supports pushing the pre-compiled `.litertlm` package via `adb` with zero storage permissions OR downloading directly inside the app from Hugging Face.
- **Resilient Context Handling**: Automatically resets and recovers conversation state when the static NPU KV-cache window (`1,280` tokens in `ekv1280`) fills up, plus one-tap session reset in the header bar.

## Prerequisites
- **Android Studio** Ladybug or newer (`compileSdk = 36`, JDK 17)
- **Qualcomm Snapdragon 8 Elite (`SM8750`)** device (e.g., Samsung Galaxy S25 series)
- `adb` installed on your workstation

## Quick Start

### Option A: One-Step Build, Push, and Launch
1. Download [`Gemma3-1B-IT_q4_ekv1280_sm8750.litertlm`](https://huggingface.co/litert-community/Gemma3-1B-IT/blob/main/Gemma3-1B-IT_q4_ekv1280_sm8750.litertlm) from Hugging Face.
2. Run:
   ```bash
   ./run.sh --model /path/to/Gemma3-1B-IT_q4_ekv1280_sm8750.litertlm
   ```

### Option B: Manual Build & ADB Push
Push the model into the app's scoped external files directory (works on Android 13–16 without extra storage permissions):
```bash
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk

adb shell mkdir -p /sdcard/Android/data/com.example.gemma3_npu/files
adb push Gemma3-1B-IT_q4_ekv1280_sm8750.litertlm \
  /sdcard/Android/data/com.example.gemma3_npu/files/
```
Alternatively, launch the app first and tap **Download Model** on the setup screen (supports optional Hugging Face token authentication for gated repositories).

### Updating Qualcomm QNN Runtime Libraries (Optional)
Prebuilt SM8750 (`v79`) QNN runtime libraries are included under `app/src/main/jniLibs/arm64-v8a/`. To refresh them or target another Hexagon DSP architecture (`69`, `73`, `75`, `79`, `81`):
```bash
./fetch_qualcomm_library.sh --hexagon-version 79
```

See [NPU_COMPILATION_GUIDE.md](NPU_COMPILATION_GUIDE.md) for details on `.litertlm` pre-compiled NPU packages vs. standalone `.tflite` AOT compilation with `compile_npu.py`.

## License
Apache License 2.0
