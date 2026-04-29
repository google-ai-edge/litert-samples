# Gemma 4 Models On-Device with NPU acceleration

A premium **multimodal** on-device LLM chat application for Android, powered by **Google LiteRT-LM**. Features **Gemma 4 models** with support for **text, image, and audio** inputs, running entirely on-device with **NPU**.

## Setup & Installation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/carrycooldude/ModelGarden-QNN-LiteRT/blob/main/google_colab/LiteRT_Gemma4_NPU_AOT_Compilation.ipynb)

### Prerequisites
*   Android Studio Ladybug (or newer)
*   Samsung S25 Ultra (or any Android 10+ device with ARM64)
*   ~3GB free storage for the model

### 1. Build & Install the Debug APK
```bash
sh ./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

### 2. Push the Model via ADB (Recommended)

Download the **LiteRT Community Gemma 4** model on your PC from HuggingFace:
```bash
# Download the LiteRT Community Gemma 4 model
Link to the model: https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/blob/main/gemma-4-E2B-it_qualcomm_sm8750.litertlm

# Push to phone
adb push gemma-4-E2B-it_qualcomm_sm8750.litertlm /sdcard/Android/data/com.example.qnn_litertlm_gemma/files/model.litertlm
```

The app will automatically detect the model in `/sdcard/Android/data/com.example.qnn_litertlm_gemma/files/model.litertlm` on launch.

### 3. Usage
1.  **Launch the App**: The app detects the Gemma 4 model and initializes (NPU → GPU → CPU)
2.  **Chat**: Type messages for text-only conversations
3.  **Image Input**: Tap the gallery button to attach an image, then ask about it
4.  **Audio Input**: Tap the mic button to record audio, tap again to stop


## License
Apache 2.0
