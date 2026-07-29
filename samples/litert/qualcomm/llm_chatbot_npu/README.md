# LLM Chatbot: Running Gemma 4/ FastVLM Models On-Device with NPU acceleration

A premium **multimodal** on-device LLM chat application for Android, powered by **Google LiteRT-LM**. Features **Gemma 4 models** and **FastVLM** with support for **text, image, and audio** input, running entirely on-device with **NPU**.

## Setup & Installation

### Prerequisites
*   Latest Android Studio (Panda4 or newer)
*   Samsung S25 Ultra

### 1. Build & Install the Debug APK
```bash
sh ./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

### 2. Push the Model via ADB (Recommended)

Download the **LiteRT Community Gemma 4** model on your PC from HuggingFace:
```bash
# Download the LiteRT Community Gemma 4 model (Use sm8750 version)
Link to the Gemma 4 model: https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/blob/main/gemma-4-E2B-it_qualcomm_sm8750.litertlm

# Download the LiteRT Community FastVLM model (Use sm8750 version)
Link to the FastVLM model: https://huggingface.co/litert-community/FastVLM-0.5B/blob/main/FastVLM-0.5B.qualcomm.sm8750.litertlm

# Push to phone
adb push gemma-4-E2B-it_qualcomm_sm8750.litertlm /sdcard/Android/data/com.example.qnn_litertlm_gemma/files/model.litertlm
adb push FastVLM-0.5B.qualcomm.sm8750.litertlm /sdcard/Android/data/com.example.qnn_litertlm_gemma/files/FastVLM-0.5B.qualcomm.sm8750.litertlm
```

The app will automatically detect the model in `/sdcard/Android/data/com.example.qnn_litertlm_gemma/files/model.litertlm` or `/sdcard/Android/data/com.example.qnn_litertlm_gemma/files/FastVLM-0.5B.qualcomm.sm8750.litertlm` on launch.

### 3. Usage
1.  **Chat**: Type messages for text-only conversations
2.  **Image Input**: Tap the gallery button to attach an image, then ask about it
3.  **Audio Input**: Tap the mic button to record audio, tap again to stop (Gemma4 2B model only)


## License
Apache 2.0
