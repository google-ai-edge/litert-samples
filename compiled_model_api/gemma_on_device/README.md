# GemmaOnDevice

GemmaOnDevice is an Android application demonstrating on-device inference of Google's Gemma 3 models (Gemma 3 1B and Gemma 3n) using the LiteRT/LM framework. It is specifically optimized to leverage hardware acceleration on the Qualcomm Snapdragon 8 Elite (sm8750) NPU, ensuring high-performance, private, multimodal interactions directly on the Samsung S25 Ultra and similar flagship devices.

## Features
- **On-Device Inference**: No cloud connectivity required. Conversations and data remain fully private.
- **NPU Acceleration**: Configured to prioritize the Snapdragon Hexagon DSP/NPU via Qualcomm's QNN delegate for maximum tokens-per-second and reduced power consumption.
- **Multimodal Support**: Supports sending images and audio alongside text prompts.
- **Seamless Model Management**: Includes an integrated model downloader to fetch quantized Int4 models directly from Hugging Face or detect locally pushed models via ADB.
- **Auto-Fallback**: Automatically falls back to GPU or CPU if NPU execution is unavailable.

## DX (Developer Experience)

### Requirements
- Android Studio Ladybug or later.
- Android SDK 36.
- A physical Android device with a compatible NPU/GPU (e.g., Samsung Galaxy S25 Ultra). Emulator execution will default to CPU and run extremely slowly.

### Building the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/carrycooldude/GemmaOnDevice.git
   ```
2. Open the project in Android Studio.
3. Sync the Gradle files. Ensure `android.useAndroidX=true` and `android.enableJetifier=true` are set in your `gradle.properties`.
4. Select the `app` configuration and click Run (or execute `./gradlew installDebug` from the command line).

### Model Provisioning
By default, the application expects the `gemma3-1b-it-int4.litertlm` container to be available. Due to Android 14+ scoped storage limitations, the most reliable way to provision models during development without using the in-app downloader is to push them directly to the application's secure storage directory.

Push the model via ADB:
```bash
adb push gemma3-1b-it-int4.litertlm /sdcard/Android/data/com.example.gemma_on_device/files/
```
Once pushed, the application will detect the model instantly upon launch.

### Accessing Gated Models
Some models on Hugging Face require authentication. If you choose to use the in-app downloader for gated models, you must provide a valid Hugging Face Access Token in the application settings.

## License
This project is licensed under the MIT License.
