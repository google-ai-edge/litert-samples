# LiteRT-LM Chatbot: Running Gemma on Google Tensor TPU

A premium **multimodal** on-device chatbot application for Android, powered by **Google LiteRT-LM**. It is designed to run **Gemma** and other LLMs entirely on-device, leveraging **Google Tensor TPU (Tensor Processing Unit)** hardware acceleration on Pixel and compatible devices.

---

## 🌟 Key Features
- **On-Device Inference**: Runs fully offline with no cloud APIs.
- **Hardware Acceleration Fallback**: Automatically tries to initialize the model on **Google Tensor TPU** first. If unsupported, it falls back to **GPU** (OpenCL) and then to **CPU**.
- **Multimodal Support**: Send text messages, select images from the gallery, and **record audio directly within the chat** for a rich multimodal experience.
- **Custom Model Uploads**: Easily add your own custom `.litertlm` model files directly from the app UI! Select your model file, configure the system prompt, choose the preferred backend, and start chatting immediately—no ADB required.
- **Real-time Metrics**: Displays Time To First Token (TTFT) and token generation speed (tokens/sec) dynamically on-screen.
- **Premium Glassmorphic UI**: Enjoy a modern, highly polished chat interface featuring glassmorphic effects, unified circular action buttons, and responsive dynamic elements.
- **Diagnostics Badge**: Displays the active backend (`NPU` (Green), `GPU` (Blue), `CPU` (Red)) currently running the inference engine.

---

## 📱 Project Components
- **`MainActivity.kt`**: Coordinates model selection dropdowns, user message input, image/audio attachments, model upload dialogs, and formats real-time metrics.
- **`LiteRTLMManager.kt`**: Singleton engine coordinator managing the LiteRT-LM `Engine` and `Conversation` lifecycles using the fallback chain.
- **`ModelResolver.kt` & `ModelConfig.kt`**: Handles model configuration, asset copying (copying bundled `.litertlm` files from raw assets), and dynamically registering new user-uploaded models.
- **`ChatAdapter.kt` & `ChatMessage.kt`**: Provides the chat bubbles and UI recyclerview binding.

---

## 🚀 Setup & Installation

### 1. Build and Run
Open the project in Android Studio or compile it from the command line:

```bash
# Build the Debug APK
./gradlew :app:assembleDebug

# Install on a connected device
adb install -r app/build/outputs/apk/debug/app-debug.apk
```
*Note: Make sure `JAVA_HOME` points to JDK 17 or JDK 21 (e.g. Android Studio's bundled JDK).*

### 2. Available Models

The application comes with three lightweight models pre-loaded in the raw assets folder:
- **no models are added yet, we can add any small models like Gemma 3 270M in Assets**
- **we can add any .litertlm models in Assets for Google Tensor TPU, after adding the model we have to update the ModelResolver.kt file and then run the app**


These models are copied to the app's cache directory and loaded on-demand.

#### Google Tensor TPU Gemma 4 Model (Custom Uploads)
To test high-performance **Gemma 4** or larger custom models using the Google Tensor TPU accelerator:
1. Obtain/compile the `.litertlm` model file targeted for Google Tensor TPU.
2. Tap the spinner dropdown in the app header and select **Upload Custom Model**.
3. Pick the `.litertlm` file from your device storage, configure its backend and multimodal settings, and upload it!
4. The model will instantly be added to the list and ready for use.

*(Advanced)* You can also push the model to the app's external files directory using ADB:
```bash
adb push model.litertlm /sdcard/Android/data/com.google.googletensortpu.googleTensorTPUApp/files/model.litertlm
```

### 3. Running Unit Tests

The project includes unit tests for core logical components like `ModelConfig`, `ChatMessage`, and `ModelResolver`. The tests utilize JUnit, Mockito, and Robolectric for Android environment mocking.

To run the unit tests from the command line, execute:
```bash
./gradlew app:testDebugUnitTest
```

**⚠️ Important Environment Note:** 
If you encounter a `java.lang.IllegalArgumentException: 26` error when running Gradle tasks, it means you are using an unsupported version of Java (JDK 26). Ensure your terminal's `JAVA_HOME` points to **JDK 17** or **JDK 21** to be compatible with the Kotlin and Android Gradle plugin versions configured in this project.

---

## ⚙️ How it Works

LiteRT-LM delegates NPU execution through a dynamic vendor library loader. 
- Setting **`Backend.NPU`** with the application's `nativeLibraryDir` instructs the engine to load the hardware-specific dispatch library (e.g. `libLiteRtDispatch_GoogleTensor.so` on Google Tensor devices).
- Legacy packaging (`useLegacyPackaging = true`) and NDK ABI filtering (`arm64-v8a`) are configured in the `app/build.gradle.kts` file to ensure the vendor shared libraries can be directly memory-mapped by the linker without compression errors.
- The project is fully streamlined for **Google Tensor TPU** with unnecessary dynamic features stripped away for optimal performance.

---

## 📜 License
Apache License 2.0
