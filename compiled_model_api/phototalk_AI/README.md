# **PhotoTalk Sample App: Co-dependent LiteRT & LiteRT-LM Sample**

**PhotoTalk Sample App** is an Android sample application demonstrating the side-by-side co-dependent integration of **LiteRT** (for classic machine learning image classification) and **LiteRT-LM** (for Large Language Model orchestration).

---

## **Architecture & Co-dependency Pattern**

On-device Vision Language Models (VLMs) can be memory-intensive on mobile devices. **PhotoTalk Sample App** uses a two-stage pipeline to achieve interactive image-based conversation efficiently:

```
┌──────────────────────────┐
│   User Image / Photo     │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  LiteRT (CompiledModel)  │ ◄── EfficientNet-Lite0 Image Classifier (TFLite)
└────────────┬─────────────┘
             │
             │ Extracted Label (e.g. "Electric Guitar", 94%)
             ▼
┌──────────────────────────┐
│        Prompting         │ ◄── Concise context: "User uploaded photo of [Label]..."
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   LiteRT-LM (Engine)     │ ◄── On-Device LLM (Gemma 4 / Gemma 2B / Gemma 3)
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Interactive Chat Session │ ◄── Concise, interactive multi-turn Q&A about the image
└──────────────────────────┘
```

1. **LiteRT (Image Classification)**: Uses the LiteRT `CompiledModel` API with **EfficientNet-Lite0** (`efficientnet_lite0.tflite`) to classify an uploaded image and extract the top detected object label.
2. **Context Handoff**: The detected label and confidence score are injected into the initial system instruction for LiteRT-LM.
3. **LiteRT-LM (Interactive Chat)**: Creates a `Conversation` session using `com.google.ai.edge.litertlm:0.10.2`. The LLM greets the user with concise insights about the identified object and handles interactive multi-turn streaming responses.

> **Note**: Gemma 4 models (e.g., `gemma-4-E2B-it.litertlm`) work exceptionally well with LiteRT-LM for fast, accurate response generation on device.

---

## **Vision Classification Model: EfficientNet-Lite0**

The image classification component uses **EfficientNet-Lite0** (`efficientnet_lite0.tflite`):

* **Model Family**: EfficientNet-Lite (developed by Google AI) optimized for edge and mobile acceleration.
* **Input Resolution**: `224x224x3` RGB pixels normalized between `[-1.0, 1.0]`.
* **Dataset**: Pretrained on ImageNet-1k (1,000 object categories).
* **Automatic Download**: Downloaded automatically during build via the Gradle task `downloadEfficientnetLite0Model` (`download_model.gradle`) from Google Cloud Storage (`storage.googleapis.com/ai-edge/...`).
* **Execution**: Executed through LiteRT's `CompiledModel` API (`com.google.ai.edge.litert.CompiledModel`) using CPU or GPU hardware delegates.

---

## **Project Structure**

```
phototalk_AI/
├── README.md
└── android/
    ├── app/
    │   ├── build.gradle.kts
    │   ├── download_model.gradle        # Automated download for EfficientNet-Lite0
    │   └── src/main/
    │       ├── AndroidManifest.xml
    │       ├── res/                     # Resources & LiteRT branding
    │       └── java/com/google/aiedge/examples/phototalk/
    │           ├── MainActivity.kt
    │           ├── MainViewModel.kt
    │           ├── ImageClassifierHelper.kt  # LiteRT CompiledModel classifier
    │           ├── LiteRtLmHelper.kt         # LiteRT-LM Engine & Conversation manager
    │           └── ui/                       # Jetpack Compose UI (PhotoTalkAppScreen.kt)
    ├── build.gradle.kts
    ├── settings.gradle.kts
    └── gradle/libs.versions.toml
```

---

## **Getting Started**

### **Prerequisites**
- Android Studio (Ladybug or later)
- Android device with API level 26+ (Android 8.0+)
- A `.litertlm` model file (Gemma 4 models like `gemma-4-E2B-it.litertlm` are recommended) placed in your device's `/sdcard/Download/` folder or selectable via the file picker.

### **Building & Running**
1. Open `compiled_model_api/phototalk_AI/android` in Android Studio or run `./gradlew installDebug`.
2. Launch **PhotoTalk Sample App**.
3. Tap **Select Image** to upload a photo. The app will run LiteRT vision classification, initialize LiteRT-LM, and start an interactive conversation!
