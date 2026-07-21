# **PhotoTalk AI: Co-dependent LiteRT & LiteRT-LM Sample**

**PhotoTalk AI** is an Android sample application demonstrating the side-by-side co-dependent integration of **LiteRT** (for classic machine learning) and **LiteRT-LM** (for Large Language Model orchestration).

---

## 💡 **Architecture & Co-dependency Pattern**

On-device Vision Language Models (VLMs) can be memory-intensive on mobile devices. **PhotoTalk AI** uses a two-stage pipeline to achieve interactive image-based conversation efficiently:

```
┌──────────────────────────┐
│   User Image / Photo     │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  LiteRT (CompiledModel)  │ ◄── EfficientNet / MobileNet Image Classifier (TFLite)
└────────────┬─────────────┘
             │
             │ Extracted Label (e.g. "Electric Guitar", 94%)
             ▼
┌──────────────────────────┐
│        Prompting         │ ◄── Formats context: "User uploaded photo of [Label]..."
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   LiteRT-LM (Engine)     │ ◄── On-Device LLM (Gemma 2B / Gemma 3)
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Interactive Chat Session │ ◄── User asks follow-up Q&A about the image
└──────────────────────────┘
```

1. **LiteRT (Image Classification)**: Uses the LiteRT `CompiledModel` API with hardware acceleration (CPU/GPU) to classify an uploaded image and extract the top detected object label.
2. **Context Handoff**: The detected label and confidence score are injected into the initial system instruction for LiteRT-LM.
3. **LiteRT-LM (Interactive Chat)**: Creates a `Conversation` session using `com.google.ai.edge.litertlm`. The LLM greets the user with insights about the identified object and handles interactive multi-turn streaming responses.

---

## 📂 **Project Structure**

```
phototalk_AI/
├── README.md
└── android/
    ├── app/
    │   ├── build.gradle.kts
    │   └── src/main/java/com/google/aiedge/examples/phototalk/
    │       ├── MainActivity.kt
    │       ├── MainViewModel.kt
    │       ├── ImageClassifierHelper.kt  # LiteRT CompiledModel classifier
    │       ├── LiteRtLmHelper.kt         # LiteRT-LM Engine & Conversation manager
    │       └── ui/                       # Jetpack Compose UI
    ├── build.gradle.kts
    ├── settings.gradle.kts
    └── gradle/libs.versions.toml
```

---

## 🛠️ **Getting Started**

### **Prerequisites**
- Android Studio (Ladybug or later)
- Android device with API level 26+ (Android 8.0+)
- A `.litertlm` model file (e.g., `Gemma 2B` or `Gemma 3`) downloaded from [HuggingFace LiteRT Community](https://huggingface.co/litert-community) pushed to device storage (e.g., `/sdcard/Download/gemma-2b-it.litertlm`).

### **Building & Running**
1. Open `compiled_model_api/phototalk_AI/android` in Android Studio.
2. Build and run the app on an Android device or emulator.
3. Select or capture an image. The app will run LiteRT classification, load LiteRT-LM, and start an interactive chat session.
