# LiteRT-LM Multimodal Chatbot: Google Tensor TPU Reference Implementation

This repository provides a reference architecture and framework for building high-performance, **on-device multimodal LLM applications** on Android. Powered by **Google LiteRT-LM**, this project demonstrates how to orchestrate local inference while leveraging hardware acceleration on **Google Tensor TPUs (G5)**, with automated fallbacks to GPU and CPU.

Developers can use this codebase as a plug-and-play sandbox to evaluate local LLMs (like Gemma) or as a foundation/starter-kit for their own on-device AI applications.

---

## 🛠️ Developer Architecture & Core Components

The framework is decoupled to separate the UI layer from the underlying inference engine, allowing you to easily rip out the UI and reuse the orchestration core.

```
├── app/src/main/java/.../
│   ├── LiteRTLMManager.kt    <-- Core Inference Orchestrator (Singleton)
│   ├── ModelResolver.kt      <-- Model Lifecycle & Asset Copier
│   ├── ModelConfig.kt        <-- Hardware Backend & Tokenizer Specs
│   └── UI Components         <-- Activity, Adapters, and View Bindings

```

### 1\. Engine & Lifecycle Management (`LiteRTLMManager.kt`)

This is the core engine singleton. It encapsulates the LiteRT-LM `Engine` and `Conversation` lifecycles, abstracting thread management and inference execution away from the UI.

* **Fallback Chain Orchestration:** Programmatically handles hardware initialization. It attempts to bind to the **Tensor TPU (NPU)** first, intercepting failures to transparently cascade down to **GPU (OpenCL)**, and ultimately **CPU**.  
* **Streaming & Performance Metrics:** Exposes hooks for real-time token streaming, capturing metrics such as Time to First Token (TTFT), Total Time for Response,  evaluation throughput (tokens/sec) and context length.

### 2\. Model & Backend Mapping (`ModelResolver.kt` & `ModelConfig.kt`)

* **Static Assets & Dynamic Registry:** Manages both bundled models (read from raw resources) and dynamically registered custom models (`.litertlm`). It handles the filesystem copying required to make raw asset descriptors readable by the native execution layer.

---

## 🚀 Quickstart: Testing LLMs

### 1\. Environment & Build Requirements

* **JDK:** Requires minimum **JDK 21** (Ensure `JAVA_HOME` is set correctly).  
* **ABI Target:** `arm64-v8a` (required for native Tensor TPU libraries).

```shell
# Clone and compile the debug APK
./gradlew :app:assembleDebug

# Deploy to your connected Pixel/Tensor device
adb install -r app/build/outputs/apk/debug/app-debug.apk

```

### 2\. Loading a Model for TPU Evaluation

To evaluate high-performance models like Gemma 4 on the Tensor TPU:

#### Method A: Bundle via Assets (Static Integration)

1. Drop your `.litertlm` file into `app/src/main/assets/`.  
2. Register the file in `ModelResolver.kt` by mapping it to a `ModelConfig` instance.  
3. Recompile. The framework will automatically handle extraction to the internal app cache on first boot.

#### Method B: Sideloading (Rapid Prototyping)

Alternatively, launch the app UI, select **Upload Custom Model** from the top dropdown, and load the `.litertlm` file directly from device storage.

---

## 🏗️ Reusing as a Framework: Key Integrations

If you are adapting this codebase into an existing project, pay attention to these critical hardware configurations:

### 1\. Native Library Linker Requirements (`build.gradle.kts`)

LiteRT-LM communicates with the Tensor TPU(NPU) via a dynamic vendor library loader (`libLiteRtDispatch_GoogleTensor.so`). To prevent the Android asset packaging tool from compressing this native library (which prevents direct memory-mapping (`mmap`) by the linker), your `app/build.gradle.kts` must include:

```kotlin
android {
    ...
    packaging {
        jniLibs {
            // Crucial: Keeps vendor shared libraries uncompressed for direct linker mapping
            useLegacyPackaging = true 
        }
    }
    ndk {
        abiFilters.add("arm64-v8a")
    }
}

```

### 2\. Initializing the NPU Backend

When instantiating the LiteRT-LM engine, the framework passes the application's `nativeLibraryDir` to point the runtime toward the device's hardware-specific dispatch libraries:

```kotlin
val config = EngineConfig().apply {
    if (preferredBackend == Backend.NPU) {
        // Points LiteRT to the hardware-specific dispatch library path
        setNativeLibraryDir(context.applicationInfo.nativeLibraryDir)
    }
}

```

---

## 🧪 Verification & Testing

The framework includes a unit testing suite targeting the non-UI business logic (`ModelResolver`, `ModelConfig`, engine configuration parsing).

Execute tests via Gradle:

```shell
./gradlew app:testDebugUnitTest

```

> **Architecture Note:** The tests leverage **Robolectric** to mock the Android runtime environment and **Mockito** for component isolation, providing a template for how to unit-test local AI orchestration pipelines without requiring a physical device connection.

---

## 📜 License

This framework wrapper is open-source software licensed under the [Apache License 2.0](https://www.google.com/search?q=LICENSE).  