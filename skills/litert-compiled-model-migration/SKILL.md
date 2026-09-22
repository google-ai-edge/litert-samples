---
name: litert-compiled-model-migration
description: Rapidly migrate an Android application from legacy TensorFlow Lite (TFLite) to the modern LiteRT CompiledModel API (2.x) in Open Source GitHub repositories. Covers buffer reuse, NPU JIT compilation, and automated 2-stage verification self-testing.
---

# Skill: LiteRT Compiled Model Migration SKILL

## Description
This skill guides an AI agent to rapidly migrate an Android application from legacy TensorFlow Lite (TFLite) to the modern LiteRT CompiledModel API (2.x) in **Open Source GitHub repositories**. It prioritizes a high-speed, 1st-pass **"Like for Like" baseline migration** with automated self-testing, and encourages advanced performance upgrades including **off-main-thread execution with buffer reuse** and **NPU JIT compilation**.

---

## 0. Automatic Discovery & Upfront Planning

Before writing code, the agent MUST inspect the project workspace and present the upfront planning interview to align migration parameters:

### A. Automatic Workspace & Toolchain Discovery
The agent must automatically inspect the repository structure:
1. **Ecosystem & Build Engine**:
   * **Gradle Build System**: Detected by `build.gradle`, `build.gradle.kts`, or `settings.gradle`. -> Enable **Gradle & GitHub PR Workflow**.
2. **Language & JNI Toolchain**:
   * **Native C++ / NDK**: Detected if `CMakeLists.txt`, `Android.mk`, or `*.cpp` files exist. -> Enable **C++ / JNI Migration Rules**.
   * **Pure Kotlin / Java**: Default to **JVM / Android SDK Migration Rules**.

> [!TIP]
> **Speed Optimization (Subagent Routing)**: When orchestrating subagents, the agent **MUST default to `DeepCoderLite`** (or `DeepInvestigatorLite`) to guarantee a 2–5 minute migration turnaround. Do NOT invoke heavy multi-layer `DeepCoder` synthesis unless the codebase features complex custom C++ NDK/CMake build systems.

### B. Upfront User Interview (Questions Asked Prior to Migration)

The agent must present the following review options to the user:

```
Before initiating the LiteRT Compiled Model Migration, please confirm your project preferences:

1. Model Workload & Domain:
   What type of data does this application process?
   - [A] Vision (Images / Video / Camera Feeds) -> Enables Zero-Copy AHardwareBuffer / direct ByteBuffer recipes.
   - [B] Audio (Speech / Sound Classification) -> Enables streaming FloatArray or ByteBuffer recipes.
   - [C] Text / NLP / GenAI -> Enables tokenized tensor buffer recipes.

2. LiteRT Runtime Target SDK:
   Which SDK distribution target should the project use?
   - [A] Standalone / Bundled LiteRT V2 (com.google.ai.edge.litert:litert) [Default]
         -> Bundles LiteRT runtime inside the APK for offline self-contained operation.
   - [B] LiteRT from Google Play services: no such artifact is published on Google Maven as of September 2026 (only the play-services-tflite-* Interpreter artifacts exist), so keep [A].

3. Hardware Acceleration & Conditional INT8 Quantization:
   Do you want to enable NPU hardware acceleration via JIT on-device compilation?
   - [A] Yes (Recommended - replaces deprecated NNAPI) [Default]
         * If the app uses a Float32 model: Would you like to generate an INT8 integer-quantized model via AI Edge Quantizer for peak NPU speed, or run the original Float32 model?
           -> Option A.1: Convert to INT8 (Generates model_int8.tflite for NPU matrix engines) [Default]
           -> Option A.2: Keep Float32 (Runs baseline float model directly on NPU)
   - [B] No (GPU and CPU acceleration only)

4. Encouraged Performance Upgrades:
   Should the agent upgrade the calling code to use LiteRT's advanced features?
   - [A] Yes (Move inference off the main thread and reuse buffers) [Default]
   - [B] No (Keep strict 1-to-1 synchronous baseline execution)

5. Automated Pull Request Provisioning:
   Should the agent automatically stage, commit, and create a GitHub PR when self-testing passes?
   - [A] Yes [Default] (Attaches verification test logs and before/after summary diff)
   - [B] No (Keep changes local in current working branch)
```

> [!IMPORTANT]
> **Mandatory Support Library Removal**: The agent must inform the user that all legacy `org.tensorflow.lite.support` libraries (`ImageProcessor`, `ResizeOp`, `NormalizeOp`, etc.) **will be completely removed and replaced** with direct LiteRT buffer APIs and native preprocessing. This is mandatory to unlock zero-copy speed and API compatibility.

---

## 1. Phase 1: "Like for Like" Baseline Migration (1st Pass Success)

Phase 1 prioritizes functional equivalence, fast compilation, and immediate 1st pass self-test success.

### Step 1: Clean & Modernize Dependencies

Inspect `libs.versions.toml` and `build.gradle.kts`:
* **Remove Legacy & Deprecated**:
  * `org.tensorflow:tensorflow-lite`
  * `org.tensorflow:tensorflow-lite-gpu`
  * `org.tensorflow:tensorflow-lite-support`
  * `org.tensorflow:tensorflow-lite-select-tf-ops` *(Legacy Flex Delegate — see Deprecated API Remediation below)*
* **Replace TFLite Support Image Preprocessing**: If the application uses legacy TFLite Support (`org.tensorflow.lite.support.image.ImageProcessor`, `ResizeOp`, `NormalizeOp`), **completely remove the Support library dependency**. Replace image scaling with `androidx.core.graphics.scale` (or `Bitmap.createScaledBitmap`) and replace normalization with direct memory-mapped pixel buffer writing (`ByteBuffer.allocateDirect` / `AHardwareBuffer`).
* **Add Modern LiteRT**:
  * *Standalone*: `implementation 'com.google.ai.edge.litert:litert:2.2.0'` (the AAR bundles the GPU accelerator; no separate `litert-gpu` artifact exists for 2.x)
* **IDE Portability**: Remove hardcoded `org.gradle.java.home` from `gradle.properties` and exclude `local.properties`.
* **Kotlin Compiler DSL**: Use top-level `kotlin { compilerOptions { ... } }` outside `android { ... }`.

### Step 2: Deprecated API & Delegate Remediation

The agent must audit and replace all deprecated delegate APIs:

1. **NNAPI Delegate (`NnApiDelegate`, `NnApiDelegate.Options`, `setUseNNAPI(true)`)**:
   * **Status**: Deprecated in Android 12+ and removed in LiteRT V2.
   * **Remediation**: Remove `org.tensorflow.lite.delegates.NnApiDelegate` imports. Replace with `CompiledModel.Options(Accelerator.NPU)` combined with `Environment.create(context, BuiltinNpuAcceleratorProvider(context), envOptions)`. Implement an explicit `NPU -> GPU -> CPU` fallback cascade to handle non-NPU hardware smoothly.

2. **Flex Delegate (`SelectDelegate`, `org.tensorflow.lite.flex`, `select-tf-ops`)**:
   * **Status**: Deprecated and incompatible with LiteRT V2 zero-copy and NPU acceleration (bloats APK size by ~30 MB with full TF runtime).
   * **Remediation**:
     * Remove `org.tensorflow:tensorflow-lite-select-tf-ops` from `build.gradle.kts`.
     * Audit model ops using `litert_gpu_toolkit` or Flatbuffer inspection to identify unsupported Flex ops.
     * Replace Flex ops by re-exporting the model via modern LiteRT converters (`litert_torch` / `LiteRT-torch` or `ai_edge_quantizer`), or, if custom C++ math is required, implement a native LiteRT custom op (see `litert/experimental/custom_ops` in the LiteRT repository).

### Step 3: Native Build Toolchain (`CMakeLists.txt` / NDK)
For native C++ modules, update `CMakeLists.txt`:
```cmake
# Replace legacy tensorflowlite_jni with LiteRt
find_library(log-lib log)
find_library(android-lib android)

target_link_libraries(your_native_lib
    LiteRt
    litert_jni
    ${log-lib}
    ${android-lib}
)
```

### Step 4: API & Lifecycle Refactoring (Rewrite Initialization & Dynamic Signatures)

> [!IMPORTANT]
> **Never Simple Swap**: Do **NOT** merely perform a search-and-replace of the `Interpreter` class. Rewrite the model initialization logic to instantiate `CompiledModel` with an explicit hardware fallback cascade (`NPU -> GPU -> CPU`). When NPU is selected, prioritize NPU JIT compilation by instantiating an explicit `Environment` object (`Environment.create(context, BuiltinNpuAcceleratorProvider(context), envOptions)`) configured with `Environment.Option.DispatchLibraryDir` and `Environment.Option.CompilerPluginLibraryDir` pointing to `context.applicationInfo.nativeLibraryDir`.

| Legacy TFLite API | Modern LiteRT V2 Drop-in Replacement |
|---|---|
| `org.tensorflow.lite.Interpreter` | `com.google.ai.edge.litert.CompiledModel` |
| `Interpreter(modelFile, options)` | `CompiledModel.create(modelPath, options, env)` *(via NPU Environment & Fallback Cascade)* |
| `interpreter.run(input, output)` | `compiledModel.run(inputBuffers, outputBuffers)` (both `List<TensorBuffer>` from `createInputBuffers()` / `createOutputBuffers()`) |
| `GpuDelegate()` / `NnApiDelegate()` | `CompiledModel.Options(Accelerator.GPU / NPU / CPU)` |
| `org.tensorflow.lite.flex.FlexDelegate` | Native LiteRT op / `CompiledModel.Options(Accelerator.NPU / GPU)` |
| `interpreter.getInputTensor(0)` | `compiledModel.getInputTensorType("args_0")` (Fallback: `"input_0"`) |
| `ImageProcessor.Builder().add(ResizeOp(...)).build()` | `androidx.core.graphics.scale(width, height)` / `Bitmap.createScaledBitmap` |
| `#include "tensorflow/lite/interpreter.h"` | `#include "litert/cc/litert_compiled_model.h"` |
| `#include "tensorflow/lite/c/c_api.h"` | `#include "litert/c/litert_compiled_model.h"` |

### Step 5: Two-Stage Fast Verification Gate (Karpathy Self-Test)
To maximize execution speed:
* **Stage 1 (Refactoring Gate)**: Run fast incremental compile checks only (`./gradlew compileDebugKotlin` or `./gradlew assembleDebug`) to verify syntax in seconds.
* **Stage 2 (Final Verification Gate)**: Copy `templates/MigrationValidationTest.kt` into `androidTest/` and execute full packaging (`./gradlew assembleDebug assembleDebugAndroidTest` and `./gradlew testDebugUnitTest`).

---

## 2. Phase 2: Encouraged Performance Upgrades

Once Phase 1 compiles and passes self-testing, the agent applies high-value performance features:

### Upgrade 2.A: Off-main-thread execution with a confined dispatcher
The Kotlin CompiledModel API has no `runAsync`: `run()` enqueues the work and the readback (`readFloat()`) is the synchronization point. Keep the UI responsive by running create, run and read on one confined dispatcher from a coroutine, and reuse the buffers created once:
```kotlin
private val modelDispatcher = Dispatchers.IO.limitedParallelism(1)

suspend fun infer(input: FloatArray): FloatArray = withContext(modelDispatcher) {
    inputBuffers[0].writeFloat(input)
    compiledModel.run(inputBuffers, outputBuffers)
    outputBuffers[0].readFloat()
}
```
Time `run()` and `readFloat()` together; timing `run()` alone under-reports GPU work.

### Upgrade 2.B: Buffer reuse and zero-copy interop
In Kotlin, tensor buffers come from the model (`createInputBuffers()` / `createOutputBuffers()`, or `createInputBuffer(name)`), are filled with `writeFloat` / `writeInt8`, and are reused for every inference; there is no `TensorBuffer.createFromAhwb` in the Kotlin API. `AHardwareBuffer` zero-copy interop is part of the C++ API (`litert::TensorBuffer::CreateFromAhwb` in `litert/cc/litert_tensor_buffer.h`), so use it from the native side when a camera pipeline needs it.
```kotlin
// Create once, reuse per frame, close before the model
val inputBuffers = compiledModel.createInputBuffers()
val outputBuffers = compiledModel.createOutputBuffers()
compiledModel.run(inputBuffers, outputBuffers)

// Cleanup lifecycle
inputBuffers.forEach { it.close() }
outputBuffers.forEach { it.close() }
compiledModel.close()
```

### Upgrade 2.C: NPU JIT Acceleration & Conditional INT8 Quantization
1. **Conditional INT8 Quantization**: If NPU JIT is selected and the user opted in, run AI Edge Quantizer (`aeq`) to generate `model_int8.tflite` in `assets/`:
   ```python
   from ai_edge_quantizer import quantizer, recipe
   qt = quantizer.Quantizer("src/main/assets/model.tflite", recipe.dynamic_wi8_afp32())
   qt.quantize().export_model("src/main/assets/model_int8.tflite")
   ```
   (`recipe.static_wi8_ai8()` needs calibration data; see the `accuracy-safe-quantization` skill in this repo for the recipe ladder and the parity gates.)
2. **NPU JIT Runtime Bundling**: Package vendor shared libraries in `app/src/main/jniLibs/arm64-v8a/` (`libLiteRtDispatch_Qualcomm.so`, `libQnnHtp.so`, etc.). Check local `LITERT_JIT_CACHE_DIR` before network downloads.
3. **Qualcomm FastRPC Permission**: Declare `<uses-native-library android:name="libcdsprpc.so" android:required="false" />` inside `<application>` in `AndroidManifest.xml`.
4. **Environment Dispatch & Fallback Cascade**: Pass `DispatchLibraryDir` pointing to `nativeLibraryDir` and implement Kotlin `NPU -> GPU -> CPU` cascade / C++ fail-fast pipeline.

---

## 3. Phase 3: Automated Pull Request Provisioning

Once self-testing succeeds:
* **GitHub Pull Request**: Create a clean git commit, exclude `local.properties` and `.gradle/`, and run `gh pr create` with an attached before/after summary diff and test execution logs.
