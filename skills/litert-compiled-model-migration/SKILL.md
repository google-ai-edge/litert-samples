---
name: litert-compiled-model-migration
description: Rapidly migrate an Android application from legacy TensorFlow Lite (TFLite) to the modern LiteRT CompiledModel API (2.x) in Open Source GitHub repositories. Covers buffer reuse, NPU JIT compilation, and a two-stage verification gate (compile check, then an instrumented test on a device).
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
   - [A] Vision (Images / Video / Camera Feeds) -> Enables the camera-frame recipe: preprocess into a reused input buffer via writeFloat / writeInt8 (AHardwareBuffer interop is C++-only).
   - [B] Audio (Speech / Sound Classification) -> Enables the streaming FloatArray recipe (writeFloat per window).
   - [C] Text / NLP / GenAI -> Enables the token-id recipe (writeInt / writeLong into the model's input buffers).

2. LiteRT Runtime Target SDK:
   Which SDK distribution target should the project use?
   - [A] Standalone / Bundled LiteRT V2 (com.google.ai.edge.litert:litert) [Default]
         -> Bundles LiteRT runtime inside the APK for offline self-contained operation.
   - [B] LiteRT from Google Play services: no such artifact is published on Google Maven as of September 2026 (only the play-services-tflite-* Interpreter artifacts exist), so keep [A].

3. Hardware Acceleration & Conditional INT8 Quantization:
   Do you want to enable NPU hardware acceleration via JIT on-device compilation?
   - [A] Yes (Recommended - replaces deprecated NNAPI; needs a supported SoC and the vendor libraries of Upgrade 2.C) [Default]
         * If the app uses a Float32 model: Would you like to generate an INT8 integer-quantized model via AI Edge Quantizer for peak NPU speed, or run the original Float32 model?
           -> Option A.1: Convert to INT8 (Generates model_int8.tflite with int8 weights and activations; needs a few representative inputs for calibration) [Default]
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
> **Mandatory Support Library Removal**: The agent must inform the user that all legacy `org.tensorflow.lite.support` libraries (`ImageProcessor`, `ResizeOp`, `NormalizeOp`, etc.) **will be completely removed and replaced** with the LiteRT buffer API (`writeFloat` / `writeInt8`) and plain Android preprocessing. This is mandatory: the Support library depends on `org.tensorflow:tensorflow-lite`, which would keep the legacy Interpreter runtime in the app next to LiteRT 2.x.

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
  * `com.google.ai.edge.litert:litert-gpu` / `litert-support` / `litert-metadata` 1.x *(the renamed TFLite artifacts; last version 1.4.2)*
* **Replace TFLite Support Image Preprocessing**: If the application uses legacy TFLite Support (`org.tensorflow.lite.support.image.ImageProcessor`, `ResizeOp`, `NormalizeOp`), **completely remove the Support library dependency**. Replace image scaling with `androidx.core.graphics.scale` (or `Bitmap.createScaledBitmap`) and write the normalized pixels into a `FloatArray` (a `ByteArray` for int8 inputs) that goes to the model's input buffer through `TensorBuffer.writeFloat` / `writeInt8`.
* **Add Modern LiteRT**:
  * *Standalone*: `implementation("com.google.ai.edge.litert:litert:2.2.0")` in Kotlin DSL, `implementation 'com.google.ai.edge.litert:litert:2.2.0'` in Groovy (pulls in `litert-api`; the AAR bundles the GPU accelerator and sets minSdk 24; no separate `litert-gpu` artifact exists for 2.x)
* **IDE Portability**: Remove hardcoded `org.gradle.java.home` from `gradle.properties` and exclude `local.properties`.
* **Kotlin Compiler DSL**: Use top-level `kotlin { compilerOptions { ... } }` outside `android { ... }`.

### Step 2: Deprecated API & Delegate Remediation

The agent must audit and replace all deprecated delegate APIs:

1. **NNAPI Delegate (`NnApiDelegate`, `NnApiDelegate.Options`, `setUseNNAPI(true)`)**:
   * **Status**: NNAPI is deprecated since Android 15, and the LiteRT 2.x Java/Kotlin API has no NNAPI entry point (`Interpreter.Options` has no `setUseNNAPI`).
   * **Remediation**: Remove `org.tensorflow.lite.nnapi.NnApiDelegate` imports. Replace with `CompiledModel.Options(Accelerator.NPU)` and `Environment.create(context, BuiltinNpuAcceleratorProvider(context))` when `BuiltinNpuAcceleratorProvider(context).isDeviceSupported()` is true (NPU-only options compile as NPU + CPU, so unsupported ops fall back to the CPU). Implement an explicit `NPU -> GPU -> CPU` fallback cascade (try each `CompiledModel.create`, catch `LiteRtException`) to handle non-NPU hardware smoothly.

2. **Flex Delegate (`FlexDelegate`, `org.tensorflow.lite.flex`, `select-tf-ops`)**:
   * **Status**: Interpreter-only: the Flex delegate ships in the legacy `tensorflow-lite-select-tf-ops` artifact (68 MB for arm64 in 2.16.1) and in no LiteRT 2.x artifact, so a model with Select TF ops needs the re-export below.
   * **Remediation**:
     * Remove `org.tensorflow:tensorflow-lite-select-tf-ops` from `build.gradle.kts`.
     * Find Flex ops by FlatBuffer inspection (custom ops whose `custom_code` starts with `Flex`); `check_gpu_compatibility` in `utilities/litert_gpu_toolkit` (this repository) reports such a model as a compile failure, not op by op.
     * Replace Flex ops by re-exporting the model without them (LiteRT-Torch for PyTorch sources; the TensorFlow converter without `SELECT_TF_OPS`), or, if custom C++ math is required, implement a `litert::CustomOpKernel` (`litert/cc/litert_custom_op_kernel.h`, shipped in `litert_cc_sdk.zip`; guide: https://ai.google.dev/edge/litert/next/custom_op_dispatcher).

### Step 3: Native Build Toolchain (`CMakeLists.txt` / NDK)
For native C++ modules, use the C++ SDK the LiteRT docs describe (https://ai.google.dev/edge/litert/next/android_cpp_sdk): download `litert_cc_sdk.zip` from the LiteRT GitHub release that matches the AAR version (https://github.com/google-ai-edge/LiteRT/releases/download/v2.2.0/litert_cc_sdk.zip), copy `libLiteRt.so` from the AAR's `jni/arm64-v8a/` into its `litert_cc_sdk/` directory (`LITERT_CC_SDK` below), and link the `litert_cc_api` target. The AAR ships no headers or prefab package, and `liblitert_jni.so` is the Kotlin binding, not a C++ target:
```cmake
# Replace legacy tensorflowlite_jni with the LiteRT C++ SDK
add_subdirectory("${LITERT_CC_SDK}" "${LITERT_CC_SDK}/build")
include_directories("${LITERT_CC_SDK}")

target_link_libraries(your_native_lib
    litert_cc_api
    android
    log
)
```

### Step 4: API & Lifecycle Refactoring (Rewrite Initialization & Dynamic Signatures)

> [!IMPORTANT]
> **Never Simple Swap**: Do **NOT** merely perform a search-and-replace of the `Interpreter` class. Rewrite the model initialization logic to instantiate `CompiledModel` with an explicit hardware fallback cascade (`NPU -> GPU -> CPU`). When NPU is selected, prioritize NPU JIT compilation by instantiating an explicit `Environment` object (`Environment.create(context, BuiltinNpuAcceleratorProvider(context))`) and passing it to `CompiledModel.create`; on a supported device the provider itself sets `Environment.Option.DispatchLibraryDir` and `Environment.Option.CompilerPluginLibraryDir` to `context.applicationInfo.nativeLibraryDir`, and the `context` overload enables the compiler cache under `context.cacheDir`.

| Legacy TFLite API | Modern LiteRT V2 Drop-in Replacement |
|---|---|
| `org.tensorflow.lite.Interpreter` | `com.google.ai.edge.litert.CompiledModel` |
| `Interpreter(modelFile, options)` | `CompiledModel.create(context.assets, assetName, options, env)` or `create(filePath, options, env)`; no ByteBuffer overload *(via NPU Environment & Fallback Cascade)* |
| `interpreter.run(input, output)` | `compiledModel.run(inputBuffers, outputBuffers)` (both `List<TensorBuffer>` from `createInputBuffers()` / `createOutputBuffers()`) |
| `interpreter.runForMultipleInputsOutputs(inputs, outputs)` | `compiledModel.run(inputBuffers, outputBuffers)` |
| `interpreter.runSignature(inputs, outputs, key)` | `compiledModel.run(inputMap, outputMap, key)` (`Map<String, TensorBuffer>`) |
| `Interpreter.Options().setNumThreads(n)` | `options.cpuOptions = CompiledModel.CpuOptions(numThreads = n)` |
| `interpreter.resizeInput(...)` | no Kotlin equivalent (the C++ API has `ResizeInputTensor`) |
| `GpuDelegate()` / `NnApiDelegate()` | `CompiledModel.Options(Accelerator.GPU / NPU / CPU)` |
| `org.tensorflow.lite.flex.FlexDelegate` | none: re-export the model without Select TF ops, or a C++ `CustomOpKernel` (Step 2) |
| `interpreter.getInputTensor(0)` | `compiledModel.getInputTensorType("args_0")` (`args_0` / `output_0` are the LiteRT-Torch export names; other converters keep their own signature names) |
| `ImageProcessor.Builder().add(ResizeOp(...)).build()` | `androidx.core.graphics.scale(width, height)` / `Bitmap.createScaledBitmap` |
| `#include "tensorflow/lite/interpreter.h"` | `#include "litert/cc/litert_compiled_model.h"` |
| `#include "tensorflow/lite/c/c_api.h"` | `#include "litert/c/litert_compiled_model.h"` |

### Step 5: Two-Stage Fast Verification Gate (Karpathy Self-Test)
To maximize execution speed:
* **Stage 1 (Refactoring Gate)**: Run fast incremental compile checks only (`./gradlew compileDebugKotlin` or `./gradlew assembleDebug`) to verify syntax in seconds.
* **Stage 2 (Final Verification Gate)**: Copy `templates/MigrationValidationTest.kt` into `app/src/androidTest/java/<package>/` (model in `app/src/main/assets/`, `androidx.test` dependencies declared), build (`./gradlew assembleDebug assembleDebugAndroidTest`) and run it on a connected device or emulator (`./gradlew connectedDebugAndroidTest`); `testDebugUnitTest` runs only `src/test`.

---

## 2. Phase 2: Encouraged Performance Upgrades

Once Phase 1 compiles and passes self-testing, the agent applies high-value performance features:

### Upgrade 2.A: Off-main-thread execution on one serial dispatcher
The Kotlin CompiledModel API has no `runAsync`. On the CPU, `run()` blocks until the outputs are ready; on the GPU it can return before the GPU finishes, and `readFloat()` blocks until it has. Keep the UI responsive by running create, run and read on one serial dispatcher from a coroutine, and reuse the buffers created once:
```kotlin
private val modelDispatcher = Dispatchers.IO.limitedParallelism(1)

suspend fun infer(input: FloatArray): FloatArray = withContext(modelDispatcher) {
    inputBuffers[0].writeFloat(input)
    compiledModel.run(inputBuffers, outputBuffers)
    outputBuffers[0].readFloat()
}
```
Time `run()` and `readFloat()` together; timing `run()` alone under-reports GPU work. `limitedParallelism(1)` serializes the calls; it does not pin them to one thread.

### Upgrade 2.B: Buffer reuse and zero-copy interop
In Kotlin, tensor buffers come from the model (`createInputBuffers()` / `createOutputBuffers()`, or `createInputBuffer(name)`), are filled with `writeFloat` / `writeInt8`, and are reused for every inference; there is no `TensorBuffer.createFromAhwb` in the Kotlin API. `AHardwareBuffer` zero-copy interop is part of the C++ API (`litert::TensorBuffer::CreateFromAhwb` in `litert/cc/litert_tensor_buffer.h`), so a camera pipeline that needs it runs the whole inference from C++ (Step 3): there is no supported way to hand a C++ buffer to Kotlin `run()`.
```kotlin
// Create once, reuse per frame, close before the model
val inputBuffers = compiledModel.createInputBuffers()
val outputBuffers = compiledModel.createOutputBuffers()
compiledModel.run(inputBuffers, outputBuffers)

inputBuffers.forEach { it.close() }
outputBuffers.forEach { it.close() }
compiledModel.close()
```
Close the buffers, then the model, then the `Environment` if the app created one (the model closes only an environment it created itself).

### Upgrade 2.C: NPU JIT Acceleration & Conditional INT8 Quantization
1. **Conditional INT8 Quantization**: If NPU JIT is selected and the user opted in, run AI Edge Quantizer (`aeq`) to generate `model_int8.tflite` in `assets/`:
   ```python
   from ai_edge_quantizer import quantizer, recipe
   qt = quantizer.Quantizer("src/main/assets/model.tflite", recipe.static_wi8_ai8())
   calibration = qt.calibrate({"serving_default": representative_inputs})  # an iterable of {input_name: ndarray}
   qt.quantize(calibration).export_model("src/main/assets/model_int8.tflite")
   ```
   (ai-edge-quantizer 0.8.0 names. `static_wi8_ai8` is the full-integer recipe that the `accuracy-safe-quantization` skill in this repo names for NPU targets; `recipe.dynamic_wi8_afp32()` needs no calibration but keeps float32 activations. Then point the app at `model_int8.tflite` and rerun the Stage 2 gate.)
2. **NPU JIT Runtime Bundling**: Package vendor shared libraries in `app/src/main/jniLibs/arm64-v8a/` (`libLiteRtDispatch_Qualcomm.so`, `libLiteRtCompilerPlugin_Qualcomm.so`, `libQnnHtp.so`, `libQnnSystem.so`, etc.); `BuiltinNpuAcceleratorProvider` reads them from `context.applicationInfo.nativeLibraryDir`. Compiled artifacts are cached under `context.cacheDir` by `Environment.create(context, …)` (`enableCompilerCache` defaults to true).
3. **Qualcomm FastRPC Permission**: The `litert` AAR's manifest already declares `<uses-native-library android:name="libcdsprpc.so" android:required="false" />` (with the OpenCL and Google Tensor entries) and manifest merging brings it into the app; add it by hand only if merging is disabled.
4. **Environment & Fallback Cascade**: Create the `Environment` with `BuiltinNpuAcceleratorProvider` (on a supported device it sets `DispatchLibraryDir` and `CompilerPluginLibraryDir` itself) and implement the Kotlin `NPU -> GPU -> CPU` cascade / C++ fail-fast pipeline.

---

## 3. Phase 3: Automated Pull Request Provisioning

Once self-testing succeeds:
* **GitHub Pull Request**: Create a clean git commit, exclude `local.properties` and `.gradle/`, and run `gh pr create` with an attached before/after summary diff and test execution logs.
