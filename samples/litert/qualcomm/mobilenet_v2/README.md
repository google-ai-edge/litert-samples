# LiteRT MobileNet-v2 JIT S25 Example

This sample runs Qualcomm MobileNet-v2 in an Android Kotlin/Compose app using LiteRT `CompiledModel`.

The app has two runtime paths:

- Emulator or laptop testing: load the Qualcomm float TFLite model from app assets and run it on `Accelerator.CPU`.
- Samsung S25 Ultra / Snapdragon SM8750 testing: load the same float TFLite model from app assets and request `Accelerator.NPU` through LiteRT's Qualcomm NPU provider.

## Important Files

Start with these files:

- `app/src/main/java/com/example/mobilenetlitert/MobilenetClassifier.kt`: core runtime logic. This creates CPU, GPU, and NPU `CompiledModel` instances, writes input buffers, calls `model.run(...)`, reads output buffers, and logs backend/timing details.
- `app/src/main/java/com/example/mobilenetlitert/DevicePolicy.kt`: chooses `CPU_EMULATOR` on emulator-like devices and `AUTO` on physical devices. `AUTO` tries NPU first, then GPU, then CPU.
- `app/src/main/jniLibs/arm64-v8a/`: Qualcomm runtime libraries packaged directly into the APK for physical arm64 devices.
- `app/build.gradle.kts`: app module setup, LiteRT dependency, Compose dependency, native-library packaging, and Gradle tasks that download the model, labels, and bundled sample image.
- `gradle/libs.versions.toml`: dependency versions for Android Gradle Plugin, Kotlin, Compose, LiteRT, and test libraries.

Useful supporting files:

- `app/src/main/java/com/example/mobilenetlitert/ImagePreprocessor.kt`: resizes images to `224x224`, extracts RGB channels, and converts pixels to float values in `[0, 1]`.
- `app/src/main/java/com/example/mobilenetlitert/ClassificationPostprocessor.kt`: maps output scores to top-k ImageNet labels.
- `app/src/main/java/com/example/mobilenetlitert/MainActivity.kt`: Compose UI for the bundled sample image, image picker, run button, backend status, timings, and top-5 predictions.
- `app/src/test/java/com/example/mobilenetlitert/ImagePreprocessorTest.kt`: unit coverage for tensor shape, channel order, and input range.
- `app/src/test/java/com/example/mobilenetlitert/ClassificationPostprocessorTest.kt`: unit coverage for top-k sorting and label mapping.
- `tools/fetch_qualcomm_libs.py`: helper script used to fetch Qualcomm QAIRT libraries into local runtime-library folders when refreshing native dependencies.

## Run

CPU emulator:

```sh
./gradlew installDebug
```

Build and unit-test:

```sh
./gradlew testDebugUnitTest assembleDebug
```

Physical Samsung S25 Ultra:

```sh
./gradlew installDebug
```

Then filter Logcat with:

```text
MobileNetLiteRT
```

Expected emulator behavior:

- backend status shows `CPU`
- app can classify the bundled sample image or a picked image
- logs show `CompiledModel.run()`
- there is no `Interpreter` API usage

Expected S25 Ultra behavior:

- app requests NPU first through `BuiltinNpuAcceleratorProvider`
- successful NPU startup logs `backend=NPU (JIT)`
- if NPU initialization fails under `AUTO`, logs show the failure and the selected fallback backend
- for strict device validation, initialize with `BackendPolicy.NPU_REQUIRED`; that path throws instead of falling back

## Flow

This is the current end-to-end flow.

1. The project uses a single Qualcomm MobileNet-v2 float TFLite model:

```text
app/src/main/assets/model/mobilenet_v2_float.tflite
```

2. Gradle downloads the model, ImageNet labels, and sample image during `preBuild`:

```kotlin
val downloadMobileNet by tasks.registering(Download::class) {
  src(
    "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/" +
      "qai-hub-models/models/mobilenet_v2/releases/v0.51.0/" +
      "mobilenet_v2-tflite-float.zip"
  )
  dest(modelZip)
  overwrite(false)
}
```

3. The app picks a backend policy at startup:

```kotlin
fun defaultBackendPolicy(): BackendPolicy {
  return if (isProbablyEmulator()) {
    BackendPolicy.CPU_EMULATOR
  } else {
    BackendPolicy.AUTO
  }
}
```

4. CPU initialization uses the same asset and requests only `Accelerator.CPU`:

```kotlin
private fun initializeCpu(): ModelSession {
  val options = CompiledModel.Options(Accelerator.CPU)
  val compiledModel = CompiledModel.create(context.assets, CPU_MODEL_ASSET, options, null)
  Log.i(TAG, "Selected model source=assets/$CPU_MODEL_ASSET accelerators=[CPU]")
  return ModelSession(compiledModel, "CPU")
}
```

5. NPU initialization creates a LiteRT environment with the Qualcomm provider and requests only `Accelerator.NPU`:

```kotlin
private fun initializeNpu(): ModelSession {
  val env = Environment.create(BuiltinNpuAcceleratorProvider(context))

  val options = CompiledModel.Options(setOf(Accelerator.NPU)).apply {
    qualcommOptions = CompiledModel.QualcommOptions(
      htpPerformanceMode = CompiledModel.QualcommOptions.HtpPerformanceMode.HIGH_PERFORMANCE
    )
  }

  val compiledModel = CompiledModel.create(context.assets, CPU_MODEL_ASSET, options, env)
  Log.i(TAG, "Selected model source=assets/$CPU_MODEL_ASSET accelerators=[NPU]")
  Log.i(TAG, "NPU model created successfully with JIT")
  return ModelSession(compiledModel, "NPU (JIT)")
}
```

6. Inference writes the preprocessed tensor, runs the compiled model, and reads the output tensor:

```kotlin
activeSession.inputBuffers[0].writeFloat(input)
activeSession.model.run(activeSession.inputBuffers, activeSession.outputBuffers)
val logits = activeSession.outputBuffers[0].readFloat()
```

7. The app logs the important runtime milestones with the stable tag `MobileNetLiteRT`:

```text
Requested backend policy: AUTO
Selected model source=assets/model/mobilenet_v2_float.tflite accelerators=[NPU]
Model initialized with backend=NPU (JIT)
Preprocessing time: ...
CompiledModel.run() time: ...
```

8. The UI displays backend status, preprocessing time, inference time, and top-5 predictions.

9. Unit tests validate preprocessing and postprocessing, while Gradle build checks validate the Android project.

## Source Docs

- [LiteRT Android Kotlin `CompiledModel` guide](https://ai.google.dev/edge/litert/next/android_kotlin): Android-side `CompiledModel`, input/output buffers, `model.run(...)`, and CPU setup.
- [LiteRT Qualcomm NPU acceleration](https://ai.google.dev/edge/litert/next/qualcomm): Qualcomm NPU provider setup and Snapdragon NPU context.
- [Qualcomm MobileNet-v2 model card](https://huggingface.co/qualcomm/MobileNet-v2): source model, float TFLite asset, ImageNet labels, and `224x224` classifier context.
