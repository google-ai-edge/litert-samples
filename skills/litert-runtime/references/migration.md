These are the migration steps of the [litert-runtime
skill](../SKILL.md#migrating-from-legacy-tflite-interpreter), with the commands
for each step. The migrated app ends with the code of the skill's Steps 1 to 5:
the `Classifier` helper of Step 2, the executor and the CPU fallback of Step 3,
and `preprocess` of Step 4. Three files in [`migration/`](migration/) are copied
into the app under migration. Each starts with `package com.example.validation`
under a `TODO` line; replace it with the app's package.

- [`GoldenCaptureTest.kt`](migration/GoldenCaptureTest.kt) records the legacy app's output for one fixed input (step 2) and is deleted afterwards.
- [`MigrationValidationTest.kt`](migration/MigrationValidationTest.kt) checks the migrated model against that output on a device (step 6).
- [`SupportBridge.kt`](migration/SupportBridge.kt) moves LiteRT Support tensors in and out of the model's buffers, for an app that keeps the Support library (step 5).

## 1. Find the runtime

Read `app/build.gradle.kts` (or `app/build.gradle`), `gradle/libs.versions.toml`,
and the output of `./gradlew :app:dependencies`, which also shows what other
libraries pull in. A legacy app gets the Interpreter runtime from one of these:

- **TensorFlow Lite** : `org.tensorflow:tensorflow-lite` and the other `org.tensorflow:tensorflow-lite*` artifacts.
- **LiteRT 1.x** : `com.google.ai.edge.litert:litert`, `litert-gpu`, `litert-support`, or `litert-metadata` at a 1.x version, the same runtime under the LiteRT name. Most apps under `samples/litert_interpreter/` in this repository use 1.0.0 on AGP 8.3 and Kotlin 1.9.
- **Google Play services** : `com.google.android.gms:play-services-tflite-java`, `-gpu`, or `-support`, with `TfLite.initialize` in the code (see [Google Play services](#google-play-services)).

Then find the code that the later steps change:

    grep -rnE 'Interpreter|GpuDelegate|CompatibilityList|NnApiDelegate|NNAPI|[nN]umThreads|FlexDelegate|org\.tensorflow\.lite\.support|TfLite' --include='*.kt' --include='*.java' app/src

Note which accelerator each code path uses (the GPU delegate, NNAPI, or the CPU
with a thread count), whether the app uses the TensorFlow Lite Support Library
(`ImageProcessor`, `TensorImage`, `TensorLabel`, `MetadataExtractor`), and
whether it has native code (`CMakeLists.txt`, `Android.mk`, or `.cpp` files; see
[Native modules](#7-native-modules)).

## 2. Capture a golden output first

The golden output is the legacy app's output for one fixed input, recorded
before any dependency changes. The verification of step 6 compares the migrated
app with it, so the input goes through the app's own preprocessing: a raw tensor
fed straight to the model would leave the preprocessing out of the comparison.

- **Input** : Use one deterministic input, never a camera frame: for a vision app the pattern of `fixedBitmap` in `GoldenCaptureTest.kt` or a small image file in `app/src/androidTest/assets/`, otherwise a generated array. A bitmap of the model's input size with rotation 0 leaves the resize and the rotation out of the comparison, so use another size and aspect ratio, and the rotation that the camera path passes, here and in step 6 alike.
- **Test setup** : Declare `testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"` and the `androidx.test.ext:junit` and `androidx.test:runner` dependencies; step 6 uses them too.
- **Capture** : Copy `GoldenCaptureTest.kt` into `app/src/androidTest/java/<package>/` and make it call the app's existing inference entry point (the class that the UI calls, with its own preprocessing) on the CPU, without the GPU or NNAPI delegate. When that entry point returns only a sorted or thresholded list, first give the class one function that returns the raw output and one that returns the preprocessed input (step 6 calls it too), still on the legacy runtime.
- **Run** : The four commands in the file's header install both APKs, run the test on the device, and copy `golden.txt` to `app/src/androidTest/assets/golden/output.txt`.
- **Clean up** : Delete the capture test. The verification of step 6 reads the file, so the test APK never needs the legacy runtime.

`SupportBridge` and `getInputTensorType` take the model's input and output
names: the names in the model's signature, or the tensor names of a model
without a signature. The Python package `ai-edge-litert` prints them on the
workstation:

    from ai_edge_litert.compiled_model import CompiledModel

    print(CompiledModel.from_file("model.tflite").get_signature_list())

A LiteRT-Torch export with one input and one output prints
`{'serving_default': {'inputs': ['args_0'], 'outputs': ['output_0']}}`.

## 3. Swap the dependencies

Remove every legacy artifact, also when another library pulls it in:

- every `org.tensorflow:tensorflow-lite*` artifact (`tensorflow-lite`, `-api`, `-gpu`, `-gpu-api`, `-support`, `-support-api`, `-metadata`, `-select-tf-ops`);
- `com.google.ai.edge.litert:litert-gpu` and `litert-gpu-api` 1.x, the TensorFlow Lite GPU delegate under the LiteRT name (`litert-gpu:2.3.0` is another library: the GPU accelerator of the `CompiledModel` runtime);
- `com.google.ai.edge.litert:litert` 1.x, which becomes `litert:2.3.0`;
- `com.google.android.gms:play-services-tflite-*` (see [Google Play services](#google-play-services)).

The LiteRT artifacts carry the `org.tensorflow.lite` runtime classes and the
model schema under their original names, so a build that mixes old and new
artifacts fails with duplicate classes. Add the dependencies of Step 1
(`litert:2.3.0` and `litert-gpu:2.3.0`) with `minSdk` 24 or higher. A
`gradle.properties` line `android.uniquePackageNames=false`, added for `litert`
2.2.0 or 2.1.6, goes too (Step 1).

### Support library

An app on the TensorFlow Lite Support Library keeps it: the library now ships as
LiteRT Support, with the same classes, methods, and behavior (see the [LiteRT
Support migration
guide](https://github.com/google-ai-edge/LiteRT/blob/main/litert/support_lib/MIGRATION_GUIDE.md)).
Swap the coordinates:

- `org.tensorflow:tensorflow-lite-support`, `litert-support` 1.x, or `play-services-tflite-support` becomes `com.google.ai.edge.litert:litert-support:2.3.0`, the Support API with the LiteRT runtime.
- `tensorflow-lite-support-api` or `litert-support-api` 1.x becomes `litert-support-api:2.3.0`, the API alone, next to `litert:2.3.0`.
- `tensorflow-lite-metadata` or `litert-metadata` 1.x becomes `litert-metadata:2.3.0` (`MetadataExtractor`).

`litert-support:2.3.0` depends on `litert:2.3.0` and raises a lower `litert`
version to 2.3.0 without a warning, so keep `litert:2.3.0` in the build file.
Then rename the package in the sources (GNU `sed`; on macOS, `sed -i ''`):

    grep -rl 'org\.tensorflow\.lite\.support' --include='*.java' --include='*.kt' app/src/ \
      | xargs sed -i 's/org\.tensorflow\.lite\.support/com.google.ai.edge.litert.support/g'

Rename it in the R8 and ProGuard rules too
(`-keep class org.tensorflow.lite.support.**`). `org.tensorflow.lite.DataType`,
the type that the Support classes take, keeps its name. An app that never used
the Support library doesn't add it.

### Labels and metadata

`FileUtil.loadLabels` and `MetadataExtractor` work as before on the 2.3.0
artifacts. Without the Support library, read the label file with
`context.assets.open(name).bufferedReader().readLines()`. A `.tflite` file with
metadata is also a zip archive: `unzip -l model.tflite` lists the label file,
and `unzip -o model.tflite labels.txt -d app/src/main/assets/` extracts it once.

### Toolchain

`litert-api` 2.3.0, which `litert` pulls in, depends on
`androidx.lifecycle:lifecycle-runtime:2.10.0`, and its classes carry Kotlin 2.4
metadata. A project on AGP 8.3 and Kotlin 1.9 needs these before its first
build, or each one becomes a build error in turn:

- `compileSdk` 35 and AGP 8.6.0 (`checkDebugAarMetadata` asks for both), and Gradle 8.7 for that AGP.
- The Kotlin plugin 2.3 or newer. Kotlin 2.2 stops with `The binary version of its metadata is 2.4.0, expected version is 2.2.0`; a project already on Kotlin 2.2 can stay there with `freeCompilerArgs += "-Xskip-metadata-version-check"`.
- With Kotlin 2.x, a Compose app applies the `org.jetbrains.kotlin.plugin.compose` plugin in place of `composeOptions`, and `kotlinOptions { jvmTarget = "1.8" }` becomes a top-level `kotlin { compilerOptions { jvmTarget.set(JvmTarget.JVM_1_8) } }` outside `android { }`.
- On AGP 9, which has Kotlin built in, the root `plugins` block pins the Kotlin version with `id("org.jetbrains.kotlin.android") version "2.3.21" apply false`.

## 4. Replace the delegates

- **GPU delegate** (`GpuDelegate`, `GpuDelegateFactory`): Create the GPU option of the Step 2 helper as `CompiledModel.Options(Accelerator.GPU, Accelerator.CPU)`, with the `litert-gpu:2.3.0` dependency. The CPU then runs the operators that the GPU doesn't support, as it did with the delegate. With `Accelerator.GPU` alone, `create` throws `LiteRtException` for such a model, and Step 3 loads it on the CPU. GPU and CPU together can still throw for a model that the GPU can't compile at all, so keep the CPU fallback of Step 3. `CompatibilityList` has no counterpart; that fallback covers a device whose GPU can't run the model. Without `litert-gpu`, GPU and CPU together still create the model and run it on the CPU, with only the log line `GPU accelerator could not be loaded and registered`, and Step 3 reports `Accelerator.GPU`.
- **NNAPI** (`NnApiDelegate`, `setUseNNAPI(true)`): NNAPI is deprecated from Android 15, and the `CompiledModel` API has no NNAPI accelerator. Remove the delegate, and give an NNAPI choice in the app the GPU option above. NPU acceleration is a separate topic that this skill doesn't cover ([NPU acceleration with LiteRT](https://ai.google.dev/edge/litert/next/npu)).
- **Select TensorFlow operators** (`FlexDelegate`, `tensorflow-lite-select-tf-ops`): No LiteRT 2.x artifact contains the Flex delegate. Find the Flex operators (custom operators whose `custom_code` starts with `Flex`) and export the model again without them: with LiteRT-Torch from a PyTorch source, or with the TensorFlow converter without `SELECT_TF_OPS`. An operator that needs its own C++ code becomes a `litert::CustomOpKernel` ([custom op dispatcher](https://ai.google.dev/edge/litert/next/custom_op_dispatcher)).

## 5. Rewrite the inference code

Don't search and replace `Interpreter`. Move the model into the `Classifier`
helper of Step 2, owned by the single-thread executor of Step 3, and keep the
app's own classes and their public functions as the callers. The helper creates
the model and its buffers once, and closes them when a step in its `init`
throws.

- **Model loading** : An `Interpreter` or `InterpreterApi` made from `FileUtil.loadMappedFile`, and the Support `Model.createModel`, become `CompiledModel.create(context.assets, name, options, environment)` in the helper, or `CompiledModel.create(filePath, options, environment)` for a model file on disk. No overload takes a `ByteBuffer`; `FileUtil.loadMappedFile` stays only where `MetadataExtractor` reads the model.
- **Inference** : `run` and `runForMultipleInputsOutputs` on `ByteBuffer` objects become `inputs[0].writeFloat(input)`, `model.run(inputs, outputs)`, and `outputs[0].readFloat()` on the buffers that `createInputBuffers()` and `createOutputBuffers()` returned (`writeInt8` and `readInt8` for int8 and uint8 tensors). `writeFloat` writes a shorter array without an error, so pass the model's input element count.
- **Signatures** : `runSignature` becomes `model.run(inputMap, outputMap, key)` with `Map<String, TensorBuffer>` maps.
- **Thread count** : `setNumThreads` becomes `cpuOptions = CompiledModel.CpuOptions(numThreads = n)` on a new `CompiledModel.Options(Accelerator.CPU)`, never on the shared `CompiledModel.Options.CPU`.
- **Tensor shapes** : `getInputTensor` and its `shape` become `model.getInputTensorType(name).layout?.dimensions`, by name only (2.3.0 throws `LiteRtException` for a uint8 tensor). `resizeInput` has no counterpart in 2.3.0.
- **Closing** : Closing the interpreter becomes the helper's `close`, which closes the buffers and then the model. The `Environment` of Step 2 stays open for the process.

### With the Support library

`ImageProcessor`, `TensorImage`, `TensorLabel`, `TensorProcessor`, `FileUtil`,
and `MetadataExtractor` stay, under `com.google.ai.edge.litert.support`. The
Support `Model` class runs on the Interpreter API and is the one Support class
to replace: the helper takes its place, and `SupportBridge.kt` moves the Support
tensors in and out of the helper's buffers.

- Copy `SupportBridge.kt` next to the helper, and create one `SupportBridge(model, inputName, outputName)` in the helper's `init` after the buffers, with the names from step 2.
- Give the helper a function that calls `bridge.write(inputs[0], image)` with the processed `TensorImage`, then `model.run(inputs, outputs)`, then `bridge.read(outputs[0], shape, DataType.FLOAT32)`, and returns the Support `TensorBuffer` that `TensorLabel` takes as before.
- The bridge reads the byte size of both tensors once. It refuses a Support tensor of another size, and a uint8 image for a float input (an image that skipped `NormalizeOp`); `writeFloat` itself accepts a shorter array, and uint8 bytes in a float input produce NaN, both without an error.
- Both libraries have a class named `TensorBuffer`; the bridge imports the Support one as `SupportTensorBuffer`.

### Without the Support library

Step 4's `preprocess` replaces the code that filled the input `ByteBuffer`. Give
it the model's input size and the legacy normalization: the `mean` and `std` of
a legacy `NormalizeOp(mean, std)`, or the app's own formula, in the same channel
order and layout.

## 6. Verify in two stages

**Stage 1, on the workstation** :
`./gradlew assembleDebug assembleDebugAndroidTest` passes, and neither of the
following prints anything. The first searches the code and the build files; the
second searches the dependencies that Gradle resolves, including those from a
version catalog and those that other libraries pull in (with product flavors,
the configuration is `<flavor>ReleaseRuntimeClasspath`):

    grep -rnE 'org\.tensorflow\.lite\.(Interpreter|support)|GpuDelegate|NnApiDelegate|FlexDelegate|support\.model\.Model|com\.google\.android\.gms\.tflite|play-services-tflite|org\.tensorflow:tensorflow-lite' --include='*.kt' --include='*.java' --include='*.kts' --include='*.gradle' --include='*.toml' --include='*.pro' --exclude-dir=build --exclude-dir=.gradle .
    ./gradlew :app:dependencies --configuration releaseRuntimeClasspath | grep -E 'org\.tensorflow:tensorflow-lite|play-services-tflite|com\.google\.ai\.edge\.litert:litert[a-z-]*:1\.'

The build alone doesn't reveal leftover Interpreter code: the `litert` 2.3.0
artifact still contains `org.tensorflow.lite.Interpreter`, and `litert-api`
contains `InterpreterApi`. `org.tensorflow.lite.DataType` and the
`org.tensorflow.lite.schema` package may stay.

**Stage 2, on a device** : Copy `MigrationValidationTest.kt` into
`app/src/androidTest/java/<package>/`; the golden file is already in
`app/src/androidTest/assets/golden/`. Make the test build the input of step 2
and run it through the app's preprocessing: the Support `ImageProcessor`
exposed as a function (with `SupportBridge` to write and read), or the
`preprocess` that replaced it. Run `./gradlew connectedDebugAndroidTest` with a
device or an emulator connected. The test runs the model on `Accelerator.CPU`
with buffers of its own and compares the first output with the golden file:

- When the file holds every output value, the maximum absolute difference stays below 1e-3, a bound for outputs of order 1; scale it for larger outputs.
- When the file holds one value, the top-1 index matches. Use this only as a last resort: a wrong normalization can keep the top-1.

The test runs neither the app's own classes nor the GPU, so also run the app
once on the device, then compare the CPU and the GPU as in Step 5 of the skill.
With no device, run Stage 1 only and report that Stage 2 didn't run.

## 7. Native modules

A native module moves to the LiteRT C++ SDK, as described in [Use prebuilt C++
library from LiteRT Maven](https://ai.google.dev/edge/litert/next/android_cpp_sdk).
Download `litert_cc_sdk.zip` from the [LiteRT GitHub
release](https://github.com/google-ai-edge/LiteRT/releases) that matches the
AAR version (`v2.3.0` for `litert:2.3.0`) and unzip it. Copy `libLiteRt.so`
from the `jni/arm64-v8a/` directory of the `litert` AAR into the unzipped
`litert_cc_sdk/` directory (`LITERT_CC_SDK` below), which holds the SDK's
`CMakeLists.txt`, and link the `litert_cc_api` target in place of
`tensorflowlite_jni`:

    add_subdirectory("${LITERT_CC_SDK}" "${LITERT_CC_SDK}/build")
    include_directories("${LITERT_CC_SDK}")

    target_link_libraries(your_native_lib
        litert_cc_api
        android
        log)

The AARs have no headers or prefab package, and `liblitert_jni.so` in
`litert-api` is the Kotlin binding, not a C++ target. The include
`tensorflow/lite/interpreter.h` becomes `litert/cc/litert_compiled_model.h`, and
the C header `tensorflow/lite/c/c_api.h` becomes
`litert/c/litert_compiled_model.h`. Zero-copy input from an `AHardwareBuffer`
(`litert::TensorBuffer::CreateFromAhwb` in `litert/cc/litert_tensor_buffer.h`)
is in the C++ API only.

## Google Play services

As of 2026-10-15, the Java and Kotlin API of LiteRT in Google Play services is
the Interpreter API: the [LiteRT in Google Play services Java
API](https://ai.google.dev/edge/litert/android/java) page adds
`com.google.android.gms:play-services-tflite-java` (with
`play-services-tflite-gpu` and `play-services-tflite-support`) at 16.5.0, calls
`TfLite.initialize`, and creates an `InterpreterApi` with
`TfLiteRuntime.FROM_SYSTEM_ONLY`. The `CompiledModel` API of this skill comes in
the `com.google.ai.edge.litert` artifacts of Step 1, which the app bundles.

- **An existing app on Google Play services** : Capture the golden output first (step 2), on the Play services runtime that the app ships with, and then follow the other steps. While swapping the dependencies (step 3), also remove the `play-services-tflite-*` lines, `TfLite.initialize` with its `TfLiteInitializationOptions` and the listener or `await` that waits for it, `TfLiteGpu.isGpuDelegateAvailable`, and `setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)`. The bundled runtime needs no initialization step: the Step 2 helper creates the model directly on the executor of Step 3. `InterpreterApi` and `GpuDelegateFactory` are replaced in steps 4 and 5. `play-services-tflite-support`, whose classes are in `org.tensorflow.lite.support`, moves to `litert-support:2.3.0` with the package rename of step 3.
- **A new app** : Steps 1 to 5 of the skill.
