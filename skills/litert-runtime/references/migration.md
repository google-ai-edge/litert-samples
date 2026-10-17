This file gives the commands and code for each step of the migration summary in
the skill. The migrated app ends with the code of Steps 1 to 5 of the skill: the
`Classifier` helper of Step 2, the executor and CPU fallback of Step 3, and
`preprocess` of Step 4.

The code in this file uses these imports:

    import android.content.Context
    import com.google.ai.edge.litert.Accelerator
    import com.google.ai.edge.litert.CompiledModel
    import com.google.ai.edge.litert.TensorBuffer
    import com.google.ai.edge.litert.support.image.TensorImage
    import com.google.ai.edge.litert.support.tensorbuffer.TensorBufferFloat
    import java.io.File
    import kotlin.math.abs

## 1. Find the runtime

Read `app/build.gradle.kts` (or `app/build.gradle`),
`gradle/libs.versions.toml`, and the output of `./gradlew :app:dependencies`,
which also shows what other libraries pull in. A legacy app gets the Interpreter
runtime from one of these:

- **TensorFlow Lite** : `org.tensorflow:tensorflow-lite` and the other `org.tensorflow:tensorflow-lite*` artifacts.
- **LiteRT 1.x** : `com.google.ai.edge.litert:litert`, `litert-gpu`, `litert-support`, or `litert-metadata` at a 1.x version.

When the app depends on LiteRT in Google Play services
(`com.google.android.gms:play-services-tflite-java`, `-gpu`, or `-support`, with
`TfLite.initialize` in the code) or on the TensorFlow Lite Task Library
(`org.tensorflow:tensorflow-lite-task-vision`, `-text`, or `-audio`), stop and
tell the user. This skill doesn't migrate either of them.

Then find the code that the later steps change:

    grep -rnE 'org\.tensorflow\.lite\.|Interpreter|GpuDelegate|CompatibilityList|NnApiDelegate|NNAPI|[nN]umThreads|FlexDelegate|TfLite' --include='*.kt' --include='*.java' app/src

Note which accelerator each code path uses (the GPU delegate, NNAPI, or the CPU
with a thread count), and whether the app uses the TensorFlow Lite Support
Library (`ImageProcessor`, `TensorImage`, `TensorLabel`, `MetadataExtractor`).

## 2. Capture a golden output first

The golden output is the legacy app's output for one fixed input, recorded
before any dependency changes. Step 6 compares the migrated app with it. Pick
one path:

- **Path A (device or emulator connected)** : Write an instrumented test with the project's existing test setup. The test feeds one deterministic input through the app's own inference class and preprocessing on the CPU, without the GPU or NNAPI delegate. Use a bitmap of another size than the model input, never a camera frame. Write the output with `writeGoldenOutput`, then copy the file to `app/src/androidTest/assets/golden/output.txt`.
- **Path B (no device)** : Don't capture. Report the capture and the device comparison of Step 6 to the user as post-implementation steps.

    // Before the dependency swap: record the legacy app's output for one fixed
    // input, one value per line.
    fun writeGoldenOutput(file: File, output: FloatArray) {
        file.writeText(output.joinToString("\n"))
    }

    // After the migration: compare the migrated output with the recorded one.
    // A difference smaller than 1e-3 passes for outputs of order 1.
    fun maxAbsDifference(golden: FloatArray, migrated: FloatArray): Float {
        check(golden.size == migrated.size) { "Size ${migrated.size} != ${golden.size}" }
        return golden.indices.maxOf { abs(golden[it] - migrated[it]) }
    }

    fun readGoldenOutput(context: Context, path: String = "golden/output.txt"): FloatArray =
        context.assets.open(path).bufferedReader().use { reader ->
            reader.readLines().filter { it.isNotBlank() }.map { it.toFloat() }.toFloatArray()
        }

## 3. Swap the dependencies

Remove every legacy artifact, also when another library pulls it in. The LiteRT
artifacts carry the `org.tensorflow.lite` runtime classes under their original
names, so a build that mixes old and new artifacts fails with duplicate classes.
Add the dependencies of Step 1 of the skill (`litert:2.3.0` and
`litert-gpu:2.3.0`) with `minSdk` 24 or higher.

### Support library

An app on the TensorFlow Lite Support Library keeps it. The library ships as
LiteRT Support, with the same classes and behavior. When the app uses
`ImageProcessor`, `TensorImage`, `TensorLabel`, or `MetadataExtractor`, you must
read the [LiteRT Support migration guide][1] before swapping the coordinates:

- `org.tensorflow:tensorflow-lite-support` or `litert-support` 1.x becomes `com.google.ai.edge.litert:litert-support:2.3.0`.
- `org.tensorflow:tensorflow-lite-support-api` or `litert-support-api` 1.x becomes `litert-support-api:2.3.0`, next to `litert:2.3.0`.
- `tensorflow-lite-metadata` or `litert-metadata` 1.x becomes `litert-metadata:2.3.0`. It depends on `com.google.flatbuffers:flatbuffers-java:25.2.10`; when the app pins an older `flatbuffers-java`, upgrade it.

Keep `litert:2.3.0` in the build file. Then rename the package in the sources,
the R8 and ProGuard rules, and any reflection strings:

    grep -rl 'org\.tensorflow\.lite\.support' --include='*.java' --include='*.kt' --include='*.pro' app/ \
      | xargs sed -i 's/org\.tensorflow\.lite\.support/com.google.ai.edge.litert.support/g'

An app that never used the Support library doesn't add it. Without the Support
library, load labels from assets:

    // Replaces FileUtil.loadLabels in an app that drops the Support library.
    fun loadLabels(context: Context, name: String = "labels.txt"): List<String> =
        context.assets.open(name).bufferedReader().use { it.readLines() }

### Toolchain

The 2.3.0 artifacts need JDK 17 or later for the Gradle build, `compileSdk`
35, AGP 8.6.0 or later, and the Kotlin plugin 2.3 or later. Their classes carry
Kotlin 2.4 metadata. Ask the user before editing `gradle/libs.versions.toml` or
the root `build.gradle.kts`:

- Kotlin 2.2 stops with `The binary version of its metadata is 2.4.0`. A project that stays on Kotlin 2.2 adds `freeCompilerArgs += "-Xskip-metadata-version-check"`.
- With Kotlin 2.x, a Compose app applies the `org.jetbrains.kotlin.plugin.compose` plugin in place of `composeOptions`.

## 4. Replace the delegates

- **GPU delegate** (`GpuDelegate`, `GpuDelegateFactory`): Create the GPU option of the Step 2 helper with the CPU as a second accelerator. `CompatibilityList` has no counterpart; the CPU fallback of Step 3 covers a device whose GPU can't run the model.
- **NNAPI** (`NnApiDelegate`, `setUseNNAPI(true)`): NNAPI is deprecated from Android 15, and the `CompiledModel` API has no NNAPI accelerator. Remove the delegate, and map an NNAPI choice in the app to the GPU option. You must tell the user about this change in the final report, because the speed and the numerical results of a quantized model that ran on a DSP or an NPU through NNAPI change on the GPU. NPU acceleration is a separate topic that this skill doesn't cover ([NPU acceleration with LiteRT][3]).
- **Select TensorFlow operators** (`FlexDelegate`): No LiteRT 2.x artifact contains the Flex delegate. This skill doesn't cover it. Tell the user that the model needs to be exported again without Select TensorFlow operators.

    // Replaces GpuDelegate: the CPU runs the operators that the GPU doesn't
    // support, as it did with the delegate. Keep the CPU fallback of Step 3 for a
    // model that the GPU can't compile at all.
    val gpuWithCpuFallback = CompiledModel.Options(Accelerator.GPU, Accelerator.CPU)

## 5. Rewrite the inference code

Never search and replace `Interpreter`. Move the model into the `Classifier`
helper of Step 2 of the skill, owned by the single-thread executor of Step 3.
Keep the app's own classes and their public functions as the callers.

- **Model loading** : `Interpreter` from `FileUtil.loadMappedFile`, and the Support `Model.createModel`, become `CompiledModel.create` in the helper. No overload takes a `ByteBuffer`.
- **Inference** : `run` on `ByteBuffer` objects becomes `writeFloat`, `run`, and `readFloat` on buffers that the helper creates once. Use `writeInt8` and `readInt8` for int8 and uint8 tensors.
- **Thread count** : `setNumThreads` becomes `CpuOptions(numThreads = n)` on new CPU options.
- **Tensor shapes** : `getInputTensor` and its `shape` become a lookup by name. `resizeInput` has no counterpart in 2.3.0.
- **Signatures** : `runSignature` becomes `run` with maps from tensor names to buffers.
- **Closing** : Closing the interpreter becomes the helper's `close`.

    // Replaces Interpreter.Options.setNumThreads. Create new options; never change
    // the shared CompiledModel.Options.CPU.
    fun cpuOptionsWithThreads(numThreads: Int): CompiledModel.Options =
        CompiledModel.Options(Accelerator.CPU).apply {
            cpuOptions = CompiledModel.CpuOptions(numThreads = numThreads)
        }

    // Replaces interpreter.getInputTensor(0).shape(). Look the tensor up by name.
    fun inputShape(model: CompiledModel, inputName: String): List<Int>? =
        model.getInputTensorType(inputName).layout?.dimensions

    // Replaces interpreter.runSignature. Create the buffers once with
    // model.createInputBuffers(signature) and model.createOutputBuffers(signature),
    // and reuse them for every call.
    fun runSignature(
        model: CompiledModel,
        signature: String,
        inputs: Map<String, TensorBuffer>,
        outputs: Map<String, TensorBuffer>
    ) {
        model.run(inputs, outputs, signature)
    }

### With the Support library

Migrated code must not import any `org.tensorflow.lite` class, including
`org.tensorflow.lite.DataType`. The LiteRT artifacts keep these classes only for
backward compatibility. Apply these rules to the Support code:

- `ImageProcessor`, `TensorImage`, and `FileUtil` stay under `com.google.ai.edge.litert.support` when the calls don't need `DataType`.
- The Support `Model` class runs on the Interpreter API. Replace it with the `Classifier` helper of Step 2 of the skill.
- Replace a Support call that needs `DataType`, such as `TensorImage(DataType.FLOAT32)`, `TensorBuffer.createFixedSize`, or `TensorLabel`, with `preprocess` of Step 4 or with the `CompiledModel` buffers.

The following code writes a processed image into the helper's buffers and pairs
the output scores with the labels, without `DataType`:

    // Replaces the Support library's Model class. ImageProcessor and TensorImage
    // stay; the processed image goes into the CompiledModel buffers directly, and
    // the scores pair with the labels without TensorLabel.
    fun runWithSupportImage(
        model: CompiledModel,
        inputs: List<TensorBuffer>,
        outputs: List<TensorBuffer>,
        image: TensorImage,
        inputElementCount: Int,
        labels: List<String>
    ): Map<String, Float> {
        // A uint8 image (one that skipped NormalizeOp) in a float input yields NaN.
        val pixels = (image.tensorBuffer as? TensorBufferFloat)?.floatArray
            ?: error("Normalize the image to float first")
        // writeFloat accepts a shorter array without an error.
        check(pixels.size == inputElementCount) { "Input size ${pixels.size}" }
        inputs[0].writeFloat(pixels)
        model.run(inputs, outputs)
        return labels.zip(outputs[0].readFloat().toList()).toMap()
    }

### Without the Support library

`preprocess` of Step 4 of the skill replaces the code that filled the input
`ByteBuffer`. Give it the model's input size and the legacy normalization: the
`mean` and `std` of a legacy `NormalizeOp(mean, std)`, or the app's own formula,
in the same channel order and layout.

## 6. Verify in two stages

**Stage 1, on the workstation**: `./gradlew assembleDebug` passes, and neither
of the following commands prints anything. The build alone doesn't reveal
leftover Interpreter code, because the `litert` 2.3.0 artifact still contains
`org.tensorflow.lite.Interpreter`:

    grep -rnE 'org\.tensorflow\.lite\.|GpuDelegate|NnApiDelegate|FlexDelegate|support\.model\.Model|org\.tensorflow:tensorflow-lite' --include='*.kt' --include='*.java' --include='*.kts' --include='*.gradle' --include='*.toml' --include='*.pro' --exclude-dir=build --exclude-dir=.gradle .
    ./gradlew :app:dependencies --configuration releaseRuntimeClasspath | grep -E 'org\.tensorflow:tensorflow-lite|com\.google\.ai\.edge\.litert:litert[a-z-]*:1\.'

**Stage 2, on a device**: Pick one path:

- **Path A (golden output captured in Step 2)** : Extend the instrumented test of Step 2. Run the same input through the migrated preprocessing and the helper on `Accelerator.CPU`, and read the golden file with `readGoldenOutput`. The test must fail when `maxAbsDifference` is 1e-3 or more. When the golden file holds only one value, the top-1 index must match; a wrong normalization often still passes this check. Then compare the CPU and the GPU as in Step 5 of the skill.
- **Path B (no device)** : Run Stage 1 only, and report that Stage 2 didn't run.

## Out of scope

This skill doesn't migrate LiteRT in Google Play services or the
TensorFlow Lite Task Library (`tensorflow-lite-task-vision`, `-text`, or
`-audio`), which has no LiteRT release.

This skill doesn't migrate C++ code. When the app has `CMakeLists.txt`,
`Android.mk`, or `.cpp` files that link `tensorflowlite_jni`, tell the user that
the C++ module moves to the LiteRT C++ SDK, and point them to
[Use prebuilt C++ library from LiteRT Maven][2].

## Pitfalls

- **`connectedDebugAndroidTest` deletes the golden file** : it uninstalls both APKs when the run ends, and the app's external files directory goes with them. For the capture of Step 2, run `./gradlew installDebug installDebugAndroidTest`, then `adb shell am instrument -w -e class <test class> <applicationId>.test/androidx.test.runner.AndroidJUnitRunner`, and `adb pull` the file from `/sdcard/Android/data/<applicationId>/files/` into `app/src/androidTest/assets/golden/`.
- **The golden file is a test asset** : `app/src/androidTest/assets/` ends up in the test APK, so `readGoldenOutput` takes `InstrumentationRegistry.getInstrumentation().context`, not the app's context, which has no such asset.
- **One golden per model** : an app with more than one model records one file per model (`golden/<model>.txt`) and compares each one in Step 6.
- **A bitmap of the model's input size hides the resize** : the comparison then leaves the resize out. Use another size and aspect ratio, and the rotation that the camera path passes, in the capture and in the comparison alike.
- **Tensor names** : `inputShape`, `runSignature`, and the other by-name calls take the names of the model's signature (`args_0` and `output_0` for a LiteRT-Torch export with one input and one output). On the workstation, `CompiledModel.from_file("model.tflite").get_signature_list()` from the Python package `ai-edge-litert` prints them; the legacy `Interpreter`'s `get_signature_list()` prints `{}` for a model without a signature definition.
- **`writeFloat` accepts a shorter array** : it writes the values it gets and leaves the rest of the input unchanged, with no error; only a longer array throws. Check the element count before writing, as `runWithSupportImage` does.
- **`writeInt8` into a float input** : the uint8 bytes of an image that skipped `NormalizeOp` go in with no error, and every output value is NaN. `runWithSupportImage` refuses such an image before it writes.
- **`getInputTensorType` and uint8** : in 2.3.0 it throws `LiteRtException` for a uint8 tensor; `getInputBufferRequirements(name).bufferSize` still gives the byte size.
- **The version catalog hides legacy artifacts from the source search** : a catalog entry splits the group, the name, and the version over three places, and an artifact that another library pulls in appears in no file. The dependencies command of Step 6 finds both.
- **`Accelerator.GPU` alone throws for a partly supported model** : when an operator stays off the GPU and the options name no CPU, `create` throws `LiteRtException` (`Failed to compile model`; Logcat logs `Following operations are not supported by GPU delegate`). The GPU option of Step 4 names the CPU as the second accelerator for that reason; the CPU fallback of Step 3 of the skill still covers a model that the GPU can't compile at all.
- **Without `litert-gpu`, the GPU option runs on the CPU** : `CompiledModel.Options(Accelerator.GPU, Accelerator.CPU)` creates the model and runs it on the CPU, with only the log line `GPU accelerator could not be loaded and registered`, and the app reports the GPU. Keep both dependencies of Step 1 of the skill.
- **`litert-support:2.3.0` raises `litert`** : it depends on `litert:2.3.0`, and Gradle resolves the higher version, so a 1.x `litert` line in the build file is overridden with no message. Write `litert:2.3.0` so that the build file says what runs.

[1]: https://github.com/google-ai-edge/LiteRT/blob/main/litert/support_lib/MIGRATION_GUIDE.md
[2]: https://ai.google.dev/edge/litert/next/android_cpp_sdk
[3]: https://ai.google.dev/edge/litert/next/npu
