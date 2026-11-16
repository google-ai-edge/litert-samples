# Device tests for the litert-runtime and litert-lm skills

This Gradle project builds the Kotlin of [`litert-runtime`](../litert-runtime/SKILL.md) and [`litert-lm`](../litert-lm/SKILL.md) into two apps and runs it on a device, through the paths where a step fails: a load that throws part-way, a `load()` while a model is loaded, and `onCleared()` or `load()` while a reply is streaming.

The apps' Kotlin is not kept here. At build time the `extractSkillCode` task takes the Kotlin code blocks out of each skill and writes them to `<module>/build/generated/skill/SkillCode.kt`, so the code under test is the text of the skill. Both `SKILL.md` files are the text [android/skills](https://github.com/android/skills) publishes: their code blocks are indented by four spaces and each shows its own imports, so the task moves the imports to the top of the file, each once, and takes only the blocks that start with an `import` or a `//` comment, which leaves out the Gradle and manifest blocks of step 1 (the module's `build.gradle.kts` and `AndroidManifest.xml` have them), the shell commands, and the calls shown inside a step. In the same way the `extractMigrationCode` task writes the Kotlin of `litert-runtime/references/migration.md`, whose code blocks start with a comment and share one block of imports, to `litert-runtime/build/generated/migration/MigrationCode.kt`, which goes into the test APK. Checked in are the package line (`skill-header.txt`), an activity that shows the skill's screen, the manifest, the tests, the models the tests load, and the golden output of the migration tests.

## Run

The Android SDK (`ANDROID_HOME`, or Android Studio) and JDK 17. Connect a device or an emulator with Android 11 or later (the `litert-runtime` tests swap the app's model with a `ResourcesLoader`, API 30).

```sh
./gradlew :litert-runtime:connectedDebugAndroidTest
```

The LiteRT-LM tests load a real model. Download `Qwen3-0.6B.litertlm` (0.6 GB) from https://huggingface.co/litert-community/Qwen3-0.6B and push it first (the tests copy it once more into the app's cache, for a load under another path):

```sh
adb push Qwen3-0.6B.litertlm /data/local/tmp/
./gradlew :litert-lm:connectedDebugAndroidTest
```

Options, each added to the `./gradlew` line:

- `-Pandroid.testInstrumentationRunnerArguments.backend=GPU` runs the LiteRT-LM tests on the GPU backend.
- `-Pandroid.testInstrumentationRunnerArguments.modelPath=<path on the device>` uses another `.litertlm` file.
- `-PlitertVersion=<version>` builds the `litert-runtime` app against another release of `litert` (`litert-gpu` is added from 2.3.0 on; for 2.2.0 or 2.1.6 add `-Pandroid.uniquePackageNames=false`, step 1 of the skill); `-PlitertlmVersion=<version>` builds the `litert-lm` app against another release of `litertlm-android`. The migration tests are built from 2.3.0 on, the first release of LiteRT Support under `com.google.ai.edge.litert.support`.
- `-PskillsDir=<dir>` builds the apps from another copy of the two skills (a relative path counts from this directory). A copy without `litert-runtime/references/migration.md` is built without the migration tests.

## What the tests check

Every test fails if anything is thrown on the ViewModel's thread, because an exception there ends the app. The screens are compiled and `MainActivity` shows them; the tests drive the ViewModels.

`MainViewModelTest` ([LiteRT](https://github.com/google-ai-edge/litert), 20 tests):

| Path | What is checked |
|---|---|
| `load()` and `classify()` on the CPU and on the GPU | the state is `ready` and the result has the model's output size |
| the warm-up `writeFloat` throws (on the CPU, and on the GPU and then on the CPU it falls back to), the warm-up `run` throws, creating the input or the output buffers throws after `create`, `create` throws on a file that is not a model | the state is `error`, and 50 such loads keep under 1 MB of native heap and under 10 threads |
| `load(Accelerator.GPU)` on a graph the GPU does not compile | the model loads on the CPU (`accelerator` is `CPU`, no `error`) and `classify()` works; 20 ViewModels in a row that load it this way and close it in `onCleared()` keep under 1 MB of native heap and under 10 threads |
| `load()` after a failed load, and `load()` while a model is loaded | the first loads; the second keeps the loaded model open |
| `classify()` throws | the state is `error` and the next `classify()` works |
| `onCleared()` after a load, right after `load()`, with no load; `load()` after `onCleared()` | the model is closed and the thread ends; nothing is created afterwards |
| 200 ViewModels in one process, each loading a model and closing it in `onCleared()`; a new ViewModel loading while the old one closes, 50 times | every one of them loads, and every model is closed |
| `preprocess()` on a `HARDWARE` bitmap, the kind `ImageDecoder` and the photo picker return | the input array comes out; `getPixels` on such a bitmap would throw |

The skill's `load()` catches the `LiteRtException` of a load on the GPU and loads the model on the CPU, and `UiState.accelerator` says which one loaded, so a GPU that cannot take the model is not an `error`. The tests that need the GPU are skipped where the app's own model ends on the CPU.

A model that was never closed cannot be reached from a test, which is why the failing loads are repeated: on the Galaxy S26, 20 unclosed CPU models kept 13.8 MB of native heap and 20 unclosed GPU models 62.9 MB and 40 threads. Whether a model was closed is read from LiteRT's `JniHandle.destroyed` field by reflection (present in 2.2.0 and 2.3.0). The models are at most 2.2 KB each, made by `tools/make_fixtures.py`, in `litert-runtime/fixtures/` (its `.gitignore` re-includes `*.tflite`). A `ResourcesLoader` serves the one under test as `assets/model.tflite`.

`MemoryOptionsTest` (7 tests) runs `stageModelAsset` and `memoryOptions`, the memory options of step 2 of the skill, with the app's own model, on one thread that creates, runs and closes every model:

| Path | What is checked |
|---|---|
| `stageModelAsset()`, twice | the first call copies the asset to `noBackupFilesDir/v1-model.tflite`; the second returns that file and does not copy again |
| `stageModelAsset(version = "v2")` after `v1` | the `v2` file is the asset, and the `v1` file is deleted |
| `stageModelAsset()` after a copy that stopped part-way (the first 100 bytes in `v1-model.tflite.tmp`) | the staged file is the whole asset, and the temporary file is gone |
| two loads from the staged file with `memoryOptions()` | the first writes `codeCacheDir/model.tflite-v1.xnnpack_cache`; the second leaves it as it is and gives the same output bit for bit, within 1e-5 of the skill's `Classifier` loading the asset |
| `memoryOptions(version = "v2")` after a load with `v1` | the `v1` cache is deleted; the cache of another model stays |
| a load from the staged file with `memoryOptions(context, Accelerator.GPU)`, on a device whose GPU takes the app's model | the output is within 1e-5 of the app's model on the GPU with the default options |
| `create` on a path with no file, with `memoryOptions()` | `LiteRtException` |

`MigrationCodeTest` (14 tests) runs the Kotlin of [`litert-runtime/references/migration.md`](../litert-runtime/references/migration.md) with the app's own model on the CPU. It is in `litert-runtime/src/migrationTest/` and goes into the test APK with `MigrationCode.kt`:

| Path | What is checked |
|---|---|
| `runWithSupportImage` with the pattern bitmap through `NormalizeOp(127.5f, 127.5f)`; the same bitmap through the skill's `preprocess` and `Classifier` | both outputs match `golden/output.txt` (`maxAbsDifference` below 1e-3), and the scores come back paired with the ten labels of `fixtures/labels.txt` (`loadLabels`) |
| `runWithSupportImage` with a uint8 image (no `NormalizeOp`), and with a normalized 32 x 32 image | each one is refused with `IllegalStateException` before the model runs |
| the uint8 image's bytes written into the float input with `writeInt8`, what the refusal prevents | the write does not throw, and every output value is NaN |
| the gate with `golden/output_wrong.txt`, whose first value is moved by 0.01; `maxAbsDifference` on arrays of two lengths | the difference is not below 1e-3; the length mismatch is refused |
| `cpuOptionsWithThreads(2)`, `inputShape(model, "args_0")`, `inputSizeBytes(model, "args_0", "serving_default")` | the model runs with the new options and the shared `CompiledModel.Options.CPU` is untouched; the shape is `[1, 224, 224, 3]`; the size is 224 x 224 x 3 x 4 bytes |
| `runSignature` on `serving_default`; `createSignatureBuffers` with `args_0` and `output_0`, then `run` with the two maps | each output equals the run without a signature key |
| `gpuWithCpuFallback` on `gpu_unsupported.tflite`, on a device whose GPU takes the app's model | `Accelerator.GPU` alone throws `LiteRtException`; with the CPU as the second accelerator the model is created and runs, and its outputs are the golden's values |
| `writeGoldenOutput` to a file, read back | the values round-trip |

`litert-runtime/fixtures/golden/` holds the output of the app's model for that input on the workstation, from `tools/make_golden.py` (the `CompiledModel` API of the Python package `ai-edge-litert` on the CPU).

`ChatViewModelTest` ([LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM), 20 tests):

| Path | What is checked |
|---|---|
| `load()`, then `send()`; `send()` before `load()`; `send()` while a reply streams | a reply streams; nothing happens; nothing happens |
| `load()` with another model path, with the same path again, with the same path on the other backend, and alternately between two paths | the engine and the conversation that were replaced are closed; the same path keeps the engine; the process does not grow by another engine |
| `load()` fails on a missing file and on a file that is not a model, also after a good load | the state is `error`, nothing stays open, 20 such loads keep under 1 MB of native heap, and the next `load()` works |
| `onCleared()` while a reply streams, before its first chunk, and right after `send()` | the reply is stopped, and the conversation and the engine are closed |
| `load()` with another model right after `send()` and while a reply streams; `send()` right after `load()` with another model | the reply is stopped, the first engine is closed, and the old reply does not reach the new chat; the message goes to the new model |
| `onCleared()` while a `load()` waits for a reply to stop, right after `load()`, with no load; `load()` after `onCleared()` and while `onCleared()` stops a reply | no engine stays open; nothing is created afterwards |

A reply that was stopped ends its coroutine as cancelled; one that ran to its end does not. The tests use that, and a limit of 20 seconds until the coroutine ends, to tell the two apart.

## Tested on

- Galaxy S26 (Android 16): `litert` 2.2.0, and 2.3.0 with `litert-gpu` 2.3.0, on the CPU and the GPU; `litertlm-android` 0.18.0 and 0.17.1 on the CPU and the GPU with `Qwen3-0.6B.litertlm`, and 0.18.0 with `gemma-4-E2B-it.litertlm`.
- Pixel 8a and Galaxy S26 (Android 16): `MemoryOptionsTest` and `MigrationCodeTest`, with `litert` 2.3.0 and `litert-gpu` 2.3.0.
- Android emulator (API 36, arm64): the `litert-runtime` tests, 17 of 20. The GPU accelerator loads there through OpenGL; the first GPU test runs, and the other three are skipped because their GPU load does not compile.
