# Device tests for the litert-runtime and litert-lm skills

This Gradle project builds the Kotlin of [`litert-runtime`](../litert-runtime/SKILL.md) and [`litert-lm`](../litert-lm/SKILL.md) into two apps and runs it on a device, through the paths where a step fails: a load that throws part-way, a `load()` while a model is loaded, and `onCleared()` or `load()` while a reply is streaming.

The apps' Kotlin is not kept here. At build time the `extractSkillCode` task takes the `kotlin` code blocks out of each skill's `SKILL.md` and `references/` (the imports, the code, and `preprocess()`) and writes them to `<module>/build/generated/skill/SkillCode.kt`, so the code under test is the text of the skill. Checked in are the package line (`skill-header.txt`), an activity that shows the skill's screen, the manifest, the tests, and the models the tests load.

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
- `-PlitertVersion=<version>` builds the `litert-runtime` app against another release of `litert` (from 2.3.0 with `litert-gpu`); `-PlitertlmVersion=<version>` builds the `litert-lm` app against another release of `litertlm-android`.
- `-PskillsDir=<dir>` builds the apps from another copy of the two skills (a relative path counts from this directory).

## What the tests check

Every test fails if anything is thrown on the ViewModel's thread, because an exception there ends the app. The screens are compiled and `MainActivity` shows them; the tests drive the ViewModels.

`MainViewModelTest` ([LiteRT](https://github.com/google-ai-edge/litert), 20 tests):

| Path | What is checked |
|---|---|
| `load()` and `classify()` on the CPU and on the GPU | the state is `ready` and the result has the model's output size |
| the warm-up `writeFloat` throws (CPU and GPU), the warm-up `run` throws, creating the input or the output buffers throws after `create`, `create` throws on a file that is not a model and on a graph the GPU does not compile | the state is `error`, and 50 such loads keep under 1 MB of native heap and under 10 threads |
| `load(Accelerator.CPU)` after the GPU did not compile the graph | the model loads and `classify()` works |
| `load()` after a failed load, and `load()` while a model is loaded | the first loads; the second keeps the loaded model open |
| `classify()` throws | the state is `error` and the next `classify()` works |
| `onCleared()` after a load, right after `load()`, with no load; `load()` after `onCleared()` | the model is closed and the thread ends; nothing is created afterwards |
| 200 ViewModels in one process, each loading a model and closing it in `onCleared()`; a new ViewModel loading while the old one closes, 50 times | every one of them loads, and every model is closed |
| `preprocess()` on a `HARDWARE` bitmap, the kind `ImageDecoder` and the photo picker return | the input array comes out; `getPixels` on such a bitmap would throw |

A model that was never closed cannot be reached from a test, which is why the failing loads are repeated: on the Galaxy S26, 20 unclosed CPU models kept 13.8 MB of native heap and 20 unclosed GPU models 62.9 MB and 40 threads. Whether a model was closed is read from LiteRT's `JniHandle.destroyed` field by reflection (present in 2.2.0 and 2.3.0). The models are at most 2.2 KB each, made by `tools/make_fixtures.py`, in `litert-runtime/fixtures/` (its `.gitignore` re-includes `*.tflite`). A `ResourcesLoader` serves the one under test as `assets/model.tflite`.

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
- Android emulator (API 36, arm64): the `litert-runtime` tests. It has no GPU accelerator, so the four tests that need one are skipped.
