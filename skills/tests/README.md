# Device tests for the litert-runtime and litert-lm skills

This Gradle project builds the Kotlin of [`litert-runtime`](../litert-runtime/SKILL.md) and [`litert-lm`](../litert-lm/SKILL.md) into two apps and runs it on a device, through the paths where a step fails: a load that throws part-way, a `load()` while a model is loaded, and `onCleared()` or `load()` while a reply is streaming.

The apps' Kotlin is not kept here. At build time the `extractSkillCode` task takes the `kotlin` code blocks out of `../litert-runtime/SKILL.md`, `../litert-runtime/references/preprocess.md` and `../litert-lm/SKILL.md` and writes them to `<module>/build/generated/skill/SkillCode.kt`, so the code under test is the text of the skill. Checked in are the package line and the imports that the skills leave to the IDE (`skill-header.txt`), an activity that shows the skill's screen, the manifest, the tests, and the models the tests load.

## Run

Connect a device or an emulator with Android 11 or later.

```
./gradlew :litert-runtime:connectedDebugAndroidTest
```

The LiteRT-LM tests load a real model. Download `Qwen3-0.6B.litertlm` (0.6 GB) from https://huggingface.co/litert-community/Qwen3-0.6B and push it first:

```
adb push Qwen3-0.6B.litertlm /data/local/tmp/
./gradlew :litert-lm:connectedDebugAndroidTest
```

Options, each added to the `./gradlew` line:

- `-Pandroid.testInstrumentationRunnerArguments.backend=GPU` runs the LiteRT-LM tests on the GPU backend.
- `-Pandroid.testInstrumentationRunnerArguments.modelPath=<path on the device>` uses another `.litertlm` file.
- `-PlitertlmVersion=<version>` builds against another release of `litertlm-android`.
- `-PskillsDir=<dir>` builds the apps from another copy of the two skills.

## What the tests check

Every test fails if anything is thrown on the ViewModel's thread, because an exception there ends the app. The screens are compiled and `MainActivity` shows them; the tests drive the ViewModels.

`MainViewModelTest` ([LiteRT](https://github.com/google-ai-edge/litert), 16 tests):

| Path | What is checked |
|---|---|
| `load()` and `classify()` on the CPU and on the GPU | the state is `ready` and the result has the model's output size |
| the warm-up `writeFloat` throws (CPU and GPU), the warm-up `run` throws, creating the input or the output buffers throws after `create` | the state is `error`, and 50 such loads keep under 1 MB of native heap and under 10 threads |
| `create` throws on a file that is not a model, and on a graph the GPU does not compile | the state is `error`; after the GPU failure `load(Accelerator.CPU)` loads |
| `load()` after a failed load, and `load()` while a model is loaded | the first loads; the second keeps the loaded model open |
| `classify()` throws | the state is `error` and the next `classify()` works |
| `onCleared()` after a load, right after `load()`, with no load; `load()` after `onCleared()` | the model is closed and the thread ends; nothing is created afterwards |

A model that was never closed cannot be reached from a test, which is why the failing loads are repeated: one unclosed model keeps 60 KB or more of native heap. The models are at most 2.2 KB each, made by `tools/make_fixtures.py`, and kept as `litert-runtime/fixtures/*.bin` because the repository ignores `*.tflite`. A `ResourcesLoader` serves the one under test as `assets/model.tflite`.

`ChatViewModelTest` ([LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM), 17 tests):

| Path | What is checked |
|---|---|
| `load()`, then `send()`; `send()` before `load()` | a reply streams; nothing happens |
| `load()` with another model path, on the other backend, with the same arguments, and alternately between two paths | the engine and the conversation that were replaced are closed; the same arguments keep the engine; the process does not grow by another engine |
| `load()` fails on a missing file and on a file that is not a model, also after a good load | the state is `error`, nothing stays open, 20 such loads keep under 1 MB of native heap, and the next `load()` works |
| `onCleared()` while a reply streams, before its first chunk, and right after `send()` | the reply is stopped, and the conversation and the engine are closed |
| `load()` with another model right after `send()` and while a reply streams | the reply is stopped, the first engine is closed, and the old reply does not reach the new chat |
| `onCleared()` while a `load()` waits for a reply to stop, right after `load()`, with no load; `load()` after `onCleared()` | no engine stays open; nothing is created afterwards |

A reply that was stopped ends its coroutine as cancelled; one that ran to its end does not. The tests use that, and a limit of 20 seconds, to tell the two apart.

## Tested on

- Galaxy S26 (Android 16): `litert` 2.2.0 on the CPU and the GPU; `litertlm-android` 0.18.0 and 0.17.1 on the CPU and the GPU with `Qwen3-0.6B.litertlm`, and 0.18.0 with `gemma-4-E2B-it.litertlm`.
- Android emulator (API 36, arm64): the `litert-runtime` tests. It has no GPU accelerator, so the three tests that need one are skipped.
