# Prompts for the litert-runtime skill

Seven prompts for the [`litert-runtime`](../litert-runtime/SKILL.md) skill ([LiteRT](https://github.com/google-ai-edge/litert); language models are the [`litert-lm` prompts](litert-lm.md), [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)). Each prompt is what a developer would type to a coding agent in a fresh checkout of [android/architecture-templates](https://github.com/android/architecture-templates) (branch `base`, commit `5467a8dc`), with the tests' fixture [`model.tflite`](../tests/litert-runtime/fixtures/model.tflite) (input 1x224x224x3, output 1x10) copied to the project root where the prompt names it. The prompt says what the developer wants and names no API. The label is the experience the task takes. The checks are the ones of the [README](README.md): whether the project builds, what the changed files (or the answer) contain and do not contain, and one sentence read against the diff. The results are in [results-litert-runtime.md](results-litert-runtime.md).

## R0 · beginner · a question
Prompt: What do I need to add to this project to run a .tflite model, and where should the model file go?
Contains: `com.google.ai.edge.litert:litert:2.2.0` (or 2.3.0 with `litert-gpu`) as the one dependency, `minSdk` 24, the model under `app/src/main/assets/` (or downloaded into the app's storage), and for 2.2.0 on AGP 9 the line `android.uniquePackageNames=false`.
Does not contain: an `org.tensorflow` artifact, a `litert-gpu` 1.x artifact.
Read: the answer is enough for a first build: dependency, SDK floor, file location, the AGP 9 line.

## R1 · beginner · build the app
Prompt: I exported an image classification model as model.tflite (it is in the project root; 224x224 RGB input, one score per class out). Add it to this app so the user can pick a photo and see the top class.
Builds: yes.
Contains: the dependency above, `minSdk = 24`, the model under `app/src/main/assets/`, `CompiledModel.create`, `android.uniquePackageNames=false` (2.2.0 on AGP 9).
Does not contain: `org.tensorflow`, a second LiteRT artifact next to `litert` 2.2.0.
Read: the input and output buffers are created once, reused for every inference and closed before the model; inference runs off the main thread; the picture is scaled to 224x224 and normalized the way the model expects, from a bitmap that `getPixels` can read.
Device: the app classifies a picked photo on a phone, on the CPU and on the GPU.

## R2 · intermediate · build the app
Prompt: Run model.tflite (project root, 224x224 RGB input) on the phone's GPU, and if the GPU can't run it, use the CPU instead without crashing.
Builds: yes.
Contains: `Accelerator.GPU`, `Accelerator.CPU`, `LiteRtException` caught where the GPU model is created.
Does not contain: `org.tensorflow`, `GpuDelegate`, `litert-gpu` next to `litert` 2.2.0, a `uses-native-library` line.
Read: a GPU load that throws leads to a CPU load, not a crash, and not to a claim that unsupported operations fall back on their own; the first inference happens at load, so the GPU's shader compilation is not in the user's first tap.
Device: the GPU run classifies the picture on a phone.

## R3 · intermediate · build the app
Prompt: Add model.tflite (project root, 224x224 RGB input) to the app behind a ViewModel, and make sure that rotating the screen, or leaving the screen and coming back, never leaks the model or crashes.
Builds: yes.
Contains: a `ViewModel` that owns the model, `onCleared`, `close()` on the buffers and the model.
Does not contain: `org.tensorflow`, a model created in a composable or an activity.
Read: the model is loaded once and kept across a rotation (a second load is a no-op); when the ViewModel is cleared, the buffers and then the model are closed on the thread that runs them, after any inference in flight; one `Environment` serves the process.
Device: two rotations and a leave-and-return on a phone, no crash, one LiteRT environment created.

## R5 · advanced · a question
Prompt: model.tflite (project root) gives the right class in Python on my laptop but a different, wrong class in my Android app for the same picture. How do I find out what is different?
Contains: a comparison with the reference output of one fixed input on the CPU first (that tells the model file from the device), then on the GPU; the preprocessing (size, mean and std, channel order, layout) as the first suspect; the bitmap's config (a `HARDWARE` bitmap copied to `ARGB_8888` before reading pixels); `writeFloat` accepting a shorter array without an error.
Does not contain: the GPU as the first suspect, a change of runtime.
Read: the answer is a procedure the developer can run, not a guess.

## R6 · intermediate · a question
Prompt: After adding the LiteRT dependency my project no longer builds. The error is: Namespace 'com.google.ai.edge.litert' is used in multiple modules and/or libraries: com.google.ai.edge.litert:litert:2.2.0, com.google.ai.edge.litert:litert-api:2.2.0. Please ensure that all modules and libraries have a unique namespace. Then: Manifest merger failed with multiple errors. What should I change?
Contains: `android.uniquePackageNames=false` in `gradle.properties` for 2.2.0, or 2.3.0 with `litert-gpu` (its two libraries have different namespaces).
Does not contain: an exclusion of `litert-api`, a downgrade of AGP, a `litert-gpu` 1.x artifact.
Read: the answer says why: `litert` and `litert-api` 2.2.0 declare the same namespace, and AGP 9 stops on it.

## R7 · advanced · build the app
Prompt: Add my audio-event model audio.tflite (project root; input = 1 second of 16 kHz mono samples) so the app classifies a 1-second clip I already hold as a FloatArray.
Builds: yes (the file named in the prompt is the fixture under another name; the run is not taken to a device).
Contains: `CompiledModel.create`, an input element count of 16000, `writeFloat` of the clip.
Does not contain: `org.tensorflow`, a bitmap step on the audio path.
Read: the inference class keeps the shape of the image one with the input element count changed; the samples are written as they are, and the output is read as one score per event.
