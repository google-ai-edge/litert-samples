# Prompts for the litert-lm skill

Six prompts for the [`litert-lm`](../litert-lm/SKILL.md) skill ([LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM); vision and audio models are the [`litert-runtime` prompts](litert-runtime.md), [LiteRT](https://github.com/google-ai-edge/litert)). Each prompt is what a developer would type to a coding agent in a fresh checkout of [android/architecture-templates](https://github.com/android/architecture-templates) (branch `base`, commit `5467a8dc`). The prompt says what the developer wants and names no API. The label is the experience the task takes. The checks are the ones of the [README](README.md): whether the project builds, what the changed files (or the answer) contain and do not contain, and one sentence read against the diff. The results are in [results-litert-lm.md](results-litert-lm.md).

## L1 · beginner · build the app
Prompt: Build a chat screen in this app that answers using a small open-weight LLM running entirely on the phone, no server, with the reply appearing as it is generated.
Builds: yes.
Contains: `com.google.ai.edge.litertlm:litertlm-android:0.18.0`, `EngineConfig`, `initialize()`, `createConversation`, `sendMessageAsync` collected chunk by chunk, a model path in the app's storage.
Does not contain: a `.litertlm` under `assets/`, `org.tensorflow`, another runtime's model format (`.task`, `.gguf`).
Read: the engine initializes off the main thread; the reply text grows as the chunks arrive; the conversation and the engine are closed when the screen's ViewModel is cleared; the answer says which litert-community file to use and how it reaches the device.
Device: a reply streams on a phone with `Qwen3-0.6B.litertlm`.

## L2 · beginner · a question
Prompt: I want to use Gemma 4 in an on-device chat app. Which model file should I use, where do I download it, and how do I get it onto the phone for development and for real users?
Contains: `litert-community/gemma-4-E2B-it-litert-lm` and its file `gemma-4-E2B-it.litertlm` (2.6 GB; `Qwen3-0.6B.litertlm` as the small alternative), the Hugging Face download URL, `adb push` and `run-as` for development, an in-app download into `filesDir` or with `DownloadManager` into `getExternalFilesDir` for users with a size check and the `INTERNET` permission, and that a file with a chip suffix is an NPU build.
Does not contain: the file under `assets/` or inside the APK, a file name that is not in the repo, a `.task` file.
Read: both routes, development and users, are there.

## L3 · intermediate · build the app
Prompt: Build a chat screen that answers with a small open-weight LLM on the phone's GPU; on the CPU of my test phone it is too slow.
Builds: yes.
Contains: `Backend.GPU()`, `libvndksupport.so` and `libOpenCL.so` as `uses-native-library` lines inside `<application>`, `cacheDir`.
Does not contain: `org.tensorflow`, an NPU backend.
Read: the two manifest lines are there (without them the engine initializes and the first reply fails with `Can not find OpenCL library on this device`); the engine initializes off the main thread.
Device: a reply streams on the GPU of a phone.

## L4 · intermediate · build the app
Prompt: Build a chat screen on an on-device LLM where the user can stop the answer while it is being generated, and start a new chat that forgets the previous messages.
Builds: yes.
Contains: `cancelProcess`, a new `createConversation` for the new chat with the old conversation closed, `sendMessageAsync`.
Does not contain: a second `Engine` for the new chat, `org.tensorflow`.
Read: Stop ends the stream and leaves the engine usable; the new chat forgets the history because the conversation is new; the send button is disabled while a reply streams.

## L5 · advanced · build the app
Prompt: Build a chat screen on an on-device LLM where the user can switch between two model files. Leaving the screen while the model is still answering, or switching models in the middle of an answer, must never crash or leak memory.
Builds: yes.
Contains: `cancelProcess`, `close()` on the conversation and then on the engine, `onCleared`, a lock that serializes the switch and the close.
Does not contain: a second `close()` on the same engine, a `close()` on an engine whose `initialize()` threw, `viewModelScope` for the close.
Read: a switch stops the reply, closes the old conversation and engine once each, and only then initializes the new engine; leaving the screen does the same under the same lock; the ViewModel's own thread outlives `viewModelScope`.

## L6 · advanced · build the app
Prompt: Build a summarizer screen: the user pastes a text, an on-device LLM returns a three-bullet summary, always in the same neutral style, and the same text gives the same summary every time.
Builds: yes.
Contains: `ConversationConfig` with `systemInstruction`, `SamplerConfig` set for a deterministic output, a new conversation per text.
Does not contain: `org.tensorflow`, the persona only in the user's message.
Read: every text goes through a fresh conversation (a kept one would carry the previous texts); the sampler is fixed; the style sits in the system instruction. The check is the configuration, not two identical outputs: the GPU computes in fp16.
