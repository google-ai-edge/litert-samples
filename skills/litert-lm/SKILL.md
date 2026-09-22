---
name: litert-lm
description: Adds on-device LLM inference to an Android app with the LiteRT-LM Kotlin API - load a .litertlm model from Hugging Face litert-community, run it on CPU, GPU or NPU, stream replies in a multi-turn conversation, add image and audio input and tool calling. Use this skill for on-device chat, summarization, extraction or agents with open models (Gemma, Qwen, Llama, Phi, FunctionGemma) instead of a cloud model.
license: Apache-2.0
metadata:
  author: Google LLC
  last-updated: '2026-09-22'
  keywords:
  - LiteRT-LM
  - litertlm
  - Gemma
  - on-device LLM
  - GPU
---

This skill provides step-by-step guidance for integrating LiteRT-LM in Android apps through its Kotlin API (`com.google.ai.edge.litertlm`). The model is one `.litertlm` file.

## Prerequisites

- Dependency in the app-level `build.gradle`: `implementation("com.google.ai.edge.litertlm:litertlm-android:0.17.1")` (Google Maven; `latest.release` also resolves).
- For the GPU backend, add both lines inside `<application>` in `AndroidManifest.xml`:

```xml
<uses-native-library android:name="libvndksupport.so" android:required="false"/>
<uses-native-library android:name="libOpenCL.so" android:required="false"/>
```

- The `.litertlm` file must be on the device's filesystem, for example under `context.filesDir`; it is not read from assets. Files are hundreds of MB to a few GB, so download at first launch, not inside the APK.

## Detailed steps

### 1. Get a model

Look in Hugging Face litert-community first (https://huggingface.co/litert-community): ready `.litertlm` files for Gemma 3 and 3n, Qwen3, Llama, Phi, FunctionGemma and others, with the quantization, the KV cache size and any NPU target in the file name. Only when the model is missing, convert it on a workstation: `litert-torch` (PyPI; the exporter, formerly `ai-edge-torch`) exports the checkpoint, `ai-edge-quantizer` quantizes it, and `litert_lm_builder` (from the `litert-lm` PyPI package) bundles tokenizer and metadata into `.litertlm`. Quantization: int8 dynamic is the safe default; int4 must be blockwise (block 32, or 128 for larger models), channelwise int4 degrades decoders. Follow [get a model](references/get-a-model.md).

### 2. Initialize the engine off the main thread

```kotlin
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig

val engine = Engine(EngineConfig(modelPath = modelFile.absolutePath, backend = Backend.GPU(), cacheDir = context.cacheDir.path))
engine.initialize()
```

`initialize()` blocks while the weights load (seconds); call it from a coroutine on `Dispatchers.IO` or a worker thread, never on the main thread. `Backend.CPU()` is the default; `Backend.NPU(nativeLibraryDir = context.applicationInfo.nativeLibraryDir)` needs the vendor libraries. Keep one `Engine` per process and call `engine.close()` when done.

### 3. Create a conversation and stream the reply

```kotlin
val conversation = engine.createConversation(ConversationConfig(systemInstruction = Contents.of("You are a helpful assistant.")))
conversation.sendMessageAsync("Hello").collect { message -> append(message.toString()) }
```

`sendMessageAsync(text)` returns a `Flow<Message>` of chunks; `sendMessageAsync(text, callback: MessageCallback)` is the callback form (`onMessage`, `onDone`, `onError`); `sendMessage(text)` blocks and returns the whole `Message`. The conversation keeps its history; wait for `onDone` (or the flow to complete) before sending the next message, and call `conversation.close()` to free native memory. Sampling: `SamplerConfig(topK, topP, temperature)` in `ConversationConfig`.

### 4. Images, audio and tools

Multimodal input needs a model that supports it (Gemma 3n): set `visionBackend` or `audioBackend` in `EngineConfig` and send `Contents.of(Content.ImageFile(path), Content.Text("Describe this"))`. Tools: a class implementing `ToolSet` with `@Tool` functions, passed as `ConversationConfig(tools = listOf(tool(MyTools())))` (FunctionGemma). Follow [multimodal and tools](references/multimodal-and-tools.md).

### 5. Lifecycle and delivery

- Download the model once from its Hugging Face URL to `filesDir` with progress; check the file size before use.
- Keep the engine in a `ViewModel` or an application-scoped holder; close the conversation, then the engine.
- Never put the model in assets or in the APK. A complete app that pairs LiteRT-LM with a CompiledModel classifier is `samples/litert/phototalk_sample_app` in litert-samples.

## Troubleshooting

- Engine creation fails on GPU: the two manifest lines are missing, or the file is a CPU-only variant; try the CPU variant to confirm the file.
- First reply is slow: `initialize()` ran on first use; initialize at app start and set `cacheDir` to speed up the second load.
- Reply is cut off or garbled: wrong file or quantization for the device; check the model card and try the int8 variant.

## Links

- LiteRT-LM on Android (Kotlin guide): https://ai.google.dev/edge/litert-lm/android
- LiteRT-LM repository (Kotlin API sources under `kotlin/java/com/google/ai/edge/litertlm`): https://github.com/google-ai-edge/LiteRT-LM
- LiteRT, the runtime underneath and the API for classic models: https://github.com/google-ai-edge/litert
