---
title: Running a model with flutter_edge_ai
source: https://pub.dev/packages/flutter_edge_ai/versions/2.1.1 (skills/flutter-edge-ai-inference/SKILL.md, README.md) ; https://pub.dev/packages/flutter_edge_ai_litertlm/versions/1.10.1 (README.md, CHANGELOG.md) ; https://github.com/DenisovAV/flutter_edge_ai/blob/8ab3ca0d53dd4fab353d5880635d95e9d888472b/website/content/docs/getting-started.md ; https://github.com/DenisovAV/flutter_edge_ai/blob/8ab3ca0d53dd4fab353d5880635d95e9d888472b/packages/flutter_edge_ai/DESKTOP_SUPPORT.md
license: MIT
---

# Running a model with flutter_edge_ai

This document explains how to load and run an on-device language model with flutter_edge_ai (formerly flutter_gemma) and its recommended LiteRT-LM engine, flutter_edge_ai_litertlm: installing Gemma 4, creating sessions and chats, streaming and stopping generation, sending images and audio, thinking mode, running several conversations, choosing a CPU, GPU or NPU backend, and the common traps that make a reply empty, repetitive or wrong.

## Rules for running a model with flutter_edge_ai

1. Depend on `flutter_edge_ai` and an engine package, and import both. Engine packages do not re-export core.
2. Register the engine in `FlutterEdgeAi.initialize(inferenceEngines: [...])`. Core ships none.
3. Declare `fileType` on `installModel`. It defaults to `ModelFileType.task`, and the declaration — never the file name — picks the engine.
4. `maxTokens` is the context window. Cap the reply with `maxOutputTokens` on the session or chat.
5. Pass `isUser: true` on every user `Message`.
6. Close a session or chat when its conversation ends. Keep the model while the feature is in use, and close it when the app no longer needs it.
7. Keep Hugging Face tokens out of source: read them with `String.fromEnvironment`. That keeps a token out of git, not out of the app.
8. On Android, set `minSdk 30` for anything built on `.litertlm` — inference, embeddings, speech.
9. Complete the platform setup before the first build on a platform: without those entries the model fails to load or the app is killed for memory.

## Setting up the recommended .litertlm engine

Add the packages with `flutter pub add flutter_edge_ai flutter_edge_ai_litertlm`, register the engine, install the model and load it:

```dart
WidgetsFlutterBinding.ensureInitialized();
await FlutterEdgeAi.initialize(inferenceEngines: [LiteRtLmEngine()]);
await FlutterEdgeAi.installModel(
  modelType: ModelType.gemma4,
  fileType: ModelFileType.litertlm,
).fromNetwork(
  'https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm',
).withProgress((p) => print('downloading: $p%')).install();
final model = await FlutterEdgeAi.getActiveModel(maxTokens: 1024);
```

`initialize()` needs the Flutter binding, so call `WidgetsFlutterBinding.ensureInitialized()` first in `main()`. Gemma 4 E2B is 2.6 GB and needs no token. On web use `gemma-4-E2B-it-web.litertlm` from the same repo (2.0 GB). `install()` skips the download when the file is already on disk, so calling it at every launch is safe. The latest install becomes the model `getActiveModel` loads. A gated repo needs a token, given once to `FlutterEdgeAi.initialize(huggingFaceToken: ...)`. Other sources on the same builder are `.fromAsset(path)`, `.fromFile(path)` and `.fromBundled(name)`.

## Installing from a Hugging Face deployment manifest

When a Hugging Face repo publishes a deployment manifest (`litertlm_manifest.json`), one call picks the variant and its tested runtime settings. The manifest describes every `.litertlm` file in the repo — which backends each is verified on, which file a given platform should pick, sha256 and size identity, and session guidance. `LitertlmManifestResolver` reads it so an app installs the right file for the device without hardcoding filenames. `LiteRtLmEngine` carries this resolver, so registering the engine is enough.

`FlutterEdgeAi.installModel(...).fromHuggingFace('litert-community/LFM2.5-230M').install()` returns an installation whose `runtime` field is passed to `getActiveModel(defaults: install.runtime)`. Everything the manifest returns is an overridable default: an explicit argument wins over the manifest, which wins over the SDK default. `notes` carries platform caveats and known issues. The two-step form — `FlutterEdgeAi.resolveHuggingFace(repo, fileType:)` followed by `fromNetwork(r.url)` — stays the offline-safe one, because manifest mode needs the network on every install. Repos without a manifest keep working through `fromHuggingFace(repo, file: ...)`.

## Why maxTokens is the context window, not the reply length

`maxTokens` on `getActiveModel` is the context window: the total budget shared by the input (system prompt, history and your message) and the generated reply — the KV-cache size. It is not the response length. Asking for `maxTokens: 100` requests a 100-token context, not a 100-token reply, and replies stay as long as ever. On native `.litertlm` the value is raised to 1024, the smallest context those models support, and only a debug-mode log says so. The web `.litertlm` engine does not take the value at all; on MediaPipe it is the real limit.

To limit how many tokens the model generates, use `maxOutputTokens` on `createSession`, `openSession`, `createChat` or `openChat`:

```dart
final model = await FlutterEdgeAi.getActiveModel(maxTokens: 1024); // context
final chat = await model.createChat(maxOutputTokens: 100);         // reply cap
```

Use 4096 or more with images or audio — one image costs hundreds of tokens.

## Why a reply comes back empty when isUser is missing

The symptom is an empty response with no error. `Message.isUser` defaults to `false`, so the prompt is read as the model's own turn. The fix is `Message(text: prompt, isUser: true)` on every user message.

Three other setup traps produce errors rather than empty replies. If only the engine package is imported, the compiler reports `Undefined name 'FlutterEdgeAi'` or `Undefined class 'InferenceModel'`; import `package:flutter_edge_ai/flutter_edge_ai.dart` as well. Code written for the old package name fails with `Undefined name 'FlutterGemma'` or an unresolved `package:flutter_gemma/...` import: rename the dependencies and imports to `flutter_edge_ai*` and run `dart fix --apply`, which renames the old class names. If no engine is registered, `getActiveModel` throws `StateError: No inference engine can handle this model (ModelFileType.litertlm)`; add the engine package and register its provider in `inferenceEngines:`, or fix `fileType` if the wrong engine is registered.

## Why the model gives the same reply every time

The symptom is identical output for identical input. `createSession` and `createChat` default to `topK: 1`, which is greedy decoding. The fix is to pass `topK` (for example 40) and a `temperature`, and to set them on the first session after `getActiveModel`: on `.litertlm` the first session's sampler settings can stay in effect for later ones. Decoding stays greedy whatever you pass on a `.litertlm` running on the NPU (`activeBackend == PreferredBackend.npu`) and on ONNX; those ignore the sampling parameters.

This is an upstream defect (LiteRT-LM issue 2080): only the first generation on an engine sets the sampler, and every later session on that engine keeps those values, whatever it asks for. An app that runs one generation with defaults is from then on locked greedy — a later `temperature: 1.5` changes nothing, with no error and no warning. It also runs the other way: an engine whose first session is stochastic keeps sampling, and a later `topK: 1` will not give argmax. The seed is not re-applied either. The workaround is to close and recreate the engine — `model.close()` followed by `getActiveModel(...)` — when you need different sampler settings.

## Session is closed: one conversation slot per model

The symptom is `StateError: Session is closed` from a session or chat that is still in use. The cause is that `createSession` and `createChat` fill one slot per model; creating another closes the one before. The fix is one conversation at a time, or `openSession` and `openChat` for several. On the web `.litertlm` engine a second `createSession` hands back the session that is already open, history and all, rather than a fresh one — close the current chat before creating the next.

## Generating a single response and streaming tokens

A session is a single exchange. Create it with sampler settings and an output cap, add the user message, and either wait for the whole reply with `getResponse()` or stream tokens with `getResponseAsync()`, then close the session:

```dart
final session = await model.createSession(temperature: 0.8, topK: 40, maxOutputTokens: 256);
try {
  await session.addQueryChunk(Message(text: prompt, isUser: true));
  await for (final token in session.getResponseAsync()) {
    reply.write(token); // update the UI here
  }
} finally {
  await session.close();
}
```

## Stopping generation early

To stop early, call `await session.stopGeneration()`, or `chat.stopGeneration()` on a chat. Cancelling the stream subscription detaches Dart but does not stop native decoding on every engine.

On `.litertlm` on Android, iOS and desktop the chat keeps working after a stop, but images and audio from earlier turns are no longer visible to the model — re-send an image if the next question is about it. The first turn after a stop runs on a fresh conversation that replays the chat's history, including whatever the stopped reply had produced. That history is replayed as text, which is why earlier images and audio are lost. Every `flutter_edge_ai_litertlm` release behaves this way; before the fix, a conversation whose generation was cancelled mid-reply stayed unusable and every later message came back empty.

## Multi-turn chat with a system instruction

A chat keeps the history: add the next user message and generate again. Create it with `model.createChat(systemInstruction: 'You are a concise assistant.', temperature: 0.8, topK: 40, maxOutputTokens: 512)`, add the user message with `addQueryChunk`, and iterate `generateChatResponseAsync()`. Close the chat when the conversation ends.

`generateChatResponseAsync()` streams `ModelResponse` values, and `generateChatResponse()` returns the whole reply as one sealed `ModelResponse` — `TextResponse`, `FunctionCallResponse`, `ParallelFunctionCallResponse` or `ThinkingResponse` — so a `switch` over it must cover all four. Text tokens arrive as `TextResponse(:final token)`.

On `.litertlm`, on every platform, native and web, the system instruction is passed as a real system turn. On MediaPipe `.task` (Android, iOS and web) it is prepended to the first user message as a fallback.

## Running several conversations at once with openChat

A single loaded model can serve several independent dialogues. `openSession()` returns a session with its own conversation history; `openChat()` is the same for the chat API. The model weights — hundreds of MB to several GB — are loaded once and shared across every session; each session only adds its own lightweight context. Typical uses are a tabbed chat UI, different roles or system instructions side by side, background summarizing alongside an active chat, and A/B prompt comparison.

`openSession` and `openChat` live only on the base class, so they inherit nothing from the installed model: pass `modelType:` (and `supportImage:` if the chat sends images) explicitly, or the chat runs as `ModelType.gemmaIt` with images off. They work on `.litertlm` (native and web), on MediaPipe Android and iOS, and on built-in AI; on ONNX and MediaPipe web they throw `UnsupportedError`. Close each one.

### Concurrent contexts but serialized inference

The sessions are logically independent, but only one session generates at a time — calling generate on a second session while another is running blocks until the first finishes. Generation is not parallel, because parallel on-device inference would contend for the accelerator and risk running out of memory. Each open session holds its own context (about 100–500 MB depending on model and `maxTokens`), so on phones with Gemma 4 E2B or larger several concurrent sessions can run out of memory. Cap the count with `maxConcurrentSessions:` on `getActiveModel`; `openSession()` throws a `StateError` past the cap.

## Thinking mode for Gemma 4, Qwen3 and DeepSeek R1

Gemma 4, Qwen3 and DeepSeek R1 can emit reasoning on their supported engines. Pass `enableThinking: true` to `createChat`, together with the matching `modelType`. Reasoning arrives as `ThinkingResponse(:final content)` only from `generateChatResponseAsync()`; `generateChatResponse()` strips it. The answer itself still arrives as `TextResponse` tokens, so a streaming loop typically shows the reasoning in one place and accumulates the answer in another.

On a native `.litertlm`, reasoning that a bundle streams on its thought channel arrives as `ThinkingResponse` whatever the `modelType`, and `enableThinking: false` sends `enable_thinking: false` to the template. With `ModelType.qwen3`, core also appends ` /no_think` to text turns; Qwen3.5 and later take `ModelType.qwen35` and no suffix. On the web, the `.litertlm` engine sends the same flag and Gemma 4 E2B returns `ThinkingResponse` in Chrome; MediaPipe web has no thinking API, and DeepSeek R1 thinks on Android and iOS only, because it ships as a mobile `.task` build. Older releases called the parameter `isThinking`.

## Sending images to a multimodal model

Load the model with image support and a larger context, open a chat with image support, and attach the image bytes to a user message:

```dart
final model = await FlutterEdgeAi.getActiveModel(maxTokens: 4096, supportImage: true);
final chat = await model.createChat(supportImage: true);
await chat.addQueryChunk(
  Message(text: 'What is in this photo?', isUser: true, imageBytes: bytes),
);
```

Other message constructors are `Message.withImage(...)`, `Message.withImages(text:, imageBytes: [...], isUser: true)` for text plus several images and `Message.imagesOnly(imageBytes: [...], isUser: true)`. The plugin handles common image formats such as JPEG and PNG.

## Sending audio to Gemma 4

Load the model with `getActiveModel(maxTokens: 4096, supportAudio: true)`, create the chat with `supportAudio: true`, and send `Message(text: 'What is said in this recording?', isUser: true, audioBytes: bytes)`.

`audioBytes` is a whole WAV file — 16 kHz mono, header included. The speech package is the opposite: its `transcribe` takes raw PCM with no header. Audio input needs Gemma 4 or Gemma 3n, on Android, iOS or desktop; the `.litertlm` web engine takes no audio.

## The model is a process-wide singleton

`getActiveModel` returns one model per process. Calling it again with different runtime arguments — `maxTokens`, `preferredBackend`, `supportImage`, `maxConcurrentSessions` and the rest — rebuilds it and closes the previous one, so a handle still held stops working. Load it once, then create and close sessions per conversation.

Sessions are cheap to create and destroy. The expensive part is `engine_create` (2–10 seconds depending on backend and model size), which happens once when the model is first opened. Upstream LiteRT-LM keeps `LiteRtEnvironment` as a process singleton for GPU paths: once it is initialized with the first model's settings — cache directory, backend, capabilities — those become fixed for the process. To swap models or switch between CPU and GPU at runtime, call `model.close()` first, then `getActiveModel(...)` again with the new settings.

## Choosing a CPU, GPU or NPU backend

Pass `preferredBackend` to `getActiveModel` and read `model.activeBackend` to see what actually loaded.

| preferredBackend | Tried in order |
|---|---|
| null or gpu | GPU, then CPU |
| npu | NPU, GPU, CPU on Windows and on Qualcomm Android; GPU, CPU everywhere else |
| cpu | CPU only |

Read `activeBackend` rather than assuming the requested one loaded; the web `.litertlm` engine reports `null`. `PreferredBackend.npu` needs a Snapdragon (Android) or Intel Lunar Lake or Panther Lake (Windows) and a model compiled for that NPU. `PreferredBackend.cpu` never falls back. The iOS Simulator is CPU-only. On web, MediaPipe is GPU-only.

On desktop with the GPU backend, the forward pass (prefill and decode) runs on the GPU accelerator — Metal, DirectX 12 or Vulkan. The per-token sampler runs on the GPU on Windows and on the CPU on macOS and Linux, where it costs roughly 1–5 ms per token.

## NPU caveats for Gemma models

The Qualcomm NPU on Android is opt-in since `flutter_edge_ai_litertlm` 1.10.0, because Qualcomm licenses its QNN runtime for redistribution inside an app only. The app's own `pubspec.yaml` sets `qualcomm_npu: true` for `flutter_edge_ai_litertlm` under `hooks: user_defines:`; the build hook then downloads the QNN runtime from Maven Central, checks its SHA-256 and bundles it (about 83 MB installed). Without the flag, a request for `PreferredBackend.npu` on Android falls back to GPU, then CPU, and the log names the flag.

On NPU, run a Gemma 4 bundle. A Gemma 3 bundle on either vendor's NPU drops every prefill chunk after the first — no error, no log line, and a fluent reply that answers from the opening of the prompt and ignores the rest (LiteRT-LM issue 3508).

`maxTokens` is not clamped up to 1024 on the NPU attempt the way it is on CPU and GPU, because the safe context is baked into the compiled bundle: pass the `cache_length` it was built for. Requesting `PreferredBackend.npu` does not guarantee the NPU runs — the engine falls back to GPU then CPU. The NPU candidate is attempted only on Windows and on Android phones with Qualcomm FastRPC (`libcdsprpc.so`) whose app opted in; on other Android phones, macOS, Linux and iOS it is skipped, because nothing there can run it. On Windows the check is per OS, so a PC without an Intel NPU still attempts it, and `activeBackend` can then report `npu` while the model runs elsewhere.

## Wrong numbers on the GPU and float32 activations

On GPU, Gemma 4 can copy numbers wrongly from a long prompt: `2026/06/23` comes back as `20226/12/17`, the same way on every run (LiteRT-LM issue 3012 on Adreno, issue 2814 on Metal). The published Gemma 4 `.litertlm` files ask for half-precision activations. Ask for full precision when the answers carry figures, dates or amounts:

```dart
final model = await FlutterEdgeAi.getActiveModel(
  preferredBackend: PreferredBackend.gpu,
  activationDataType: ActivationDataType.float32,
);
```

Prefill gets slower (about 3 times on a Snapdragon 8 Elite and an iPhone 11, under 1.5 times on an Apple M3 Max); decode speed barely changes. Left unset, the model file decides. It applies to the text decoder of `.litertlm` models on Android, iOS and desktop — not to the vision or audio encoders, which keep what the model file asks for. MediaPipe, ONNX, built-in AI and the web engines ignore it.

### Memory cost and caveats of float32 activations

`float32` needs more GPU memory than the default, and a GPU engine that cannot be created falls back to CPU without an error — right digits, a much slower run. After loading, check `model.activeBackend == PreferredBackend.gpu` before concluding the setting did anything. On Android the GPU shares system memory, so on a 4–6 GB phone running out of it at `float32` can end the app rather than fall back to CPU. Both precisions share one compiled GPU program cache per model, so switching recompiles the GPU programs (about 600 MB for Gemma 4 E2B): pick one precision per install rather than per request. Every `flutter_edge_ai_litertlm` release supports the setting.

## The flutter_edge_ai_litertlm engine package

`flutter_edge_ai_litertlm` (renamed from `flutter_gemma_litertlm`) is the LiteRT-LM engine for flutter_edge_ai: it runs `.litertlm` models and LiteRT `.tflite` embeddings, through `dart:ffi` on Android, iOS, macOS, Linux and Windows, and as an early preview on the web through `@litert-lm/core` and LiteRT.js. It owns the shared LiteRT-LM native library (`libLiteRtLm`), which `flutter_edge_ai_speech` also uses, and it ships the LiteRT C API embedding backend `LiteRtEmbeddingBackend`. `LiteRtLmEngine` handles `ModelFileType.litertlm` models; pass it alongside other engines, such as `MediaPipeEngine`, if the app uses both formats.

| Platform | Support |
|---|---|
| Android | FFI (GPU via OpenCL, NPU via `.litertlm` on Qualcomm) |
| iOS | FFI (GPU via Metal on device; CPU on simulator) |
| macOS and Linux | FFI (GPU via Metal or Vulkan) |
| Windows | FFI (CPU, GPU via DirectX 12, Intel NPU) |
| Web | via `@litert-lm/core` (CDN, early preview) |

The native library is fetched at build time by `hook/build.dart` (Native Assets) from a SHA256-verified GitHub release, so native platforms need no manual setup. Version 1.10.0 moved to LiteRT-LM v0.18.0 and renamed its Metal accelerator to `LiteRtLmMetalAccelerator`, so on iOS and macOS the GPU of `flutter_litert`, a separate LiteRT plugin, keeps working beside it. A stale Native Assets cache after a native version bump can leave the library unbundled, surfacing as a `dlopen` "no such file" error on the first inference; a clean rebuild fixes it.

## Fixed engine issues worth knowing

- Android GPU crash on Mali (fixed in engine version 1.8.2): `PreferredBackend.gpu` on phones with a Mali GPU (Samsung A-series, MediaTek, Google Tensor) killed the process while the model loaded, because the OpenCL accelerators called `AHardwareBuffer_allocate` without declaring `libandroid.so`. Adreno GPUs and the CPU backend were unaffected.
- Google Play rejection over 16 KB page sizes (fixed in engine version 1.8.0): the bundled Qualcomm Hexagon DSP blobs for the NPU path arrived with a 4 KB alignment, and Play refused the release at submission. With the opt-in NPU, the build hook now raises the alignment of the libraries it bundles to 16 KB.
- Garbled or empty streams on Android (fixed in engine version 1.5.2): an app that embedded or transcribed anything before its first generation could receive corrupted streams, because the first `dlopen` of `libLiteRtLm` decides for the whole process whether its exports are globally visible.

Every `flutter_edge_ai_litertlm` release includes these fixes.
