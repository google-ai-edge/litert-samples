---
title: Running a model with flutter_edge_ai
source: https://pub.dev/packages/flutter_edge_ai/versions/2.5.1 (skills/flutter-edge-ai-inference/SKILL.md, README.md, CHANGELOG.md, lib/src/sampling.dart, lib/src/domain/multimodal_order.dart, lib/src/api/flutter_edge_ai.dart) ; https://pub.dev/packages/flutter_edge_ai_litertlm/versions/1.17.0 (README.md: Sampling, Context and reply length, Multimodal order, Platforms, Where a model runs, Strict backend, Several models at once, Inspecting a model file, Google Tensor NPU, MediaTek NPU, Qualcomm NPU, Troubleshooting; CHANGELOG.md; lib/src/litert_lm_load_errors.dart, lib/src/bundle/litertlm_load_plan.dart, lib/src/ffi/backend_preference.dart, lib/src/ffi/ffi_inference_model.dart) ; https://github.com/DenisovAV/flutter_edge_ai/blob/d35c575ab91e078c47201131800cf33dd7630833/website/content/docs/getting-started.md ; https://github.com/DenisovAV/flutter_edge_ai/blob/d35c575ab91e078c47201131800cf33dd7630833/packages/flutter_edge_ai/DESKTOP_SUPPORT.md
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

`initialize()` needs the Flutter binding, so call `WidgetsFlutterBinding.ensureInitialized()` first in `main()`. Gemma 4 E2B is 2.6 GB and needs no token. On web use `gemma-4-E2B-it-web.litertlm` from the same repo (2.0 GB). `install()` skips the download when the file is already installed from the same source, so calling it at every launch is safe; installing the same file name again from another source replaces the install. The latest install becomes the model `getActiveModel` loads. A gated repo needs a token, given once to `FlutterEdgeAi.initialize(huggingFaceToken: ...)`. Other sources on the same builder are `.fromAsset(path)`, `.fromFile(path)` and `.fromBundled(name)`.

## Installing from a Hugging Face deployment manifest

When a Hugging Face repo publishes a deployment manifest (`litertlm_manifest.json`), one call picks the variant and its tested runtime settings. The manifest describes every `.litertlm` file in the repo — which backends each is verified on, which file a given platform should pick, sha256 and size identity, and session guidance. `LitertlmManifestResolver` reads it so an app installs the right file for the device without hardcoding filenames. `LiteRtLmEngine` carries this resolver, so registering the engine is enough.

`FlutterEdgeAi.installModel(...).fromHuggingFace('litert-community/LFM2.5-230M').install()` returns an installation whose `runtime` field is passed to `getActiveModel(defaults: install.runtime)`. Everything the manifest returns is an overridable default: an explicit argument wins over the manifest, which wins over the SDK default. `notes` carries platform caveats and known issues. The two-step form — `FlutterEdgeAi.resolveHuggingFace(repo, fileType:)` followed by `fromNetwork(r.url)` — stays the offline-safe one, because manifest mode needs the network on every install. Repos without a manifest keep working through `fromHuggingFace(repo, file: ...)`.

## Why maxTokens is the context window, not the reply length

`maxTokens` on `getActiveModel` is the context window: the total budget shared by the input (system prompt, history and your message) and the generated reply — the KV-cache size. It is not the response length. Asking for `maxTokens: 100` requests a 100-token context, not a 100-token reply, and replies stay as long as ever. On native `.litertlm` on the CPU and the GPU a value below 1024 is raised to 1024, because the published bundles fail to allocate below it, and only a debug-mode log says so; on the NPU the value is passed through, so pass the context the bundle was compiled for. The web `.litertlm` engine does not take the value at all; on MediaPipe it is the real limit.

To limit how many tokens the model generates, use `maxOutputTokens` on `createSession`, `openSession`, `createChat` or `openChat`:

```dart
final model = await FlutterEdgeAi.getActiveModel(maxTokens: 1024); // context
final chat = await model.createChat(maxOutputTokens: 100);         // reply cap
```

Use 4096 or more with images or audio — one image costs hundreds of tokens.

## Why a reply comes back empty when isUser is missing

The symptom is an empty response with no error. `Message.isUser` defaults to `false`, so the prompt is read as the model's own turn. The fix is `Message(text: prompt, isUser: true)` on every user message.

Four other setup traps produce errors rather than empty replies. If only the engine package is imported, the compiler reports `Undefined name 'FlutterEdgeAi'` or `Undefined class 'InferenceModel'`; import `package:flutter_edge_ai/flutter_edge_ai.dart` as well. Code written for the old package name fails with `Undefined name 'FlutterGemma'` or an unresolved `package:flutter_gemma/...` import: rename the dependencies and imports to `flutter_edge_ai*` and run `dart fix --apply`, which renames the old class names. An import of core's internals, such as `package:flutter_edge_ai/core/model.dart`, fails with `Target of URI doesn't exist` since `flutter_edge_ai` 2.3.0 made them private under `lib/src`: import `package:flutter_edge_ai/flutter_edge_ai.dart` instead. If no engine is registered, `getActiveModel` throws `StateError: No inference engine can handle this model (ModelFileType.litertlm)`; add the engine package and register its provider in `inferenceEngines:`, or fix `fileType` if the wrong engine is registered.

## Why the model gives the same reply every time

The symptom is identical output for identical input. Since `flutter_edge_ai` 2.2.0, sampling values left unset on `createSession` and `createChat` come from the sampler the `.litertlm` file ships, else from the model family's defaults (`SamplingParams.forModelType`), else `temperature: 0.8, topK: 40, topP: 0.95, randomSeed: 1`; a value you pass wins, field by field. Up to `flutter_edge_ai` 2.1 an unset `topK` was 1, so every session decoded greedily. The causes today are `topK: 1` passed somewhere, a family that decodes greedily by default (`ModelType.phi`, `ModelType.hammer`), or a fixed seed: an unset `randomSeed` is 1, or the bundle's own, so a freshly loaded model repeats its first sampled answer.

The fix is to leave `temperature`, `topK` and `topP` unset, which gives the model's own sampler or its family's defaults, or to pass a `temperature`: with no `topK`, a greedy family or bundle then contributes no `topK: 1`. Decoding stays greedy whatever you pass on a `.litertlm` running on the NPU (`activeBackend == PreferredBackend.npu`), and a one-time warning says so when you set any value. ONNX follows the model's `genai_config.json`, which is greedy unless it sets `do_sample`; values you pass override it.

On `.litertlm`, LiteRT-LM keeps the sampler of an engine's first generation, random state included, for every later conversation on that engine (LiteRT-LM issue 2080), so set the sampling values on the first session after `getActiveModel`. A later session that asks for other values keeps the first ones; since `flutter_edge_ai_litertlm` 1.12.0 a warning says so once per loaded model when either set was chosen by the app. Because the random state carries on, a new chat on the same model already gets a new answer to the same prompt. To change the sampler, close and reload the model — `model.close()` followed by `getActiveModel(...)`. MediaPipe starts each session from its seed, so vary `randomSeed` there for a different answer to the same prompt.

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

On `.litertlm` on Android, iOS and desktop the chat keeps working after a stop, whether the app called `stopGeneration()` or cancelled its subscription. Since `flutter_edge_ai_litertlm` 1.15.0 the first turn after a stop rewinds the conversation to before the stopped turn and puts that turn back — the message with its images and audio, and whatever the reply had produced — so the model keeps the whole chat, media included. On the NPU, whose executor rolls back one token at most, and in 1.14.0 and earlier, that turn runs instead on a fresh conversation that replays the history as text, so images and audio from earlier turns are gone: re-send an image there if the next question is about it. A rewind that fails falls back to the same replay and says so in the log, in release builds too. Before the fix in `flutter_gemma_litertlm` 1.8.1, the package's old name, a conversation whose generation was cancelled mid-reply stayed unusable and every later message came back empty.

## Multi-turn chat with a system instruction

A chat keeps the history: add the next user message and generate again. Create it with `model.createChat(systemInstruction: 'You are a concise assistant.', temperature: 0.8, topK: 40, maxOutputTokens: 512)`, add the user message with `addQueryChunk`, and iterate `generateChatResponseAsync()`. Close the chat when the conversation ends.

`generateChatResponseAsync()` streams `ModelResponse` values, and `generateChatResponse()` returns the whole reply as one sealed `ModelResponse` — `TextResponse`, `FunctionCallResponse`, `ParallelFunctionCallResponse` or `ThinkingResponse` — so a `switch` over it must cover all four. Text tokens arrive as `TextResponse(:final token)`.

On `.litertlm`, on every platform, native and web, the system instruction is passed as a real system turn. On MediaPipe `.task` (Android, iOS and web) it is prepended to the first user message as a fallback.

## Running several conversations at once with openChat

A single loaded model can serve several independent dialogues. `openSession()` returns a session with its own conversation history; `openChat()` is the same for the chat API. The model weights — hundreds of MB to several GB — are loaded once and shared across every session; each session only adds its own lightweight context. Typical uses are a tabbed chat UI, different roles or system instructions side by side, background summarizing alongside an active chat, and A/B prompt comparison.

`openSession` and `openChat` live only on the base class, so they inherit nothing from the installed model: pass `modelType:` (and `supportImage:` or `supportAudio:` if the chat sends images or audio) explicitly, or the chat runs as `ModelType.gemmaIt` with images and audio off. They work on `.litertlm` (native and web), on MediaPipe Android and iOS, on built-in AI, and on `flutter_edge_ai_llama` (`.gguf`), whose `openChat` does take the installed model's type and image and audio support; on ONNX and MediaPipe web they throw `UnsupportedError`. Close each one.

### Concurrent contexts but serialized inference

The sessions are logically independent, but only one session generates at a time — calling generate on a second session while another is running blocks until the first finishes. Generation is not parallel, because parallel on-device inference would contend for the accelerator and risk running out of memory. Each open session holds its own context (about 100–500 MB depending on model and `maxTokens`), so on phones with Gemma 4 E2B or larger several concurrent sessions can run out of memory. Cap the count with `maxConcurrentSessions:` on `getActiveModel`; `openSession()` throws a `StateError` past the cap.

## Thinking mode for Gemma 4, Qwen3, DeepSeek R1 and thought-channel models

Gemma 4, Qwen3, DeepSeek R1 and `.litertlm` bundles with a thought channel (for example SmolLM3 and Phi-4 Mini Reasoning) can emit reasoning on their supported engines. Pass `enableThinking: true` to `createChat`, together with the matching `modelType`. Reasoning arrives as `ThinkingResponse(:final content)` only from `generateChatResponseAsync()`; `generateChatResponse()` strips it. The answer itself still arrives as `TextResponse` tokens, so a streaming loop typically shows the reasoning in one place and accumulates the answer in another.

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

`audioBytes` is a whole WAV file — 16 kHz mono, header included. The speech package is the opposite: its `transcribe` takes raw PCM with no header. Audio input needs Gemma 4 or Gemma 3n, on Android, iOS or desktop; VibeVoice-ASR also takes audio, for transcription only (measured by the package on Android and macOS). On the `.litertlm` web engine an audio message throws `UnsupportedError`.

## The order of images, text and audio in a message

A user message with an image or audio reaches the model as parts, and the model's chat template renders them in the order they arrive. Since `flutter_edge_ai` 2.4.0 with `flutter_edge_ai_litertlm` 1.14.0 the `.litertlm` engine chooses that order by model type: a Gemma 4 message is sent as images, text, then audio, as the Gemma 4 model card asks, and every other model type as images, audio, then text, the order of earlier releases. `flutter_edge_ai_litertlm` 1.13.x ignores the setting.

A model whose chat template expects another order takes `getActiveModel(multimodalOrder: MultimodalOrder.imagesAudioText)` or `MultimodalOrder.imagesTextAudio`. `MultimodalOrder(images:, audio:)` places each medium before or after the text with `MediaPosition.beforeText` or `MediaPosition.afterText`; media on the same side keep images before audio. Changing the order rebuilds the model. The web engine sends no media to a language model, so it has nothing to order.

The package measured the Gemma 4 order on Galaxy S24 and S25 Ultra phones: over 244 paired spoken turns, 22 were answered only with the audio after the text and 2 only with it before, while VibeVoice-ASR transcribed far worse with its audio after the prompt. That is why the multimodal order follows the model type rather than changing for every model.

## The model is a process-wide singleton

`getActiveModel` returns one model per process. Calling it again with the same arguments returns that model; calling it with different runtime arguments — `maxTokens`, `preferredBackend`, `supportImage`, `multimodalOrder`, `maxConcurrentSessions` and the rest — or after another model was installed rebuilds it and closes the previous one, so a handle still held stops working. `model.isClosed` is true from the moment `close()` starts, and a `getActiveModel` call made meanwhile waits for the close to finish before it builds the next model. With no model installed it throws `NoActiveModelError`, a `StateError`. Load it once, then create and close sessions per conversation.

Sessions are cheap to create and destroy. The expensive part is `engine_create` (2–10 seconds depending on backend and model size), which happens once when the model is first opened. To give the active model other settings, call `model.close()` first, then `getActiveModel(...)` again with the new settings.

To keep several models loaded at once — one file on the CPU and the GPU, or two files — use `FlutterEdgeAi.openModel(spec, ...)` (`flutter_edge_ai` 2.5.0) with `LiteRtLmEngine` (`flutter_edge_ai_litertlm` 1.16.0) on Android, iOS and desktop; the web engine and the other engines throw `UnsupportedError`. Install the file with `installModel(...).install(activate: false)` and pass the `spec` it returns. Every call builds a new model that the app owns and closes: it is never cached, a `getActiveModel` rebuild leaves it open, and the plugin has no memory policy — each open model holds its own weights and KV cache, and on iOS a process past its memory limit is killed with no exception to catch. Engine creations run one at a time, and a model's sampler is fixed by its first generation, so sessions sharing one model should use the same sampling. The package measured it on macOS (CPU and GPU engines on one file, two GPU engines) and on Windows Lunar Lake (CPU, GPU and NPU engines together); on Windows a second GPU engine can fail with `wgpu::Instance already set`.

## Choosing a CPU, GPU or NPU backend

Pass `preferredBackend` to `getActiveModel` and read `model.activeBackend` to see what loaded. Since `flutter_edge_ai_litertlm` 1.14.0 the file decides where it can run: before any engine exists, the engine reads the bundle's header and each compiled section's stamp.

| The file | No preferredBackend | preferredBackend given |
|---|---|---|
| Generic | GPU, then CPU | `gpu`, `cpu`: it, then the next; `npu`: not tried (GPU, then CPU, with a notice) |
| Text model with `backend_constraint` | first of GPU, CPU allowed | only the declared ones |
| Compiled for an NPU (`…_qualcomm_sm8750`) | the NPU | `npu`; `gpu`, `cpu` refused |
| Shorter than its sections | refused | refused |

A generic bundle has no NPU graph, so it never runs on an NPU, with or without a vendor flag: LiteRT would try to compile it on the device with a compiler plugin, which no NPU stack here ships, and run it on the CPU while reporting `npu`. Since `flutter_edge_ai_litertlm` 1.15.0 a request for `npu` on such a file goes to the GPU, then the CPU, and the log says why. Only the default `LiteRtLmEngine()` falls back to the next backend. `LiteRtLmEngine(backendFallback: false)`, chosen once in `FlutterEdgeAi.initialize` for the whole process, makes exactly one attempt: on `preferredBackend`, or, when none is given, on the GPU for a generic bundle, the NPU for a bundle compiled for one, and the first backend a constrained text model allows; it refuses `npu` on a generic bundle with a `LiteRtLmBundleException`. A bundle compiled for an NPU runs on the NPU only, in either mode: it starts there when `preferredBackend` is unset or `npu`, a request for `gpu` or `cpu` is refused, and a device whose NPU cannot be reached fails at once, without a GPU attempt.

A load that fails on its backends throws a `BackendInitException`. Its `attempts` list holds one `BackendInitAttemptFailure` per backend tried, in order — one with the strict engine or an NPU-compiled bundle, one per backend of the fallback order otherwise — each with the `backend` and the `error`, and `lastAttempt` is the last of them. The error is a `LiteRtLmEngineCreateException` when the native engine refused the backend, carrying LiteRT-LM's own reason in `nativeMessage` (read from its C API on every platform, release builds included), or an `NpuDispatchUnavailableException` when the NPU could not be reached.

A refusal before any engine exists is a `LiteRtLmBundleException`, with a message that says what to do. Its `problem` is `incomplete` for a download that stopped early, `wrongBackend` for a backend the file rules out, and `unsupportedNpu` for a text model or an encoder the load turns on that is compiled for an NPU this package has no dispatch for here (an NPU vendor it does not know, a Qualcomm bundle on Windows, a Google Tensor or MediaTek bundle off Android), for sections the load turns on that are compiled for two different NPUs, or for an NPU encoder next to a text model that is not on the NPU. A file whose header the reader does not understand, such as a `.task` bundle or a newer container, loads as before.

Encoders never fall back, and the engine places them: an encoder left unset (`preferredVisionBackend`, `preferredAudioBackend`) runs where its section says — its declared `backend_constraint` (the CPU when that allows it), the NPU when the section is compiled for one, else the CPU. So the Gemma 4 vision and audio encoders run on the CPU, the Gemma 3n vision encoder on the GPU (its section allows only the GPU), and the official `…_qualcomm_sm8750` bundle, whose encoders are not compiled for the NPU, runs them on the CPU. A bundle with compiled encoders runs them on the NPU without the app choosing: `gemma-4-E2B-it_Qualcomm_QCS8275.litertlm`, for QCS8275 boards such as the Arduino VENTUNO Q, is one, and Gemma 4 E2B builds for phones with the towers compiled for the NPU too are provided by Google, not yet published. An encoder runs on the NPU only together with a text model on the NPU. Asking an encoder for a backend its section excludes is refused, for example `gpu` for the published Gemma 4 and Gemma 3n audio encoders, which declare cpu. `activeVisionBackend` and `activeAudioBackend` (filled since `flutter_edge_ai_litertlm` 1.14.0) report where each encoder runs, and are null for a model without that encoder.

The web `.litertlm` engine reports `activeBackend` as `null`. `PreferredBackend.npu` needs a Snapdragon phone, a Qualcomm Linux arm64 board, a Google Tensor G5 or G6 Pixel (Android 16 or later, not a BP2A build), a MediaTek Android 12+ device whose NeuroPilot runtime opens in the app, or Intel Lunar Lake or Panther Lake (Windows), and a model compiled for that NPU. The Intel NPU needs `flutter_edge_ai_litertlm` 1.15.0 or later. `PreferredBackend.cpu` never falls back. The iOS Simulator is CPU-only. On web, MediaPipe is GPU-only.

On desktop with the GPU backend, the forward pass (prefill and decode) runs on the GPU accelerator — Metal, DirectX 12 or Vulkan. The per-token sampler runs on the GPU on Windows and on the CPU on macOS and Linux, where it costs roughly 1–5 ms per token.

## Reading a .litertlm file before loading it

`readLiteRtLmBundleInfo(path)`, added in `flutter_edge_ai_litertlm` 1.14.0, returns what the load planner reads from a `.litertlm` file: `isComplete` (false for a download that stopped early), `isCompiledForNpu` (the file runs on the NPU only), `npuTarget` (for example Qualcomm SM8750, or null), the `visionEncoder` and `audioEncoder` sections with their own `npuTarget` and `backendConstraint`, `hasMtpDrafter` (the file can decode speculatively) and the list of `sections`. It reads the header and a few hundred bytes per compiled section, never the weights.

`readLiteRtLmBundleInfo` is native only and throws `UnsupportedError` on the web. A file that is not a `.litertlm` container, has another container version or has an unreadable header makes it throw `LiteRtLmBundleException`.

## NPU caveats for Gemma models

Each Android and Linux NPU vendor is opt-in, under `hooks: user_defines: flutter_edge_ai_litertlm:` in the app's `pubspec.yaml`. `qualcomm_npu: true` (since 1.10.0; Qualcomm licenses its QNN runtime for redistribution inside an app only) bundles the runtime into Android (about 83 MB installed) and Linux arm64 builds, for Snapdragon phones and Qualcomm Linux boards (QCS6490, QCS8275, QCS9075). `google_tensor_npu: true` (1.15.0) bundles the Tensor dispatch, built from LiteRT's source; the runtime it drives is part of the Pixel system image. `mediatek_npu: true` (1.17.0) fetches Google's `litert-npu-runtime-mediatek` AAR from Google Maven and bundles its dispatch, about 330 KB; the NeuroPilot runtime is part of the phone, and a phone whose vendor does not expose it to apps is refused with the reason. An app can set several flags: the bundle's stamp picks the vendor, not its file name. Without the vendor's flag, a bundle compiled for that NPU is refused with a `BackendInitException` that says how to enable it.

On NPU, run a Gemma 4 bundle. Most Gemma 3 bundles on the Qualcomm or the Intel NPU drop every prefill chunk after the first — no error, no log line, and a fluent reply that answers from the opening of the prompt and ignores the rest (LiteRT-LM issue 3508); of the published Qualcomm 1B files only `…_sm8850` keeps them. Google Tensor and MediaTek are not measured, and Google publishes no Gemma 4 bundle for MediaTek.

`maxTokens` is not clamped up to 1024 on the NPU: pass the `cache_length` the bundle was compiled for. The NPU is attempted only for a bundle compiled for an NPU, and only on Windows, on Google Tensor G5 and G6 Pixels with Android 16 or later (not a BP2A build), on MediaTek devices with Android 12 or later whose NeuroPilot adapters open in the app, on Android phones whose Qualcomm FastRPC (`libcdsprpc.so`) opens, and on Qualcomm Linux arm64 boards — all but Windows only when the app sets that vendor's flag. On a Linux board the probe needs `/dev/fastrpc-cdsp` to exist and open for the user, `/dev/dma_heap/system` to open as well where the kernel has it (group `fastrpc` grants both), and Qualcomm's FastRPC library `libcdsprpc.so.1` (`qcom-fastrpc1` on Ubuntu). Elsewhere a bundle compiled for an NPU is refused at once with a `BackendInitException` whose one attempt is an `NpuDispatchUnavailableException`, and it is never tried on the GPU. Windows has no device probe: a bundle compiled for the Intel NPU is not refused on a PC without one, and what LiteRT does then is not measured. Intel's Gemma 4 12B for Panther Lake needs 1.17.0, which ships OpenVINO 2026.4.1, and each load copies its weights to a 5.55 GB `%TEMP%\litert_ov_bank_<hex>.bin`; from 1.17.0 the next Intel NPU load deletes the banks an exited 1.17.0 app left behind.

The plugin does not read the device's SoC before the load: when a bundle cannot run on the device's SoC, engine create fails and the error names the SoC the bundle was compiled for. Only a load tells whether a bundle for another Qualcomm SoC runs.

## Wrong numbers on the GPU and float32 activations

On GPU, Gemma 4 can copy numbers wrongly from a long prompt: `2026/06/23` comes back as `20226/12/17`, the same way on every run (LiteRT-LM issue 3012 on Adreno, issue 2814 on Metal). The published Gemma 4 `.litertlm` files ask for half-precision activations. Ask for full precision when the answers carry figures, dates or amounts:

```dart
final model = await FlutterEdgeAi.getActiveModel(
  preferredBackend: PreferredBackend.gpu,
  activationDataType: ActivationDataType.float32,
);
```

Prefill gets slower (about 3 times on a Snapdragon 8 Elite and an iPhone 11, under 1.5 times on an Apple M3 Max); decode speed barely changes. Left unset, the model file decides, with one exception since `flutter_edge_ai_litertlm` 1.14.0: a FunctionGemma bundle that declares no `prefer_activation_type` gets `float32` on its GPU attempt, because at `float16` it answers with nothing. A plain Gemma 3 270M (`ModelType.gemmaIt`) answers with `<pad>` on the GPU too (LiteRT-LM issue 3280) and needs `float32` passed explicitly. A value you pass always wins. It applies to the text decoder of `.litertlm` models on Android, iOS and desktop — not to the vision or audio encoders, which keep what the model file asks for. MediaPipe, ONNX, built-in AI and the web engines ignore it.

### Memory cost and caveats of float32 activations

`float32` needs more GPU memory than the default, and with the default engine a GPU engine that cannot be created falls back to CPU without an exception, only a warning line in the log — right digits, a much slower run (the strict engine throws `BackendInitException` instead). After loading, check `model.activeBackend == PreferredBackend.gpu` before concluding the setting did anything. On Android the GPU shares system memory, so on a 4–6 GB phone running out of it at `float32` can end the app rather than fall back to CPU. Both precisions share one compiled GPU program cache per model, so switching recompiles the GPU programs (about 600 MB for Gemma 4 E2B): pick one precision per install rather than per request. Every `flutter_edge_ai_litertlm` release supports the setting.

## The flutter_edge_ai_litertlm engine package

`flutter_edge_ai_litertlm` (renamed from `flutter_gemma_litertlm`) is the LiteRT-LM engine for flutter_edge_ai: it runs `.litertlm` models, EmbeddingGemma 2 `.litertlm` embeddings and LiteRT `.tflite` embeddings, through `dart:ffi` on Android, iOS, macOS, Linux and Windows, and as an early preview on the web through `@litert-lm/core` and LiteRT.js. It owns the shared LiteRT-LM native library (`libLiteRtLm`), which `flutter_edge_ai_speech` also uses, and it ships the LiteRT C API embedding backend `LiteRtEmbeddingBackend`. `LiteRtLmEngine` handles `ModelFileType.litertlm` models; pass it alongside other engines, such as `MediaPipeEngine`, if the app uses both formats.

| Platform | Support |
|---|---|
| Android | FFI (GPU via OpenCL; NPU on Qualcomm, Google Tensor G5/G6 or MediaTek, each opt-in) |
| iOS | FFI (GPU via Metal on device; CPU on simulator) |
| macOS and Linux | FFI (GPU via Metal or Vulkan; NPU on Qualcomm Linux arm64, opt-in) |
| Windows | FFI on x64 (CPU, GPU via DirectX 12, Intel NPU); no ARM64 native archive |
| Web | via `@litert-lm/core` (CDN, early preview; language models text-only) |

The native library (`native-v0.18.0-g`, LiteRT-LM v0.18.0, in `flutter_edge_ai_litertlm` 1.17.0) is fetched at build time by `hook/build.dart` (Native Assets) from a SHA256-verified GitHub release, so nothing has to be downloaded by hand; the app still needs its platform setup (Android `minSdk 30`, iOS 15.0 and the memory entitlements, the macOS Podfile build phase). Without the macOS phase the first model load fails, and since 1.17.0 its error contains `flutter_edge_ai_litertlm: LiteRT-LM did not load on macOS` and names the fix. Version 1.10.0 moved to LiteRT-LM v0.18.0 and renamed its Metal accelerator to `LiteRtLmMetalAccelerator`, so that the GPU of `flutter_litert`, a separate LiteRT plugin, keeps working beside it on iOS and macOS; in 1.10.0 the GPU backend still crashed in `LiteRtCreateCompiledModel` when the app also used `flutter_litert`, and 1.10.1 fixed that. A stale Native Assets cache after a native version bump can leave the library unbundled, surfacing as a `dlopen` "no such file" error on the first inference; a clean rebuild fixes it.

## Fixed engine issues worth knowing

- Intel NPU on Windows failing to load (fixed in `flutter_edge_ai_litertlm` 1.15.0): 1.14.0, like every release since `native-v0.17.0`, failed the Intel NPU load or ended the process (`0xC0000409`).
- Linux tool calls aborting the app (fixed in `flutter_edge_ai_litertlm` 1.11.1): up to 1.11.0 a chat or session with `tools` aborted on its first constrained token on Linux x64 and arm64, because the prebuilt constraint provider uses another C++ ABI than the runtime.
- Android GPU crash on Mali (fixed in `flutter_gemma_litertlm` 1.8.2): in 1.7.0–1.8.1, `PreferredBackend.gpu` on phones with a Mali GPU (Samsung A-series, MediaTek, Google Tensor) killed the process while the model loaded, because the OpenCL accelerators called `AHardwareBuffer_allocate` without declaring `libandroid.so`. Adreno GPUs and the CPU backend were unaffected.
- Google Play rejection over 16 KB page sizes (fixed in `flutter_gemma_litertlm` 1.8.0): the bundled Qualcomm Hexagon DSP blobs for the NPU path arrived with a 4 KB alignment, and Play refused the release at submission. With the opt-in NPU, the build hook now raises the alignment of the libraries it bundles to 16 KB.
- Garbled or empty streams on Android (fixed in `flutter_gemma_litertlm` 1.5.2): an app that embedded or transcribed anything before its first generation could receive corrupted streams, because the first `dlopen` of `libLiteRtLm` decides for the whole process whether its exports are globally visible.

The Mali, 16 KB and stream fixes landed under the package's old name, `flutter_gemma_litertlm`, and every `flutter_edge_ai_litertlm` release includes them. The Linux tool-call fix needs `flutter_edge_ai_litertlm` 1.11.1 or later, and the Intel NPU fix 1.15.0 or later.
