---
title: flutter_edge_ai overview
source: https://pub.dev/packages/flutter_edge_ai/versions/2.1.1 (README.md, CHANGELOG.md, lib/fix_data.yaml, skills/flutter-edge-ai-inference/SKILL.md, example/lib/models/) ; https://github.com/DenisovAV/flutter_edge_ai
license: MIT
---

# flutter_edge_ai overview

This document explains what the flutter_edge_ai Flutter plugin (formerly flutter_gemma) is, which packages it is split into, which engines and model families it supports, where models can be installed from, how gated Hugging Face models are authenticated, and how the feature set differs between Android, iOS, web and desktop.

## What flutter_edge_ai is

flutter_edge_ai brings on-device AI to Flutter on Android, iOS, web, macOS, Windows and Linux. It runs Gemma and other open models, or the model the operating system already ships, with text, vision, audio, function calling, embeddings, RAG and speech — no server and no cloud. The core is small: an app adds only the runtimes and stores it uses. Through version 1.11.3 the package shipped as `flutter_gemma`; it was renamed to `flutter_edge_ai` in 1.11.4.

The supported model families are Gemma 4 E2B and E4B, Gemma 3n E2B and E4B, Gemma 3 1B, Gemma 3 270M, FunctionGemma 270M, Qwen3 0.6B, Qwen3.5 and later, Qwen 2.5 0.5B and 1.5B, DeepSeek R1, Phi-4 Mini, FastVLM, Qwen2-VL, SmolVLM2, LLaVA-OneVision, SmolLM, SmolLM3, LFM2.5 and Phi-4 Mini Reasoning. Gemma 4 and Gemma 3n take image and audio input; FastVLM, Qwen2-VL, SmolVLM2 and LLaVA-OneVision take images. Desktop platforms (macOS, Windows, Linux) run LiteRT-LM `.litertlm` files, ONNX models or the operating system's own model; MediaPipe `.task` files do not run on desktop.

Models run locally on the user's device, for privacy and offline use.

## Main inference features

- Pluggable engines: LiteRT-LM (`.litertlm`), MediaPipe (`.task`), ONNX Runtime, and the operating system's own models — Gemini Nano, Apple Foundation Models, Windows AI Foundry and the Chrome Prompt API.
- Multimodal input: images and audio with Gemma 4, Gemma 3n and the other vision models. Audio input works on Android, iOS and desktop, not on the web.
- Function calling: models can call the app's functions on the models that support them.
- Thinking mode: Gemma 4, Qwen3 and DeepSeek R1 can stream their reasoning separately from the answer.
- Stop generation: cancel a reply mid-stream with `stopGeneration()`.
- CPU, GPU and NPU backends per model, with an opt-in Qualcomm NPU on Android and the Intel NPU on Windows.
- Desktop support: LiteRT-LM is called directly from Dart through `dart:ffi` — no JVM and no separate process.
- Model management: installs from the network, app assets, bundled files or local paths, Hugging Face installs in one call, retries, typed download errors and LoRA weights.
- Genkit integration for on-device and hybrid cloud flows.

## Speech, agent skills, embeddings and RAG features

Beyond text generation, flutter_edge_ai offers opt-in packages for the rest of an on-device AI stack:

- On-device speech-to-text with `flutter_edge_ai_speech`: transcribe audio fully offline with a selectable ASR model (moonshine, Whisper, Parakeet) via the LiteRT C API. Whisper is multilingual.
- On-device text-to-speech with `flutter_edge_ai_speech`: synthesize speech fully offline with a selectable model (Matcha, Qwen3-TTS, Inflect-Nano-v2).
- On-device voice loop: `VoiceSession` chains STT, LLM and TTS into one push-to-talk turn with barge-in (native platforms only).
- On-device agent skills with `flutter_edge_ai_agent`: give the model `SKILL.md` skills (text, JavaScript, native intent or MCP) that it invokes through function calling, fully offline. It is verified on Android, iOS, macOS and Windows and is not supported on the web.
- Text embeddings: 768-dimensional vectors from EmbeddingGemma or Gecko through the LiteRT backend in `flutter_edge_ai_litertlm`, with the tokenizers from `flutter_edge_ai_embeddings`.
- On-device RAG with `flutter_edge_ai_rag` and a storage provider: `flutter_edge_ai_qdrant` (qdrant-edge, native) or `flutter_edge_ai_sqlite` (`sqlite-vec` / `vec0` KNN inside SQLite on all six platforms including the web), with payload filters.

flutter_edge_ai also ships agent skills for coding assistants. Running `dart run skills@ get --all` installs the skills that the Flutter Edge AI packages in the app's dependencies bundle, where coding assistants look for them.

## Modular packages: a small core plus opt-in engines

`flutter_edge_ai` is a small core package plus opt-in packages for each engine or backend, so an app only pulls the native weight it actually uses. You add the core package, then the packages for the model formats and features you need:

- `flutter_edge_ai` — core, always required; it registers no engine on its own.
- `flutter_edge_ai_litertlm` — `.litertlm` models and LiteRT embeddings (FFI on mobile and desktop, early preview on the web).
- `flutter_edge_ai_mediapipe` — `.task` and `.bin` models (MediaPipe; mobile and web).
- `flutter_edge_ai_builtin_ai` — the model built into the OS or browser: Gemini Nano, Apple Foundation Models, Windows AI Foundry and the Chrome Prompt API.
- `flutter_edge_ai_onnx` — ONNX Runtime text generation and embeddings.
- Further opt-in packages for embeddings tokenizers, RAG, agent skills, speech and memory diagnostics, listed in the table below.

Moving from the old names means renaming each `flutter_gemma*` dependency and import to its `flutter_edge_ai*` counterpart (`flutter_gemma_rag_sqlite` and `flutter_gemma_rag_qdrant` became `flutter_edge_ai_sqlite` and `flutter_edge_ai_qdrant`) and running `dart fix --apply`, which renames `FlutterGemma` to `FlutterEdgeAi`, `FlutterGemmaPlugin` to `FlutterEdgeAiPlugin`, `FlutterGemmaDesktop` to `FlutterEdgeAiDesktop` and `GemmaLogLevel` to `EdgeAiLogLevel`. Version 2.0.0 removed the old aliases and moved the RAG APIs out of core into `flutter_edge_ai_rag`.

## Which flutter_edge_ai package to add for each need

| You want to | Add |
|---|---|
| Run `.litertlm` models (Gemma 4, Qwen3, FastVLM, and everything on desktop) and LiteRT embeddings | `flutter_edge_ai_litertlm` |
| Run `.task` or `.bin` models (MediaPipe; mobile and web) | `flutter_edge_ai_mediapipe` |
| Run ONNX models — text generation and embeddings | `flutter_edge_ai_onnx` |
| Use the model built into the OS or browser | `flutter_edge_ai_builtin_ai` |
| Tokenizers for text embeddings (needed with the LiteRT or ONNX embedding backend) | `flutter_edge_ai_embeddings` |
| On-device RAG | `flutter_edge_ai_rag` plus `flutter_edge_ai_qdrant` (native) or `flutter_edge_ai_sqlite` (all six platforms) |
| Speech-to-text, text-to-speech, voice loop | `flutter_edge_ai_speech` |
| Agent skills over function calling | `flutter_edge_ai_agent` |
| Measure the memory a model costs, read from the OS | `flutter_edge_ai_diagnostics` |
| Genkit, and on-device or cloud routing | `genkit_flutter_edge_ai`, `genkit_hybrid` |

## Registering engines with FlutterEdgeAi.initialize

Call `await FlutterEdgeAi.initialize(...)` once in `main()`, after `WidgetsFlutterBinding.ensureInitialized()`, and register the opt-in packages you added to `pubspec.yaml`. Core registers no engine on its own, so without this step `getActiveModel()` throws `StateError: No inference engine can handle this model`, and the first embedding fails the same way.

Each list comes from one package: `inferenceEngines:` takes `LiteRtLmEngine()`, `MediaPipeEngine()`, `BuiltInAiEngine()` or `OnnxEngine()`; `embeddingBackends:` (for example `LiteRtEmbeddingBackend()`) and `embeddingTokenizers:` (for example `GemmaEmbeddingTokenizers()`) set up embeddings; `sttBackends:` and `ttsBackends:` come from `flutter_edge_ai_speech`; `skillExecutors:` from `flutter_edge_ai_agent`; and `huggingFaceResolvers:` overrides the Hugging Face resolvers that engines bring with them. RAG is not configured here: it lives in `flutter_edge_ai_rag`.

Add only the engines you ship. Passing both `LiteRtLmEngine()` and `MediaPipeEngine()` lets one app run both formats — the registry routes each model to the engine that handles its file type. Common settings are `huggingFaceToken` for gated models, `maxDownloadRetries` (default 10), and the web-only `webStorageMode`.

## Model file types and how ModelFileType selects the engine

flutter_edge_ai groups model file formats by how chat templates are handled.

SDK-managed templates: `.task` files are the MediaPipe format for mobile and web, and `.litertlm` files are the LiteRT-LM format for Android, iOS, desktop and the web preview. The runtime applies the chat template — MediaPipe for `.task`, LiteRT-LM for `.litertlm` — so your code sends plain text on every platform.

Manual template formatting: `.bin` and `.tflite` files require manual chat template formatting in your code.

`ModelFileType` is what selects the engine — it is not inferred from the file name. `installModel` defaults it to `ModelFileType.task`, so declare it explicitly: `ModelFileType.litertlm` for `.litertlm` files (omitting it routes the model to MediaPipe, which cannot read that format), `ModelFileType.task` for `.task` files, `ModelFileType.binary` for `.bin` and `.tflite` files, `ModelFileType.onnx` for ONNX Runtime GenAI model directories, and `ModelFileType.builtIn` for OS-provided models such as Gemini Nano and Apple Foundation Models.

## Engines by platform

| Engine | Android | iOS | Web | macOS | Windows | Linux |
|---|---|---|---|---|---|---|
| LiteRT-LM (`.litertlm`) | yes | yes | preview | yes | yes | yes |
| MediaPipe (`.task`) | yes | yes | yes | no | no | no |
| ONNX Runtime | yes | yes | yes | yes | yes | yes |
| Built-in AI | yes | yes | yes | yes | yes | no |

LiteRT-LM ships `arm64` on Android, iOS and macOS, `x86_64` on Windows, and `x86_64` and `arm64` on Linux. On Android, MediaPipe `.task` also runs on `x86_64` and `armeabi-v7a`. Web `.litertlm` is text and function calling only, with no vision, audio or LoRA. The iOS Simulator runs on the CPU only.

## Model capabilities by model family

| Model family | ModelType | Function calling | Thinking | Vision / audio |
|---|---|---|---|---|
| Gemma 4 E2B, E4B | `gemma4` | yes | yes | yes / yes |
| Gemma 3n E2B, E4B | `gemmaIt` | yes | no | yes / yes |
| Gemma 3 1B, Gemma 3 270M | `gemmaIt` | no | no | no |
| FunctionGemma 270M | `functionGemma` | yes | no | no |
| Qwen3 0.6B | `qwen3` | yes | yes | no |
| Qwen3.5, 3.6, 3.8 | `qwen35` | no | no | no |
| Qwen 2.5 0.5B, 1.5B | `qwen` | yes | no | no |
| DeepSeek R1 | `deepSeek` | yes | yes | no |
| Phi-4 Mini | `phi` | yes | no | no |
| FastVLM, Qwen2-VL, SmolVLM2, LLaVA-OneVision | `general` | no | no | yes / no |
| SmolLM, SmolLM3, LFM2.5, Phi-4 Mini Reasoning | `general` | no | no | no |

Gemma 4 E2B as a `.litertlm` file is 2.6 GB and needs no Hugging Face token. On the web there is no audio input, Gemma 3n vision is native-only, and Gemma 4 thinking needs the `.litertlm` web build.

## Choosing the right ModelType

When installing a model you specify a `ModelType`. It tells flutter_edge_ai how the model writes tool calls and reasoning, and on some engines it also picks the prompt format. The full set is `general`, `gemmaIt`, `gemma4`, `deepSeek`, `qwen`, `qwen3`, `qwen35`, `llama`, `hammer`, `functionGemma` and `phi`.

Gemma 3 and Gemma 3n are `ModelType.gemmaIt` — there is no `gemma3`. A wrong type still generates text; tool calls and reasoning then arrive as raw text. Gemma 4 and FunctionGemma on a `.litertlm` route their native tool-call tokens through the LiteRT-LM SDK's chat-template path.

## Installing a model from Hugging Face, the network, assets or files

Models are installed with a builder: `FlutterEdgeAi.installModel(modelType:, fileType:)` followed by a source and `.install()`. The latest install becomes the active model that `getActiveModel` loads, and `install()` skips the download when the file is already on disk.

- `.fromHuggingFace(repo)` — the engine's resolver reads the repo's deployment manifest at install time and installs the right variant; pass `file:` to pin an explicit file.
- `.fromNetwork(url, token:)` — downloads from an HTTP or HTTPS URL.
- `.fromAsset(path)` — copies a model declared in `pubspec.yaml` assets.
- `.fromBundled(name)` — uses a native platform resource bundled with the app.
- `.fromFile(path)` — references a file already on disk, for example one picked with a file picker.

`FlutterEdgeAi.resolveHuggingFace(repo, fileType:)` returns the resolved file and overridable runtime defaults without installing, so you can inspect the variant and its notes first.

## Model source types compared

| Source | Platform | Progress | Resume | Authentication | Use case |
|---|---|---|---|---|---|
| NetworkSource | All | Detailed | Server-dependent | Supported | Hugging Face, CDNs, private servers |
| AssetSource | All | End only | No | Not applicable | Models bundled in app assets |
| BundledSource | All | End only | No | Not applicable | Native platform resources |
| FileSource | Native; on the web, URLs and asset paths only | End only | No | Not applicable | User-selected files |

Network installs report progress from 0 to 100 percent through `withProgress`, retry transient errors with backoff (`maxDownloadRetries`), can be cancelled with a `CancelToken`, and on Android can opt into a foreground service with `fromNetwork(url, foreground: true)` for large downloads.

## Bundling a small model inside the app

Bundling suits small models that must be available instantly and offline, such as Gemma 3 270M at about 300 MB or LFM2.5 230M at about 170 MB. It is not for large models, because every byte adds to the app's download size; host large models for download instead, or let the user pick a file already on the device with `fromFile`.

## Handling download errors with DownloadException

`installModel(...).install()` throws a public `DownloadException` carrying a sealed `DownloadError`, so an app can react to gated Hugging Face models without matching error strings. The cases are `UnauthorizedError` (401, a missing or invalid token), `ForbiddenError` (403, the token lacks access to a gated model), `NotFoundError` (404, a wrong URL), `RateLimitedError` (429), `ServerError` (5xx), `NetworkError` (connectivity), `CanceledError` (the user cancelled) and `UnknownError`. Each `DownloadError` exposes `toUserMessage()`, `toTitle()`, `isRetryable` and `requiresUserAction` helpers for building UI.

To remove a model, call `model.close()`, then `FlutterEdgeAi.uninstallModel(fileName)`; when it was the active model, also call `FlutterEdgeAi.clearActiveInferenceIdentity()`.

## Hugging Face authentication for gated models

Some models need a Hugging Face token to download. Never commit tokens to version control. The recommended pattern is to read the token with `const String.fromEnvironment('HUGGINGFACE_TOKEN')`, build with `--dart-define=HUGGINGFACE_TOKEN=hf_...`, and pass it once to `FlutterEdgeAi.initialize(huggingFaceToken: ...)` — `null` when it is empty.

A token read this way stays out of git, not out of the app: it is compiled into the binary, and on the web into `main.dart.js`, where every visitor can read it. A shipped app should download from a repo that needs no token.

## Which models require a Hugging Face token

Gated models include Gemma 3n E2B and E4B, Gemma 3 1B and Gemma 3 270M, and EmbeddingGemma in `litert-community/`. Gemma 4 E2B and E4B, Qwen3, Qwen 2.5, DeepSeek R1, SmolLM, SmolLM3, FastVLM, Qwen2-VL, SmolVLM2, Phi-4 Mini Reasoning and LFM2.5 230M download without a token. To get access to a gated repo, open the model page on Hugging Face and accept its licence; tokens are created at huggingface.co/settings/tokens.

## Logging and privacy of prompts

The plugin's internal logs are silent in release builds — model output, prompts and conversation history are never written to logcat or syslog there. In debug builds they follow `FlutterEdgeAi.logLevel`. `EdgeAiLogLevel.none` prints nothing. `EdgeAiLogLevel.info`, the default, prints lifecycle, errors and diagnostics but no model output or prompts. `EdgeAiLogLevel.verbose` adds model output, prompts and conversation history. Release builds stay silent regardless of this setting.

## Feature comparison across Android, iOS, web and desktop

| Feature | Android | iOS | Web | Desktop |
|---|---|---|---|---|
| Text generation | yes | yes | yes | yes |
| Image input | yes | yes | MediaPipe `.task` | yes |
| Audio input | yes | device only | no | `.litertlm` only |
| Speech-to-text and text-to-speech | yes | yes | no | yes |
| Function calling | yes | yes | yes | yes |
| Agent skills | yes | yes | no | macOS, Windows, Linux |
| GPU acceleration | yes | yes (device) | yes | yes (Metal, Vulkan, DirectX 12) |
| NPU acceleration | Qualcomm, `.litertlm`, opt-in | no | no | Windows Intel |
| Text embeddings and RAG | yes | yes | yes | yes |

On the web, the `.litertlm` engine is an early preview limited to text and function calling, so images on the web need a MediaPipe `.task` build. JavaScript agent skills do not run on Linux, which has no embeddable webview.

## Web support and the early-preview web .litertlm engine

Web `.litertlm` inference runs Gemma `.litertlm` web builds in the browser through the upstream `@litert-lm/core` package, using WebGPU and WASM. It is an early preview: it supports text generation, multi-turn chat and function calling, but no vision, audio or LoRA, and `activeBackend` reports `null` there. MediaPipe `.task` on the web remains supported and runs on the GPU; for vision on the web today, use a MediaPipe `.task` web model.

## Troubleshooting multimodal and performance issues

Multimodal issues: make sure you are using a multimodal model (Gemma 4 E2B or E4B, Gemma 3n E2B or E4B, FastVLM, Qwen2-VL, SmolVLM2 or LLaVA-OneVision), pass `supportImage: true` to `getActiveModel` and to `createChat`, give the model a larger context (4096 tokens or more), and check device memory, because multimodal models need more RAM.

Performance: use the GPU backend for better performance with multimodal models, and consider the CPU backend for text-only models on lower-end devices. Read `model.activeBackend` to see which backend actually loaded.

Reasoning text: the chat API separates reasoning such as `<think>...</think>` (DeepSeek, Qwen) and Gemma's thought channel from the answer and streams it as `ThinkingResponse`; `generateChatResponse()` strips it.
