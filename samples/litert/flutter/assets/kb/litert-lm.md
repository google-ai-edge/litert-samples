---
title: LiteRT-LM and the .litertlm format
source: https://github.com/google-ai-edge/LiteRT-LM/blob/a5d53ea12050386fab996944312481a009a55fed/README.md ; https://github.com/google-ai-edge/LiteRT-LM/blob/a5d53ea12050386fab996944312481a009a55fed/docs/README.md ; https://github.com/google-ai-edge/LiteRT-LM/blob/a5d53ea12050386fab996944312481a009a55fed/models/README.md ; https://github.com/google-ai-edge/LiteRT-LM/blob/a5d53ea12050386fab996944312481a009a55fed/models/gemma4/README.md ; https://github.com/google-ai-edge/LiteRT-LM/blob/a5d53ea12050386fab996944312481a009a55fed/docs/litert_lm_builder.md ; https://github.com/google-ai-edge/LiteRT-LM/blob/a5d53ea12050386fab996944312481a009a55fed/docs/api/cpp/conversation.md ; https://github.com/google-ai-edge/LiteRT-LM/blob/a5d53ea12050386fab996944312481a009a55fed/docs/api/cpp/tool-use.md ; https://github.com/google-ai-edge/LiteRT-LM/blob/a5d53ea12050386fab996944312481a009a55fed/docs/api/cpp/constrained-decoding.md ; https://github.com/google-ai-edge/LiteRT-LM/blob/a5d53ea12050386fab996944312481a009a55fed/docs/api/kotlin/getting_started.md ; https://github.com/google-ai-edge/LiteRT-LM/blob/a5d53ea12050386fab996944312481a009a55fed/docs/getting-started/build-and-run.md ; https://github.com/google-ai-edge/LiteRT-LM/blob/a5d53ea12050386fab996944312481a009a55fed/samples/ios_and_mac/README.md ; https://github.com/google-ai-edge/litert-samples/blob/367ddbdee22a8d73fba6e519c4a3f18d19008b05/skills/litert-lm/SKILL.md ; https://github.com/google-ai-edge/litert-samples/blob/367ddbdee22a8d73fba6e519c4a3f18d19008b05/skills/litert-lm/references/get-a-model.md ; https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/blob/b3ca0d2f076785a8f4b2219ddbd2bdb99954eae1/README.md
license: Apache-2.0
---

# LiteRT-LM and the .litertlm format

This document explains LiteRT-LM, Google's runtime layer for running large language models on device with LiteRT: what it is, its language APIs, the `.litertlm` model file and its builder, chat templates, the Conversation API, tool use, constrained decoding, and the Kotlin and Swift APIs.

## What LiteRT-LM is

LiteRT-LM is Google's production-ready orchestration layer to run LLMs with LiteRT, engineered for high-performance, cross-platform execution. Its key features are:

- Cross-platform support: Android, iOS, Web, Desktop, and IoT (e.g. Raspberry Pi).
- Hardware acceleration: peak performance via GPU and NPU accelerators.
- Multi-modality: support for vision and audio inputs.
- Tool use: function calling support for agentic workflows.
- Broad model support: Gemma, Llama, Phi-4, Qwen, and more.

LiteRT-LM powers on-device GenAI experiences in Chrome, Chromebook Plus, Pixel Watch, and more. You can also try the Google AI Edge Gallery app, available on Google Play and the App Store, to run models immediately on your device.

## How LiteRT-LM relates to LiteRT

LiteRT-LM is a specialized orchestration layer built directly on top of LiteRT, Google's high-performance multi-platform runtime. LiteRT provides the foundational hardware acceleration via XNNPack for CPU and ML Drift for GPU. LiteRT-LM adds the specialized GenAI libraries and APIs, such as KV-cache management, prompt templating, and function calling. This integrated stack is the same technology powering the Google AI Edge Gallery showcase app. Models for LiteRT-LM are provided in the `.litertlm` format.

## Supported language APIs and their status

LiteRT-LM offers language-specific guides and setup instructions:

| Language | Status | Best for |
| --- | --- | --- |
| Python | Stable | Prototyping and scripting |
| Kotlin | Stable | Android apps and JVM |
| Swift | Early preview | Native iOS and macOS |
| JavaScript (web) | Early preview | Browser environments |
| Flutter | Community | Cross-platform mobile |
| C++ | Stable | High-performance native |

The v0.16.0 release added the first versioned C API shared library prebuilts for all supported platforms, which allows natively integrating LiteRT-LM into applications and creating language bindings without building shared libraries. It also added the experimental YNNPACK delegate, enabled for Linux arm64 builds in the LiteRT-LM CLI and Python API. The preceding v0.15.0 release brought Apple Foundation Framework integration, CLI configuration, and JavaScript API updates.

## Trying a model from the command line

You can try LiteRT-LM from a terminal without writing code, using `uv`:

```bash
uv tool install litert-lm
litert-lm run \
  --from-huggingface-repo=litert-community/gemma-4-E4B-it-litert-lm \
  gemma-4-E4B-it.litertlm \
  --backend=gpu \
  --enable-speculative-decoding=true \
  --prompt="What is the capital of France?"
```

This runs Gemma 4 E4B with multi-token prediction (MTP) on Linux, macOS, Windows or Raspberry Pi through the LiteRT-LM CLI.

## Building LiteRT-LM from source

App developers do not need to build the project from source: Kotlin, Swift and Python users should use the pre-built SDKs. Building the core C++ framework is meant for core contributors and for native C++ developers who need custom compilation flags for an embedded system. Bazel (version 7.6.1) is the recommended build system; the CMake build system was added recently and is still under active development. You should check out the latest stable release tag. LiteRT-LM can be deployed on Android, Linux, macOS and Windows, and `runtime/engine/litert_lm_main.cc` is a demo that shows how to initialize and interact with the model. To run on GPU on all platforms, add `--define=litert_runtime_link_mode=dynamic` to the build command and keep the prebuilt shared libraries (`.so`, `.dll` or `.dylib`) in the same directory as the `litert_lm_main` binary. Running GPU on Windows needs DirectXShaderCompiler.

## Building and inspecting .litertlm files

The `litert-lm-builder` Python package (installed with `uv pip install litert-lm-builder`) builds, inspects and unpacks LiteRT-LM files. It provides two terminal commands, `litert-lm-builder` and `litert-lm-peek`. A key feature of the builder CLI is chaining subcommands, each adding a section to the file: system metadata, a TFLite model with its model type, a SentencePiece tokenizer, and the output path.

```bash
litert-lm-builder \
  system_metadata --str Authors "ODML Team" \
  tflite_model --path schema/testdata/attention.tflite --model_type prefill_decode \
  sp_tokenizer --path runtime/components/testdata/sentencepiece.model \
  output --path demo.litertlm
```

The builder can also be driven by a TOML configuration, and `litert-lm-builder unpack` extracts an existing `.litertlm` file into a directory with its sections and a reconstructed `model.toml`. `litert-lm-peek` reads a file's header, system metadata and section information and prints them, and can dump the contained files byte for byte. The same operations are available from Python through `LitertLmFileBuilder` and `peek_litertlm_file`.

## Choosing a model file from litert-community

The Hugging Face litert-community organization holds one repo per model, such as `litert-community/gemma-4-E2B-it-litert-lm` and `litert-community/Qwen3-0.6B`. The files to download from these two are `gemma-4-E2B-it.litertlm` (2.6 GB) and `Qwen3-0.6B.litertlm` (0.6 GB), which run on the CPU and the GPU. Other files carry their variant in the name: `wi4b32` (int4 weights, block 32), `q8` or `q4` for the quantization, `ekv1280` or `ekv4096` for the context length (prompt plus reply, in tokens), and a SoC suffix such as `_Google_Tensor_G5` or `.mediatek.mt6993` for an NPU build. The model card says which backend each file was tested on and its size. The download URL is `https://huggingface.co/<repo>/resolve/main/<file>`.

Converting a model yourself happens only when the model is missing, and on a workstation, not in the app: export with `litert-torch`, quantize with `ai-edge-quantizer` (int8 dynamic as the default; int4 blockwise), and bundle tokenizer and metadata into `.litertlm` with `litert_lm_builder` from the `litert-lm` package.

## Chat templates and their input variables

LiteRT-LM uses Jinja2 templates to transform structured conversation turns and tool declarations into model-specific prompt strings. The models directory says they are rendered hermetically via Minijinja; the C++ Conversation API documentation describes `PromptTemplate` as a thin wrapper around Minja, a C++ implementation of Jinja. The Jinja template used by a model is provided by the model file metadata. When rendering, LiteRT-LM passes these top-level fields:

- `messages` (required): the conversation history as a list of message objects.
- `tools` (optional): available tools (function declarations) that the model can invoke.
- `enable_thinking` (optional): whether the model should produce reasoning thoughts before answering.
- `add_generation_prompt` (optional): whether to append the model turn prefix (defaults to true).
- `bos_token` (optional): the beginning-of-sequence token.

Message roles are `system` (instructions, personality, guidelines and tool schemas), `user`, `assistant` (text and/or tool calls) and `tool` (outputs of executed functions). The `content` field is strictly a list of multimodal parts, such as a text part or an image part. Tool definitions follow the OpenAPI standard. A subtle change in prompt because of incorrect formatting can lead to significant model degradation, so the template should strictly match the structure the instruction-tuned model expects.

### Gemma 4 chat template differences

For the Gemma 4 family, the canonical template for the 2B and 4B models is `chat_template_e2b_e4b.jinja`, and `chat_template.jinja` covers 12B, 26B and 31B. Compared with the Hugging Face template, the LiteRT-LM templates expect tool responses inside the `content` field of a message with role `tool` (Hugging Face expects a `tool_responses` field), support both a string and a list of parts for system message content (Hugging Face only works with a string), and use the previous message type to decide whether to append the turn terminator `<turn|>`.

## The Conversation API workflow

`Conversation` is a high-level API representing a single, stateful conversation with the LLM, and is the recommended entry point for most users. It internally manages a `Session` and handles maintaining the initial context, managing tool definitions, preprocessing multimodal data, and applying Jinja prompt templates with role-based message formatting. The typical lifecycle is:

1. Create an `Engine` with the model path and configuration. This is a heavyweight object that holds the model weights.
2. Use the `Engine` to create one or more lightweight `Conversation` objects.
3. Send messages through the `Conversation` and receive responses.

`SendMessage` is a blocking call that returns the complete model response, and `SendMessageAsync` streams the response token by token through callbacks; this mirrors the Gemini Chat APIs. For multimodality, the engine must be created with a vision and/or audio backend, for example a CPU main backend with a GPU vision backend and a CPU audio backend.

The core input and output format is `Message`, a type alias for `ordered_json`, a flexible nested key-value structure. A message must contain `role`; `content` can be a plain text string or a list of parts. Supported parts are text (`type: text`), image by `path` or base64 `blob`, and audio by `path` or base64 `blob`. One message can mix several images, audio clips and text parts.

## Preface, history and ConversationConfig

`Preface` sets the initial context: `messages` (system instructions, few-shot examples or history), `tools` (mostly following the Gemini API FunctionDeclaration format) and `extra_context`, for example `enable_thinking` for models with a thinking mode. This is similar to a Gemini API system instruction and tools.

`Conversation` keeps the history of all messages because the Jinja template usually needs the whole history. Since the `Session` is stateful and processes input incrementally, the Conversation renders the template twice, with the history up to the previous turn and with the current message, and sends only the new portion. `ConversationConfig` can be created from an `Engine` (using its default `SessionConfig`) or from a specific `SessionConfig`, and can provide a Preface or override the prompt template and the `DataProcessorConfig`, which is useful for fine-tuned models.

With `SendMessageAsync`, the callback receives only the latest chunk of output, not the whole message; it is called for each new chunk, on error, and with an empty `Message` to signal the end of the response. The full response is available as the last history entry once the call completes.

### Model data processors

`ModelDataProcessor` is the model-specific component that converts the generic `Message` into the `InputData` the `Session` needs and converts the session's responses back, similar to Hugging Face multimodal processors. For example, the Gemma 3 data processor includes image and audio preprocessors, and the Qwen3 data processor handles `tool_calls` and tool responses. It is initialized from a `DataProcessorConfig` that corresponds to the `LlmModelType` stored in the model file metadata. Supporting a new LLM type typically means implementing a new data processor.

## Tool use in LiteRT-LM

LiteRT-LM handles tool calling in the Conversation API. The flow is:

1. The application declares the available tools; each declaration has a name, parameters and description in JSON.
2. The user's message is sent to LiteRT-LM, which feeds it to the model and starts generation.
3. The model outputs a string indicating a tool call.
4. LiteRT-LM detects the tool call and parses it into a JSON object.
5. The application executes the tool and gets a result.
6. The application sends the result back to the model.
7. The model answers in natural language or makes another tool call.

The application provides the tool specifications, implements and executes the tools, and manages the chat loop. LiteRT-LM translates messages into the format the model was trained on, runs inference, detects and parses tool calls, and maintains the conversation history between user, model and tools.

### Declaring tools and returning results

Tools are declared by setting the `tools` field of the `Preface` to a JSON array of declarations, each a JSON schema with the tool's name, description and parameters. The model's reply then contains a `tool_calls` list with the function name and arguments. The application calls the real function and passes the result back as a message with `role` set to `tool`; the model then produces a natural-language interpretation. With `SendMessageAsync`, text chunks stream as usual, but when a tool call starts, LiteRT-LM waits for the rest of the call, parses it, and sends the parsed JSON to the callback in the `tool_calls` field. Internally, tool declarations are formatted by `ModelDataProcessor::FormatTools`, tool calls are parsed by `ModelDataProcessor::ToMessage`, and calls and responses are formatted in `ModelDataProcessor::MessageToTemplateInput`.

## Constrained decoding

Constrained decoding enforces structure on the model's output, which is useful for function calling, structured data extraction and grammar enforcement. There are two ways to use it, and only one can be chosen per `Conversation`:

1. Constrained decoding for tool calling: enable it with `SetEnableConstrainedDecoding(true)` on the `ConversationConfig::Builder`; LiteRT-LM reads the tool declarations and constrains function-call strings to the model's function-calling syntax.
2. Custom constrained decoding: set a constraint provider. `LlGuidanceConfig` uses the LLGuidance library and supports regex, JSON Schema and Lark grammars; `ExternalConstraintConfig` lets you pass your own pre-built `Constraint` object per request.

Constraints are applied per message through the `decoding_constraint` field of the optional arguments passed to `SendMessage` or `SendMessageAsync`.

## The Kotlin API for Android and the JVM

The Kotlin API targets Android and the JVM (Linux, macOS, Windows), with GPU and NPU acceleration, multi-modality and tool use. Maven packages are `com.google.ai.edge.litertlm:litertlm-android` and `litertlm-jvm`. The `Engine` is the entry point; `engine.initialize()` can take a significant amount of time (up to 10 seconds) to load the model, so call it on a background thread or coroutine, and close the engine when done. An optional writable `cacheDir` can improve the second load time.

On Android, the GPU backend needs `<uses-native-library>` entries for `libvndksupport.so` and `libOpenCL.so` (with `required="false"`) inside the `<application>` tag. The NPU backend may need the directory containing the NPU libraries, such as `context.applicationInfo.nativeLibraryDir`. Messages are sent with `sendMessage` (synchronous), `sendMessageAsync` with a callback, or `sendMessageAsync` returning a Kotlin `Flow`, the recommended approach for coroutine users. For multimodal models, `Content` can be `Text`, `ImageBytes`, `ImageFile`, `AudioBytes` or `AudioFile`, and the engine takes a `visionBackend` and an `audioBackend`.

### Tools and template variables in Kotlin

Tools can be defined with Kotlin functions, a class implementing `ToolSet` with methods annotated `@Tool` and parameters annotated `@ToolParam`, from which the API generates an OpenAPI-style schema. Parameter types can be `String`, `Int`, `Boolean`, `Float`, `Double` or a `List` of these; return values are converted to JSON. Alternatively, implement `OpenApiTool` with a JSON description. By default tool calls are executed automatically and their results sent back to the model; setting `automaticToolCalling = false` lets the app execute tools manually. `extraContext`, a map passed to `sendMessage`, supplies extra variables to the Jinja template, such as `enable_thinking`. Errors surface as `LiteRtLmJniException` or standard Kotlin exceptions.

### Building an Android chat app with LiteRT-LM

The litert-samples `litert-lm` skill builds an Android app with `litertlm-android` 0.17.1, whose AAR declares `minSdk` 24. The model file lives on the device's filesystem (hundreds of MB to a few GB), never in assets or in the APK: for a first run copy it with `adb push`, and for users download it in the app into `filesDir` and check its size against the model card. The engine is kept for the app's lifetime and closed after the conversation; `SamplerConfig(topK, topP, temperature)` in `ConversationConfig` sets sampling. If engine creation fails with `Backend.GPU()`, the two manifest lines are missing; if the first reply is slow, load the engine at app start and set `cacheDir`.

## The Swift API for iOS and macOS

The LiteRT-LM Swift API requires iOS 15.0 or later, macOS 12.0 or later, Xcode 15.0 or later, and a `.litertlm` model file such as Gemma 4 E2B. Add the package with Swift Package Manager from `https://github.com/google-ai-edge/LiteRT-LM.git`; if Xcode reports `no such module LiteRTLM`, add the LiteRTLM library under Frameworks, Libraries, and Embedded Content. Drag the model file into the project with the app target checked, or add it to Copy Bundle Resources. The usage flow is to find the model in the app bundle, configure and initialize the `Engine`, create a `Conversation`, and stream the response into the UI. A locally built, unsigned library may be blocked by macOS with a "Malware" warning; for local testing, remove the quarantine attribute from `CLiteRTLM.xcframework` with `xattr -rd com.apple.quarantine`.
