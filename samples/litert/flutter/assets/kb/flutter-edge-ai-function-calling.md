---
title: Function calling with flutter_edge_ai
source: https://pub.dev/packages/flutter_edge_ai/versions/2.5.1 (skills/flutter-edge-ai-function-calling/SKILL.md, skills/flutter-edge-ai-inference/SKILL.md, README.md, CHANGELOG.md, lib/src/message.dart, lib/src/model_response.dart) ; https://pub.dev/packages/flutter_edge_ai_litertlm/versions/1.17.0 (README.md: Troubleshooting; CHANGELOG.md) ; https://github.com/DenisovAV/flutter_edge_ai/blob/d35c575ab91e078c47201131800cf33dd7630833/website/content/docs/function-calling.md
license: MIT
---

# Function calling with flutter_edge_ai

This document explains function calling, also called tool calling, in flutter_edge_ai (formerly flutter_gemma): letting an on-device model call the app's own Dart functions. It covers which models can call tools, how to declare a tool, the response types a chat returns, the built-in tool loop, handling calls and parallel calls by hand, returning errors, what happens on the LiteRT-LM tool path, and the usual failure symptoms.

## Rules for function calling in flutter_edge_ai

1. `createChat` needs `tools` and `supportsFunctionCalls: true`. Without the flag no call is parsed and only a debug warning is logged — and on `.litertlm` with Gemma 4 or FunctionGemma the declarations still reach the SDK, so the model answers with raw tool-call JSON inside the text stream.
2. Pass `modelType` on web and on ONNX. `createChat` on native `.litertlm`, on MediaPipe Android and iOS, and on built-in AI uses the installed model's type when it is left out; the web engines and ONNX fall back to `ModelType.gemmaIt`, and another model's calls then arrive as raw text. `openChat` always falls back — pass it there on every platform. On `flutter_edge_ai_llama` (`.gguf`) the calls are read through the file's own chat template: `modelType`, which `createChat` and `openChat` both take from the install there, only picks sampling defaults, and the tools are on unless `supportsFunctionCalls: false`.
3. Switch over all four `ModelResponse` subtypes. It is sealed — a switch that leaves out `ThinkingResponse` does not compile.
4. Return tool results as data, errors included. Never throw from a tool.
5. Prefer `generateChatResponseWithTools` to a hand-written loop.
6. Use a model that can call tools: Gemma 4, FunctionGemma, Phi-4 Mini, Qwen 2.5, Qwen3, DeepSeek R1, Gemma 3n E4B `.litertlm`, or the OS built-in model. Gemma 3 1B, Gemma 3 270M, SmolLM, SmolLM3, LFM2.5, Phi-4 Mini Reasoning and the vision models (FastVLM, Qwen2-VL, SmolVLM2, LLaVA-OneVision) cannot — they describe the action in prose instead. A `.gguf` model through `flutter_edge_ai_llama` can call tools when its chat template supports them.

## Which models support function calling

Models with function calling support:

- Gemma 4 (E2B, E4B) — full function calling support, with native function-call tokens.
- Gemma 3n E4B — function calling on the downloadable E4B `.litertlm` build, not on E2B or the MediaPipe `.task` builds.
- FunctionGemma 270M — Google's specialized function calling model.
- DeepSeek R1 — function calling plus thinking mode support.
- Qwen models (0.5B, 0.6B, 1.5B) — full function calling support.
- Phi-4 Mini — advanced reasoning with function calling support.

Models without function calling support: Gemma 3 270M, Gemma 3 1B, SmolLM 135M and LFM2.5 230M (text generation only), FastVLM 0.5B, Qwen2-VL 2B, SmolVLM2 500M and LLaVA-OneVision 0.5B (vision models), SmolLM3 3B (text generation with reasoning) and Phi-4 Mini Reasoning (reasoning model).

If you pass `tools` with `supportsFunctionCalls: false`, the chat logs a warning and does not inject them; the model still works normally for text generation. Function calling works on Android, iOS, web and desktop. Gemma 4 calls tools only from a `.litertlm`, through the native SDK chat template: with a MediaPipe `.task` (the web `-web.task` builds included) or with ONNX its tools never reach the model, so on the web Gemma 4 calls tools through the `.litertlm` web engine. The operating systems' built-in models (Gemini Nano, Apple Foundation Models and others) also call tools, but prompt-based: core weaves the tool definitions into the prompt and parses the calls back out of the model's text.

## Declaring a tool with a JSON Schema

A tool is a `Tool` object with a `name`, a `description` and `parameters`. `parameters` is a JSON Schema object. The model matches the user's intent against `description`, so write it as an action and describe every parameter.

```dart
const changeColor = Tool(
  name: 'change_color',
  description: 'Change the app background colour.',
  parameters: {
    'type': 'object',
    'properties': {
      'color': {'type': 'string', 'description': 'A colour name, e.g. red.'},
    },
    'required': ['color'],
  },
);
```

## Opening a chat with tools enabled

Function calling starts from a loaded `InferenceModel`. Open the chat with the tool list, the function-calls flag and the model type:

```dart
final InferenceChat chat = await model.createChat(
  tools: myTools,
  supportsFunctionCalls: true,
  modelType: ModelType.gemma4,
);
```

The `ModelType` matters because it tells flutter_edge_ai how the model writes tool calls. Gemma 4 uses `ModelType.gemma4`, FunctionGemma uses `ModelType.functionGemma`, Phi-4 Mini has its own `ModelType.phi` that parses Phi's tool-call markers, and Gemma 3 and Gemma 3n use `ModelType.gemmaIt`.

## Response types a chat can return

A chat returns a sealed `ModelResponse`, with four subtypes:

- `TextResponse` contains a text token (`response.token`) for regular model output; use it to update the UI incrementally.
- `FunctionCallResponse` contains the function name (`response.name`) and arguments (`response.args`) when the model wants to call a function.
- `ParallelFunctionCallResponse` contains several calls (`calls`) in one turn.
- `ThinkingResponse` contains the model's reasoning process (`response.content`) for models with thinking mode enabled.

Because the type is sealed, a `switch` must cover all four subtypes or it does not compile. A model that answers in text instead of calling a tool has made a valid choice — the model may decide to answer directly.

## The built-in tool loop generateChatResponseWithTools

The built-in loop calls the handler for each tool call, feeds the result back, and continues until the model answers in text or `maxToolTurns` is reached.

```dart
await chat.addQueryChunk(Message(text: prompt, isUser: true));
final answer = StringBuffer();
await for (final r in chat.generateChatResponseWithTools(
  onToolCall: (FunctionCallResponse call) {
    answer.clear(); // text streamed before a call is not the answer
    return runTool(call.name, call.args);
  },
  maxToolTurns: 8,
  onMaxToolTurns: () => print('stopped after 8 tool turns'),
)) {
  if (r is TextResponse) answer.write(r.token);
}
```

Gemma 4 often writes a sentence, then calls a tool. The sentence streams as `TextResponse`, so the loop above clears the buffer when a call arrives, as `flutter_edge_ai_agent` does. Since `flutter_edge_ai` 2.4.0 the call's own JSON no longer streams as text after such a sentence, and the chat's history keeps only the call, not the sentence. Without `supportsFunctionCalls: true`, or with `toolChoice: ToolChoice.none`, the call's JSON stays in the text instead.

`onToolCall` receives a `FunctionCallResponse` with `name` and `args`, and returns the map the model reads back. `args` keeps the JSON types the model wrote — numbers, booleans, null, nested maps and lists, not only strings — so read a number as `num`. Its `id` is null on the `.litertlm`, MediaPipe, ONNX and built-in AI engines; `flutter_edge_ai_llama` gives every call one, and genai `callId`s and Genkit `ref`s carry theirs. The stream carries `TextResponse` and `ThinkingResponse`. Reaching `maxToolTurns` ends the stream without an error — `onMaxToolTurns` is the only signal. An exception from `onToolCall` is answered to the model as `{"error": …}`, and a result `jsonEncode` cannot write (a `DateTime`, say) as `{"status": "ran", …}`, because the tool did run; the turn's other calls are answered as not run, and the error is rethrown on the stream. When an answer cannot be fed, for example to a session closed by a barge-in, the tool's own error is still the one rethrown: recreate the chat, or call `clearHistory(replayHistory:)`, before the next turn.

## Handling a tool call yourself

Without the built-in loop, call `generateChatResponse()` and switch over the result. For a `FunctionCallResponse(:final name, :final args)`, run the tool, add the result with `Message.toolResponse(toolName: name, response: result)`, and call `generateChatResponse()` again to get the follow-up. A `TextResponse` means the model chose to answer directly, and a `ThinkingResponse` can be skipped.

```dart
case FunctionCallResponse(:final name, :final args):
  final result = await runTool(name, args);
  await chat.addQueryChunk(Message.toolResponse(toolName: name, response: result));
  final followUp = await chat.generateChatResponse();
```

`Message.toolResponse` is one of several message constructors; others are `Message.text`, `Message.withImages`, `Message.imagesOnly`, `Message.withAudio`, `Message.toolCall`, `Message.systemInfo` and `Message.thinking`. `Message.toolResponse` also takes `toolCallId:`, the call's `id`, which stays in the history. Only `flutter_edge_ai_llama` reads it — pass it there when two calls share a name and their results go back out of order; the other engines pair a result with its call by `toolName`, in order.

## Handling parallel function calls

A `ParallelFunctionCallResponse(:final calls)` carries several calls in one model turn. Run each call, add one `Message.toolResponse(toolName: call.name, response: result)` per call, and only then call `generateChatResponse()` once to get the answer that uses all the results.

```dart
case ParallelFunctionCallResponse(:final calls):
  for (final call in calls) {
    final result = await runTool(call.name, call.args);
    await chat.addQueryChunk(Message.toolResponse(toolName: call.name, response: result));
  }
  final afterAll = await chat.generateChatResponse();
```

## Tool calls in a saved chat history

Since `flutter_edge_ai` 2.4.0 a chat records each tool call as `Message.toolCall(calls:)`. Its `toolCalls` hold the structured `FunctionCallResponse`s with their ids, and its `text` is what text engines replay — for a `.litertlm` call, its `tool_calls` objects without the text the model wrote before them. To restore a conversation with `clearHistory(replayHistory:)`, keep both, and give each `Message.toolResponse` the `toolCallId` of its call. A `Message.toolCall` built from `calls` alone writes its `text` as `{"tool_calls":[{"id","name","arguments"}]}`.

## Returning errors as tool results

Errors are results. When a tool fails, return the error as data instead of throwing:

```dart
await chat.addQueryChunk(
  Message.toolResponse(
    toolName: 'change_color',
    response: {'error': 'unknown colour: mauvish'},
  ),
);
```

The model can recover from an error it can read. An exception thrown out of a tool ends the turn instead.

## What happens after you send a tool result

On a `.litertlm`, both Gemma 4 and FunctionGemma go through LiteRT-LM's own tool path: the declarations travel to the runtime as structured data, the call comes back parsed, and `Message.toolResponse(...)` goes back as one role-`tool` message that continues the same model turn. Every release of `flutter_edge_ai` with `flutter_edge_ai_litertlm` does this for FunctionGemma too; legacy pairs older than `flutter_gemma` 1.8.4 with `flutter_gemma_litertlm` 1.7.1 sent its tool results as an ordinary user message, and the model answered them by repeating the call it had just made.

Where a call comes back as text rather than structured tool calls — the web SDK, or a `.litertlm` exported without the FunctionGemma model type, whose runtime opens no tool-call channel — flutter_edge_ai parses that text itself, so the app still receives a `FunctionCallResponse`. Nothing in app code changes: the wire format is chosen from the model type and the file type together.

### Three things to know about the LiteRT-LM tool path

- `ToolChoice.none` cannot take the declarations back out on a `.litertlm`, because the runtime holds them.
- FunctionGemma is an action model — it often ends its turn at the call rather than narrating the result. Render the tool's own result in the UI, and reach for Gemma 4 when you want the model to talk about what came back.
- `.task` models through MediaPipe have no native tool path, so they keep the text wire format flutter_edge_ai renders itself.

## Constrained decoding and the tool-call crashes fixed in the LiteRT-LM engine

On the LiteRT-LM engine, constrained decoding for tool calls is implemented by a prebuilt companion library, `libGemmaModelConstraintProvider`, that ships with the LiteRT-LM release. In `flutter_gemma_litertlm` 1.7.0, a chat or session created with `tools` died on the first decoded token — `EXC_BAD_ACCESS` or `SIGSEGV` inside the runtime, on every platform, CPU and GPU alike — while generation without tools was unaffected. The cause was that upstream replaced the `Constraint` interface, and the companion published at tag v0.17.0 still implemented the old one, so the runtime called into the wrong vtable slot. Version 1.7.1 of `flutter_gemma_litertlm`, the engine package's old name, fixed it by pinning a native bundle with the companion rebuilt from upstream main, and every `flutter_edge_ai_litertlm` release includes that fix.

Linux had a second cause. Up to `flutter_edge_ai_litertlm` 1.11.0, a chat or session with `tools` aborted on its first constrained token on Linux x64 and arm64, because the prebuilt constraint provider uses another C++ ABI than the runtime. Version 1.11.1 (`native-v0.18.0-d`) bridges it, so tool calls on Linux need 1.11.1 or later.

## When the model answers in prose or shows raw tool-call markers

Model answers in prose: the symptom is a reply such as "I would change the colour to red" instead of a call. Check `supportsFunctionCalls: true`, pass `modelType` on web and ONNX, then check that the model is tool-capable.

Raw markers in the text: the symptom is `<|tool_call>` or `<tool_call|>` appearing in `TextResponse` tokens. The cause is a `modelType` that does not match the installed model, or a call that was cut off or malformed: since `flutter_edge_ai` 2.4.1 a raw Gemma 4 call runs with its typed arguments or not at all, so when one call of the turn does not parse in full — cut off by `maxOutputTokens`, say — none of that turn's calls runs and the text stays as it came. Raise `maxOutputTokens` then. The same mismatch on Gemma 4 or FunctionGemma without `supportsFunctionCalls` produces raw tool-call JSON in the text stream.

## When the model repeats a call or says nothing after the tool result

Model calls the same tool again instead of answering the result: the symptom is FunctionGemma on `.litertlm` repeating the call it just made after `Message.toolResponse`. The fix is to upgrade `flutter_edge_ai` and `flutter_edge_ai_litertlm` together — both halves. Core picks the wire format and the engine sends it; the current packages include both, while legacy pairs older than `flutter_gemma` 1.8.4 with `flutter_gemma_litertlm` 1.7.1 send the result as an ordinary user message.

Nothing after the tool result: the symptom is that the call arrives, the result goes back, and the stream ends with no text. On FunctionGemma there is nothing to fix — it is an action model, and ending the turn at the call is what it was trained for. Render the tool's own result, and use Gemma 4 when the model should talk about what came back.

## Function calling on the web

Function calling works on the `.litertlm` web engine, which is otherwise an early-preview, text-only engine. Constrained decoding is left off there (LiteRT-LM issue 2434 aborts the next turn after a tool call when it is on), so the call's JSON shape is not grammar-enforced on the web. Pass `modelType`, because the web engines fall back to `ModelType.gemmaIt` when it is left out. On the web `createSession` and `createChat` hold one slot: a second `createChat` while the first is open hands back the open session, history and all. Close the chat before creating the next, or use `openChat` for conversations that run side by side. For agent-style skills defined in `SKILL.md` files, which run on top of this function-calling loop, see the separate `flutter_edge_ai_agent` package.
