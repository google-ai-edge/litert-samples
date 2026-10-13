// Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import 'dart:async';
import 'dart:convert';

import 'package:flutter_edge_ai/flutter_edge_ai.dart' hide ModelSpec;

/// One generation the fake session plays when it is asked for a response.
sealed class ScriptedTurn {
  const ScriptedTurn({this.gate});

  /// When set, the turn waits for it before its first token (a slow
  /// prefill: a stop can land while nothing streams yet).
  final Completer<void>? gate;
}

/// A Gemma 4 tool-call turn as LiteRT-LM delivers it: the `tool_calls`
/// JSON streamed as text (core swallows it: the SDK-passthrough branch of
/// `InferenceChat.generateChatResponseAsync`) and the same JSON as the
/// session's `lastRawResponse`.
final class ToolCallTurn extends ScriptedTurn {
  const ToolCallTurn(this.name, this.args, {super.gate, this.midGate});

  final String name;
  final Map<String, Object?> args;

  /// When set, the JSON stops halfway until it completes (a stop can land
  /// mid tool-call generation).
  final Completer<void>? midGate;

  String get rawJson => jsonEncode({
    'role': 'assistant',
    'tool_calls': [
      {
        'type': 'function',
        'function': {'name': name, 'arguments': args},
      },
    ],
  });
}

/// A plain answer, streamed token by token.
final class TextTurn extends ScriptedTurn {
  const TextTurn(this.tokens, {super.gate});

  final List<String> tokens;
}

/// A native decode error mid-turn.
final class FailTurn extends ScriptedTurn {
  const FailTurn(this.error, {super.gate});

  final Object error;
}

/// A `{`-leading turn whose tool call the SDK could not parse: the JSON
/// streams as text and `lastRawResponse` holds no `tool_calls`, so core
/// surfaces the swallowed text as one TextResponse (the swallowed-tool-call
/// fallback of `InferenceChat.generateChatResponseAsync`).
final class UnparsedToolCallTurn extends ScriptedTurn {
  const UnparsedToolCallTurn(this.raw, {super.gate});

  final String raw;
}

abstract class _SessionBase implements InferenceModelSession {}

/// The `.litertlm` FFI session for an agent chat: it buffers queries,
/// plays [script] one turn per `getResponseAsync`, and exposes
/// `lastRawResponse` like the real one. [stopGeneration] cancels natively
/// only while a reply streams and ends that stream cleanly; with
/// [closeOnCancel] false the test ends it with [finish].
class FakeToolSession extends _SessionBase with RawSdkResponseSession {
  FakeToolSession(this._script);

  final List<ScriptedTurn> _script;
  final List<Message> queries = [];
  int responseRequests = 0;
  int stopCalls = 0;
  int nativeCancels = 0;
  bool closed = false;
  bool closeOnCancel = true;
  int nativeInputTokens = 0;
  int nativeOutputTokens = 0;

  StreamController<String>? _reply;
  bool _streaming = false;
  String? _lastRaw;

  @override
  String? get lastRawResponse => _lastRaw;

  bool get streaming => _streaming;

  /// User turns and tool responses, as the session received them.
  List<Message> get toolResponses => [
    for (final m in queries)
      if (m.type == MessageType.toolResponse) m,
  ];

  /// When set, [addQueryChunk] waits for it (a stop can land after the
  /// prompt is staged but before the first generation).
  Completer<void>? queryGate;

  @override
  Future<void> addQueryChunk(Message message) async {
    if (closed) throw StateError('Session is closed');
    await queryGate?.future;
    queries.add(message);
  }

  @override
  Stream<String> getResponseAsync() {
    responseRequests++;
    _lastRaw = null;
    final reply = StreamController<String>(
      onListen: () => _streaming = true,
      onCancel: () => _streaming = false,
    );
    _reply = reply;
    if (_script.isNotEmpty) unawaited(_play(reply, _script.removeAt(0)));
    return reply.stream;
  }

  Future<void> _play(StreamController<String> reply, ScriptedTurn turn) async {
    await Future<void>.delayed(Duration.zero); // listened by now
    await turn.gate?.future;
    if (!identical(_reply, reply)) return; // stopped meanwhile
    switch (turn) {
      case ToolCallTurn(:final rawJson, :final midGate):
        final half = rawJson.length ~/ 2;
        reply.add(rawJson.substring(0, half));
        if (midGate != null) {
          await midGate.future;
          if (!identical(_reply, reply)) return;
        }
        reply.add(rawJson.substring(half));
        _lastRaw = rawJson;
      case FailTurn(:final error):
        _reply = null;
        _streaming = false;
        reply.addError(error);
        unawaited(reply.close());
        return;
      case UnparsedToolCallTurn(:final raw):
        reply.add(raw);
        _lastRaw = '{"role":"assistant","content":"garbled"}';
      case TextTurn(:final tokens):
        for (final token in tokens) {
          if (!identical(_reply, reply)) return;
          reply.add(token);
          await Future<void>.delayed(Duration.zero);
        }
        if (!identical(_reply, reply)) return;
        _lastRaw = jsonEncode({
          'role': 'assistant',
          'content': [
            {'type': 'text', 'text': tokens.join()},
          ],
        });
    }
    nativeOutputTokens += 8;
    finish();
  }

  /// Ends the current reply normally.
  void finish() {
    final reply = _reply;
    if (reply == null) return;
    _reply = null;
    _streaming = false;
    unawaited(reply.close());
  }

  @override
  Future<void> stopGeneration() async {
    stopCalls++;
    final reply = _reply;
    if (!_streaming || reply == null) return;
    nativeCancels++;
    if (!closeOnCancel) return;
    _reply = null;
    _streaming = false;
    unawaited(reply.close());
  }

  @override
  Future<String> getResponse() => throw UnimplementedError();

  @override
  Future<int> sizeInTokens(String text) async => (text.length / 4).ceil();

  @override
  SessionMetrics getSessionMetrics() => SessionMetrics(
    inputTokens: nativeInputTokens,
    outputTokens: nativeOutputTokens,
    totalTokens: nativeInputTokens + nativeOutputTokens,
  );

  @override
  Future<void> close() async {
    closed = true;
    _streaming = false;
    await _reply?.close();
    _reply = null;
  }
}

/// What one `createChat` call asked for.
final class const ChatArgs({
  required final List<Tool> tools,
  required final bool? supportsFunctionCalls,
  required final ModelType? modelType,
  required final bool? supportImage,
  required final String? systemInstruction,
  required final ToolChoice toolChoice,
  required final double temperature,
  required final int topK,
});

/// An [InferenceModel] without an engine whose sessions play [script] (one
/// list for the model: a rebuilt chat continues where the last one
/// stopped). One session slot, like the FFI model.
class FakeToolModel extends InferenceModel {
  final List<ScriptedTurn> script = [];
  final List<FakeToolSession> created = [];
  final List<ChatArgs> chats = [];
  FakeToolSession? _session;

  FakeToolSession get lastSession =>
      created.isEmpty ? throw StateError('No session yet') : created.last;

  @override
  PreferredBackend? get activeBackend => PreferredBackend.gpu;

  @override
  InferenceModelSession? get session => _session;

  @override
  int get maxTokens => 4096;

  @override
  ModelFileType get fileType => ModelFileType.litertlm;

  @override
  Future<InferenceModelSession> createSession({
    double temperature = .8,
    int randomSeed = 1,
    int topK = 1,
    double? topP,
    String? loraPath,
    bool? enableVisionModality,
    bool? enableAudioModality,
    String? systemInstruction,
    bool enableThinking = false,
    List<Tool> tools = const [],
    int? maxOutputTokens,
  }) async {
    await _session?.close();
    final session = FakeToolSession(script);
    created.add(session);
    return _session = session;
  }

  @override
  Future<InferenceChat> createChat({
    double temperature = .8,
    int randomSeed = 1,
    int topK = 1,
    double? topP,
    int tokenBuffer = 256,
    String? loraPath,
    bool? supportImage,
    bool? supportAudio,
    List<Tool> tools = const [],
    bool? supportsFunctionCalls,
    bool enableThinking = false,
    ModelType? modelType,
    ToolChoice toolChoice = ToolChoice.auto,
    int? maxFunctionBufferLength,
    String? systemInstruction,
    int? maxOutputTokens,
  }) {
    chats.add(
      ChatArgs(
        tools: tools,
        supportsFunctionCalls: supportsFunctionCalls,
        modelType: modelType,
        supportImage: supportImage,
        systemInstruction: systemInstruction,
        toolChoice: toolChoice,
        temperature: temperature,
        topK: topK,
      ),
    );
    return super.createChat(
      temperature: temperature,
      randomSeed: randomSeed,
      topK: topK,
      topP: topP,
      tokenBuffer: tokenBuffer,
      loraPath: loraPath,
      supportImage: supportImage,
      supportAudio: supportAudio,
      tools: tools,
      supportsFunctionCalls: supportsFunctionCalls,
      enableThinking: enableThinking,
      // Like the FFI model: the installed type (Gemma 4) when not given.
      modelType: modelType ?? ModelType.gemma4,
      toolChoice: toolChoice,
      maxFunctionBufferLength: maxFunctionBufferLength,
      systemInstruction: systemInstruction,
      maxOutputTokens: maxOutputTokens,
    );
  }

  @override
  void addCloseListener(void Function() listener) {}

  @override
  Future<void> close() async => _session?.close();
}
