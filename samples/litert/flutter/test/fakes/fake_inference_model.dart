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

import 'package:flutter_edge_ai/flutter_edge_ai.dart' hide ModelSpec;

/// [InferenceModelSession] driven by the test. It mimics the `.litertlm` FFI
/// session: the prompt is buffered until [getResponseAsync] is listened to,
/// and [stopGeneration] cancels natively only while a reply is streaming —
/// the stream then closes cleanly, like native `CANCELLED`.
class FakeInferenceSession implements InferenceModelSession {
  final List<Message> queries = [];
  StreamController<String>? _reply;
  bool _streaming = false;
  int responseRequests = 0;
  int stopCalls = 0;
  int nativeCancels = 0;
  bool closed = false;
  double? nativeTokensPerSecond;

  /// LiteRT-LM's prefill and decode counts for this conversation, as
  /// `getSessionMetrics` reports them (the test moves them).
  int nativeInputTokens = 0;
  int nativeOutputTokens = 0;

  /// False: a native cancel lets already-queued chunks through and the test
  /// ends the stream with [finish], like late native callbacks.
  bool closeOnCancel = true;

  /// When set, [addQueryChunk] waits for it, so a test can land a stop after
  /// the turn began but before native generation started.
  Completer<void>? queryGate;

  StreamController<String> get _activeReply =>
      _reply ?? (throw StateError('No reply is streaming'));

  /// Whether a reply stream is being listened to.
  bool get streaming => _streaming;

  void emit(String token) => _activeReply.add(token);

  /// Ends the reply normally.
  void finish() {
    final reply = _activeReply;
    _reply = null;
    _streaming = false;
    unawaited(reply.close());
  }

  /// Ends the reply with a native error.
  void failWith(Object error) {
    final reply = _activeReply;
    _reply = null;
    _streaming = false;
    reply.addError(error);
    unawaited(reply.close());
  }

  @override
  Future<void> addQueryChunk(Message message) async {
    if (closed) throw StateError('Session is closed');
    await queryGate?.future;
    queries.add(message);
  }

  @override
  Stream<String> getResponseAsync() {
    responseRequests++;
    _reply = StreamController<String>(
      onListen: () => _streaming = true,
      onCancel: () => _streaming = false,
    );
    return _activeReply.stream;
  }

  @override
  Future<void> stopGeneration() async {
    stopCalls++;
    final reply = _reply;
    if (!_streaming || reply == null) return; // nothing in flight: no-op
    nativeCancels++;
    if (!closeOnCancel) return;
    _reply = null;
    _streaming = false;
    // Not awaited: the consumer may call this from inside its await-for body,
    // where the done event cannot be delivered until the body returns.
    unawaited(reply.close());
  }

  @override
  Future<String> getResponse() => throw UnimplementedError();

  @override
  Future<int> sizeInTokens(String text) async => (text.length / 4).ceil();

  @override
  SessionMetrics getSessionMetrics() => SessionMetrics(
    tokensPerSecond: nativeTokensPerSecond,
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

/// [InferenceModel] without an engine. Like the FFI model it has one session
/// slot: a new session closes the previous one. Chats get `ModelType.gemma4`.
class FakeInferenceModel extends InferenceModel {
  FakeInferenceModel({this.activeBackend = PreferredBackend.gpu});

  @override
  final PreferredBackend? activeBackend;

  final List<FakeInferenceSession> created = [];
  final List<({double temperature, int topK, int? maxOutputTokens})>
  sessionSettings = [];
  int chatsCreated = 0;
  int closeCalls = 0;

  /// When set, [createSession] waits for it, like a slow conversation create.
  Completer<void>? createGate;
  int _creating = 0;

  /// Most [createSession] calls ever in flight at once.
  int maxConcurrentCreates = 0;
  double? nativeTokensPerSecond;
  FakeInferenceSession? _session;

  FakeInferenceSession get lastSession =>
      created.isEmpty ? throw StateError('No session yet') : created.last;

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
    _creating++;
    if (_creating > maxConcurrentCreates) maxConcurrentCreates = _creating;
    try {
      await createGate?.future;
    } finally {
      _creating--;
    }
    await _session?.close();
    final session = FakeInferenceSession()
      ..nativeTokensPerSecond = nativeTokensPerSecond;
    created.add(session);
    sessionSettings.add((
      temperature: temperature,
      topK: topK,
      maxOutputTokens: maxOutputTokens,
    ));
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
    chatsCreated++;
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
  Future<void> close() async {
    closeCalls++;
    await _session?.close();
  }
}
