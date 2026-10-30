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

import 'package:flutter_edge_ai/flutter_edge_ai.dart';

/// One conversation on a loaded engine. Created per prompt, closed after.
abstract class EngineTurn {
  /// Multi-turn: every call continues the same native conversation, so the
  /// KV cache of earlier turns is reused (prefill-before-question).
  Stream<String> generate(String prompt);

  /// Token count of [text] with the engine's tokenizer (null if unknown).
  Future<int?> countTokens(String text);

  /// Runtime-reported numbers for the last [generate]. Call after the stream
  /// completes.
  SessionMetrics get metrics;

  /// Stops the [generate] in progress; its stream then ends.
  Future<void> stop();

  Future<void> close();
}

/// One loaded LiteRT-LM engine bound to one requested backend.
///
/// `load()` records wall-clock engine creation time and never downgrades
/// silently: whatever the runtime actually initialised is exposed through
/// [activeBackend] so a table can show "requested NPU, ran on CPU".
abstract class EngineSlot {
  PreferredBackend get backend;
  PreferredBackend? get activeBackend;
  bool get isLoaded;
  Duration? get loadTime;

  Future<void> load({required String modelPath, required int maxTokens});
  Future<EngineTurn> openTurn({
    double temperature = 0.8,
    int topK = 1,
    int? maxOutputTokens,
  });

  /// A chat that declares [tools] and lets the model call them (Act 4).
  Future<InferenceChat> openChat({
    required List<Tool> tools,
    required String systemInstruction,
    required bool enableThinking,
    required double temperature,
    required int topK,
    required int randomSeed,
    int? maxOutputTokens,
  });
  Future<void> unload();
}

/// One model opened with `FlutterEdgeAi.openModel`: this slot owns it, it
/// stays open next to every other slot's model, and [unload] closes it.
/// `main()` registers the engine with `backendFallback: false`, so the
/// requested backend either initialises or [load] throws, and a failed
/// backend is a failed row rather than a quiet reload on the CPU.
class LiteRtLmEngineSlot implements EngineSlot {
  LiteRtLmEngineSlot(this.backend);

  @override
  final PreferredBackend backend;

  InferenceModel? _model;
  Duration? _loadTime;

  /// The load in flight, so [unload] can wait for it and close what it opens.
  Future<void>? _loading;

  @override
  PreferredBackend? get activeBackend => _model?.activeBackend;

  @override
  bool get isLoaded => _model != null;

  @override
  Duration? get loadTime => _loadTime;

  @override
  Future<void> load({required String modelPath, required int maxTokens}) =>
      _loading ??= _load(modelPath, maxTokens).whenComplete(() => _loading = null);

  Future<void> _load(String modelPath, int maxTokens) async {
    if (_model != null) return;
    final sw = Stopwatch()..start();
    _model = await FlutterEdgeAi.openModel(
      await _register(modelPath),
      maxTokens: maxTokens,
      preferredBackend: backend,
    );
    sw.stop();
    _loadTime = sw.elapsed;
  }

  @override
  Future<EngineTurn> openTurn({
    double temperature = 0.8,
    int topK = 1,
    int? maxOutputTokens,
  }) async {
    final model = _model;
    if (model == null) throw StateError('EngineSlot(${backend.name}) is not loaded');
    // Explicit, so every backend samples alike rather than by the bundle's
    // own defaults. topP 0.95 and seed 1 are what LiteRT-LM 0.17 used unset.
    // createSession (not openSession) keeps one native conversation, so the
    // KV cache of earlier turns is reused.
    final session = await model.createSession(
      temperature: temperature,
      topK: topK,
      topP: 0.95,
      randomSeed: 1,
      maxOutputTokens: maxOutputTokens,
    );
    return _SessionTurn(session);
  }

  @override
  Future<InferenceChat> openChat({
    required List<Tool> tools,
    required String systemInstruction,
    required bool enableThinking,
    required double temperature,
    required int topK,
    required int randomSeed,
    int? maxOutputTokens,
  }) {
    final model = _model;
    if (model == null) throw StateError('EngineSlot(${backend.name}) is not loaded');
    return model.createChat(
      temperature: temperature,
      topK: topK,
      topP: 0.95,
      randomSeed: randomSeed,
      tools: tools,
      supportsFunctionCalls: true,
      enableThinking: enableThinking,
      modelType: ModelType.gemma4,
      systemInstruction: systemInstruction,
      maxOutputTokens: maxOutputTokens,
    );
  }

  @override
  Future<void> unload() async {
    // Leaving an act mid-load must not leave the model it was opening behind.
    try {
      await _loading;
    } catch (_) {
      // Nothing was opened.
    }
    final model = _model;
    _model = null;
    _loadTime = null;
    if (model != null) await model.close();
  }
}

/// Registers [path] in place without making it the active model; `openModel`
/// takes the spec it returns. Cheap: nothing is copied or hashed, and the same
/// path again is skipped. Done before every load, because the plugin names a
/// model by its file name: another file with the same name in between would
/// leave an earlier spec pointing at the wrong file.
Future<InferenceModelSpec> _register(String path) async => (await FlutterEdgeAi.installModel(
      modelType: ModelType.gemma4,
      fileType: ModelFileType.litertlm,
    ).fromFile(path).install(activate: false))
        .spec;

class _SessionTurn implements EngineTurn {
  _SessionTurn(this._session);
  final InferenceModelSession _session;

  @override
  Stream<String> generate(String prompt) async* {
    await _session.addQueryChunk(Message.text(text: prompt, isUser: true));
    yield* _session.getResponseAsync();
  }

  @override
  Future<int?> countTokens(String text) => _session.sizeInTokens(text);

  @override
  SessionMetrics get metrics => _session.getSessionMetrics();

  @override
  Future<void> stop() => _session.stopGeneration();

  @override
  Future<void> close() => _session.close();
}
