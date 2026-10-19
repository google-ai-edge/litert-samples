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

// ignore_for_file: implementation_imports

import 'dart:io';

import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:flutter_edge_ai_litertlm/src/ffi/litert_lm_client.dart';
import 'package:flutter_edge_ai_litertlm/src/ffi/litert_lm_bindings.dart';
import 'package:flutter_edge_ai_litertlm/src/ffi/npu_vendor_host.dart'
    show npuDisablesHwMasking, windowsNpuDispatchDir;
import 'package:flutter_edge_ai_litertlm/src/npu_stacks.dart' show NpuVendor;
import 'dart:ffi' as ffi;

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
  Future<void> unload();
}

// ---------------------------------------------------------------------------
// Public-API slot: FlutterEdgeAi.getActiveModel() — ONE active model per
// process, with the plugin's own npu→gpu→cpu fallback. Used by
// EnginePolicy.sequential.
// ---------------------------------------------------------------------------

class LiteRtLmEngineSlot implements EngineSlot {
  LiteRtLmEngineSlot(this.backend);

  @override
  final PreferredBackend backend;

  InferenceModel? _model;
  Duration? _loadTime;

  @override
  PreferredBackend? get activeBackend => _model?.activeBackend;

  @override
  bool get isLoaded => _model != null;

  @override
  Duration? get loadTime => _loadTime;

  @override
  Future<void> load({required String modelPath, required int maxTokens}) async {
    if (_model != null) return;
    // getActiveModel() opens whatever model is active, so make [modelPath]
    // the active one first: Act 2 asks for its own file (the 12B), and
    // without this it silently got the model Act 1 had installed. fromFile
    // registers the file in place; nothing is copied.
    final sw = Stopwatch()..start();
    await FlutterEdgeAi.installModel(modelType: ModelType.gemma4, fileType: ModelFileType.litertlm)
        .fromFile(modelPath)
        .install();
    _model = await FlutterEdgeAi.getActiveModel(
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
    final session = await model.createSession(
      temperature: temperature,
      topK: topK,
      maxOutputTokens: maxOutputTokens,
    );
    return _SessionTurn(session);
  }

  @override
  Future<void> unload() async {
    final model = _model;
    _model = null;
    _loadTime = null;
    if (model != null) await model.close();
  }
}

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
  Future<void> close() => _session.close();
}

// ---------------------------------------------------------------------------
// Raw-FFI slot: one LiteRtLmFfiClient per slot, so several engines can be
// alive at once (lib/experiments/coexistence.dart). No fallback: the requested
// backend either initialises or load() throws. Used by EnginePolicy.resident.
// ---------------------------------------------------------------------------

class FfiEngineSlot implements EngineSlot {
  FfiEngineSlot(this.backend);

  @override
  final PreferredBackend backend;

  LiteRtLmFfiClient? _client;
  Duration? _loadTime;

  /// The raw client has no fallback: it runs on the requested backend or
  /// load() throws. "Requested NPU, ran on CPU" can only happen on the
  /// plugin path ([LiteRtLmEngineSlot]).
  @override
  PreferredBackend? get activeBackend => _client == null ? null : backend;

  @override
  bool get isLoaded => _client != null;

  @override
  Duration? get loadTime => _loadTime;

  @override
  Future<void> load({required String modelPath, required int maxTokens}) async {
    if (_client != null) return;
    _maybeEnableVerboseNativeLog();
    final client = LiteRtLmFfiClient();
    final sw = Stopwatch()..start();
    try {
      await client.initialize(
        modelPath: modelPath,
        backend: backend.name,
        maxTokens: maxTokens,
        // The client no longer picks an NPU vendor itself. Intel's stack sits
        // beside the exe; elsewhere npu stays null and the row fails, as wanted.
        npuDispatch: backend == PreferredBackend.npu && Platform.isWindows
            ? (
                dir: windowsNpuDispatchDir,
                disableHwMasking: npuDisablesHwMasking(NpuVendor.intel),
              )
            : null,
      );
    } catch (_) {
      await client.shutdown();
      rethrow;
    }
    sw.stop();
    _client = client;
    _loadTime = sw.elapsed;
  }

  @override
  Future<EngineTurn> openTurn({
    double temperature = 0.8,
    int topK = 1,
    int? maxOutputTokens,
  }) async {
    final client = _client;
    if (client == null) throw StateError('EngineSlot(${backend.name}) is not loaded');
    // Explicit, so every backend samples alike rather than by the bundle's
    // own defaults. topP 0.95 and seed 1 are what LiteRT-LM 0.17 used unset.
    final handle = await client.createConversationHandle(
      sampling: ResolvedSampling(
        temperature: temperature,
        topK: topK,
        topP: 0.95,
        randomSeed: 1,
      ),
      samplingExplicit: true,
      maxOutputTokens: maxOutputTokens,
      // Text-only turns; the order only matters once media is attached.
      multimodalOrder: MultimodalOrder.imagesAudioText,
    );
    return _HandleTurn(handle);
  }

  @override
  Future<void> unload() async {
    final client = _client;
    _client = null;
    _loadTime = null;
    if (client != null) await client.shutdown();
  }
}

bool _verboseSet = false;

/// `LITERT_VERBOSE=1` in the environment turns LiteRT-LM's own logger to
/// VERBOSE (stderr) before the first engine is created. Windows only: the
/// DLL sits next to the exe; on macOS it is a framework the client resolves.
void _maybeEnableVerboseNativeLog() {
  if (_verboseSet || !Platform.isWindows) return;
  if (Platform.environment['LITERT_VERBOSE'] != '1') return;
  _verboseSet = true;
  try {
    final exeDir = File(Platform.resolvedExecutable).parent.path;
    final lib = ffi.DynamicLibrary.open('$exeDir${Platform.pathSeparator}LiteRtLm.dll');
    LiteRtLmBindings(lib).litert_lm_set_min_log_level(0);
    stderr.writeln('LiteRT-LM log level: VERBOSE');
  } catch (e) {
    stderr.writeln('LiteRT-LM log level unchanged: $e');
  }
}

class _HandleTurn implements EngineTurn {
  _HandleTurn(this._handle);
  final LiteRtLmConversationHandle _handle;

  @override
  Stream<String> generate(String prompt) => _handle.chat(prompt);

  @override
  Future<int?> countTokens(String text) => _handle.tokenCount(text);

  @override
  SessionMetrics get metrics => _handle.getSessionMetrics();

  @override
  Future<void> close() async => _handle.close();
}
