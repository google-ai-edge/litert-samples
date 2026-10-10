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
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart' hide ModelSpec;

import '../../../config/model_catalog.dart';
import '../../../domain/models/chat_model_config.dart';
import '../../../domain/models/npu_availability.dart';
import '../../../utils/result.dart';
import '../images/image_normalizer.dart';
import 'npu_availability.dart';

/// What [LlmService.load] reports.
final class const LlmInfo({
  required final String modelId,
  required final PreferredBackend backend,
  required final Duration loadTime,

  /// The context window the engine was built with (`InferenceModel
  /// .maxTokens`): flutter_edge_ai_litertlm raises a GPU/CPU value below
  /// 1024.
  required final int contextTokens,
});

/// The engine came up on a different backend than requested. Never a silent
/// fallback: setup stops and the UI shows this, with why the requested
/// backend failed when flutter_edge_ai said so ([nativeReasons]).
final class BackendMismatchException implements Exception {
  const BackendMismatchException({
    required this.requested,
    required this.active,
    this.modelName = 'Gemma',
    this.nativeReasons = const [],
  });

  final PreferredBackend requested;
  final PreferredBackend? active;
  final String modelName;

  /// flutter_edge_ai's own lines about the failed attempts (`npu backend
  /// failed, trying the next candidate: …`).
  final List<String> nativeReasons;

  @override
  String toString() =>
      '$modelName: requested ${requested.name} but the engine loaded on '
      '${active?.name ?? 'an unknown backend'}. Fallback is disabled, so it '
      'was closed.${_reasons(requested, nativeReasons)}';
}

/// `PreferredBackend.npu` on a device where flutter_edge_ai would not offer
/// it ([NpuUnavailable]): fails before anything loads, instead of loading on
/// the GPU or CPU flutter_edge_ai would put it on.
final class NpuUnavailableException implements Exception {
  const NpuUnavailableException({
    required this.modelName,
    required this.reason,
  });

  final String modelName;
  final String reason;

  @override
  String toString() =>
      '$modelName: the NPU is not available on this device ($reason). Run '
      'it on the GPU or CPU, or use a Snapdragon phone with an NPU build for '
      'its SoC.';
}

/// The load itself failed: wrong SoC, a context the build does not have, no
/// vision encoder, a broken file. [cause] is the engine's error;
/// [nativeReasons] what flutter_edge_ai printed about each attempt.
final class ChatModelLoadException implements Exception {
  const ChatModelLoadException({
    required this.modelName,
    required this.requested,
    required this.cause,
    this.nativeReasons = const [],
  });

  final String modelName;
  final PreferredBackend requested;
  final Object cause;
  final List<String> nativeReasons;

  @override
  String toString() =>
      '$modelName did not load on ${requested.name}: $cause'
      '${_reasons(requested, nativeReasons)}';
}

String _reasons(PreferredBackend requested, List<String> reasons) =>
    reasons.isEmpty ? '' : ' flutter_edge_ai reported: ${reasons.join(' | ')}';

/// flutter_edge_ai's warnings during a model build: its fallback notices go
/// through `print` (`[flutter_edge_ai] WARNING: …`, flutter_edge_ai_litertlm
/// 1.9.0 `lib/src/ffi/backend_preference.dart`), the only place the reason a
/// requested backend failed shows up when another one loads.
@visibleForTesting
List<String> nativeReasonsIn(List<String> printed) => [
  for (final line in printed)
    if (line.contains('[flutter_edge_ai] WARNING:'))
      _clip(
        line.substring(line.indexOf('WARNING:') + 'WARNING:'.length).trim(),
      ),
];

String _clip(String line) =>
    line.length > 400 ? '${line.substring(0, 400)}…' : line;

/// The Gemma file to install is not there: `GEMMA_MODEL_PATH` points at a
/// file the app cannot see, or a store file vanished after verification.
final class ModelFileMissingException implements Exception {
  const ModelFileMissingException(this.path);

  final String path;

  @override
  String toString() =>
      'Model file not found or not readable: $path (check GEMMA_MODEL_PATH; '
      'a sandboxed macOS build reads only its own container), or choose the '
      'model again on the Models screen.';
}

/// Builds the model for a [LlmConfig]. The app uses [getActiveModelFor];
/// tests pass a fake.
typedef ModelLoader = Future<InferenceModel> Function(LlmConfig config);

/// `FlutterEdgeAi.getActiveModel` with exactly the [LlmConfig] arguments.
Future<InferenceModel> getActiveModelFor(LlmConfig config) =>
    FlutterEdgeAi.getActiveModel(
      maxTokens: config.maxTokens,
      preferredBackend: config.backend,
      supportImage: config.supportImage,
      maxNumImages: config.maxNumImages,
    );

/// The service was closed while an operation was in flight.
final class LlmServiceClosedException implements Exception {
  const LlmServiceClosedException();

  @override
  String toString() => 'The LLM service was closed';
}

/// Owns the chat model handle: install, load on exactly the requested
/// backend, warm up, unload, close. Called only by `ModelRepository` (and the
/// self-test); together they are the only place that calls
/// `FlutterEdgeAi.getActiveModel`.
class LlmService {
  LlmService({
    this._loadModel = getActiveModelFor,
    this._warmUpImage = warmUpPng,
    this._npu = probeNpu,
  });

  final ModelLoader _loadModel;

  /// The small PNG the warm-up sends ([warmUpPng] in the app).
  final Future<Uint8List> Function() _warmUpImage;

  /// Whether flutter_edge_ai would accept the NPU here ([probeNpu]).
  final NpuAvailability Function() _npu;

  InferenceModel? _model;
  String? _modelId;
  ChatModelConfig? _loaded;
  bool _closed = false;

  /// Whether [load] succeeded and neither [unload] nor [close] ran since.
  bool get isLoaded => _model != null;

  /// What the loaded model was loaded with (images, tools); null when none
  /// is loaded.
  ChatModelConfig? get loaded => _model == null ? null : _loaded;

  /// The loaded model. Only valid after a successful [load].
  InferenceModel get model =>
      _model ?? (throw StateError('LlmService.load() has not succeeded'));

  /// Registers the `.litertlm` at [path] (absolute: a verified model-store
  /// file, or the resolved `GEMMA_MODEL_PATH`) and makes it the active
  /// model; returns the model id. Nothing is downloaded here: the model store
  /// does that, with resume and verification.
  ///
  /// flutter_edge_ai builds the active spec's own path, but reuses a loaded
  /// model of the same file name with the same arguments (flutter_edge_ai
  /// 2.1.0 `createModel` in `lib/mobile/flutter_edge_ai_mobile.dart` and
  /// `lib/desktop/flutter_edge_ai_desktop.dart`): [unload] before installing
  /// another file. [modelType] is the install-time type every chat inherits.
  Future<Result<String>> install({
    required String path,
    ModelType modelType = ModelType.gemma4,
    required void Function(int percent) onProgress,
  }) async {
    try {
      if (!await File(path).exists()) {
        return Result.error(ModelFileMissingException(path));
      }
      final installation = await FlutterEdgeAi.installModel(
        modelType: modelType,
        fileType: ModelFileType.litertlm,
      ).fromFile(path).withProgress(onProgress).install();
      _modelId = installation.modelId;
      debugPrint('[LlmService] installed ${installation.modelId} from $path');
      return Result.ok(installation.modelId);
    } catch (e, st) {
      debugPrint('[LlmService] install failed: $e\n$st');
      return Result.error(asException(e));
    }
  }

  /// Loads the installed model with [model]'s [LlmConfig]. Fails, and closes
  /// what was loaded, when the engine comes up on another backend than
  /// requested; an NPU request where flutter_edge_ai would not offer the NPU
  /// fails before loading. flutter_edge_ai's prints during the build are
  /// captured (and still printed) so the error can say why the requested
  /// backend failed.
  Future<Result<LlmInfo>> load(ChatModelConfig model) async {
    final config = model.llm;
    if (config.backend == PreferredBackend.npu) {
      if (_npu() case NpuUnavailable(:final reason)) {
        final error = NpuUnavailableException(
          modelName: model.name,
          reason: reason,
        );
        debugPrint('[LlmService] $error');
        return Result.error(error);
      }
    }
    // A model still loaded is closed first: flutter_edge_ai would close it
    // anyway when the arguments differ, and reuse it (the old file) when
    // they do not.
    await unload();
    final watch = Stopwatch()..start();
    final printed = <String>[];
    try {
      final InferenceModel loaded;
      try {
        loaded = await runZoned(
          () => _loadModel(config),
          zoneSpecification: ZoneSpecification(
            print: (self, parent, zone, line) {
              printed.add(line);
              parent.print(zone, line);
            },
          ),
        );
      } catch (e, st) {
        debugPrint('[LlmService] load failed: $e\n$st');
        return Result.error(
          ChatModelLoadException(
            modelName: model.name,
            requested: config.backend,
            cause: e,
            nativeReasons: nativeReasonsIn(printed),
          ),
        );
      }
      watch.stop();
      if (_closed) {
        // close() ran during the load (e.g. quit during the first GPU
        // compile): this model has no owner, so close it here.
        await _closeQuietly(loaded);
        return const Result.error(LlmServiceClosedException());
      }
      final active = loaded.activeBackend;
      if (active != config.backend) {
        final error = BackendMismatchException(
          requested: config.backend,
          active: active,
          modelName: model.name,
          nativeReasons: nativeReasonsIn(printed),
        );
        debugPrint('[LlmService] $error');
        _model = null;
        await _closeQuietly(loaded);
        return Result.error(error);
      }
      _model = loaded;
      _loaded = model;
      final info = LlmInfo(
        modelId: _modelId ?? model.name,
        backend: active!,
        loadTime: watch.elapsed,
        contextTokens: loaded.maxTokens,
      );
      debugPrint(
        '[LlmService] loaded ${model.name} (${info.modelId}) '
        'backend=${active.name} load=${watch.elapsedMilliseconds}ms '
        'maxTokens=${loaded.maxTokens} (requested ${config.maxTokens}) '
        'image=${config.supportImage} tools=${model.tools} '
        'type=${model.modelType.name}',
      );
      return Result.ok(info);
    } catch (e, st) {
      debugPrint('[LlmService] load failed: $e\n$st');
      return Result.error(asException(e));
    }
  }

  /// One tiny generation so the first real turn does not pay for lazy native
  /// setup. Uses the shared chat sampler: on `.litertlm` the
  /// first session's sampler can stay in effect for later ones. The session
  /// is closed before any chat opens.
  ///
  /// [withImage] (the model was loaded with image support): the prompt also
  /// carries a 32×32 PNG, so the vision encoder's first use (~1.3 s extra)
  /// happens here and not on the user's first photo. It is per engine, so it
  /// survives every later chat.
  Future<Result<Duration>> warmUp(
    SamplerConfig sampler, {
    required bool withImage,
  }) async {
    final model = _model;
    if (_closed) return const Result.error(LlmServiceClosedException());
    if (model == null) {
      return Result.error(asException(StateError('warmUp() before load()')));
    }
    final watch = Stopwatch()..start();
    InferenceModelSession? session;
    try {
      final image = withImage ? await _warmUpImage() : null;
      session = await model.createSession(
        temperature: sampler.temperature,
        topK: sampler.topK,
        maxOutputTokens: 1,
      );
      await session.addQueryChunk(
        Message(text: 'Hi', isUser: true, imageBytes: image),
      );
      await session.getResponseAsync().drain<void>();
      debugPrint(
        '[LlmService] warm-up ${watch.elapsedMilliseconds}ms '
        '(image: ${image == null ? 'no' : '${image.length} B PNG'})',
      );
      return Result.ok(watch.elapsed);
    } catch (e, st) {
      debugPrint('[LlmService] warm-up failed: $e\n$st');
      return Result.error(asException(e));
    } finally {
      await session?.close();
    }
  }

  /// Closes the loaded model and keeps the service usable: the next
  /// install + [load] builds a fresh engine (a closed model resets
  /// flutter_edge_ai's singleton). Before switching to another file, and
  /// before the self-test loads its own. A no-op when nothing is loaded.
  Future<void> unload() async {
    final model = _model;
    _model = null;
    _loaded = null;
    if (model != null) {
      debugPrint('[LlmService] unloading the chat model');
      await _closeQuietly(model);
    }
  }

  /// Closes the model, and any model a load in flight delivers later. Safe
  /// to call more than once; a native close failure is logged, not thrown.
  Future<void> close() async {
    _closed = true;
    final model = _model;
    _model = null;
    if (model != null) await _closeQuietly(model);
  }

  /// Closes a model without throwing: a close failure is logged, not allowed
  /// to hide why the model is being closed.
  static Future<void> _closeQuietly(InferenceModel model) async {
    try {
      await model.close();
    } catch (e, st) {
      debugPrint('[LlmService] closing the model failed: $e\n$st');
    }
  }
}
