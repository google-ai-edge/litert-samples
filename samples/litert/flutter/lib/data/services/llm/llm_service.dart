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
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart' hide ModelSpec;
import 'package:flutter_edge_ai_litertlm/flutter_edge_ai_litertlm.dart'
    show
        BackendInitAttemptFailure,
        BackendInitException,
        LiteRtLmBundleException,
        LiteRtLmBundleProblem,
        LiteRtLmEngineCreateException,
        NpuDispatchUnavailableException;

import '../../../config/model_catalog.dart';
import '../../../domain/models/chat_model_config.dart';
import '../../../utils/result.dart';
import '../../../utils/wav.dart';
import '../images/image_normalizer.dart';

/// What [LlmService.load] reports.
final class const LlmInfo({
  required final String modelId,

  /// `InferenceModel.activeBackend`: the backend the engine runs on. The
  /// strict engine either comes up on the requested one or fails the load;
  /// null when the engine does not report one (the web engine, not built
  /// here).
  required final PreferredBackend? backend,

  /// `InferenceModel.activeVisionBackend` / `activeAudioBackend`: where the
  /// engine put each encoder (its own choice: the app asks for none); null
  /// for one that was not loaded or that the engine does not report.
  final PreferredBackend? visionBackend,
  final PreferredBackend? audioBackend,
  required final Duration loadTime,

  /// The context window the engine was built with (`InferenceModel
  /// .maxTokens`): flutter_edge_ai_litertlm raises a GPU/CPU value below
  /// 1024.
  required final int contextTokens,

  /// The process's resident memory before and after the load
  /// (`ProcessInfo.currentRss`): the resource gate's figures, the audio
  /// encoder's share included when it is loaded. GPU memory the driver keeps
  /// outside the process's pages is not in them.
  final int? rssBeforeBytes,
  final int? rssAfterBytes,
});

/// The chat model did not load: [cause] is the plugin's typed error, and
/// [reason] says what it says. The engine is registered with
/// `LiteRtLmEngine(backendFallback: false)` (`initEdgeAi`), so a load makes
/// one attempt, on [requested], and never comes up on another backend:
/// - a [LiteRtLmBundleException]: the file cannot run as asked, said before
///   any engine exists (its [LiteRtLmBundleProblem]: an NPU build asked for
///   the GPU or the CPU, an incomplete file, an NPU this package has no
///   dispatch for, …);
/// - a [BackendInitException]: the attempt failed; each
///   [BackendInitAttemptFailure.error] is a [LiteRtLmEngineCreateException]
///   (with LiteRT-LM's own [LiteRtLmEngineCreateException.nativeMessage]) or
///   a [NpuDispatchUnavailableException] (the NPU cannot be reached here);
/// - anything else as it was thrown.
final class ChatModelLoadException implements Exception {
  const ChatModelLoadException({
    required this.modelName,
    required this.requested,
    required this.cause,
  });

  final String modelName;
  final PreferredBackend requested;
  final Object cause;

  /// The plugin's reason, in its own words.
  String get reason => switch (cause) {
    LiteRtLmBundleException(:final problem, :final message) =>
      '$message (${problem.name})',
    BackendInitException(:final attempts) => [
      for (final attempt in attempts)
        '${attempt.backend.name}: ${_attemptReason(attempt.error)}',
    ].join('; '),
    final other => '$other',
  };

  /// An attempt's error in the plugin's words. Its engine-create message
  /// usually has LiteRT-LM's own reason in it already:
  /// `Failed to create engine on gpu: {nativeMessage}. Model: {path}`. One
  /// that does not (a declared `section_backend_constraint`) gets it
  /// appended.
  static String _attemptReason(Object error) => switch (error) {
    LiteRtLmEngineCreateException(:final message, :final nativeMessage?)
        when !message.contains(nativeMessage) =>
      '$message (LiteRT-LM: $nativeMessage)',
    LiteRtLmEngineCreateException(:final message) => message,
    NpuDispatchUnavailableException(:final message) => message,
    final other => '$other',
  };

  @override
  String toString() => '$modelName did not load on ${requested.name}: $reason';
}

/// A 1 s 16 kHz mono PCM16 WAV of a quiet 440 Hz tone: what an audio warm-up
/// sends, so the audio encoder runs once before the first spoken question.
Uint8List warmUpWav() {
  const rate = 16000;
  final samples = Int16List(rate);
  for (var i = 0; i < rate; i++) {
    samples[i] = (1000 * math.sin(2 * math.pi * 440 * i / rate)).round();
  }
  return wavFromPcm16(samples.buffer.asUint8List(), sampleRate: rate);
}

/// The Gemma file to install is not there: `GEMMA_MODEL_PATH` points at a
/// file the app cannot see, or a store file vanished after verification.
final class ModelFileMissingException implements Exception {
  const ModelFileMissingException(this.path);

  final String path;

  @override
  String toString() =>
      'Model file not found or not readable: $path (check GEMMA_MODEL_PATH; '
      'a sandboxed macOS build reads its own container, debug builds also '
      '~/Downloads, and files chosen in the app), or choose the model again '
      'on the Models screen.';
}

/// Builds the model for a [LlmConfig]. The app uses [getActiveModelFor];
/// tests pass a fake.
typedef ModelLoader = Future<InferenceModel> Function(LlmConfig config);

/// The arguments of `FlutterEdgeAi.getActiveModel` the app passes (a fake in
/// tests).
typedef ActiveModelGetter = Future<InferenceModel> Function({
  int? maxTokens,
  PreferredBackend? preferredBackend,
  PreferredBackend? preferredVisionBackend,
  PreferredBackend? preferredAudioBackend,
  bool? supportImage,
  int? maxNumImages,
  bool? supportAudio,
});

/// `FlutterEdgeAi.getActiveModel` with exactly the [LlmConfig] arguments.
/// No tower backend is asked for: with `preferredVisionBackend` and
/// `preferredAudioBackend` null the engine places each encoder from the file
/// (flutter_edge_ai_litertlm 1.14.0: its declared `backend_constraint`, the
/// NPU for an encoder compiled for one, else the CPU) and reports where
/// (`InferenceModel.activeVisionBackend` / `activeAudioBackend`).
Future<InferenceModel> getActiveModelFor(
  LlmConfig config, {
  ActiveModelGetter getActiveModel = FlutterEdgeAi.getActiveModel,
}) => getActiveModel(
  maxTokens: config.maxTokens,
  preferredBackend: config.backend,
  preferredVisionBackend: null,
  preferredAudioBackend: null,
  supportImage: config.supportImage,
  maxNumImages: config.maxNumImages,
  supportAudio: config.supportAudio,
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
    this._warmUpAudio = warmUpWav,
  });

  final ModelLoader _loadModel;

  /// The small PNG the warm-up sends ([warmUpPng] in the app).
  final Future<Uint8List> Function() _warmUpImage;

  /// The short WAV an audio warm-up sends ([warmUpWav] in the app).
  final Uint8List Function() _warmUpAudio;

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
  /// flutter_edge_ai 2.4.0 reuses a loaded model only for the same spec
  /// (name, source path, type) and the same arguments, and closes it
  /// otherwise (`InferenceModelCache._whyRebuild`,
  /// `lib/src/lifecycle/inference_model_cache.dart`); [unload] before
  /// installing another file anyway. [modelType] is the install-time type
  /// every chat inherits.
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

  /// Loads the installed model with [model]'s [LlmConfig], on exactly its
  /// backend: the engine is strict (`LiteRtLmEngine(backendFallback: false)`),
  /// so a backend that cannot run it fails the load with the plugin's typed
  /// error, wrapped in a [ChatModelLoadException]. No other check is made
  /// here: an NPU this device cannot reach is the plugin's
  /// [NpuDispatchUnavailableException], an NPU build asked for the GPU its
  /// [LiteRtLmBundleException].
  Future<Result<LlmInfo>> load(ChatModelConfig model) async {
    final config = model.llm;
    // A model still loaded is closed first: flutter_edge_ai 2.4.0 would
    // close it anyway when the spec or the arguments differ, and reuse it
    // when neither does.
    await unload();
    final rssBefore = _rss();
    final watch = Stopwatch()..start();
    try {
      final InferenceModel loaded;
      try {
        loaded = await _loadModel(config);
      } catch (e, st) {
        debugPrint('[LlmService] load failed: $e\n$st');
        return Result.error(
          ChatModelLoadException(
            modelName: model.name,
            requested: config.backend,
            cause: e,
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
      _model = loaded;
      _loaded = model;
      final active = loaded.activeBackend;
      final rssAfter = _rss();
      final info = LlmInfo(
        modelId: _modelId ?? model.name,
        backend: active,
        visionBackend: loaded.activeVisionBackend,
        audioBackend: loaded.activeAudioBackend,
        loadTime: watch.elapsed,
        contextTokens: loaded.maxTokens,
        rssBeforeBytes: rssBefore,
        rssAfterBytes: rssAfter,
      );
      final towers = [
        if (config.supportImage)
          'vision:${info.visionBackend?.name ?? 'not reported'}',
        if (config.supportAudio)
          'audio:${info.audioBackend?.name ?? 'not reported'}',
      ];
      debugPrint(
        '[LlmService] loaded ${model.name} (${info.modelId}) '
        'backend=${active?.name ?? 'not reported'} '
        'load=${watch.elapsedMilliseconds}ms '
        'maxTokens=${loaded.maxTokens} (requested ${config.maxTokens}) '
        'towers=${towers.isEmpty ? 'none' : towers.join(',')} '
        'tools=${model.tools} '
        'type=${model.modelType.name} '
        'rss ${_mb(rssBefore)} -> ${_mb(rssAfter)}'
        '${rssBefore == null || rssAfter == null ? '' : ' (${rssAfter - rssBefore >= 0 ? '+' : ''}${_mb(rssAfter - rssBefore)})'}',
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
  /// survives every later chat. [withAudio] (loaded with the audio encoder):
  /// a 1 s WAV goes with it too, for the audio encoder's first use. A
  /// failure of either fails the warm-up (and the load).
  Future<Result<Duration>> warmUp(
    SamplerConfig sampler, {
    required bool withImage,
    bool withAudio = false,
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
      final audio = withAudio ? _warmUpAudio() : null;
      session = await model.createSession(
        temperature: sampler.temperature,
        topK: sampler.topK,
        maxOutputTokens: 1,
      );
      await session.addQueryChunk(
        Message(text: 'Hi', isUser: true, imageBytes: image, audioBytes: audio),
      );
      await session.getResponseAsync().drain<void>();
      debugPrint(
        '[LlmService] warm-up ${watch.elapsedMilliseconds}ms '
        '(image: ${image == null ? 'no' : '${image.length} B PNG'}, '
        'audio: ${audio == null ? 'no' : '${audio.length} B WAV'})',
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

  static int? _rss() {
    try {
      return ProcessInfo.currentRss;
    } catch (_) {
      return null;
    }
  }

  static String _mb(int? bytes) =>
      bytes == null ? 'n/a' : '${(bytes / (1 << 20)).round()} MB';

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
