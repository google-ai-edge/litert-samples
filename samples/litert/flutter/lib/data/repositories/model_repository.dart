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

import 'package:flutter/foundation.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart' show EmbeddingModality;
import 'package:flutter_edge_ai_litertlm/flutter_edge_ai_litertlm.dart'
    show readLiteRtLmBundleInfo;

import '../../config/env.dart';
import '../../config/model_catalog.dart';
import '../../domain/hardware/accelerator_inference.dart';
import '../../domain/hardware/native_log_parser.dart';
import '../../domain/models/chat_model.dart';
import '../../domain/models/chat_model_config.dart';
import '../../domain/models/detection.dart';
import '../../domain/models/detector_choice.dart';
import '../../domain/models/detector_spec.dart';
import '../../domain/models/hardware_profile.dart';
import '../../domain/models/model_id.dart';
import '../../domain/models/model_source_resolver.dart';
import '../../domain/models/model_state.dart';
import '../../domain/ports/chat_model_planner.dart';
import '../../domain/ports/model_states.dart';
import '../../utils/result.dart';
import '../services/detector/detector_engine.dart' show isDetectorFileProblem;
import '../services/detector/detector_service.dart';
import '../services/hardware/native_log_tap.dart';
import '../services/knowledge/embedder_service.dart';
import '../services/llm/llm_service.dart';
import '../services/llm/npu_availability.dart' show currentAbi;
import '../services/model_store/bundled_model_files.dart';
import '../services/model_store/local_files.dart';
import '../services/model_store/models_folder.dart' show BundleReader;
import '../services/speech/stt_service.dart';
import '../services/speech/tts_service.dart';
import 'chat_model/named_chat_model.dart';
import 'model/model_load_pipeline.dart';

/// A model has no files built into this app.
final class ModelNotProvisionedException implements Exception {
  const ModelNotProvisionedException(this.id);

  final ModelId id;

  @override
  String toString() => '${id.spec.displayName} is not built into this app.';
}

/// No chat model is chosen yet (the app ships none): the demos stay off and
/// the Models screen's Chat model card says how to choose one. [note] is a
/// one-time explanation (a retired saved choice).
final class ChatModelNotChosenException implements Exception {
  const ChatModelNotChosenException([this.note]);

  final String? note;

  @override
  String toString() => [
    'No chat model yet: choose a .litertlm in the Chat model card.',
    ?note,
  ].join(' ');
}

/// The user's own chat model is chosen but cannot be loaded; [reason] says
/// why and what to do.
final class ChatModelBlockedException implements Exception {
  const ChatModelBlockedException(this.reason);

  final String reason;

  @override
  String toString() => reason;
}

/// Installs, loads and warms up every model, the largest (the chat model) first
/// so it fails first, and publishes each model's state. Required models first;
/// a required failure stops setup. Optional models ([ModelSpec.required] false)
/// follow, and their failure is published without failing [prepareAll].
///
/// The detector, the embedder and the speech models are built into the app; the
/// chat model is the one chosen in the Chat model card, or `GEMMA_MODEL_PATH`
/// while none is ([ModelSourceResolver]'s rule). After install the app needs no
/// network.
///
/// Together with its services this is the only code that calls
/// `FlutterEdgeAi.getActive*`.
class ModelRepository implements ModelStates {
  ModelRepository({
    required this._llm,
    required this._stt,
    required this._tts,
    required this._bundled,
    String gemmaModelPath = kGemmaModelPath,
    this._warmUpSampler = kSampler,
    required this._detector,
    this._detectorBackend = kDetectorBackend,
    this._detectorBackendChoice,
    this._bundledDetector = loadBundledDetector,
    required this._embedder,
    this._logTap = const NoNativeLogTap(),
    this._hardwareProfile,
    this._chatModels,
    Set<ModelId>? requiredModels,
    this._closeWait = const Duration(seconds: 5),
    this._readBundle = readLiteRtLmBundleInfo,
    String? abi,
  }) : _abi = abi ?? currentAbi(),
       _sources = ModelSourceResolver(
         // `GEMMA_MODEL_PATH` loads with the resolver's default settings,
         // the ones the self-test uses too.
         gemmaModelPath: gemmaModelPath,
       ),
       _required = requiredModels ?? _catalogRequired;

  final LlmService _llm;
  final SttService _stt;
  final TtsService _tts;

  /// Reads what `GEMMA_MODEL_PATH`'s file says about itself
  /// ([namedChatModelFor]); flutter_edge_ai_litertlm's
  /// `readLiteRtLmBundleInfo` (a fake in tests).
  final BundleReader _readBundle;

  /// This process's ABI: where a file compiled for an NPU can run.
  final String _abi;

  /// Which file the chat model loads: the chosen one, or
  /// `GEMMA_MODEL_PATH` while none is. Every other model is built in.
  final ModelSourceResolver _sources;
  final DetectorService _detector;
  final SamplerConfig _warmUpSampler;

  /// `DETECTOR_BACKEND`; empty is the GPU. Used only without
  /// [_detectorBackendChoice].
  final String _detectorBackend;

  /// Demo 3's Detector setting under the define's precedence
  /// (`LiveCameraSettingsRepository.readBackend`), read at every detector
  /// load.
  final Future<Result<DetectorBackendChoice>> Function()?
  _detectorBackendChoice;

  /// The built-in detector's bytes (the asset bundle in the app).
  final Future<Uint8List> Function() _bundledDetector;
  final EmbedderService _embedder;

  /// The model files built into the app (the embedder's, the recognizers',
  /// the TTS bundle).
  final BundledModelFiles _bundled;

  /// Each GPU-capable load (Gemma, the detector) reads its own window of the
  /// native log, kept in [LoadedModelInfo.nativeLog].
  final NativeLogTap _logTap;

  /// The probed device (`HardwareRepository`: probed once, never throws;
  /// null when the probe failed), read with the native log after a GPU
  /// detector load: a software rasterizer fails it. Null: only the log is
  /// read.
  final Future<HardwareProfile?> Function()? _hardwareProfile;

  /// What the chat model slot ([ModelId.chat]) loads: the user's own
  /// `.litertlm`. Null: none chosen (`GEMMA_MODEL_PATH` still loads).
  final ChatModelPlanner? _chatModels;

  /// Models setup cannot succeed without; the catalog's by default.
  final Set<ModelId> _required;

  /// How long [close] waits for the loads in flight to stop before it
  /// closes the models anyway.
  final Duration _closeWait;

  static final Set<ModelId> _catalogRequired = {
    for (final id in ModelId.values)
      if (id.spec.required) id,
  };

  final ValueNotifier<Map<ModelId, ModelState>> _states = ValueNotifier(
    Map.unmodifiable({
      for (final id in ModelId.values) id: const ModelPending(),
    }),
  );
  Future<Result<void>>? _inFlight;
  final ValueNotifier<bool> _preparing = ValueNotifier(false);
  bool _closed = false;
  Future<void>? _closing;

  /// Setup runs and detector reloads in flight: [close] waits for them.
  int _loads = 0;
  Completer<void>? _loadsIdle;

  /// Every model's install → load → warm-up, publishing on [_states].
  late final ModelLoadPipeline _pipeline = ModelLoadPipeline(
    publish: _set,
    isClosed: () => _closed,
  );

  /// True while a setup run ([prepareAll], also inside [reloadChatModel])
  /// goes.
  @override
  ValueListenable<bool> get preparing => _preparing;

  @override
  ValueListenable<Map<ModelId, ModelState>> get states => _states;

  @override
  bool get requiredReady =>
      _required.every((id) => _states.value[id] is ModelReady);

  /// Prepares every model that is not ready yet. Concurrent calls share one
  /// run; calling it again after a failure is the Retry.
  Future<Result<void>> prepareAll() => _inFlight ??= () {
    if (!_closed) _preparing.value = true;
    return _tracked(_prepareAll).whenComplete(() {
      _inFlight = null;
      if (!_closed) _preparing.value = false;
    });
  }();

  Future<Result<void>> _prepareAll() async {
    if (_closed) return _closedError();
    final llm = await _prepareLlm();
    // No chat model chosen yet: everything else loads (built in), so choosing
    // one later only loads it; the run still fails at _checkRequired.
    if (llm case Error(:final error)
        when error is! ChatModelNotChosenException) {
      return llm;
    }
    if (_closed) return _closedError();
    // Speech after the LLM: the largest model fails first.
    final stt = await _prepareStt(ModelId.whisperBase);
    if (stt is Error<void>) return stt;
    if (_closed) return _closedError();
    final tts = await _prepareTts();
    if (tts is Error<void>) return tts;
    if (_closed) return _closedError();
    await _prepareOptional(ModelId.yolo26n, _prepareDetector);
    if (_closed) return _closedError();
    // Demo 3's recognizer: downloaded, loaded and warmed up once so a switch
    // on entry only reloads it. It stays the active one after setup.
    await _prepareOptional(ModelId.moonshineTiny, () async {
      await _prepareStt(ModelId.moonshineTiny);
      return _states.value[ModelId.moonshineTiny] ?? const ModelPending();
    });
    if (_closed) return _closedError();
    // Last; the knowledge base indexes after setup, not in it.
    await _prepareOptional(ModelId.embeddingGemma, _prepareEmbedder);
    if (_closed) return _closedError();
    return _checkRequired();
  }

  /// The steps above are listed by hand while [requiredReady] follows the
  /// catalog: a required model that no step made ready fails setup here. One
  /// still pending (no step for it) becomes a failure with a Retry rather than
  /// a row stuck at "Waiting".
  Result<void> _checkRequired() {
    final missing = [
      for (final id in ModelId.values)
        if (_required.contains(id) && _states.value[id] is! ModelReady) id,
    ];
    if (missing.isEmpty) return const Result.ok(null);
    for (final id in missing) {
      if (_states.value[id]
          case ModelPending() ||
              ModelInstalling() ||
              ModelLoading() ||
              ModelWarmingUp()) {
        _set(id, ModelFailed('No setup step prepared ${id.spec.displayName}'));
      }
    }
    return Result.error(
      asException(
        StateError(
          'Required models not ready: '
          '${missing.map((id) => id.spec.displayName).join(', ')}',
        ),
      ),
    );
  }

  /// The active speech recognizer (the overlay shows it); null before setup
  /// loaded one and while a switch replaces it.
  ValueListenable<ActiveStt?> get activeStt => _stt.active;

  /// Why the last recognizer switch failed; null after a successful one.
  ValueListenable<String?> get sttSwitchError => _stt.switchError;

  /// Makes [id] (a demo's [ModelId] recognizer) the active one: the STT
  /// model is a singleton, so Demo 1 (Whisper) and Demo 3 (moonshine)
  /// switch it on entry, in parallel with opening the chat. A no-op when it
  /// is active already; fails when setup did not make [id] ready.
  Future<Result<void>> activateStt(ModelId id) async {
    if (_closed) return _closedError();
    if (_states.value[id] is! ModelReady) {
      return Result.error(
        SpeechNotReadyException('${id.spec.displayName} (not ready)'),
      );
    }
    return switch (await _stt.activate(id, warmUp: kSttWarmUpOnSwitch)) {
      Ok() => const Result.ok(null),
      Error(:final error) => Result.error(error),
    };
  }

  /// A recognizer from the files built into the app. A built-in file that
  /// cannot be found or extracted (a full disk) is a failure with Retry, for
  /// an optional recognizer too; only a recognizer this build has no files
  /// for at all ([ModelNotProvisionedException]) is unavailable.
  Future<Result<void>> _prepareStt(ModelId id) async {
    if (_states.value[id] is ModelReady) return const Result.ok(null);
    final SttSource source;
    switch (await _sttSource(id)) {
      case Ok(:final value):
        source = value;
      case Error(:final error)
          when _required.contains(id) || error is! ModelNotProvisionedException:
        return _fail(id, error);
      case Error(:final error):
        _set(id, ModelUnavailable(error.toString()));
        return Result.error(error);
    }
    if (_closed) return _closedError();
    return _prepareSpeech(
      id,
      install: (onProgress) =>
          _stt.install(id, source: source, onProgress: onProgress),
      load: () => _stt.load(id),
      warmUp: _stt.warmUp,
      unload: _stt.unload,
      modelId: () => _stt.modelIdOf(id),
      backend: _stt.configOf(id).backend.name,
    );
  }

  Future<Result<SttSource>> _sttSource(ModelId id) async {
    final files = kBundledSttFiles[id];
    if (files == null) return Result.error(ModelNotProvisionedException(id));
    // An Android first launch extracts here (no percent).
    _set(id, const ModelInstalling());
    final model = await _bundled.pathOf(files.model);
    final tokenizer = await _bundled.pathOf(files.tokenizer);
    return switch ((model, tokenizer)) {
      (Ok(value: final m), Ok(value: final t)) => Result.ok(
        SttFromFiles(modelPath: m, tokenizerPath: t),
      ),
      (Error(:final error), _) || (_, Error(:final error)) => Result.error(
        BundledFileException('The built-in ${id.spec.displayName}: $error'),
      ),
    };
  }

  /// Inflect TTS from its bundle built into the app (one directory; on
  /// Android extracted once).
  Future<Result<void>> _prepareTts() async {
    const id = ModelId.inflectNano;
    if (_states.value[id] is ModelReady) return const Result.ok(null);
    _set(id, const ModelInstalling());
    final String directory;
    switch (await _bundled.directoryOf(kBundledInflectFiles)) {
      case Ok(:final value):
        directory = value;
      case Error(:final error):
        return _fail(
          id,
          BundledFileException('The built-in ${id.spec.displayName}: $error'),
        );
    }
    if (_closed) return _closedError();
    return _prepareSpeech(
      id,
      install: (onProgress) =>
          _tts.install(directory: directory, onProgress: onProgress),
      load: _tts.load,
      warmUp: _tts.warmUp,
      unload: _tts.unload,
      modelId: () => _tts.modelId,
      backend: kTtsConfig.backend.name,
    );
  }

  /// Runs an optional model's step. Whatever happens is published on that
  /// model's row; it never fails [prepareAll].
  Future<void> _prepareOptional(
    ModelId id,
    Future<ModelState> Function() step,
  ) async {
    if (_states.value[id] is ModelReady) return;
    try {
      _set(id, await step());
    } catch (e, st) {
      debugPrint('[ModelRepository] optional $id failed: $e\n$st');
      _set(id, ModelFailed(e.toString()));
    }
  }

  /// The detector row's software-GPU refusal ([_softwareGpuFailure]) while
  /// it is still the row. Setup runs ([prepareAll]: a Retry, a chat model
  /// reload) keep it rather than load on the same GPU again (seconds on a
  /// software rasterizer, refused again); a CPU choice, or [reloadDetector]
  /// (it resets the row first), loads again.
  ModelFailed? _softwareGpuRefusal;

  /// YOLO26n in its worker isolate, with its load checks (strict GPU, or the
  /// explicit CPU mode; see `DetectorEngine.load`), then, on the GPU, the
  /// check that the GPU is not a software rasterizer
  /// ([_softwareGpuFailure]). Any failure is this row's, shown on the Demo 3
  /// tile; it never falls back to another backend.
  ///
  /// The model: the asset built into the app.
  Future<ModelState> _prepareDetector() async {
    const id = ModelId.yolo26n;
    final DetectorBackend backend;
    switch (await (_detectorBackendChoice?.call() ??
        Future.value(resolveDetectorBackend(define: _detectorBackend)))) {
      case Ok(:final value):
        backend = value.backend;
      case Error(error: final InvalidDetectorBackendDefineException error):
        return ModelUnavailable(error.toString());
      case Error(:final error):
        return ModelFailed(error.toString());
    }
    if (_softwareGpuRefusal case final refusal?
        when backend == DetectorBackend.gpu &&
            identical(_states.value[id], refusal)) {
      debugPrint(
        '[ModelRepository] detector: the GPU was refused as a software '
        'rasterizer; not loading on it again',
      );
      return refusal;
    }
    final DetectorModelSource source;
    try {
      source = DetectorBytes(await _bundledDetector());
    } catch (e, st) {
      debugPrint('[ModelRepository] the bundled detector: $e\n$st');
      return ModelFailed(
        'The built-in detector ($kDetModelAsset) could not be read: $e',
      );
    }
    final outcome = await _pipeline.run(
      id,
      // Nothing to install: the file or the bytes load as they are.
      load: () => _detector.load(source: source, backend: backend),
      // close() ran during the load: whatever loaded has no owner now.
      releaseOrphan: _detector.close,
      // No warm-up step: the load runs the model (verify, first run).
      logTap: _logTap,
      describe: (info, _, nativeLog) => LoadedModelInfo(
        modelId: kDetModelName,
        backend: info.backend.name,
        loadTime: info.createTime + info.verifyTime,
        warmUpTime: info.firstRunTime,
        detail: info.label,
        explicitCpu: info.backend == DetectorBackend.cpu,
        nativeLog: nativeLog,
      ),
    );
    if (outcome case LoadReady(:final info)
        when backend == DetectorBackend.gpu) {
      if (await _softwareGpuFailure(info) case final reason?) {
        debugPrint('[ModelRepository] detector refused: $reason');
        // Released at once, the service stays usable for "Run detector on
        // CPU". After close() the service is closed already.
        await _detector.unload();
        // No Retry on the row: the same GPU is refused again. Its message
        // says what helps; Demo 3 offers the CPU.
        return _softwareGpuRefusal = ModelFailed(
          reason,
          backend: backend.name,
          retryable: false,
        );
      }
    }
    return _finalState(
      outcome,
      failed: (failure) => ModelFailed(
        failure.error.toString(),
        // Demo 3 offers the other backend, unless the file itself is bad.
        backend: isDetectorFileProblem(failure.error.toString())
            ? null
            : backend.name,
      ),
    );
  }

  /// Why a GPU detector load ([info]) is refused: its evidence shows a
  /// software rasterizer (llvmpipe & co.). Read as the "This device" card
  /// reads it ([inferEvidence]): the adapter line in the load's native log,
  /// else the probe's likely adapter. Such a "GPU" takes seconds per frame,
  /// so the live pipeline's stall watchdog would fail it with no way out:
  /// a GPU failure instead, and Demo 3 offers the CPU. Null otherwise.
  Future<String?> _softwareGpuFailure(LoadedModelInfo info) async {
    final evidence = inferEvidence(
      requested: DetectorBackend.gpu.name,
      reported: info.backend,
      hardware: await _hardwareProfile?.call(),
      log: parseNativeLog(info.nativeLog),
    );
    return switch (softwareGpuReason(evidence)) {
      final reason? =>
        '$reason No fallback: install the GPU driver, or run the detector '
            "on the CPU (Live camera's settings, or DETECTOR_BACKEND=cpu).",
      null => null,
    };
  }

  /// Loads the detector again from the current backend choice (Demo 3's
  /// Detector setting): after a setup run in flight, `DetectorService.load`
  /// replaces the worker and the row is republished. The caller stops live
  /// detection first, so no frame is in the old worker.
  Future<Result<void>> reloadDetector() async {
    if (_closed) return _closedError();
    final running = _inFlight;
    if (running != null) await running;
    if (_closed) return _closedError();
    _set(ModelId.yolo26n, const ModelPending());
    await _tracked(() => _prepareOptional(ModelId.yolo26n, _prepareDetector));
    return switch (_states.value[ModelId.yolo26n]) {
      ModelReady() => const Result.ok(null),
      ModelFailed(:final message) => Result.error(
        DetectorUnavailableException(message),
      ),
      ModelUnavailable(:final reason) => Result.error(
        DetectorUnavailableException(reason),
      ),
      _ => const Result.error(
        DetectorUnavailableException('The detector did not load'),
      ),
    };
  }

  /// EmbeddingGemma 2 for the knowledge base: the `.litertlm` built into the
  /// app (in place on desktop and iOS, extracted once and verified on
  /// Android). Loaded on the CPU for text and checked (backend, 768 dims),
  /// warmed up with one query embedding. Any failure is this row's: Demo 1
  /// then chats without the knowledge base and says so.
  Future<ModelState> _prepareEmbedder() async {
    const id = ModelId.embeddingGemma;
    final EmbedderSource source;
    _set(id, const ModelInstalling());
    // An Android first launch extracts here (no percent: the copy runs
    // natively without progress events).
    switch (await _bundled.pathOf(kBundledEmbedderModel)) {
      case Ok(value: final path):
        source = EmbedderFromFiles(modelPath: path);
      case Error(:final error):
        return ModelFailed('The built-in EmbeddingGemma: $error');
    }
    if (_closed) return const ModelFailed('ModelRepository closed');
    final outcome = await _pipeline.run(
      id,
      install: (onProgress) =>
          _embedder.install(source, onProgress: onProgress),
      load: _embedder.load,
      warmUp: _embedder.warmUp,
      describe: (info, warmUpTime, _) => LoadedModelInfo(
        modelId: info.modelId,
        backend: info.backend.name,
        loadTime: info.loadTime,
        warmUpTime: warmUpTime,
        detail: _embedderDetail(info.backend.name, info.modalities),
      ),
    );
    return _finalState(
      outcome,
      failed: (failure) => switch (failure) {
        LoadFailed(
          step: LoadStep.install,
          error: final EmbedderFilesMissingException error,
        ) =>
          ModelUnavailable(error.toString()),
        LoadFailed(:final error) => ModelFailed(error.toString()),
      },
    );
  }

  /// Adds EmbeddingGemma's image encoder for the visual memory
  /// ([EmbedderService.enableImages]): one reload, the first time the camera
  /// demo starts its memory; the knowledge base's embeddings wait meanwhile.
  /// The embedder's row then lists image among its media. A failed reload
  /// that also lost the text and audio embedder fails the row
  /// (the knowledge base says so); one that kept it leaves the row ready.
  /// Never narrowed back: the images stay until the app exits.
  Future<Result<EmbedderImagesLoad>> enableEmbedderImages() async {
    const id = ModelId.embeddingGemma;
    if (_closed) {
      return Result.error(asException(StateError('ModelRepository closed')));
    }
    final state = _states.value[id];
    if (state is! ModelReady) {
      return Result.error(
        asException(StateError('EmbeddingGemma is not ready ($state)')),
      );
    }
    final result = await _embedder.enableImages();
    if (_closed) return result;
    switch (result) {
      case Ok(value: EmbedderImagesLoad(alreadyLoaded: false)):
        // The row as loaded, with image now among its media.
        if (_states.value[id] case ModelReady(:final info)) {
          _set(
            id,
            ModelReady(
              LoadedModelInfo(
                modelId: info.modelId,
                backend: info.backend,
                loadTime: info.loadTime,
                warmUpTime: info.warmUpTime,
                detail: _embedderDetail(info.backend, _embedder.modalities),
                explicitCpu: info.explicitCpu,
                backendReported: info.backendReported,
                nativeLog: info.nativeLog,
              ),
            ),
          );
        }
      case Ok():
        break;
      case Error(error: EmbedderImagesException(restored: false) && final e):
        _set(id, ModelFailed(e.toString()));
      case Error():
        break;
    }
    return result;
  }

  /// The embedder row's detail: `cpu · 768-d · text+audio`.
  String _embedderDetail(String backend, Set<EmbeddingModality> media) =>
      '$backend · ${_embedder.dimension}-d · '
      '${[for (final m in EmbeddingModality.values)
        if (media.contains(m)) m.name].join('+')}';

  /// Loads a chat model again: closes the loaded one, then prepares the
  /// slot from the current plan (another file, backend or setting) and every
  /// model still not ready. The caller releases the open chat first (it
  /// holds a session on the model being closed). Refused after
  /// [refuseChatModelLoads].
  Future<Result<void>> reloadChatModel() async {
    if (_chatLoadsRefused case final reason?) {
      return Result.error(ChatModelBlockedException(reason));
    }
    await unloadChatModel();
    if (_closed) return _closedError();
    return prepareAll();
  }

  /// Closes the chat model and marks its slot pending (the self-test loads
  /// its own; flutter_edge_ai has one model per process). Waits for a setup
  /// run in flight first.
  Future<void> unloadChatModel() async {
    if (_closed) return;
    final running = _inFlight;
    if (running != null) await running;
    await _llm.unload();
    if (_chatLoadsRefused == null) _set(ModelId.chat, const ModelPending());
  }

  /// Why no chat model may load any more ([refuseChatModelLoads]); null
  /// while loads are allowed.
  String? _chatLoadsRefused;

  /// From now until the app restarts no chat model loads: [prepareAll]
  /// fails the chat slot and [reloadChatModel] refuses, and the slot shows
  /// [reason] without a Retry. For an engine the app can no longer close (a
  /// self-test past its time limit that did not end: its own engine may
  /// still be loaded, and flutter_edge_ai holds one per process).
  void refuseChatModelLoads(String reason) {
    _chatLoadsRefused = reason;
    _set(ModelId.chat, ModelFailed(reason, retryable: false));
  }

  Future<Result<void>> _prepareLlm() async {
    const id = ModelId.chat;
    if (_states.value[id] is ModelReady) return const Result.ok(null);
    if (_chatLoadsRefused case final reason?) {
      _set(id, ModelFailed(reason, retryable: false));
      return Result.error(ChatModelBlockedException(reason));
    }

    final String path;
    final ChatModelConfig config;
    final ChatModelFacts Function(LlmInfo info) facts;
    switch (_sources.chat(_chatModels?.plan ?? const NoChatModelPlan())) {
      case BlockedChatSource(:final reason):
        return _fail(id, ChatModelBlockedException(reason));
      case NoChatSource(:final note):
        final error = ChatModelNotChosenException(note);
        _set(id, ModelUnavailable(error.toString()));
        return Result.error(error);
      case CustomChatSource(:final plan):
        final model = plan.model;
        path = plan.path;
        config = plan.config;
        facts = (info) => ChatModelFacts(
          name: model.displayName,
          custom: true,
          source: model.sourceLine,
          sha256: model.file?.sha256,
          checksumMatched: model.file?.checksumMatched ?? false,
          requestedBackend: model.backend.name,
          requestedContext: model.maxTokens,
          contextTokens: info.contextTokens,
          visionBackend: info.visionBackend?.name,
          audioBackend: info.audioBackend?.name,
          tools: model.tools,
          modelType: model.modelType.name,
          rssBeforeBytes: info.rssBeforeBytes,
          rssAfterBytes: info.rssAfterBytes,
        );
      case DefineChatSource(path: final given, :final label, config: final c):
        path = await resolveLocalPath(given);
        // Gemma 4 E2B's settings, or for a build compiled for an NPU the
        // app's defaults for it (the self-test reads it the same way).
        switch (await namedChatModelFor(
          path,
          c,
          read: _readBundle,
          abi: _abi,
        )) {
          case Ok(:final value):
            config = value.config;
          case Error(:final error):
            // No backend runs it here: the same load is refused again.
            _set(id, ModelFailed('$label: $error', retryable: false));
            return Result.error(error);
        }
        facts = (info) => ChatModelFacts(
          name: config.name,
          custom: false,
          source: label,
          requestedBackend: config.llm.backend.name,
          requestedContext: config.llm.maxTokens,
          contextTokens: info.contextTokens,
          visionBackend: info.visionBackend?.name,
          audioBackend: info.audioBackend?.name,
          tools: config.tools,
          modelType: config.modelType.name,
          rssBeforeBytes: info.rssBeforeBytes,
          rssAfterBytes: info.rssAfterBytes,
        );
    }
    final outcome = await _pipeline.run(
      id,
      install: (onProgress) => _llm.install(
        path: path,
        modelType: config.modelType,
        onProgress: onProgress,
      ),
      load: () => _llm.load(config),
      // Its model has no owner after close() or a failed warm-up: unloaded
      // at once, not held (GBs) until the services close or a Retry.
      releaseOrphan: _llm.unload,
      warmUp: () => _llm.warmUp(
        _warmUpSampler,
        withImage: config.llm.supportImage,
        withAudio: config.llm.supportAudio && kLlmAudioWarmUp,
      ),
      releaseFailed: _llm.unload,
      // Through the warm-up: the sampler and the vision encoder report there
      // (the self-test reads the same window).
      logTap: _logTap,
      describe: (info, warmUpTime, nativeLog) => LoadedModelInfo(
        modelId: info.modelId,
        // The strict engine runs on the backend asked for or fails the load;
        // one that reports none shows the request, marked as such.
        backend: (info.backend ?? config.llm.backend).name,
        backendReported: info.backend != null,
        loadTime: info.loadTime,
        warmUpTime: warmUpTime,
        nativeLog: nativeLog,
        chat: facts(info),
      ),
    );
    // A file the plugin refused before any engine (incomplete, wrongBackend,
    // unsupportedNpu) is refused again by the same load: no Retry; the
    // Models screen offers what the plugin's message advises.
    if (outcome case LoadFailed(
      error: ChatModelLoadException(
            cause: LiteRtLmBundleException(:final problem),
          ) &&
          final error,
    )) {
      _set(id, ModelFailed('$error', retryable: false, bundleProblem: problem));
      return Result.error(error);
    }
    return _settle(id, outcome);
  }

  /// Install (every launch: it restores the active spec), load, warm up one
  /// speech model; [unload] releases it after a failed warm-up, or when
  /// close() ran meanwhile. The backend is the one requested; no API
  /// reports it (flutter_edge_ai_speech has no getter for it), so the row
  /// says "requested".
  Future<Result<void>> _prepareSpeech(
    ModelId id, {
    required Future<Result<String>> Function(
      void Function(int percent) onProgress,
    )
    install,
    required Future<Result<Duration>> Function() load,
    required Future<Result<Duration>> Function() warmUp,
    required Future<void> Function() unload,
    required String? Function() modelId,
    required String backend,
  }) async {
    final outcome = await _pipeline.run(
      id,
      install: install,
      load: load,
      releaseOrphan: unload,
      warmUp: warmUp,
      releaseFailed: unload,
      describe: (loadTime, warmUpTime, _) => LoadedModelInfo(
        modelId: modelId() ?? id.name,
        backend: backend,
        detail: '${backend.toUpperCase()} (requested)',
        loadTime: loadTime,
        warmUpTime: warmUpTime,
        backendReported: false,
      ),
    );
    return _settle(id, outcome);
  }

  /// A required model's outcome, published; its error is setup's (setup
  /// stops on it).
  Result<void> _settle(ModelId id, LoadOutcome outcome) {
    switch (outcome) {
      case LoadReady(:final info):
        _set(id, ModelReady(info));
        return const Result.ok(null);
      case LoadFailed(:final error):
        return _fail(id, error);
      case LoadStopped():
        return _closedError();
    }
  }

  /// An optional model's final state, which [_prepareOptional] publishes;
  /// [failed] says what a failure means for that model.
  static ModelState _finalState(
    LoadOutcome outcome, {
    required ModelState Function(LoadFailed failure) failed,
  }) => switch (outcome) {
    LoadReady(:final info) => ModelReady(info),
    final LoadFailed failure => failed(failure),
    // Never shown: close() ran, and nothing is published after it.
    LoadStopped() => const ModelFailed('ModelRepository closed'),
  };

  /// close() ran while a step was awaited: stop before the next one.
  static Result<void> _closedError() =>
      Result.error(asException(StateError('ModelRepository closed')));

  Result<void> _fail(ModelId id, Exception error) {
    _set(id, ModelFailed(error.toString()));
    return Result.error(error);
  }

  void _set(ModelId id, ModelState state) {
    if (_closed) return;
    _states.value = Map.unmodifiable({..._states.value, id: state});
  }

  /// Runs [body] as a load [close] waits for.
  Future<T> _tracked<T>(Future<T> Function() body) async {
    _loads++;
    try {
      return await body();
    } finally {
      if (--_loads == 0) {
        final idle = _loadsIdle;
        _loadsIdle = null;
        idle?.complete();
      }
    }
  }

  /// Closes every model in reverse load order, once the setup run or
  /// detector reload in flight has stopped at its next step (it checks
  /// after every install, load and warm-up): no service closes under a
  /// warm-up still running on its model. The wait is bounded ([_closeWait],
  /// logged when it runs out; a first GPU compile can take longer), and each
  /// service closes a model a load delivers after that itself. Safe to call
  /// more than once; a second call waits for the first.
  Future<void> close() => _closing ??= _close();

  Future<void> _close() async {
    _closed = true;
    if (_loads > 0) {
      final idle = _loadsIdle ??= Completer<void>();
      final watch = Stopwatch()..start();
      await idle.future.timeout(
        _closeWait,
        onTimeout: () => debugPrint(
          '[ModelRepository] close: a model load still runs after '
          '${_closeWait.inMilliseconds} ms; closing the models anyway',
        ),
      );
      debugPrint(
        '[ModelRepository] close: waited ${watch.elapsedMilliseconds} ms for '
        'the load in flight',
      );
    }
    await _embedder.close();
    await _detector.close();
    await _tts.close();
    await _stt.close();
    await _llm.close();
    _states.dispose();
    _preparing.dispose();
  }
}
