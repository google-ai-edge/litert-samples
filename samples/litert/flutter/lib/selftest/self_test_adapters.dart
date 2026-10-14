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
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_edge_ai/flutter_edge_ai.dart' hide ModelSpec;

import '../config/model_catalog.dart';
import '../config/voice_config.dart' show kMicAccessMessage;
import '../data/repositories/audio_repository.dart'
    show
        AudioRepository,
        MicAccessException,
        MicFormatException,
        PlaybackException;
import '../data/repositories/audio_repository_device.dart';
import '../data/services/audio/audio_device_service.dart';
import '../data/services/audio/audio_session_service.dart';
import '../data/services/audio/mic_service.dart';
import '../data/services/audio/pcm_player_service.dart';
import '../data/services/detector/detector_service.dart';
import '../data/services/detector/frame_message.dart';
import '../data/services/frames/frame_source.dart';
import '../data/services/knowledge/embedder_service.dart';
import '../data/services/llm/llm_service.dart';
import '../data/services/model_store/bundled_model_files.dart';
import '../data/services/model_store/local_files.dart';
import '../domain/audio/audio_device_checks.dart';
import '../domain/models/audio_devices.dart';
import '../domain/models/chat_model.dart';
import '../domain/models/chat_model_config.dart';
import '../domain/models/detection.dart';
import '../domain/models/frame_source_info.dart';
import '../domain/models/hardware_profile.dart' show HostPlatform;
import '../domain/models/knowledge_query.dart';
import '../domain/models/model_id.dart';
import '../domain/models/model_source_resolver.dart';
import '../domain/models/model_state.dart';
import '../utils/result.dart';
import 'cats_golden.dart';
import 'embedder_steps.dart' show kSelfTestAudioQuestionAsset;
import 'self_test_ports.dart';

/// The self-test's chat model: an explicit `--gemma` with the app's Gemma 4
/// E2B settings; otherwise exactly what the app would load, by the app's own
/// rule ([ModelSourceResolver]):
/// the chat model chosen on the Models screen ([plan]) — the user's own
/// `.litertlm` with its saved settings and its header's towers, as the app
/// read them (`GEMMA_MODEL_PATH` does not override it) — or, with none
/// chosen, `GEMMA_MODEL_PATH` ([define]) when set, else "No chat model yet".
/// `--gemma-backend` places the header's towers for that backend
/// ([SelfTestChatModel.withBackend]), as the app does.
Future<SelfTestChatModel> resolveSelfTestChatModel({
  required String? argument,
  required String define,
  required ChatModelPlan plan,
}) => _resolveChatModel(argument: argument, define: define, plan: plan);

Future<SelfTestChatModel> _resolveChatModel({
  required String? argument,
  required String define,
  required ChatModelPlan plan,
}) async {
  if (argument != null) {
    return SelfTestChatModel(
      file: checkReadable(argument, '--gemma'),
      config: kDefineChatModel,
      settingsSource: 'the app\'s Gemma 4 E2B settings',
    );
  }
  final sources = ModelSourceResolver(gemmaModelPath: define);
  switch (sources.chat(plan)) {
    case CustomChatSource(plan: final custom):
      final model = custom.model;
      return SelfTestChatModel(
        file: checkReadable(
          custom.path,
          model.source is LocalModelSource
              ? 'in place'
              : 'model store, custom/',
        ),
        config: custom.config,
        settingsSource: 'your own .litertlm, saved settings',
        sha256: model.file?.sha256,
        custom: model,
        header: custom.header,
      );
    case BlockedChatSource(:final reason):
      return SelfTestChatModel(
        file: ModelFileChoice(source: 'your own .litertlm', problem: reason),
        config: ChatModelConfig(
          name: 'your own .litertlm',
          modelType: kDefineChatModel.modelType,
          llm: kDefineChatModel.llm,
          tools: false,
        ),
        settingsSource: 'your own .litertlm',
      );
    case NoChatSource(:final note):
      return SelfTestChatModel(
        file: ModelFileChoice(
          source: 'no chat model',
          problem: [
            'No chat model yet: choose a .litertlm in the Chat model card, '
                'or pass --gemma.',
            ?note,
          ].join(' '),
        ),
        config: kDefineChatModel,
        settingsSource: 'no chat model',
      );
    case DefineChatSource(:final path, :final config):
      return SelfTestChatModel(
        // Checked up front, so a sandboxed path fails with a reason
        // instead of deep inside a load.
        file: checkReadable(
          await resolveLocalPath(path),
          kGemmaModelPathDefine,
        ),
        config: config,
        settingsSource: 'the app\'s Gemma 4 E2B settings',
      );
  }
}

/// [path] if it opens for reading; else why not.
ModelFileChoice checkReadable(String path, String source) {
  try {
    File(path).openSync().closeSync();
    return ModelFileChoice(path: path, source: source);
  } on FileSystemException catch (e) {
    final sandbox = Platform.isMacOS
        ? ' macOS builds are sandboxed: they read only their own container '
              '(the model store and the models folder), so put the file '
              'there or choose it in the app.'
        : '';
    return ModelFileChoice(
      source: source,
      problem:
          'not readable: $path (${e.osError?.message ?? e.message}).$sandbox',
    );
  }
}

/// The detector for the self-test: `--detector`, else the one built into
/// the app (what the app loads).
Future<ModelFileChoice> resolveSelfTestDetector({
  required String? argument,
}) async => argument != null
    ? checkReadable(argument, '--detector')
    : ModelFileChoice.bundledDetector;

/// The report's image label for the bundled cats.
const kSelfTestCatsLabel = 'bundled cats (COCO val2017 39769) · golden';

/// The image step 3 detects on: [path], or the bundled cats (the golden).
Future<Result<SelfTestImage>> loadSelfTestImage(String? path) async {
  try {
    final bytes = path == null
        ? (await rootBundle.load(kCatsAsset)).buffer.asUint8List()
        : await File(path).readAsBytes();
    final codec = await ui.instantiateImageCodec(bytes);
    try {
      final image = (await codec.getNextFrame()).image;
      try {
        final data = await image.toByteData();
        if (data == null) {
          return Result.error(asException(StateError('no RGBA from decode')));
        }
        return Result.ok(
          SelfTestImage(
            frame: _RgbaFrame(
              image.width,
              image.height,
              data.buffer.asUint8List(),
            ),
            label: path ?? kCatsAsset,
            golden: path == null,
          ),
        );
      } finally {
        image.dispose();
      }
    } finally {
      codec.dispose();
    }
  } catch (e) {
    return Result.error(asException(e));
  }
}

final class _RgbaFrame implements FrameView {
  _RgbaFrame(this.width, this.height, Uint8List rgba)
    : planes = [
        FramePlane(bytes: rgba, bytesPerRow: width * 4, bytesPerPixel: 4),
      ];

  @override
  final int width;
  @override
  final int height;
  @override
  final List<FramePlane> planes;
  @override
  FramePixelFormat get format => FramePixelFormat.rgba8888;
  @override
  int get rotationDeg => 0;
}

/// The app's `DetectorService`: worker isolate and the fail-fast load
/// checks (strict accelerator, full acceleration on the GPU, tensor sizes,
/// output verified against a CPU reference run). Its own instance: the
/// app's detector stays as it is.
final class AppSelfTestDetector implements SelfTestDetector {
  final DetectorService _service = DetectorService();

  @override
  Future<Result<DetectorInfo>> load(
    ModelFileChoice model,
    DetectorBackend backend,
  ) async {
    final DetectorModelSource source;
    if (model.bundled) {
      try {
        source = DetectorBytes(await loadBundledDetector());
      } catch (e) {
        return Result.error(asException(e));
      }
    } else {
      source = DetectorFile(model.path!);
    }
    return _service.load(source: source, backend: backend);
  }

  @override
  Future<Result<DetectionFrame>> detect(FrameMessage frame) =>
      _service.detect(frame);

  @override
  Future<void> close() => _service.close();
}

/// The app's `LlmService` with the chat model's configuration: its backend
/// check rejects any other backend than requested, and its NPU gate refuses
/// the NPU where flutter_edge_ai has no NPU stack. flutter_edge_ai has one
/// model per process: the app's chat model must be unloaded first (the in-app
/// self-test does that).
final class AppSelfTestGemma implements SelfTestGemma {
  /// [llm]: the app's own service by default (a fake in tests).
  AppSelfTestGemma({LlmService? llm}) : _llm = llm ?? LlmService();

  final LlmService _llm;

  @override
  Future<Result<GemmaLoad>> load(String path, ChatModelConfig config) async {
    if (await _llm.install(
          path: path,
          modelType: config.modelType,
          onProgress: (_) {},
        )
        case Error(:final error)) {
      return Result.error(error);
    }
    final LlmInfo info;
    switch (await _llm.load(config)) {
      case Ok(:final value):
        info = value;
      case Error(:final error):
        return Result.error(error);
    }
    final Duration warmUp;
    // As the app warms it up (`ModelRepository`): an audio tower that fails
    // only on its first use fails here too.
    switch (await _llm.warmUp(
      kSampler,
      withImage: config.llm.supportImage,
      withAudio: config.llm.supportAudio && kLlmAudioWarmUp,
    )) {
      case Ok(:final value):
        warmUp = value;
      case Error(:final error):
        return Result.error(error);
    }
    return Result.ok(
      GemmaLoad(
        modelId: info.modelId,
        backend: info.backend,
        loadTime: info.loadTime,
        warmUpTime: warmUp,
        contextTokens: info.contextTokens,
        config: gemmaLoadConfigLine(config, engineContext: info.contextTokens),
      ),
    );
  }

  @override
  Future<Result<GemmaGeneration>> generate(String prompt, int maxTokens) async {
    InferenceModelSession? session;
    try {
      session = await _llm.model.createSession(
        temperature: kSampler.temperature,
        topK: kSampler.topK,
        maxOutputTokens: maxTokens,
      );
      await session.addQueryChunk(Message(text: prompt, isUser: true));
      final watch = Stopwatch()..start();
      Duration? first;
      var chunks = 0;
      final text = StringBuffer();
      await for (final chunk in session.getResponseAsync().timeout(
        const Duration(minutes: 5),
      )) {
        first ??= watch.elapsed;
        chunks++;
        text.write(chunk);
      }
      final total = watch.elapsed;
      SessionMetrics? metrics;
      try {
        metrics = session.getSessionMetrics();
      } catch (e) {
        debugPrint('[SelfTest] session metrics unavailable: $e');
      }
      return Result.ok(
        GemmaGeneration(
          chunks: chunks,
          engineTokens: metrics == null || metrics.outputTokens == 0
              ? null
              : metrics.outputTokens,
          engineTokensPerSecond: metrics?.tokensPerSecond,
          firstChunk: first ?? total,
          total: total,
          text: text.toString(),
        ),
      );
    } catch (e) {
      return Result.error(asException(e));
    } finally {
      await session?.close();
    }
  }

  @override
  Future<void> close() => _llm.close();
}

/// The bundled spoken question of step 7b ([kSelfTestAudioQuestionAsset]),
/// whole (a few tens of KB).
/// The photo step 7c embeds: the bundled cats JPEG as it is ([kCatsAsset],
/// 640×480, upright).
Future<Result<Uint8List>> loadSelfTestPhoto() async {
  try {
    final data = await rootBundle.load(kCatsAsset);
    return Result.ok(
      data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes),
    );
  } catch (e) {
    return Result.error(
      asException(StateError('$kCatsAsset is not in this build ($e)')),
    );
  }
}

Future<Result<Uint8List>> loadSelfTestAudioQuestion() async {
  try {
    final data = await rootBundle.load(kSelfTestAudioQuestionAsset);
    return Result.ok(
      data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes),
    );
  } catch (e) {
    return Result.error(
      asException(
        StateError(
          '$kSelfTestAudioQuestionAsset is not in this build ($e): '
          'tool/make_question_audio.sh makes it',
        ),
      ),
    );
  }
}

/// `audio+text`.
String _mediaOf(Set<EmbeddingModality> modalities) =>
    (modalities.map((m) => m.name).toList()..sort()).join('+');

/// The headless self-test's embedder: the app's [EmbedderService]
/// ([appEmbedderService]) over the built-in EmbeddingGemma 2 file,
/// installed, loaded (audio declared, as setup loads it) and warmed up for
/// the run with the app's text question and spoken clip (so step 7a runs a
/// warm audio encoder, as in the app), the image encoder added for step 7c
/// as the camera demo adds it, closed after the run.
final class AppSelfTestEmbedder implements SelfTestEmbedder {
  AppSelfTestEmbedder({BundledModelFiles? bundled, EmbedderService? service})
    : _bundled = bundled ?? BundledModelFiles(),
      _service = service ?? appEmbedderService();

  final BundledModelFiles _bundled;
  final EmbedderService _service;

  @override
  Future<Result<SelfTestEmbedderLoad>> prepare() async {
    final String path;
    switch (await _bundled.pathOf(kBundledEmbedderModel)) {
      case Ok(:final value):
        path = value;
      case Error(:final error):
        return Result.error(error);
    }
    final installed = await _service.install(
      EmbedderFromFiles(modelPath: path),
      onProgress: (_) {},
    );
    if (installed case Error(:final error)) return Result.error(error);
    final EmbedderInfo info;
    switch (await _service.load()) {
      case Ok(:final value):
        info = value;
      case Error(:final error):
        return Result.error(error);
    }
    switch (await _service.warmUp()) {
      case Error(:final error):
        return Result.error(error);
      case Ok(value: final warmUp):
        return Result.ok(
          SelfTestEmbedderLoad(
            modelId: info.modelId,
            backend: info.backend.name,
            media: _mediaOf(info.modalities),
            loadTime: info.loadTime,
            warmUpTime: warmUp,
          ),
        );
    }
  }

  @override
  Future<Result<List<double>>> embed(KnowledgeQuery query) =>
      _service.embedQuery(query);

  @override
  Future<Result<SelfTestImagesLoad>> enableImages() async =>
      _imagesLoad(await _service.enableImages());

  @override
  Future<Result<List<double>>> embedImage(Uint8List encoded) =>
      _service.embedImage(encoded);

  @override
  Future<Result<List<double>>> embedMemoryText(String text) =>
      _service.embedText(text);

  @override
  Future<void> close() => _service.close();
}

/// [EmbedderService.enableImages]'s report as the self-test's.
Result<SelfTestImagesLoad> _imagesLoad(Result<EmbedderImagesLoad> result) =>
    switch (result) {
      Ok(:final value) => Result.ok(
        SelfTestImagesLoad(
          loadTime: value.loadTime,
          rssBeforeBytes: value.rssBeforeBytes,
          rssAfterBytes: value.rssAfterBytes,
          alreadyLoaded: value.alreadyLoaded,
        ),
      ),
      Error(:final error) => Result.error(error),
    };

/// The in-app self-test's embedder: the app's own [EmbedderService], as
/// setup loaded it (one embedder per process: loading another would hand
/// back this one, and closing it would take the knowledge base down). Used
/// as it is and never closed; a run while it is not ready fails 7a with its
/// state.
final class BorrowedSelfTestEmbedder implements SelfTestEmbedder {
  /// [enableImages]: the app's own way to add the image encoder
  /// (`ModelRepository.enableEmbedderImages`, which also updates the
  /// embedder's row), as the camera demo does.
  BorrowedSelfTestEmbedder(
    this._service,
    this._models, {
    required this._enableImages,
  });

  final EmbedderService _service;
  final ValueListenable<Map<ModelId, ModelState>> _models;
  final Future<Result<EmbedderImagesLoad>> Function() _enableImages;

  @override
  Future<Result<SelfTestEmbedderLoad>> prepare() async {
    final state = _models.value[ModelId.embeddingGemma];
    if (state is! ModelReady || !_service.isLoaded) {
      final why = switch (state) {
        ModelUnavailable(:final reason) => 'unavailable: $reason',
        ModelFailed(:final message) => 'failed: $message',
        ModelReady() => 'closed',
        null || ModelPending() => 'not loaded yet',
        ModelInstalling() || ModelLoading() || ModelWarmingUp() => 'loading',
      };
      return Result.error(
        asException(StateError('The app\'s EmbeddingGemma is $why')),
      );
    }
    return Result.ok(
      SelfTestEmbedderLoad(
        modelId: _service.modelId ?? state.info.modelId,
        backend: state.info.backend,
        media: _mediaOf(_service.modalities),
        borrowed: true,
      ),
    );
  }

  @override
  Future<Result<List<double>>> embed(KnowledgeQuery query) =>
      _service.embedQuery(query);

  @override
  Future<Result<SelfTestImagesLoad>> enableImages() async =>
      _imagesLoad(await _enableImages());

  @override
  Future<Result<List<double>>> embedImage(Uint8List encoded) =>
      _service.embedImage(encoded);

  @override
  Future<Result<List<double>>> embedMemoryText(String text) =>
      _service.embedText(text);

  @override
  Future<void> close() async {}
}

/// The app's audio path: a [DeviceAudioRepository] for the output (session,
/// soloud, the output check, the reply player) and `record` with the app's
/// input check for the microphone, each on its own so a broken output does
/// not hide a working microphone. Linux records the sink's monitor with
/// `parecord`.
///
/// [shared]: the running app's own audio repository (the in-app self-test).
/// soloud is one engine per process, and closing a second repository would
/// shut it down under the app, so the shared one is used and never closed.
final class AppSelfTestAudio implements SelfTestAudio {
  AppSelfTestAudio(this._platform, {AudioRepository? shared})
    : _devices = audioDeviceServiceFor(_platform),
      _ownsRepo = shared == null {
    _repo =
        shared ??
        DeviceAudioRepository(
          session: audioSessionServiceFor(_platform),
          mic: _mic,
          player: SoloudPcmPlayerService(),
          deviceService: _devices,
          platform: _platform,
        );
  }

  final HostPlatform _platform;
  final AudioDeviceService _devices;
  final RecordMicService _mic = RecordMicService();
  final bool _ownsRepo;
  late final AudioRepository _repo;
  _ParecordMonitor? _monitor;

  @override
  Future<Result<DeviceReady>> prepareOutput() async {
    if (await _repo.prepare() case Error(:final error)) {
      return Result.error(error);
    }
    return switch (_repo.devices.value.output) {
      final DeviceReady ready => Result.ok(ready),
      final other => Result.error(
        PlaybackException(
          'the output was not checked (${deviceCheckText(other)})',
        ),
      ),
    };
  }

  @override
  String? get monitorUnavailable => switch (_platform) {
    HostPlatform.linux => null,
    final other =>
      '${other == HostPlatform.macos ? 'macOS' : other.name} has no monitor '
          'source to record the output from (Linux records it with parecord '
          '-d @DEFAULT_MONITOR@); the step checks the device and the playback '
          'only',
  };

  @override
  Future<Result<SelfTestMonitor>> startMonitor() async {
    switch (await _ParecordMonitor.start()) {
      case Error(:final error):
        return Result.error(error);
      case Ok(:final value):
        _monitor = value;
        return Result.ok(value);
    }
  }

  @override
  Future<Result<void>> play(Uint8List pcm, int sampleRate) async {
    switch (_repo.beginPlayback(sampleRate)) {
      case Error(:final error):
        return Result.error(error);
      case Ok(value: final playback):
        try {
          playback.enqueue(pcm);
        } on PlaybackException catch (e) {
          return Result.error(e);
        }
        playback.end();
        await playback.drained;
        return const Result.ok(null);
    }
  }

  @override
  Future<Result<MicCapture>> recordMic(Duration duration) async {
    final input = await _devices.checkInput();
    if (input case DeviceUnavailable(:final message)) {
      return Result.error(MicAccessException(message));
    }
    try {
      if (!await _mic.hasPermission()) {
        return Result.error(MicAccessException(kMicAccessMessage));
      }
    } catch (e) {
      return Result.error(MicAccessException('$kMicAccessMessage ($e)'));
    }
    String? formatChanged;
    final Stream<Uint8List> stream;
    try {
      stream = await _mic.startPcm16(
        sampleRate: 16000,
        onFormatChanged: (actual) => formatChanged = actual,
      );
    } catch (e) {
      return Result.error(
        MicAccessException(micStartErrorMessage(e, _platform)),
      );
    }
    // [duration] of audio, not of wall time: the first chunk comes ~0.1 s
    // after the start. A device that delivers less within 2 s more is
    // reported with what it delivered.
    final want = duration.inMicroseconds * 16000 ~/ 1000000 * 2;
    final bytes = BytesBuilder(copy: false);
    final done = Completer<void>();
    final enough = Completer<void>();
    var stopping = false;
    var endedEarly = false;
    Object? streamError;
    final sub = stream.listen(
      (chunk) {
        bytes.add(chunk);
        if (bytes.length >= want && !enough.isCompleted) enough.complete();
      },
      onError: (Object e) => streamError ??= e,
      onDone: () {
        endedEarly = !stopping;
        if (!done.isCompleted) done.complete();
      },
    );
    try {
      await Future.any([
        enough.future,
        done.future,
        Future<void>.delayed(duration + const Duration(seconds: 2)),
      ]);
      stopping = true;
      await _mic.stop();
      await done.future.timeout(const Duration(seconds: 1), onTimeout: () {});
    } finally {
      await sub.cancel();
    }
    if (formatChanged case final actual?) {
      return Result.error(MicFormatException(actual));
    }
    if (streamError case final e?) return Result.error(asException(e));
    // Nothing at all (record 7.1.1 keeps the stream open when parecord
    // exits, so an empty capture is the only sign): ask why.
    if (bytes.isEmpty) {
      final again = await _devices.checkInput();
      return Result.error(
        MicAccessException(switch (again) {
          DeviceUnavailable(:final message) => message,
          _ =>
            'No audio arrived from the microphone '
                '${endedEarly ? '(its stream ended)' : 'in ${(duration + const Duration(seconds: 2)).inSeconds} s'} '
                '(${deviceCheckText(again)})',
        }),
      );
    }
    final pcm = bytes.takeBytes();
    return Result.ok(
      MicCapture(
        device: deviceCheckText(input),
        pcm: pcm.length > want ? Uint8List.sublistView(pcm, 0, want) : pcm,
      ),
    );
  }

  @override
  Future<void> close() async {
    await _monitor?.kill();
    if (_ownsRepo) {
      await _repo.close();
    } else {
      await _mic.dispose();
    }
  }
}

/// `parecord -d @DEFAULT_MONITOR@`: what the default sink plays, as 16 kHz
/// mono PCM16 on stdout.
final class _ParecordMonitor implements SelfTestMonitor {
  _ParecordMonitor._(this._process, this._bytes, this._drained);

  final Process _process;
  final BytesBuilder _bytes;
  final Future<void> _drained;

  /// How long the stream gets to connect; a parecord that exits within it
  /// (no server, no such source) is an error.
  static const _connect = Duration(milliseconds: 300);

  /// The tone's tail still in the server's buffers when playback reports
  /// its end.
  static const _tail = Duration(milliseconds: 400);

  static Future<Result<_ParecordMonitor>> start() async {
    final Process process;
    try {
      process = await Process.start('parecord', const [
        '--device=@DEFAULT_MONITOR@',
        '--raw',
        '--format=s16le',
        '--rate=16000',
        '--channels=1',
        '--latency-msec=50',
      ]);
    } on ProcessException {
      return const Result.error(MicAccessException(kParecordMissing));
    }
    final bytes = BytesBuilder(copy: false);
    final stderrText = StringBuffer();
    final drained = Future.wait([
      process.stdout.listen(bytes.add).asFuture<void>(),
      process.stderr
          .transform(const Utf8Decoder(allowMalformed: true))
          .listen(stderrText.write)
          .asFuture<void>(),
    ]);
    final exited = await process.exitCode
        .then<int?>((code) => code)
        .timeout(_connect, onTimeout: () => null);
    if (exited != null) {
      await drained;
      final why = stderrText.toString().trim();
      return Result.error(
        asException(
          StateError(
            'parecord -d @DEFAULT_MONITOR@ exited with $exited'
            '${why.isEmpty ? '' : ': $why'}',
          ),
        ),
      );
    }
    return Result.ok(_ParecordMonitor._(process, bytes, drained));
  }

  @override
  Future<Result<Uint8List>> stop() async {
    await Future<void>.delayed(_tail);
    await kill();
    return Result.ok(_bytes.takeBytes());
  }

  /// Ends parecord (SIGTERM, then SIGKILL) and waits for its output.
  Future<void> kill() async {
    _process.kill();
    await _process.exitCode.timeout(
      const Duration(seconds: 2),
      onTimeout: () {
        _process.kill(ProcessSignal.sigkill);
        return -1;
      },
    );
    await _drained.timeout(const Duration(seconds: 2), onTimeout: () => []);
  }
}
