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
import '../data/services/llm/llm_service.dart';
import '../data/services/model_store/local_files.dart';
import '../domain/audio/audio_device_checks.dart';
import '../domain/models/audio_devices.dart';
import '../domain/models/chat_model.dart';
import '../domain/models/chat_model_config.dart';
import '../domain/models/detection.dart';
import '../domain/models/frame_source_info.dart';
import '../domain/models/hardware_profile.dart' show HostPlatform;
import '../domain/models/model_source_resolver.dart';
import '../utils/result.dart';
import 'cats_golden.dart';
import 'self_test_ports.dart';

/// The self-test's chat model: an explicit `--gemma` with the app's Gemma 4
/// E2B settings; otherwise exactly what the app would load, by the app's own
/// rule ([ModelSourceResolver]):
/// the chat model chosen on the Models screen ([plan]) — the user's own
/// `.litertlm` with its saved settings (`GEMMA_MODEL_PATH` does not override
/// it) — or, with none chosen, `GEMMA_MODEL_PATH` ([define]) when set, else
/// "No chat model yet".
Future<SelfTestChatModel> resolveSelfTestChatModel({
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
  final LlmService _llm = LlmService();

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
    switch (await _llm.warmUp(kSampler, withImage: config.llm.supportImage)) {
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
