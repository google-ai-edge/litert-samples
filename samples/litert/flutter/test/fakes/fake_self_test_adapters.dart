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
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_edge_ai/flutter_edge_ai.dart' show PreferredBackend;
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/audio_repository.dart'
    show MicAccessException, PlaybackException;
import 'package:litert_edge_demos/data/services/detector/frame_message.dart';
import 'package:litert_edge_demos/data/services/frames/frame_source.dart';
import 'package:litert_edge_demos/data/services/hardware/hardware_info_service.dart';
import 'package:litert_edge_demos/data/services/llm/llm_service.dart'
    show BackendMismatchException;
import 'package:litert_edge_demos/domain/models/audio_devices.dart';
import 'package:litert_edge_demos/domain/models/chat_model_config.dart';
import 'package:litert_edge_demos/domain/models/detection.dart';
import 'package:litert_edge_demos/domain/models/frame_source_info.dart';
import 'package:litert_edge_demos/domain/models/hardware_profile.dart';
import 'package:litert_edge_demos/selftest/cats_golden.dart';
import 'package:litert_edge_demos/selftest/self_test_options.dart';
import 'package:litert_edge_demos/selftest/self_test_runner.dart';
import 'package:litert_edge_demos/utils/result.dart' show Result;

import 'fake_hardware.dart';

/// The build every fake self-test reports.
const kFakeSelfTestBuild = BuildInfo(
  appVersion: '0.1.0+1',
  buildMode: 'release',
  flutterVersion: '3.47.3',
  dartVersion: '3.13.3',
  packages: {'flutter_edge_ai': '2.1.0'},
);

/// A model file in the model store.
const kFakeSelfTestFile = ModelFileChoice(
  path: '/models/file',
  source: 'model store',
);

/// A Linux box with a real GPU that Vulkan names (a T4 VM).
const kFakeLinuxT4Profile = HardwareProfile(
  platform: HostPlatform.linux,
  os: 'Ubuntu 24.04.3 LTS',
  cpu: CpuInfo(model: 'Intel Xeon CPU @ 2.20GHz', cores: 4),
  vulkan: [
    VulkanDevice(name: 'Tesla T4', type: 'DISCRETE_GPU'),
    VulkanDevice(name: 'llvmpipe (LLVM 19.1.1, 256 bits)', type: 'CPU'),
  ],
);

/// Only llvmpipe: the "GPU" is the CPU.
const kFakeLlvmpipeProfile = HardwareProfile(
  platform: HostPlatform.linux,
  os: 'Ubuntu 24.04.3 LTS',
  cpu: CpuInfo(model: 'AMD EPYC 7B12', cores: 2),
  vulkan: [VulkanDevice(name: 'llvmpipe (LLVM 19.1.1, 256 bits)', type: 'CPU')],
);

/// Linux with nothing that names a GPU (no vulkaninfo, no GPU found).
const kFakeUnnamedLinuxProfile = HardwareProfile(
  platform: HostPlatform.linux,
  os: 'Ubuntu 22.04.5 LTS',
  cpu: CpuInfo(model: '4× Neoverse-N1', cores: 4),
);

/// The golden detections as a frame (what a healthy detector returns).
DetectionFrame fakeGoldenCatsFrame(
  int frameId, {
  DetectorBackend backend = DetectorBackend.gpu,
  int runMicros = 5000,
}) => DetectionFrame(
  frameId: frameId,
  width: 640,
  height: 480,
  boxes: Float32List.fromList([
    for (final g in kCatsGolden) ...[...g.box, g.score, g.cls.toDouble()],
  ]),
  preMicros: 100,
  runMicros: runMicros,
  postMicros: 100,
  backend: backend,
);

/// One dog where the golden has cats; its score moves with [frameId], so no
/// two frames are bit-identical.
DetectionFrame fakeWrongFrame(int frameId) => DetectionFrame(
  frameId: frameId,
  width: 640,
  height: 480,
  boxes: Float32List.fromList([10, 20, 300, 400, 0.5 + frameId / 10000, 16]),
  preMicros: 100,
  runMicros: 4000 + frameId * 10,
  postMicros: 100,
  backend: DetectorBackend.gpu,
);

DetectorInfo fakeDetectorInfo(DetectorBackend backend) => DetectorInfo(
  backend: backend,
  fullyAccelerated: backend == DetectorBackend.gpu,
  verifyAbsolute: 1.77e-3,
  verifyRelative: 2e-6,
  verifyReference: VerifyReference.interpreter,
  createTime: const Duration(milliseconds: 116),
  verifyTime: const Duration(milliseconds: 124),
  firstRunTime: const Duration(milliseconds: 5),
);

/// The detector as the self-test drives it.
final class FakeSelfTestDetector implements SelfTestDetector {
  FakeSelfTestDetector({
    this.failOn = const {},
    this.reportAs,
    this.tap,
    this.logLines = const [],
    this.failDetects = const {},
    DetectionFrame Function(int call)? frame,
    this.closeLog,
  }) : _frame = frame ?? fakeGoldenCatsFrame;

  /// Backends whose load fails.
  final Set<DetectorBackend> failOn;

  /// The backend a successful load reports; null: the one requested.
  final DetectorBackend? reportAs;
  final FakeNativeLogTap? tap;

  /// Written to [tap] during a load, like LiteRT's stderr.
  final List<String> logLines;

  /// 1-based detect calls that fail.
  final Set<int> failDetects;
  final DetectionFrame Function(int call) _frame;

  /// Gets `detector` when closed (shared with the other fakes: the order).
  final List<String>? closeLog;

  final List<DetectorBackend> loads = [];
  final List<ModelFileChoice> models = [];
  int detects = 0;
  bool closed = false;
  bool _loaded = false;

  /// When set, [load] waits for it (a native call that hangs).
  Completer<void>? loadGate;

  @override
  Future<Result<DetectorInfo>> load(
    ModelFileChoice model,
    DetectorBackend backend,
  ) async {
    loads.add(backend);
    models.add(model);
    await loadGate?.future;
    tap?.lines.addAll(logLines);
    if (failOn.contains(backend)) {
      return Result.error(Exception('${backend.name} rejected YOLO26n'));
    }
    _loaded = true;
    return Result.ok(fakeDetectorInfo(reportAs ?? backend));
  }

  @override
  Future<Result<DetectionFrame>> detect(FrameMessage frame) async {
    detects++;
    if (!_loaded) return Result.error(Exception('not loaded'));
    if (failDetects.contains(detects)) {
      return Result.error(Exception('the worker isolate died'));
    }
    return Result.ok(_frame(detects));
  }

  @override
  Future<void> close() async {
    closed = true;
    closeLog?.add('detector');
  }
}

/// The chat model as the self-test drives it.
final class FakeSelfTestGemma implements SelfTestGemma {
  FakeSelfTestGemma({
    this.active,
    this.loadError,
    this.tap,
    this.logLines = const [],
    this.generateError,
    this.chunks,
    this.engineMetrics = true,
    this.text = 'Elias lived on the edge of the world.',
    this.closeLog,
    this.engineContext,
  });

  /// The backend the engine comes up on; null: as requested.
  final PreferredBackend? active;

  /// The context the engine is built with; null: the one requested.
  final int? engineContext;
  final Exception? loadError;
  final FakeNativeLogTap? tap;

  /// Written to [tap] during a load.
  final List<String> logLines;
  final Exception? generateError;

  /// Streamed chunks; null: as many as asked for.
  final int? chunks;

  /// The engine reports its own token count and rate.
  final bool engineMetrics;
  final String text;

  /// Gets `gemma` when closed.
  final List<String>? closeLog;

  final List<PreferredBackend> loads = [];
  final List<ChatModelConfig> configs = [];
  int generations = 0;
  bool closed = false;

  /// When set, [load] waits for it (a native call that hangs).
  Completer<void>? loadGate;

  @override
  Future<Result<GemmaLoad>> load(String path, ChatModelConfig config) async {
    final backend = config.llm.backend;
    loads.add(backend);
    configs.add(config);
    await loadGate?.future;
    tap?.lines.addAll(logLines);
    if (loadError case final e?) return Result.error(e);
    final actual = active ?? backend;
    if (actual != backend) {
      return Result.error(
        BackendMismatchException(requested: backend, active: actual),
      );
    }
    return Result.ok(
      GemmaLoad(
        modelId: 'gemma-4-E2B-it',
        backend: actual,
        loadTime: const Duration(milliseconds: 3600),
        warmUpTime: const Duration(milliseconds: 2200),
        contextTokens: engineContext ?? config.llm.maxTokens,
        config: gemmaLoadConfigLine(
          config,
          engineContext: engineContext ?? config.llm.maxTokens,
        ),
      ),
    );
  }

  @override
  Future<Result<GemmaGeneration>> generate(String prompt, int maxTokens) async {
    generations++;
    if (generateError case final e?) return Result.error(e);
    final n = chunks ?? maxTokens;
    return Result.ok(
      GemmaGeneration(
        chunks: n,
        engineTokens: engineMetrics ? n : null,
        engineTokensPerSecond: engineMetrics ? 73.3 : null,
        firstChunk: const Duration(milliseconds: 180),
        total: const Duration(milliseconds: 1030),
        text: text,
      ),
    );
  }

  @override
  Future<void> close() async {
    closed = true;
    closeLog?.add('gemma');
  }
}

/// A 2×2 RGBA frame (the fake detector ignores the pixels).
final class FakeTinyFrame implements FrameView {
  @override
  int get width => 2;
  @override
  int get height => 2;
  @override
  FramePixelFormat get format => FramePixelFormat.rgba8888;
  @override
  int get rotationDeg => 0;
  @override
  List<FramePlane> get planes => [
    FramePlane(bytes: Uint8List(16), bytesPerRow: 8, bytesPerPixel: 4),
  ];
}

/// 16 kHz PCM16 at [dbfs] RMS (a square wave: RMS = peak).
Uint8List fakePcmAt(
  double dbfs, {
  Duration length = const Duration(seconds: 1),
}) {
  final samples = length.inMicroseconds * 16000 ~/ 1000000;
  final amplitude = (32767 * math.pow(10, dbfs / 20)).round();
  final data = ByteData(samples * 2);
  for (var i = 0; i < samples; i++) {
    data.setInt16(i * 2, i.isEven ? amplitude : -amplitude, Endian.little);
  }
  return data.buffer.asUint8List();
}

final class _FakeMonitor implements SelfTestMonitor {
  _FakeMonitor(this._pcm, this._error);

  final Uint8List _pcm;
  final Exception? _error;

  @override
  Future<Result<Uint8List>> stop() async =>
      _error == null ? Result.ok(_pcm) : Result.error(_error);
}

/// The audio path: [monitorPcm] is what the sink's monitor records, [micPcm]
/// what the microphone delivers; every `…Error` fails that call, every
/// `hang…` never answers it.
final class FakeSelfTestAudio implements SelfTestAudio {
  FakeSelfTestAudio({
    this.monitorUnavailable,
    Uint8List? monitorPcm,
    Uint8List? micPcm,
    this.outputError,
    this.outputCaution = false,
    this.monitorStartError,
    this.monitorStopError,
    this.playError,
    this.micError,
    this.hangOutput = false,
    this.hangMic = false,
    this.hangClose = false,
    this.closeLog,
  }) : monitorPcm = monitorPcm ?? fakePcmAt(-9),
       micPcm = micPcm ?? fakePcmAt(-35);

  @override
  final String? monitorUnavailable;
  final Uint8List monitorPcm;
  final Uint8List micPcm;
  final Exception? outputError;
  final bool outputCaution;
  final Exception? monitorStartError;
  final Exception? monitorStopError;
  final Exception? playError;
  final Exception? micError;
  final bool hangOutput;
  final bool hangMic;
  final bool hangClose;

  /// Gets `audio` when closed.
  final List<String>? closeLog;

  final List<int> played = [];
  bool monitorStarted = false;
  bool micRecorded = false;
  bool closed = false;

  @override
  Future<Result<DeviceReady>> prepareOutput() async {
    if (hangOutput) return Completer<Result<DeviceReady>>().future;
    return switch (outputError) {
      final e? => Result.error(e),
      null => Result.ok(
        outputCaution
            ? const DeviceReady(
                'Dummy Output',
                detail: 'pulseaudio dummy sink (auto_null): nothing is audible',
                caution: true,
              )
            : const DeviceReady(
                'Built-in Audio Analog Stereo',
                detail: 'PulseAudio (on PipeWire 1.0.5)',
              ),
      ),
    };
  }

  @override
  Future<Result<SelfTestMonitor>> startMonitor() async {
    monitorStarted = true;
    if (monitorStartError case final e?) return Result.error(e);
    return Result.ok(_FakeMonitor(monitorPcm, monitorStopError));
  }

  @override
  Future<Result<void>> play(Uint8List pcm, int sampleRate) async {
    played.add(pcm.length);
    if (playError case final e?) return Result.error(e);
    return const Result.ok(null);
  }

  @override
  Future<Result<MicCapture>> recordMic(Duration duration) async {
    micRecorded = true;
    if (hangMic) return Completer<Result<MicCapture>>().future;
    if (micError case final e?) return Result.error(e);
    return Result.ok(
      MicCapture(
        device: 'Built-in Audio Analog Stereo · PulseAudio (on PipeWire 1.0.5)',
        pcm: micPcm,
      ),
    );
  }

  @override
  Future<void> close() async {
    closed = true;
    closeLog?.add('audio');
    if (hangClose) await Completer<void>().future;
  }
}

/// Fails every probe with [error] (a bug in the probe).
final class ThrowingHardwareInfoService implements HardwareInfoService {
  ThrowingHardwareInfoService(this.error);

  final Error error;

  @override
  Future<HardwareProfile> probe() async => throw error;
}

/// The chat model the app would load for `GEMMA_MODEL_PATH`.
const kFakeDefineChatModel = SelfTestChatModel(
  file: kFakeSelfTestFile,
  config: kDefineChatModel,
  settingsSource: 'the app\'s Gemma 4 E2B settings',
);

/// A runner over fakes; [progress] gets its progress lines.
SelfTestRunner fakeSelfTestRunner({
  SelfTestOptions options = const SelfTestOptions(),
  required SelfTestDetector detector,
  required SelfTestGemma gemma,
  FakeNativeLogTap? tap,
  HardwareInfoService? hardware,
  HardwareProfile profile = kFakeMacProfile,
  SelfTestChatModel chatModel = kFakeDefineChatModel,
  ModelFileChoice detectorFile = kFakeSelfTestFile,
  Future<Result<SelfTestImage>> Function()? loadImage,
  String imageLabel = 'bundled cats',
  SelfTestAudio? audio,
  Duration audioTimeout = kSelfTestAudioTimeout,
  void Function(String line)? progress,
  List<String> unhandled = const [],
}) => SelfTestRunner(
  options: options,
  build: kFakeSelfTestBuild,
  hardware: hardware ?? FakeHardwareInfoService(profile),
  detector: detector,
  gemma: gemma,
  memory: FakeMemoryProbe(),
  logTap: tap ?? FakeNativeLogTap(),
  chatModel: chatModel,
  detectorFile: detectorFile,
  loadImage:
      loadImage ??
      () async => Result.ok(
        SelfTestImage(frame: FakeTinyFrame(), label: 'cats', golden: true),
      ),
  imageLabel: imageLabel,
  audio: audio,
  audioTimeout: audioTimeout,
  progress: progress,
  now: () => DateTime.utc(2026, 10, 6),
  unhandledErrors: unhandled,
);

/// The errors the audio fakes are usually given.
const kFakeNoOutput = PlaybackException(
  'No audio output (no PulseAudio/PipeWire/ALSA device): …',
);
const kFakeNoMic = MicAccessException(
  'Microphone unavailable: install pulseaudio-utils (parecord)',
);
