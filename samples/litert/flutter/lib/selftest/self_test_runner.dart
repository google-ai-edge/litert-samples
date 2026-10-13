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

import '../data/services/hardware/hardware_info_service.dart';
import '../data/services/hardware/memory_probe.dart';
import '../data/services/hardware/native_log_tap.dart';
import '../domain/models/hardware_profile.dart';
import '../utils/result.dart';
import 'audio_steps.dart';
import 'chat_model_steps.dart';
import 'detector_steps.dart';
import 'evidence_judge.dart';
import 'self_test_options.dart';
import 'self_test_ports.dart';
import 'step_recorder.dart';

export 'audio_steps.dart' show kSelfTestAudioTimeout;
export 'self_test_ports.dart';
export 'step_recorder.dart' show SelfTestStep, StepStatus;

/// The whole run.
final class const SelfTestReport({
  required final DateTime startedAt,
  required final Duration elapsed,
  required final BuildInfo build,
  required final SelfTestOptions options,
  required final SelfTestChatModel chatModel,
  required final ModelFileChoice detectorFile,
  required final String imageLabel,
  required final String nativeLog,
  required final HardwareProfile? hardware,
  required final List<SelfTestStep> steps,

  /// A GPU step passed on a software GPU only because of
  /// `--allow-software-gpu`; the result line says so.
  final bool softwareGpuAllowed = false,

  /// Dart errors no code caught (`PlatformDispatcher.onError`) during the
  /// run: any one fails it.
  final List<String> unhandledErrors = const [],
}) {
  /// Every step passed (or ran for information, or passed with a warning);
  /// a skipped step means an earlier one failed.
  bool get passed =>
      steps.isNotEmpty &&
      unhandledErrors.isEmpty &&
      steps.every(
        (s) =>
            s.status == StepStatus.pass ||
            s.status == StepStatus.info ||
            s.status == StepStatus.warn,
      );

  int get exitCode => passed ? 0 : 1;
}

/// Runs the self-test steps in order (integration_test/detector_coex_test
/// inside the app): 1 hardware probe; 2 detector load on the requested
/// backend (no fallback; 2b an explicit CPU attempt when asked); 3 cats
/// golden, ten identical runs; 4a Gemma load + warm-up on the requested
/// backend (no fallback), 4b 64 tokens timed (the chat model: Gemma 4 E2B or
/// the user's own `.litertlm`); 5 the detector again,
/// bit-identical; 6a a tone on the default output (recorded from the sink's
/// monitor where possible), 6b one second from the default microphone
/// (without `--skip-audio`). Memory is sampled around each step. Closes the
/// models and the audio at the end.
///
/// The orchestrator: it runs step 1 (the hardware probe) itself; steps 2 to 6
/// are [DetectorSteps], [ChatModelSteps] and [AudioSteps]. One
/// [StepRecorder] records them all and one [EvidenceJudge] judges them.
final class SelfTestRunner {
  SelfTestRunner({
    required this._options,
    required this._build,
    required this._hardware,
    required this._detector,
    required this._gemma,
    required this._memory,
    required this._logTap,
    required SelfTestChatModel chatModel,
    required this._detectorFile,
    required this._loadImage,
    required this._imageLabel,
    this._audio,
    this._audioTimeout = kSelfTestAudioTimeout,
    void Function(String line)? progress,
    this._now = DateTime.now,
    this._unhandledErrors = const [],
  }) : _progress = progress ?? ((_) {}),
       _chatModel = chatModel.withBackend(_options.gemmaBackend);

  final SelfTestOptions _options;
  final BuildInfo _build;
  final HardwareInfoService _hardware;
  final SelfTestDetector _detector;
  final SelfTestGemma _gemma;
  final MemoryProbe _memory;
  final NativeLogTap _logTap;

  /// The chat model with `--gemma-backend` applied.
  final SelfTestChatModel _chatModel;
  final ModelFileChoice _detectorFile;
  final Future<Result<SelfTestImage>> Function() _loadImage;
  final String _imageLabel;

  /// Null: no audio steps (`--skip-audio`).
  final SelfTestAudio? _audio;
  final Duration _audioTimeout;
  final void Function(String line) _progress;
  final DateTime Function() _now;

  /// Filled by the entry's `PlatformDispatcher.onError` while the run goes;
  /// read when the report is built.
  final List<String> _unhandledErrors;

  late final StepRecorder _recorder = StepRecorder(
    memory: _memory,
    progress: _progress,
  );
  late final EvidenceJudge _judge = EvidenceJudge(
    allowSoftwareGpu: _options.allowSoftwareGpu,
  );
  late final DetectorSteps _detectorSteps = DetectorSteps(
    recorder: _recorder,
    judge: _judge,
    detector: _detector,
    file: _detectorFile,
    backend: _options.detectorBackend,
    cpuRetry: _options.detectorCpuRetry,
    logTap: _logTap,
    loadImage: _loadImage,
    hardware: () => _profile,
  );
  late final ChatModelSteps _chatSteps = ChatModelSteps(
    recorder: _recorder,
    judge: _judge,
    gemma: _gemma,
    chatModel: _chatModel,
    logTap: _logTap,
    hardware: () => _profile,
  );

  /// Null: no audio steps (`--skip-audio`).
  late final AudioSteps? _audioSteps = switch (_audio) {
    final audio? => AudioSteps(
      recorder: _recorder,
      judge: _judge,
      audio: audio,
      timeout: _audioTimeout,
      progress: _progress,
    ),
    null => null,
  };

  /// Step 1's probe; null before it or when it failed.
  HardwareProfile? _profile;

  /// [run] has closed the chat model.
  bool _chatModelClosed = false;
  DateTime? _startedAt;
  final Stopwatch _total = Stopwatch();

  /// Asks a run past its time limit to wind down: the step in flight ends
  /// on its own (a native call cannot be interrupted), every later step is
  /// skipped, and [run] then closes the models and the audio as usual. The
  /// in-app self-test waits for that before it loads the app's chat model
  /// again (one engine per process).
  void stop() => _recorder.stop();

  /// The run may hold a chat model engine: step 4a has started and the chat
  /// model has not been closed since. flutter_edge_ai holds one engine per
  /// process, so while a run that did not end after [stop] says so, the app
  /// must not load its own; one that hung before step 4a never will (after
  /// [stop] no step starts).
  bool get chatEngineMayBeLoaded => _chatSteps.loadStarted && !_chatModelClosed;

  Future<SelfTestReport> run() async {
    _startedAt = _now();
    _total.start();
    try {
      await _recorder.run('1', 'hardware probe', _probe);
      await _detectorSteps.load();
      await _detectorSteps.golden();
      await _chatSteps.run();
      await _detectorSteps.again();
      await _audioSteps?.run();
    } finally {
      await _detector.close();
      await _gemma.close();
      _chatModelClosed = true;
      await _audioSteps?.close();
    }
    return report();
  }

  /// The report so far. With [timedOut], the step still running becomes a
  /// failed `timeout` step (the watchdog's report: a native call that never
  /// returns must not hang a headless run).
  SelfTestReport report({Duration? timedOut}) => SelfTestReport(
    startedAt: _startedAt ?? _now(),
    elapsed: _total.elapsed,
    build: _build,
    options: _options,
    chatModel: _chatModel,
    detectorFile: _detectorFile,
    imageLabel: _imageLabel,
    nativeLog: _logTap.description,
    hardware: _profile,
    softwareGpuAllowed:
        _detectorSteps.softwareGpuAllowed || _chatSteps.softwareGpuAllowed,
    unhandledErrors: List.unmodifiable(_unhandledErrors),
    steps: _recorder.steps(timedOut: timedOut),
  );

  Future<StepOutcome> _probe() async {
    final profile = _profile = await _hardware.probe();
    return (
      StepStatus.pass,
      [
        '${profile.os} · ${profile.cpu.model} · '
            '${profile.gpus.length} GPU(s) · ${profile.notes.length} note(s)',
      ],
    );
  }
}
