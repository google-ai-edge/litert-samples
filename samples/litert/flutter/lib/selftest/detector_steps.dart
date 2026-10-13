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

import '../data/services/detector/frame_message.dart';
import '../data/services/hardware/native_log_tap.dart';
import '../domain/hardware/diagnostics_report.dart' show evidenceText;
import '../domain/models/detection.dart';
import '../domain/models/hardware_profile.dart';
import '../domain/vision/coco_vocabulary.dart';
import '../utils/result.dart';
import 'cats_golden.dart';
import 'evidence_judge.dart';
import 'self_test_ports.dart';
import 'step_recorder.dart';
import 'tapped_load.dart';

/// Detections per golden run (the coex test's ten).
const _goldenRuns = 10;

/// Steps 2 (2b), 3 and 5 through [SelfTestDetector]: the detector loaded on
/// exactly the requested backend (no fallback; 2b an explicit CPU attempt,
/// information only, when asked); ten detections of the image, identical,
/// against the cats golden when it applies; and, after the chat model ran,
/// the same detection again, bit-identical.
final class DetectorSteps {
  DetectorSteps({
    required this._recorder,
    required this._judge,
    required this._detector,
    required this._file,
    required this._backend,
    required this._cpuRetry,
    required this._logTap,
    required this._loadImage,
    required this._hardware,
  });

  final StepRecorder _recorder;
  final EvidenceJudge _judge;
  final SelfTestDetector _detector;
  final ModelFileChoice _file;

  /// `--detector-backend`.
  final DetectorBackend _backend;

  /// `--detector-cpu-retry`.
  final bool _cpuRetry;
  final NativeLogTap _logTap;
  final Future<Result<SelfTestImage>> Function() _loadImage;

  /// The probed host (step 1); null before it or when the probe failed.
  final HardwareProfile? Function() _hardware;

  /// After [load]: null when no detector is loaded, else whether the loaded
  /// one is the explicit CPU attempt (its steps are information only).
  bool? _ready;

  /// After [golden]: the image and its first detection, for step 5.
  SelfTestImage? _image;
  DetectionFrame? _reference;

  bool _softwareGpuAllowed = false;

  /// A load passed on a software (or unnamed Linux) GPU only because
  /// `--allow-software-gpu` asked for it.
  bool get softwareGpuAllowed => _softwareGpuAllowed;

  /// Step 2 and, after a failed GPU load with `--detector-cpu-retry`, 2b.
  Future<void> load() async {
    var loaded = false;
    await _recorder.run('2', 'detector load (${_backend.name})', () async {
      final (status, details) = await _loadOn(_backend);
      loaded = status == StepStatus.pass;
      return (status, details);
    });
    if (loaded) {
      _ready = false;
      return;
    }
    if (!_cpuRetry || _backend != DetectorBackend.gpu || _file.path == null) {
      return;
    }
    var cpuLoaded = false;
    await _recorder.run('2b', 'detector load (cpu, explicit attempt, information only)', () async {
      final (status, details) = await _loadOn(DetectorBackend.cpu);
      cpuLoaded = status == StepStatus.pass;
      return (
        StepStatus.info,
        [
          cpuLoaded
              ? 'the CPU loads it; the requested GPU did not (still a failure)'
              : 'the CPU failed too',
          ...details,
        ],
      );
    });
    if (cpuLoaded) _ready = true;
  }

  /// Step 3; skipped when no detector loaded.
  Future<void> golden() async {
    switch (_ready) {
      case null:
        _recorder.skip('3', 'cats golden', 'no detector loaded');
      case final onCpuAttempt:
        await _recorder.run(
          '3',
          'cats golden${onCpuAttempt ? ' (on the CPU attempt)' : ''}',
          () => _cats(onCpuAttempt: onCpuAttempt),
        );
    }
  }

  /// Step 5; skipped when step 3 left no reference.
  Future<void> again() async {
    if ((_ready, _image, _reference) case (
      final onCpuAttempt?,
      final image?,
      final reference?,
    )) {
      await _recorder.run(
        '5',
        'detector again (bit-identical)',
        () => _again(image, reference, onCpuAttempt: onCpuAttempt),
      );
    } else {
      _recorder.skip(
        '5',
        'detector again',
        'step 3 did not produce a reference',
      );
    }
  }

  Future<StepOutcome> _loadOn(DetectorBackend backend) async {
    if (_file.path == null) {
      return (StepStatus.fail, [_file.problem ?? 'no detector file']);
    }
    final load = await loadWithNativeLog(
      _logTap,
      () => _detector.load(_file, backend),
    );
    switch (load.result) {
      case Error(:final error):
        return (StepStatus.fail, load.failure(error));
      case Ok(value: final info):
        final evidence = load.evidence(
          requested: backend.name,
          reported: info.backend.name,
          hardware: _hardware(),
        );
        final verdict = _judge.accelerator(
          evidence,
          platform: _hardware()?.platform,
        );
        if (verdict.softwareGpuAllowed) _softwareGpuAllowed = true;
        return (
          verdict.status,
          [
            '$info',
            evidenceText(evidence),
            ...verdict.errors,
            ...load.logLines,
          ],
        );
    }
  }

  /// Ten detections of the image; the golden when it applies; all
  /// identical. Keeps the image and the first frame for step 5.
  Future<StepOutcome> _cats({required bool onCpuAttempt}) async {
    final SelfTestImage image;
    switch (await _loadImage()) {
      case Error(:final error):
        return (StepStatus.fail, ['image: $error']);
      case Ok(:final value):
        image = _image = value;
    }
    final frames = <DetectionFrame>[];
    for (var i = 0; i < _goldenRuns; i++) {
      switch (await _detector.detect(
        FrameMessage.copyOf(image.frame, frameId: i + 1),
      )) {
        case Error(:final error):
          return (StepStatus.fail, ['detect ${i + 1} failed: $error']);
        case Ok(:final value):
          frames.add(value);
      }
    }
    _reference = frames.first;
    final verdict = _judge.cats(frames, golden: image.golden);
    return (
      onCpuAttempt ? StepStatus.info : _passIf(verdict.passed),
      [
        _describe(frames.first),
        'run ${verdict.medianRunMs.toStringAsFixed(1)} ms median '
            '(first ${(frames.first.runMicros / 1000).toStringAsFixed(1)} ms) · '
            '$_goldenRuns runs '
            '${verdict.identical ? 'identical' : 'NOT identical'}',
        if (verdict.golden case final check?)
          '${check.classesMatch ? 'golden classes ✓' : 'golden classes ✗ '
                        '(want ${[for (final g in kCatsGolden) cocoName(g.cls)].join(', ')})'} · '
              'max box ${check.maxBoxPx.toStringAsFixed(2)} px (≤ $kCatsMaxBoxPx) · '
              'max Δscore ${check.maxScoreDelta.toStringAsFixed(4)} (≤ $kCatsMaxScoreDelta)'
        else
          'custom image: no golden, consistency only',
      ],
    );
  }

  Future<StepOutcome> _again(
    SelfTestImage image,
    DetectionFrame reference, {
    required bool onCpuAttempt,
  }) async {
    switch (await _detector.detect(
      FrameMessage.copyOf(image.frame, frameId: 1000),
    )) {
      case Error(:final error):
        return (StepStatus.fail, ['detect failed: $error']);
      case Ok(:final value):
        final same = _judge.sameAsReference(value, reference);
        return (
          onCpuAttempt ? StepStatus.info : _passIf(same),
          [
            same
                ? 'bit-identical to step 3'
                : 'DIFFERS from step 3: ${_describe(value)}',
            'run ${(value.runMicros / 1000).toStringAsFixed(1)} ms',
          ],
        );
    }
  }

  static StepStatus _passIf(bool ok) => ok ? StepStatus.pass : StepStatus.fail;

  static String _describe(DetectionFrame f) => f.count == 0
      ? 'no detections'
      : [
          for (var i = 0; i < f.count; i++)
            '${cocoName(f.classId(i))} ${f.score(i).toStringAsFixed(3)}',
        ].join(', ');
}
