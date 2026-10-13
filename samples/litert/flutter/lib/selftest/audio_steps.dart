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

import '../domain/audio/audio_device_checks.dart' show deviceCheckText;
import '../domain/hardware/diagnostics_report.dart' show seconds;
import '../utils/pcm.dart';
import '../utils/result.dart';
import 'evidence_judge.dart';
import 'self_test_ports.dart';
import 'step_recorder.dart';

/// Step 6a's tone: 1 s of 440 Hz at −9 dBFS RMS, at a TTS-like rate.
const kSelfTestToneHz = 440;
const kSelfTestTone = Duration(seconds: 1);
const kSelfTestToneRate = 24000;

/// Step 6b records this long.
const kSelfTestMicLength = Duration(seconds: 1);

/// Each audio step's limit: a machine without audio must fail, not hang.
const kSelfTestAudioTimeout = Duration(seconds: 15);

/// 100 ms at the capture rate (16 kHz).
const _window100ms = 1600;

/// Steps 6a and 6b through [SelfTestAudio]: a tone on the default output
/// (recorded from the sink's monitor where there is one), then one second
/// from the default microphone, independent of each other. Each fails after
/// [_timeout] instead of hanging.
final class AudioSteps {
  AudioSteps({
    required this._recorder,
    required this._judge,
    required this._audio,
    required this._timeout,
    required this._progress,
  });

  final StepRecorder _recorder;
  final EvidenceJudge _judge;
  final SelfTestAudio _audio;
  final Duration _timeout;
  final void Function(String line) _progress;

  /// Steps 6a and 6b.
  Future<void> run() async {
    final note = _audio.monitorUnavailable;
    await _recorder.run(
      '6a',
      note == null
          ? 'audio output (tone → sink monitor)'
          : 'audio output (played, not measured)',
      () => _withTimeout('the output', () => _output(note)),
    );
    await _recorder.run(
      '6b',
      'microphone (${kSelfTestMicLength.inSeconds} s, default input)',
      () => _withTimeout('the microphone', _mic),
    );
  }

  /// Releases the devices. A device that hung a step may hang its close
  /// too: the report must not wait for it (the process exits right after,
  /// which releases it).
  Future<void> close() => _audio.close().timeout(
    _timeout,
    onTimeout: () => _progress(
      'the audio did not close within ${seconds(_timeout)}; exiting '
      'releases it',
    ),
  );

  Future<StepOutcome> _withTimeout(
    String what,
    Future<StepOutcome> Function() body,
  ) => body().timeout(
    _timeout,
    onTimeout: () => (
      StepStatus.fail,
      [
        '$what did not answer within ${seconds(_timeout)}: the sound server '
            'or the device hung (no audio stack?)',
      ],
    ),
  );

  Future<StepOutcome> _output(String? monitorNote) async {
    final String device;
    final List<String> flagged;
    switch (await _audio.prepareOutput()) {
      case Error(:final error):
        return (StepStatus.fail, ['$error']);
      case Ok(:final value):
        device = deviceCheckText(value);
        // Usable but worth a look (the auto_null dummy sink, an inferred
        // ALSA output): a pass becomes a WARN.
        flagged = [
          if (value.caution) 'the output check flagged this device: $device',
        ];
    }
    final ok = flagged.isEmpty ? StepStatus.pass : StepStatus.warn;
    final tone = sineTonePcm16(
      hz: kSelfTestToneHz,
      duration: kSelfTestTone,
      sampleRate: kSelfTestToneRate,
    );
    final played =
        'played ${seconds(kSelfTestTone)} of $kSelfTestToneHz Hz at '
        '${_db(rmsDbfs(tone))} on $device';
    if (monitorNote != null) {
      if (await _audio.play(tone, kSelfTestToneRate) case Error(:final error)) {
        return (StepStatus.fail, ['output $device', 'playback failed: $error']);
      }
      return (ok, [played, 'not measured: $monitorNote', ...flagged]);
    }
    final SelfTestMonitor monitor;
    switch (await _audio.startMonitor()) {
      case Error(:final error):
        return (StepStatus.fail, ['output $device', 'monitor capture: $error']);
      case Ok(:final value):
        monitor = value;
    }
    final playback = await _audio.play(tone, kSelfTestToneRate);
    final recorded = await monitor.stop();
    if (playback case Error(:final error)) {
      return (StepStatus.fail, ['output $device', 'playback failed: $error']);
    }
    switch (recorded) {
      case Error(:final error):
        return (StepStatus.fail, [played, 'monitor capture: $error']);
      case Ok(value: final pcm):
        final stats = analyzePcm16(pcm, frameSamples: _window100ms);
        final level =
            'sink monitor: loudest 100 ms ${_db(stats.peakFrameDbfs)}, whole '
            '${_db(rmsDbfs(pcm))} over '
            '${seconds(pcm16Duration(pcm.length, 16000))} '
            '(pass ≥ ${_db(kMonitorPassDbfs)})';
        final headline = switch (_judge.monitor(stats)) {
          MonitorVerdict.heard => null,
          MonitorVerdict.nothing =>
            'played but nothing reached the sound server',
          MonitorVerdict.tooQuiet =>
            'played, but it reached the sound server too quietly',
        };
        if (headline == null) return (ok, [played, level, ...flagged]);
        return (
          StepStatus.fail,
          [
            headline,
            played,
            level,
            'is the sink muted or near 0 volume, or did the stream go to '
                'another sink than the default?',
            ...flagged,
          ],
        );
    }
  }

  Future<StepOutcome> _mic() async {
    switch (await _audio.recordMic(kSelfTestMicLength)) {
      case Error(:final error):
        return (StepStatus.fail, ['no microphone: $error']);
      case Ok(value: MicCapture(:final device, :final pcm)):
        final stats = analyzePcm16(pcm, frameSamples: _window100ms);
        String level() =>
            '${seconds(pcm16Duration(pcm.length, 16000))} · RMS '
            '${_db(rmsDbfs(pcm))} · loudest 100 ms ${_db(stats.peakFrameDbfs)}';
        return switch (_judge.mic(stats)) {
          MicVerdict.empty => (
            StepStatus.fail,
            [device, 'no audio arrived in ${seconds(kSelfTestMicLength)}'],
          ),
          MicVerdict.digitalZeros => (
            StepStatus.warn,
            [
              device,
              level(),
              'all digital zeros: the input is muted, or the OS hands the app '
                  'no audio',
            ],
          ),
          MicVerdict.silent => (
            StepStatus.warn,
            [
              device,
              level(),
              'silent (below ${_db(kMicSilenceDbfs)}): the room may be quiet, '
                  'or the input is muted',
            ],
          ),
          MicVerdict.heard => (StepStatus.pass, [device, level()]),
        };
    }
  }

  static String _db(double dbfs) => '${dbfs.toStringAsFixed(1)} dBFS';
}
