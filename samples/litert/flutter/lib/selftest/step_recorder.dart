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

import '../data/services/hardware/memory_probe.dart';
import '../domain/models/hardware_profile.dart' show MemorySnapshot;

enum StepStatus {
  pass('PASS'),
  fail('FAIL'),
  skip('SKIP'),

  /// Ran for information only (an explicit CPU attempt); never decides the
  /// result.
  info('INFO'),

  /// Passed with a finding worth a look that does not fail the run (a
  /// silent microphone: the room may be quiet).
  warn('WARN');

  const StepStatus(this.label);

  final String label;
}

/// One step's outcome.
final class const SelfTestStep({
  /// `1`, `2`, `2b`, `4a`.
  required final String id,
  required final String title,
  required final StepStatus status,
  final List<String> details = const [],
  final Duration elapsed = Duration.zero,
  final MemorySnapshot? before,
  final MemorySnapshot? after,
});

/// What a step's body returns: its status and its detail lines (the first
/// one is also its progress line).
typedef StepOutcome = (StepStatus, List<String>);

/// The run's steps in order, as they are recorded: each one timed, memory
/// sampled before and after, a progress line when it starts and when it
/// ends. After [stop], every step not started yet is skipped instead.
final class StepRecorder {
  StepRecorder({required this._memory, required this._progress});

  final MemoryProbe _memory;
  final void Function(String line) _progress;
  final List<SelfTestStep> _steps = [];

  /// The step running now (`step 4a chat model load + warm-up (gpu)`); null
  /// between steps.
  String? _current;

  /// [stop] was called: every step not started yet is skipped.
  bool _stopRequested = false;

  /// Why a step is skipped after [stop].
  static const _stopped = 'the run was stopped after its time limit';

  /// Every step not started yet is skipped; the one in flight ends on its
  /// own.
  void stop() => _stopRequested = true;

  /// Runs [body] as step [id]: timing, memory before and after, progress
  /// lines. A throw is a failed step (a bug), not a crashed self-test.
  Future<void> run(
    String id,
    String title,
    Future<StepOutcome> Function() body,
  ) async {
    if (_stopRequested) {
      skip(id, title, _stopped);
      return;
    }
    _progress('step $id $title …');
    _current = 'step $id $title';
    final before = _memory.snapshot();
    final watch = Stopwatch()..start();
    StepStatus status;
    List<String> details;
    try {
      (status, details) = await body();
    } catch (e, st) {
      status = StepStatus.fail;
      details = ['threw: $e', ...st.toString().split('\n').take(6)];
    }
    watch.stop();
    _steps.add(
      SelfTestStep(
        id: id,
        title: title,
        status: status,
        details: details,
        elapsed: watch.elapsed,
        before: before,
        after: _memory.snapshot(),
      ),
    );
    _current = null;
    _progress(
      'step $id ${status.label}${details.isEmpty ? '' : ': ${details.first}'}',
    );
  }

  /// Records step [id] as skipped, with [why] — or, after [stop], with the
  /// stop: a step after it is skipped because of it, whatever it depends
  /// on (the dependency itself may have been skipped by the stop).
  void skip(String id, String title, String why) {
    final reason = _stopRequested ? _stopped : why;
    _steps.add(
      SelfTestStep(
        id: id,
        title: title,
        status: StepStatus.skip,
        details: [reason],
      ),
    );
    _progress('step $id SKIP: $reason');
  }

  /// The steps so far. With [timedOut], the step still running becomes a
  /// failed `timeout` step (the watchdog's report).
  List<SelfTestStep> steps({Duration? timedOut}) => List.unmodifiable([
    ..._steps,
    if (timedOut != null)
      SelfTestStep(
        id: 'T',
        title: 'timeout',
        status: StepStatus.fail,
        details: [
          '${_current ?? 'the run'} did not finish within '
              '${timedOut.inSeconds} s (--timeout)',
        ],
      ),
  ]);
}
