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

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/selftest/step_recorder.dart';

import '../fakes/fake_hardware.dart';

void main() {
  late List<String> progress;
  late FakeMemoryProbe memory;
  late StepRecorder recorder;

  setUp(() {
    progress = [];
    memory = FakeMemoryProbe();
    recorder = StepRecorder(memory: memory, progress: progress.add);
  });

  test('a step: timed, memory sampled before and after, a progress line '
      'when it starts and one with its first detail when it ends', () async {
    await recorder.run('2', 'detector load (gpu)', () async {
      await Future<void>.delayed(const Duration(milliseconds: 5));
      return (StepStatus.pass, ['DetectorInfo(GPU)', 'second line']);
    });

    final step = recorder.steps().single;
    expect(step.id, '2');
    expect(step.title, 'detector load (gpu)');
    expect(step.status, StepStatus.pass);
    expect(step.details, ['DetectorInfo(GPU)', 'second line']);
    expect(step.elapsed, greaterThanOrEqualTo(const Duration(milliseconds: 5)));
    expect(step.before!.availableBytes, 9 << 30);
    expect(step.after!.availableBytes, (9 << 30) - (100 << 20));
    expect(progress, [
      'step 2 detector load (gpu) …',
      'step 2 PASS: DetectorInfo(GPU)',
    ]);
  });

  test('a step without details ends with its status alone', () async {
    await recorder.run('1', 'probe', () async => (StepStatus.warn, <String>[]));
    expect(progress.last, 'step 1 WARN');
  });

  test('a body that throws is a failed step with the error and at most six '
      'stack lines, and the run goes on', () async {
    await recorder.run('1', 'hardware probe', () => throw StateError('boom'));
    await recorder.run('2', 'next', () async => (StepStatus.pass, ['ok']));

    final failed = recorder.steps().first;
    expect(failed.status, StepStatus.fail);
    expect(failed.details.first, 'threw: Bad state: boom');
    expect(failed.details.length, inInclusiveRange(2, 7));
    expect(failed.after, isNotNull);
    expect(progress, contains('step 1 FAIL: threw: Bad state: boom'));
    expect(recorder.steps().last.status, StepStatus.pass);
  });

  test('skip: a SKIP step with the reason, no time, no memory', () {
    recorder.skip('5', 'detector again', 'step 3 did not produce a reference');

    final step = recorder.steps().single;
    expect(step.status, StepStatus.skip);
    expect(step.details, ['step 3 did not produce a reference']);
    expect(step.elapsed, Duration.zero);
    expect(step.before, isNull);
    expect(step.after, isNull);
    expect(progress, ['step 5 SKIP: step 3 did not produce a reference']);
  });

  test('stop: the step in flight ends on its own; a step not started yet is '
      'skipped without running its body', () async {
    final gate = Completer<void>();
    final inFlight = recorder.run('4a', 'chat model load', () async {
      await gate.future;
      return (StepStatus.pass, ['loaded']);
    });
    await pumpEventQueue();
    recorder.stop();
    gate.complete();
    await inFlight;
    var ran = false;
    await recorder.run('4b', 'generate', () async {
      ran = true;
      return (StepStatus.pass, <String>[]);
    });

    expect(ran, isFalse);
    expect(recorder.steps().map((s) => s.status), [
      StepStatus.pass,
      StepStatus.skip,
    ]);
    expect(recorder.steps().last.title, 'generate');
    expect(
      recorder.steps().last.details.single,
      'the run was stopped after its time limit',
    );
    expect(
      progress.last,
      'step 4b SKIP: the run was stopped after its time '
      'limit',
    );
  });

  test('stop: a step skipped for a dependency after the stop names the stop, '
      'not the dependency (it was skipped because of the stop)', () {
    recorder
      ..stop()
      ..skip(
        '4b',
        'chat model generate',
        'the chat model did not load as requested',
      );

    expect(recorder.steps().single.details, [
      'the run was stopped after its time limit',
    ]);
    expect(progress, [
      'step 4b SKIP: the run was stopped after its time limit',
    ]);
  });

  test('timed out: a failed T step naming the step in flight, or the run '
      'between steps', () async {
    expect(
      recorder.steps(timedOut: const Duration(seconds: 90)).single.details,
      ['the run did not finish within 90 s (--timeout)'],
    );

    final gate = Completer<void>();
    final inFlight = recorder.run('4a', 'chat model load (gpu)', () async {
      await gate.future;
      return (StepStatus.pass, <String>[]);
    });
    await pumpEventQueue();
    final late = recorder.steps(timedOut: const Duration(seconds: 90));
    gate.complete();
    await inFlight;

    expect(late.single.id, 'T');
    expect(late.single.title, 'timeout');
    expect(late.single.status, StepStatus.fail);
    expect(late.single.details, [
      'step 4a chat model load (gpu) did not finish within 90 s (--timeout)',
    ]);
    expect(recorder.steps().single.id, '4a', reason: 'T is never recorded');
  });

  test('the steps are a read-only copy', () async {
    await recorder.run('1', 'probe', () async => (StepStatus.pass, ['x']));
    final steps = recorder.steps();
    expect(() => steps.clear(), throwsUnsupportedError);
    recorder.skip('2', 'next', 'why');
    expect(steps, hasLength(1));
  });
}
