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
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/utils/command.dart';
import 'package:litert_edge_demos/utils/result.dart';

/// What the view sees of a command at one notification.
typedef _Seen = ({bool running, bool error, bool completed});

void main() {
  /// Records the command's state at every notification.
  List<_Seen> watch(Command<Object?> command) {
    final seen = <_Seen>[];
    command.addListener(
      () => seen.add((
        running: command.running,
        error: command.error,
        completed: command.completed,
      )),
    );
    return seen;
  }

  /// What `FlutterError.reportError` reports during the test (the handler
  /// that would otherwise fail it, see flutter_test_config.dart).
  List<FlutterErrorDetails> captureReported() {
    final reported = <FlutterErrorDetails>[];
    final previous = FlutterError.onError;
    FlutterError.onError = reported.add;
    addTearDown(() => FlutterError.onError = previous);
    return reported;
  }

  test('a run notifies at its start and its end; the result is kept '
      'afterwards', () async {
    final command = Command0<int>(() async => const Result.ok(7));
    addTearDown(command.dispose);
    final seen = watch(command);

    await command.execute();

    expect(seen, [
      (running: true, error: false, completed: false),
      (running: false, error: false, completed: true),
    ]);
    expect((command.result! as Ok<int>).value, 7);
  });

  test('a second execute while one runs is ignored: one action, one '
      'result', () async {
    final gate = Completer<void>();
    final arguments = <String>[];
    final command = Command1<String, String>((argument) async {
      arguments.add(argument);
      await gate.future;
      return Result.ok(argument);
    });
    addTearDown(command.dispose);

    final first = command.execute('first');
    final second = command.execute('second');
    await second;
    expect(command.running, isTrue, reason: 'the first still runs');

    gate.complete();
    await first;

    expect(arguments, ['first']);
    expect((command.result! as Ok<String>).value, 'first');
  });

  test('a Result.error is an error', () async {
    final command = Command0<void>(
      () async => Result.error(Exception('refused')),
    );
    addTearDown(command.dispose);

    await command.execute();

    expect(command.error, isTrue);
    expect(command.completed, isFalse);
  });

  test('an Exception that escapes the action is an error, not an unhandled '
      'zone error, and is not reported as a bug', () async {
    final reported = captureReported();
    final command = Command0<void>(
      () async => throw const FormatException('x'),
    );
    addTearDown(command.dispose);
    final seen = watch(command);

    await command.execute();

    expect(command.running, isFalse);
    expect(command.error, isTrue);
    expect((command.result! as Error<void>).error, isA<FormatException>());
    expect(seen.last, (running: false, error: true, completed: false));
    expect(reported, isEmpty);
  });

  test('an Error that escapes the action (a programmer bug, or a plugin\'s '
      'StateError) is reported to FlutterError, so a test fails on it, and '
      'is an error result too: the view never sees a finished run without '
      'a result', () async {
    final reported = captureReported();
    final command = Command0<void>(() async => throw StateError('worker'));
    addTearDown(command.dispose);
    final seen = watch(command);

    await command.execute();

    expect(command.running, isFalse);
    expect(command.result, isNotNull);
    expect(command.error, isTrue);
    expect(
      (command.result! as Error<void>).error,
      isA<UnexpectedError>().having((e) => e.error, 'error', isA<StateError>()),
    );
    expect(seen.last, (running: false, error: true, completed: false));
    expect(reported, hasLength(1));
    expect(reported.single.exception, isA<StateError>());
    expect(reported.single.library, 'command');
    expect(reported.single.stack, isNotNull);
  });

  testWidgets('in a widget test the reported Error is the test\'s exception', (
    tester,
  ) async {
    final command = Command0<void>(() async => throw RangeError('index 3'));
    addTearDown(command.dispose);

    await command.execute();

    expect(tester.takeException(), isA<RangeError>());
    expect(command.error, isTrue);
  });

  test('disposed while it runs: the run finishes without notifying a '
      'disposed notifier', () async {
    final gate = Completer<Result<void>>();
    final command = Command0<void>(() => gate.future);
    final seen = watch(command);

    final run = command.execute();
    expect(seen, hasLength(1), reason: 'the start');
    command.dispose();
    gate.complete(const Result.ok(null));
    await run;

    expect(seen, hasLength(1), reason: 'nothing after the dispose');
    expect(command.running, isFalse);
  });

  test('disposed while it runs and the action then throws: still no '
      'notification and no unhandled error (the Error is reported)', () async {
    final reported = captureReported();
    final gate = Completer<void>();
    final command = Command0<void>(() async {
      await gate.future;
      throw StateError('late');
    });
    final seen = watch(command);

    final run = command.execute();
    command.dispose();
    gate.complete();
    await run;

    expect(seen, hasLength(1));
    expect(command.error, isTrue);
    expect(reported.single.exception, isA<StateError>());
  });
}
