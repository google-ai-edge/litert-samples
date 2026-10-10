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

// Waits for the integration tests: real frames pumped on the wall clock
// while the app's isolates, plugins and sockets do their work.

import 'package:flutter_test/flutter_test.dart';

/// How often the waits pump a frame unless a test asks for another step.
const kPumpStep = Duration(milliseconds: 20);

/// Pumps a frame every [step] until [condition] holds; false when [timeout]
/// passes first. [onPoll] runs on every poll that finds [condition] false,
/// before the timeout check (a test fails fast from it, for example on a
/// model that failed to load).
Future<bool> tryPumpUntil(
  WidgetTester tester,
  bool Function() condition, {
  required Duration timeout,
  void Function()? onPoll,
  Duration step = kPumpStep,
}) async {
  final watch = Stopwatch()..start();
  while (!condition()) {
    onPoll?.call();
    if (watch.elapsed > timeout) return false;
    await tester.pump(step);
  }
  return true;
}

/// Pumps a frame every [step] until [condition] holds; fails after
/// [timeout] with [reason] and, when given, [describe]'s account of the
/// state at that moment. [onPoll] as in [tryPumpUntil].
Future<void> pumpUntil(
  WidgetTester tester,
  bool Function() condition, {
  required Duration timeout,
  required String reason,
  void Function()? onPoll,
  String Function()? describe,
  Duration step = kPumpStep,
}) async {
  if (await tryPumpUntil(
    tester,
    condition,
    timeout: timeout,
    onPoll: onPoll,
    step: step,
  )) {
    return;
  }
  final state = describe?.call();
  fail('Timed out after $timeout: $reason${state == null ? '' : '. $state'}');
}

/// Pumps a frame every [step] for [duration] of wall-clock time.
Future<void> pumpFor(
  WidgetTester tester,
  Duration duration, {
  Duration step = kPumpStep,
}) async {
  final watch = Stopwatch()..start();
  while (watch.elapsed < duration) {
    await tester.pump(step);
  }
}
