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

/// Completes once [condition] holds: checked now and on every notification
/// of [listenable], so no state is missed between polls and nothing waits
/// longer than the change takes. Fails with [what] after [timeout] of real
/// time, well inside the test's own timeout, so a regression says what it
/// was waiting for instead of hanging.
Future<void> untilNotified(
  Listenable listenable,
  bool Function() condition, {
  required String what,
  Duration timeout = const Duration(seconds: 10),
}) {
  if (condition()) return Future.value();
  final done = Completer<void>();
  void check() {
    if (!done.isCompleted && condition()) done.complete();
  }

  listenable.addListener(check);
  final timer = Timer(timeout, () {
    if (done.isCompleted) return;
    done.completeError(TestFailure('Timed out after $timeout: $what'));
  });
  return done.future.whenComplete(() {
    timer.cancel();
    listenable.removeListener(check);
  });
}
