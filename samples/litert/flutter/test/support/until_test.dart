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

import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';

import 'until.dart';

/// A [ValueNotifier] that says whether anyone still listens.
final class _Value extends ValueNotifier<int> {
  _Value(super.value);

  bool get listened => hasListeners;
}

void main() {
  test('holds already: completes at once, without a notification', () async {
    final value = _Value(1);
    addTearDown(value.dispose);

    await untilNotified(value, () => value.value == 1, what: 'one');
  });

  test('completes on the notification that makes it hold, and stops '
      'listening', () async {
    final value = _Value(0);
    addTearDown(value.dispose);
    var done = false;
    final waiting = untilNotified(
      value,
      () => value.value == 2,
      what: 'two',
    ).then((_) => done = true);

    value.value = 1;
    await pumpEventQueue();
    expect(done, isFalse);
    value.value = 2;
    await waiting;

    expect(done, isTrue);
    expect(value.listened, isFalse);
  });

  test('fails with what it waited for once the timeout passes', () async {
    final value = _Value(0);
    addTearDown(value.dispose);

    await expectLater(
      untilNotified(
        value,
        () => value.value == 1,
        what: 'one',
        timeout: const Duration(milliseconds: 20),
      ),
      throwsA(
        isA<TestFailure>().having(
          (f) => f.message,
          'message',
          contains('Timed out after 0:00:00.020000: one'),
        ),
      ),
    );
    expect(value.listened, isFalse);
  });
}
