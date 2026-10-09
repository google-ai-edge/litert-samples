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
import 'package:litert_edge_demos/utils/waits.dart';

/// A flag that counts its listeners (a wait must leave none behind).
final class _Flag extends ValueNotifier<bool> {
  _Flag({required bool value}) : super(value);

  int listeners = 0;

  @override
  void addListener(VoidCallback listener) {
    listeners++;
    super.addListener(listener);
  }

  @override
  void removeListener(VoidCallback listener) {
    listeners--;
    super.removeListener(listener);
  }
}

void main() {
  group('whenFalse', () {
    test('a flag that is already false: done at once, no listener', () async {
      final flag = _Flag(value: false);
      addTearDown(flag.dispose);
      expect(await whenFalse(flag), WaitEnd.done);
      expect(flag.listeners, 0);
    });

    test('done at the change that turns it false, not before; the listener '
        'is removed', () async {
      final flag = _Flag(value: true);
      addTearDown(flag.dispose);
      WaitEnd? end;
      unawaited(whenFalse(flag).then((e) => end = e));
      await pumpEventQueue();
      expect(end, isNull);
      expect(flag.listeners, 1);

      flag.value = true; // notifies nothing: still true
      await pumpEventQueue();
      expect(end, isNull);

      flag.value = false;
      await pumpEventQueue();
      expect(end, WaitEnd.done);
      expect(flag.listeners, 0);
    });

    test('a flag disposed while true never notifies again: the wait ends '
        'when its owner closes', () async {
      final flag = _Flag(value: true);
      final closed = Completer<void>();
      WaitEnd? end;
      unawaited(whenFalse(flag, closed: closed.future).then((e) => end = e));
      await pumpEventQueue();
      flag.dispose();
      await pumpEventQueue();
      expect(end, isNull);

      closed.complete();
      await pumpEventQueue();
      expect(end, WaitEnd.closed);
      expect(flag.listeners, 0);
    });

    test('with a timeout it ends when that passes; the listener is '
        'removed', () async {
      final flag = _Flag(value: true);
      addTearDown(flag.dispose);
      final end = await whenFalse(
        flag,
        timeout: const Duration(milliseconds: 10),
      );
      expect(end, WaitEnd.timedOut);
      expect(flag.listeners, 0);
      flag.value = false; // a late change reaches no waiter
    });

    test('the first of the change, the close and the timeout wins', () async {
      final flag = _Flag(value: true);
      addTearDown(flag.dispose);
      final closed = Completer<void>();
      final waiting = whenFalse(
        flag,
        closed: closed.future,
        timeout: const Duration(seconds: 30),
      );
      flag.value = false;
      closed.complete();
      expect(await waiting, WaitEnd.done);
    });
  });

  group('waitFor', () {
    test('done when the future completes', () async {
      final work = Completer<void>();
      final waiting = waitFor(work.future, closed: Completer<void>().future);
      work.complete();
      expect(await waiting, WaitEnd.done);
    });

    test('a future that never completes: closed when its owner '
        'closes', () async {
      final closed = Completer<void>();
      final waiting = waitFor(Completer<void>().future, closed: closed.future);
      closed.complete();
      expect(await waiting, WaitEnd.closed);
    });

    test('an owner already closed ends the wait at once', () async {
      expect(
        await waitFor(Completer<void>().future, closed: Future<void>.value()),
        WaitEnd.closed,
      );
    });

    test('a timeout ends it', () async {
      expect(
        await waitFor(
          Completer<void>().future,
          timeout: const Duration(milliseconds: 10),
        ),
        WaitEnd.timedOut,
      );
    });

    test("the future's error is the wait's when it ends it", () async {
      final work = Completer<void>();
      final waiting = waitFor(work.future);
      work.completeError(StateError('broken'));
      await expectLater(waiting, throwsA(isA<StateError>()));
    });

    test('an error after the wait ended is only logged, never '
        'unhandled', () async {
      final work = Completer<void>();
      final logged = <String>[];
      final print = debugPrint;
      debugPrint = (message, {wrapWidth}) => logged.add(message ?? '');
      addTearDown(() => debugPrint = print);

      final waiting = waitFor(work.future, closed: Future<void>.value());
      expect(await waiting, WaitEnd.closed);
      work.completeError(StateError('late'));
      await pumpEventQueue();
      expect(logged.single, startsWith('[wait] failed after the wait ended'));
    });
  });
}
