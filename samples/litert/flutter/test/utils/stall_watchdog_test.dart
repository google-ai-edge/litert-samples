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

import 'package:fake_async/fake_async.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/utils/stall_watchdog.dart';

const _ms = Duration(milliseconds: 1);
const _interval = Duration(milliseconds: 25);
const _timeout = Duration(milliseconds: 100);

/// Fake time, and a clock that can jump ahead without any timer running (the
/// app suspended).
final class _Time {
  _Time(this.async);

  final FakeAsync async;
  int suspendedMicros = 0;

  int clock() => async.elapsed.inMicroseconds + suspendedMicros;

  /// The isolate stops for [d]: the clock moves, no timer runs.
  void suspend(Duration d) => suspendedMicros += d.inMicroseconds;
}

void main() {
  group('StallWatchdog', () {
    late _Time time;
    late StallWatchdog watchdog;
    late List<int> stalls;
    late int healthy;
    late Duration timeout;
    late bool cancelled;

    void inFakeTime(void Function(FakeAsync async) body) => fakeAsync((async) {
      time = _Time(async);
      watchdog = StallWatchdog(interval: _interval, clockMicros: time.clock);
      stalls = [];
      healthy = 0;
      timeout = _timeout;
      cancelled = false;
      body(async);
      watchdog.disarm();
    });

    int arm() => watchdog.arm(
      timeout: () => timeout,
      onStall: stalls.add,
      cancelIf: () => cancelled,
      onHealthy: () => healthy++,
    );

    test('no activity: a stall at the first check that sees the timeout of '
        'silence, not before, with the silence in µs', () {
      inFakeTime((async) {
        arm();
        async.elapse(_timeout - _ms);
        expect(stalls, isEmpty);
        expect(healthy, 3, reason: 'checks at 25, 50 and 75 ms');
        async.elapse(_ms);
        expect(stalls, [_timeout.inMicroseconds]);
      });
    });

    test('activity keeps it healthy; the silence counts from the newest '
        'activity', () {
      inFakeTime((async) {
        arm();
        for (var i = 0; i < 10; i++) {
          async.elapse(const Duration(milliseconds: 20));
          watchdog.activity(time.clock());
        }
        expect(stalls, isEmpty);
        final last = time.clock(); // 200 ms
        // Checks at 225 ... 275 ms see 25 ... 75 ms of silence; 300 sees 100.
        async.elapse(const Duration(milliseconds: 99));
        expect(stalls, isEmpty);
        async.elapse(_ms);
        expect(stalls, [time.clock() - last]);
        expect(stalls.single, _timeout.inMicroseconds);
      });
    });

    test('a stall is reported once, and the watchdog disarms first', () {
      inFakeTime((async) {
        bool? armedInCallback;
        watchdog.arm(
          timeout: () => _timeout,
          onStall: (silent) {
            stalls.add(silent);
            armedInCallback = watchdog.isArmed;
          },
        );
        expect(watchdog.isArmed, isTrue);
        async.elapse(const Duration(seconds: 1));
        expect(stalls, hasLength(1));
        expect(armedInCallback, isFalse);
        expect(watchdog.isArmed, isFalse);
      });
    });

    test('the timeout is read at every check', () {
      inFakeTime((async) {
        arm();
        async.elapse(const Duration(milliseconds: 50));
        timeout = const Duration(milliseconds: 300);
        async.elapse(const Duration(milliseconds: 249));
        expect(stalls, isEmpty);
        async.elapse(_ms);
        expect(stalls, [const Duration(milliseconds: 300).inMicroseconds]);
      });
    });

    test('cancelIf disarms at the next check without a report', () {
      inFakeTime((async) {
        arm();
        async.elapse(const Duration(milliseconds: 50));
        expect(healthy, 2);
        cancelled = true;
        async.elapse(_interval);
        expect(watchdog.isArmed, isFalse);
        async.elapse(const Duration(seconds: 1));
        expect(stalls, isEmpty);
        expect(healthy, 2, reason: 'the cancelling check is not healthy');
      });
    });

    test('a check over two intervals after the previous one (the app was '
        'suspended) starts a fresh window instead of counting the '
        'suspension', () {
      inFakeTime((async) {
        arm();
        async.elapse(const Duration(milliseconds: 50));
        expect(healthy, 2);

        time.suspend(const Duration(milliseconds: 500));
        async.elapse(_interval); // the late check
        expect(stalls, isEmpty, reason: 'not 500 ms of silence');
        expect(healthy, 2, reason: 'a late check is neither');

        // The window counts from the late check.
        async.elapse(_timeout - _ms);
        expect(stalls, isEmpty);
        async.elapse(_ms);
        expect(stalls, [_timeout.inMicroseconds]);
      });
    });

    test('a check exactly two intervals after the previous one is on time; '
        'one µs more is late', () {
      inFakeTime((async) {
        arm();
        time.suspend(_interval);
        async.elapse(_interval); // on time: silence 50 ms
        expect(healthy, 1);
        // Exactly on time again, with enough silence for a stall.
        time.suspend(_interval);
        async.elapse(_interval);
        expect(stalls, [const Duration(milliseconds: 100).inMicroseconds]);

        arm();
        time.suspend(_interval + const Duration(microseconds: 1));
        async.elapse(_interval);
        expect(healthy, 1, reason: 'late: not counted');
        expect(stalls, hasLength(1));
      });
    });

    test('arm starts a fresh window, returns its start time, and replaces '
        'an earlier arming', () {
      inFakeTime((async) {
        arm();
        async.elapse(const Duration(milliseconds: 90));
        final start = arm();
        expect(start, time.clock());
        async.elapse(const Duration(milliseconds: 99));
        expect(stalls, isEmpty, reason: 'silence before the arming is gone');
        expect(healthy, 3 + 3, reason: 'one series of checks at a time');
        async.elapse(const Duration(milliseconds: 1));
        expect(stalls, [_timeout.inMicroseconds]);
      });
    });

    test('disarm stops the checks; activity while disarmed is harmless', () {
      inFakeTime((async) {
        arm();
        async.elapse(const Duration(milliseconds: 30));
        watchdog
          ..disarm()
          ..disarm()
          ..activity(time.clock());
        expect(watchdog.isArmed, isFalse);
        async.elapse(const Duration(seconds: 1));
        expect(stalls, isEmpty);
        expect(healthy, 1);
      });
    });
  });

  group('DeadlineWatchdog', () {
    late _Time time;
    late DeadlineWatchdog deadline;
    late int expired;
    late bool cancelled;
    const tolerance = Duration(milliseconds: 25);

    void inFakeTime(void Function(FakeAsync async) body) => fakeAsync((async) {
      time = _Time(async);
      deadline = DeadlineWatchdog(
        timeout: _timeout,
        lateTolerance: tolerance,
        clockMicros: time.clock,
      );
      expired = 0;
      cancelled = false;
      body(async);
      deadline.disarm();
    });

    void arm() =>
        deadline.arm(onExpired: () => expired++, cancelIf: () => cancelled);

    test('expires at the timeout, not before, once', () {
      inFakeTime((async) {
        arm();
        expect(deadline.isArmed, isTrue);
        async.elapse(_timeout - _ms);
        expect(expired, 0);
        async.elapse(_ms);
        expect(expired, 1);
        expect(deadline.isArmed, isFalse);
        async.elapse(const Duration(seconds: 1));
        expect(expired, 1);
      });
    });

    test('disarm first: no expiry', () {
      inFakeTime((async) {
        arm();
        async.elapse(_timeout - _ms);
        deadline
          ..disarm()
          ..disarm();
        expect(deadline.isArmed, isFalse);
        async.elapse(const Duration(seconds: 1));
        expect(expired, 0);
      });
    });

    test('cancelIf at the deadline: no expiry, and nothing re-armed', () {
      inFakeTime((async) {
        arm();
        cancelled = true;
        async.elapse(_timeout);
        expect(expired, 0);
        expect(deadline.isArmed, isFalse);
      });
    });

    test('a deadline that fires more than the tolerance late (the app was '
        'suspended) is re-armed with a fresh window', () {
      inFakeTime((async) {
        arm();
        time.suspend(const Duration(milliseconds: 300));
        async.elapse(_timeout);
        expect(expired, 0, reason: 'fired late: re-armed');
        expect(deadline.isArmed, isTrue);

        async.elapse(_timeout - _ms);
        expect(expired, 0);
        async.elapse(_ms);
        expect(expired, 1);
      });
    });

    test('late by exactly the tolerance still expires; 1 µs more does not', () {
      inFakeTime((async) {
        arm();
        time.suspend(tolerance);
        async.elapse(_timeout);
        expect(expired, 1);

        arm();
        time.suspend(tolerance + const Duration(microseconds: 1));
        async.elapse(_timeout);
        expect(expired, 1);
        expect(deadline.isArmed, isTrue);
      });
    });

    test('arm replaces the earlier deadline, counted from the new arming', () {
      inFakeTime((async) {
        arm();
        async.elapse(const Duration(milliseconds: 60));
        arm();
        async.elapse(_timeout - _ms);
        expect(expired, 0);
        async.elapse(_ms);
        expect(expired, 1, reason: 'one deadline, not two');
        async.elapse(const Duration(seconds: 1));
        expect(expired, 1);
      });
    });
  });
}
