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

/// Whether a check due [delayMicros] after [sinceMicros] that ran at
/// [nowMicros] ran more than [toleranceMicros] past its deadline: the
/// isolate was not running (a debugger pause, system sleep, a backgrounded
/// app), so the time that passed says nothing about what is watched.
bool _overdue({
  required int nowMicros,
  required int sinceMicros,
  required int delayMicros,
  required int toleranceMicros,
}) => nowMicros - sinceMicros - delayMicros > toleranceMicros;

/// Watches that something keeps happening (frames from a camera, JPEGs from
/// a stream): every `interval` it checks how long ago the newest [activity]
/// was, and reports a stall once that silence reaches the timeout.
///
/// - [arm] starts a fresh window: silence from before it (paused, starting)
///   does not count.
/// - A check that runs more than one interval late (over two intervals
///   after the previous one) means the app itself was suspended: it starts a
///   fresh window instead of counting the suspension as silence.
/// - A stall is reported once, and the watchdog disarms first.
///
/// Times are the injected monotonic clock's, in µs.
final class StallWatchdog {
  StallWatchdog({required this._interval, required int Function() clockMicros})
    : _clock = clockMicros;

  final Duration _interval;
  final int Function() _clock;
  Timer? _timer;
  int _lastActivityAt = 0;
  int _lastCheckAt = 0;

  /// Whether checks are running.
  bool get isArmed => _timer != null;

  /// Starts a check every interval with a fresh window, replacing an
  /// earlier arming, and returns the clock time the window starts at.
  ///
  /// Each check first asks [cancelIf] (the watched thing is gone: disarm,
  /// no report), then reads [timeout] (it may change while armed). A stall
  /// calls [onStall] with the silence in µs; a check that finds activity
  /// within the timeout calls [onHealthy] (a check that ran late calls
  /// neither).
  int arm({
    required Duration Function() timeout,
    required void Function(int silentMicros) onStall,
    bool Function()? cancelIf,
    void Function()? onHealthy,
  }) {
    _timer?.cancel();
    final now = _lastActivityAt = _lastCheckAt = _clock();
    _timer = Timer.periodic(_interval, (_) {
      if (cancelIf?.call() ?? false) {
        disarm();
        return;
      }
      final previousCheck = _lastCheckAt;
      final now = _lastCheckAt = _clock();
      final interval = _interval.inMicroseconds;
      if (_overdue(
        nowMicros: now,
        sinceMicros: previousCheck,
        delayMicros: interval,
        toleranceMicros: interval,
      )) {
        _lastActivityAt = now; // suspended: a fresh window
        return;
      }
      final silentMicros = now - _lastActivityAt;
      if (silentMicros < timeout().inMicroseconds) {
        onHealthy?.call();
        return;
      }
      disarm();
      onStall(silentMicros);
    });
    return now;
  }

  /// Something happened at [atMicros] (the clock's time). Harmless while
  /// disarmed: [arm] starts a fresh window anyway.
  void activity(int atMicros) => _lastActivityAt = atMicros;

  /// Stops the checks. Safe to call when not armed.
  void disarm() {
    _timer?.cancel();
    _timer = null;
  }
}

/// Watches that one thing finishes in time (a frame the detector must
/// answer): [arm] reports once when `timeout` passes, unless [disarm] came
/// first.
///
/// A deadline that fires more than `lateTolerance` late means the app was
/// suspended, and the answer may be queued right behind this timer: it is
/// re-armed with a fresh window instead of reported.
///
/// Times are the injected monotonic clock's, in µs.
final class DeadlineWatchdog {
  DeadlineWatchdog({
    required this._timeout,
    required this._lateTolerance,
    required int Function() clockMicros,
  }) : _clock = clockMicros;

  final Duration _timeout;
  final Duration _lateTolerance;
  final int Function() _clock;
  Timer? _timer;

  /// Whether a deadline is pending.
  bool get isArmed => _timer != null;

  /// Starts the deadline from now, replacing an earlier one. When it
  /// passes, [cancelIf] is asked first (the watched thing is gone: no
  /// report); then [onExpired] runs, unless the deadline fired late.
  void arm({required void Function() onExpired, bool Function()? cancelIf}) {
    _timer?.cancel();
    final armedAt = _clock();
    _timer = Timer(_timeout, () {
      _timer = null;
      if (cancelIf?.call() ?? false) return;
      if (_overdue(
        nowMicros: _clock(),
        sinceMicros: armedAt,
        delayMicros: _timeout.inMicroseconds,
        toleranceMicros: _lateTolerance.inMicroseconds,
      )) {
        arm(onExpired: onExpired, cancelIf: cancelIf);
        return;
      }
      onExpired();
    });
  }

  /// Cancels the pending deadline. Safe to call when not armed.
  void disarm() {
    _timer?.cancel();
    _timer = null;
  }
}
