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

/// How a wait from [whenFalse] or [waitFor] ended.
enum WaitEnd {
  /// What was waited for happened.
  done,

  /// `closed` completed first: the waiter's owner was disposed or closed.
  closed,

  /// The timeout passed first.
  timedOut,
}

/// Completes when [flag] is false: at once when it already is, else at the
/// notification that turns it false.
///
/// A disposed notifier never notifies again, so a wait on it alone can
/// hang: this one also ends when [closed] completes (the waiter's owner was
/// disposed or closed) and, with a [timeout], once that passes. The
/// listener is removed however the wait ends.
Future<WaitEnd> whenFalse(
  ValueListenable<bool> flag, {
  Future<void>? closed,
  Duration? timeout,
}) async {
  if (!flag.value) return WaitEnd.done;
  final turnedFalse = Completer<void>();
  void listener() {
    if (!flag.value && !turnedFalse.isCompleted) turnedFalse.complete();
  }

  flag.addListener(listener);
  try {
    return await waitFor(turnedFalse.future, closed: closed, timeout: timeout);
  } finally {
    flag.removeListener(listener);
  }
}

/// Completes when [future] does, or when [closed] completes first (the
/// waiter's owner was disposed or closed), or once a [timeout] passes.
///
/// An error of [future] that ends the wait is rethrown; one that comes
/// after the wait ended is logged.
Future<WaitEnd> waitFor(
  Future<void> future, {
  Future<void>? closed,
  Duration? timeout,
}) {
  final ended = Completer<WaitEnd>();
  Timer? timer;
  void end(WaitEnd how) {
    if (ended.isCompleted) return;
    timer?.cancel();
    ended.complete(how);
  }

  unawaited(
    future.then(
      (_) => end(WaitEnd.done),
      onError: (Object error, StackTrace stack) {
        if (ended.isCompleted) {
          debugPrint('[wait] failed after the wait ended: $error\n$stack');
          return;
        }
        timer?.cancel();
        ended.completeError(error, stack);
      },
    ),
  );
  if (closed != null) unawaited(closed.then((_) => end(WaitEnd.closed)));
  if (timeout != null) timer = Timer(timeout, () => end(WaitEnd.timedOut));
  return ended.future;
}
