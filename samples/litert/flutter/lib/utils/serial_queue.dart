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

/// Runs asynchronous operations one at a time, in the order they were
/// queued: each starts once the one queued before it has ended, whether that
/// one succeeded or failed. An operation's error goes to its own caller (and
/// to [onError]); the queue goes on.
///
/// Who wins when requests overlap (a newer request superseding a queued one)
/// is the owner's rule, checked inside the operation; the queue only orders.
/// The constructors differ only in when an operation queued while none is
/// pending starts.
final class SerialQueue {
  /// Every operation starts on a later microtask, after the code that queued
  /// it: a request made right after can still supersede it. The first one
  /// waits on a completed future made here, so it starts in the zone the
  /// queue was made in (a repository made outside a widget test's fake zone
  /// runs its first operation in real time).
  SerialQueue({this.onError})
    : _tail = Future<void>.value(),
      _atOnceWhenIdle = false;

  /// Like [SerialQueue.new], except that the very first operation starts at
  /// once, inside [run]. No future is made here: one made in a widget test's
  /// fake zone would hold up every later operation.
  SerialQueue.firstAtOnce({this.onError})
    : _tail = null,
      _atOnceWhenIdle = false;

  /// An operation queued while none is pending starts at once, inside
  /// [run]; one queued behind another waits for it.
  SerialQueue.atOnceWhenIdle({this.onError})
    : _tail = null,
      _atOnceWhenIdle = true;

  /// Sees each failed operation, typically to log it; the error still
  /// completes that operation's future. With it, the error counts as
  /// handled: a caller that drops the future (`unawaited`) gets no uncaught
  /// error, one that awaits it still gets the error.
  final void Function(Object error, StackTrace stack)? onError;
  final bool _atOnceWhenIdle;

  /// Completes when the last operation queued so far has ended, never with
  /// an error; null before the first one ([SerialQueue.firstAtOnce]) or
  /// while none is pending ([SerialQueue.atOnceWhenIdle]).
  Future<void>? _tail;

  /// Completes once every operation queued so far has ended; never with an
  /// error.
  Future<void> get idle => _tail ?? Future<void>.value();

  /// Queues [op]; the returned future completes with its result or error.
  Future<T> run<T>(Future<T> Function() op) {
    final previous = _tail;
    final done = Completer<void>();
    final tail = _tail = done.future;
    Future<T> body() async {
      try {
        return await op();
      } catch (e, st) {
        onError?.call(e, st);
        rethrow;
      } finally {
        if (_atOnceWhenIdle && identical(_tail, tail)) _tail = null;
        done.complete();
      }
    }

    final result = previous == null ? body() : previous.then((_) => body());
    if (onError != null) result.ignore();
    return result;
  }
}
