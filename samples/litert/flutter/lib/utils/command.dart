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

import 'result.dart';

typedef CommandAction0<T> = Future<Result<T>> Function();
typedef CommandAction1<T, A> = Future<Result<T>> Function(A);

/// An async user action with observable state, in the style of compass_app.
///
/// The view listens to [running], [error] and [completed]; the action itself
/// returns a [Result]. A second [execute] while one is running is ignored.
abstract class Command<T> extends ChangeNotifier {
  bool _running = false;
  bool _disposed = false;
  Result<T>? _result;

  /// True while the action runs.
  bool get running => _running;

  /// True when the last run returned [Result.error].
  bool get error => _result is Error<T>;

  /// True when the last run returned [Result.ok].
  bool get completed => _result is Ok<T>;

  /// The last run's result, or null before the first run finishes.
  Result<T>? get result => _result;

  Future<void> _execute(CommandAction0<T> action) async {
    if (_running) return;
    _running = true;
    _result = null;
    _notify();
    try {
      _result = await action();
    } catch (e, st) {
      // Actions return Results; anything that escapes one (an Exception, or
      // an Error such as a plugin's StateError) is still a failure the view
      // must see, not an unhandled zone error and not a null result.
      debugPrint('[Command] the action threw: $e\n$st');
      if (e is! Exception) {
        // An Error is a bug (a TypeError, a RangeError, a failed assert):
        // reported as well, so it is not lost behind the failure the view
        // shows, and a test fails on it as it did when it escaped.
        FlutterError.reportError(
          FlutterErrorDetails(exception: e, stack: st, library: 'command'),
        );
      }
      _result = Result.error(asException(e));
    } finally {
      _running = false;
      _notify();
    }
  }

  void _notify() {
    if (!_disposed) notifyListeners();
  }

  @override
  void dispose() {
    _disposed = true;
    super.dispose();
  }
}

/// A [Command] without an argument.
final class Command0<T> extends Command<T> {
  Command0(this._action);

  final CommandAction0<T> _action;

  Future<void> execute() => _execute(_action);
}

/// A [Command] with one argument.
final class Command1<T, A> extends Command<T> {
  Command1(this._action);

  // Private, and called only through [execute], whose covariant parameter is
  // checked on entry: the pattern unsafe_variance's documentation
  // recommends. Making A truly invariant needs statically checked variance,
  // which Dart does not have yet.
  // ignore: unsafe_variance
  final CommandAction1<T, A> _action;

  Future<void> execute(A argument) => _execute(() => _action(argument));
}
