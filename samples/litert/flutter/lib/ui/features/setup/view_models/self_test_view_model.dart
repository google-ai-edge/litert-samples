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

import '../../../../domain/models/self_test.dart';
import '../../../../domain/ports/self_test_launcher.dart';
import '../../../../utils/command.dart';
import '../../../../utils/result.dart';

/// The Models screen's "Run self-test": runs the `--selftest` steps in the
/// app and shows the report with a Copy button. Not while models load or a
/// download runs (the run unloads the chat model).
class SelfTestViewModel extends ChangeNotifier {
  SelfTestViewModel({required this._launcher, this._blockers = const []}) {
    run = Command0<SelfTestOutcome>(_run)..addListener(notifyListeners);
    for (final blocker in _blockers) {
      blocker.addListener(notifyListeners);
    }
  }

  final SelfTestLauncher _launcher;
  final List<ValueListenable<bool>> _blockers;
  final ValueNotifier<List<String>> _progress = ValueNotifier(const []);
  bool _disposed = false;

  late final Command0<SelfTestOutcome> run;

  /// The running test's progress lines (`step 4a … PASS`).
  ValueListenable<List<String>> get progress => _progress;

  bool get canRun =>
      !run.running && !_launcher.stuck && !_blockers.any((b) => b.value);

  /// Why "Run self-test" is disabled while it does not run; null when it
  /// can run.
  String? get blockedReason {
    if (run.running || canRun) return null;
    if (_launcher.stuck) {
      // Whether the chat model is usable meanwhile, its own row says
      // (kSelfTestStillRunning when the run may hold its engine).
      return 'The last self-test did not end in time: restart the app before '
          'running it again.';
    }
    return 'Wait until the models have loaded, no download runs and the chat '
        'model is not being switched.';
  }

  /// The last run's report.
  SelfTestOutcome? get outcome => switch (run.result) {
    Ok(:final value) => value,
    _ => null,
  };

  /// Why the last run could not produce a report.
  String? get error => switch (run.result) {
    Error(:final error) => 'The self-test could not run: $error',
    _ => null,
  };

  Future<Result<SelfTestOutcome>> _run() {
    _progress.value = const [];
    return _launcher.run(
      progress: (line) {
        if (!_disposed) _progress.value = [..._progress.value, line];
      },
    );
  }

  @override
  void dispose() {
    _disposed = true;
    for (final blocker in _blockers) {
      blocker.removeListener(notifyListeners);
    }
    run
      ..removeListener(notifyListeners)
      ..dispose();
    _progress.dispose();
    super.dispose();
  }
}
