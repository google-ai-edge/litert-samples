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

import 'package:litert_edge_demos/domain/models/self_test.dart';
import 'package:litert_edge_demos/domain/ports/self_test_launcher.dart';
import 'package:litert_edge_demos/utils/result.dart';

/// [SelfTestLauncher] for screens that show the "Run self-test" card: a run
/// reports one progress line and [outcome] (by default, that it cannot run
/// in a widget test).
final class FakeSelfTestLauncher implements SelfTestLauncher {
  FakeSelfTestLauncher([Result<SelfTestOutcome>? outcome])
    : outcome =
          outcome ?? Result.error(Exception('no self-test in a widget test'));

  final Result<SelfTestOutcome> outcome;
  int runs = 0;

  /// After a run, the launcher's answer to [SelfTestLauncher.stuck].
  bool stuckAfterRun = false;

  @override
  bool stuck = false;

  @override
  Future<Result<SelfTestOutcome>> run({
    required void Function(String line) progress,
  }) async {
    runs++;
    progress('step 1 hardware probe …');
    stuck = stuckAfterRun;
    return outcome;
  }
}
