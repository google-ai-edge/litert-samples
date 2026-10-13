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

// A port: an interface that the domain and the view models depend on and an
// outer layer implements. Every port lives in lib/domain/ports/;
// function-typed dependencies stay next to their one consumer.

import '../../utils/result.dart';
import '../models/self_test.dart';

/// What the Models screen's "Run self-test" runs (`InAppSelfTest`; tests
/// use a fake).
abstract interface class SelfTestLauncher {
  /// A run passed its time limit and never ended (`kSelfTestStillRunning`):
  /// no other run may start, nor the chat model load, until the app
  /// restarts.
  bool get stuck;

  Future<Result<SelfTestOutcome>> run({
    required void Function(String line) progress,
  });
}
