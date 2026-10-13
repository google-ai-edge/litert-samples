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

/// One finished in-app self-test.
final class const SelfTestOutcome({
  /// The delimited SELFTEST block, as `--selftest` prints it.
  required final String text,
  required final bool passed,

  /// Where the block was written; null when it could not be.
  final String? reportPath,

  /// The run passed its time limit and had still not ended after the stop
  /// grace (`kSelfTestStopGrace`): no other run starts until the app
  /// restarts.
  final bool stillRunning = false,

  /// With [stillRunning]: the run's own chat model engine may still be
  /// loaded (`SelfTestRunner.chatEngineMayBeLoaded`), so the app's is not
  /// loaded again (one engine per process). False when it hung before its
  /// chat model load or after closing it: the app's chat model loads again
  /// as usual.
  final bool chatEngineMayBeLoaded = true,
});
