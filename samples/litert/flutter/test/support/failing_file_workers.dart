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

import 'dart:isolate';

/// Worker entry points for `ModelFileOps(worker: …)` that fail the way a
/// real worker can: they run in a real isolate, so the test exercises the
/// real port protocol.

/// An uncaught error in the worker isolate (its `onError` message).
void crashingFileWorker((Object, SendPort) args) =>
    throw StateError('the worker blew up');

/// The worker's own error report, as its generic `catch` sends it.
void erroringFileWorker((Object, SendPort) args) =>
    args.$2.send(('error', 'the worker reported a failure'));

/// The worker exits without sending a result (its `onExit` message only).
void silentFileWorker((Object, SendPort) args) {}
