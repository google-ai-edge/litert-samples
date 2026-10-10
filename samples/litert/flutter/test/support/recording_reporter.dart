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

import 'package:litert_edge_demos/data/services/model_store/store_operation.dart';
import 'package:litert_edge_demos/domain/models/provisioning.dart';

/// A [FileStateReporter] that keeps every update, marked as reported at once
/// or as progress, in order.
final class RecordingReporter implements FileStateReporter {
  final List<StoreFileState> reported = [];
  final List<StoreFileState> progressed = [];

  /// Every update in order: `('report', state)` or `('progress', state)`.
  final List<(String, StoreFileState)> all = [];

  @override
  void report(StoreFileState state) {
    reported.add(state);
    all.add(('report', state));
  }

  @override
  void progress(StoreFileState state) {
    progressed.add(state);
    all.add(('progress', state));
  }
}
