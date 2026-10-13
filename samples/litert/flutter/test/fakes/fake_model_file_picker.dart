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

import 'package:litert_edge_demos/domain/ports/model_file_picker.dart';

/// [ModelFilePicker] that returns what the test set.
final class FakeModelFilePicker implements ModelFilePicker {
  FakeModelFilePicker({
    this.support = const ImportFromFolder(),
    this.temporaryCopies,
    this.modelFile,
  });

  @override
  ImportSupport support;
  String? temporaryCopies;

  /// What [pickModelFile] returns (null: cancelled).
  String? modelFile;
  int picks = 0;

  /// When set, every pick throws it (a picker that cannot open).
  Exception? error;

  @override
  Future<String?> pickModelFile() async {
    picks++;
    if (error case final e?) throw e;
    return modelFile;
  }

  @override
  Future<String?> temporaryCopiesDirectory() async => temporaryCopies;
}
