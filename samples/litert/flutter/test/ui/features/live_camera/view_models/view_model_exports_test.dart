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

import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

/// Demo 3's view-model libraries re-export nothing: every type is imported
/// from the file that declares it, so no importer leans on a re-export kept
/// only for compatibility.
void main() {
  test("Demo 3's view-model libraries re-export nothing", () {
    final directory = Directory('lib/ui/features/live_camera/view_models');
    final exports = [
      for (final file in directory.listSync().whereType<File>())
        if (file.path.endsWith('.dart'))
          for (final line in file.readAsLinesSync())
            if (line.trimLeft().startsWith('export ')) '${file.path}: $line',
    ];
    expect(directory.existsSync(), isTrue);
    expect(exports, isEmpty);
  });
}
