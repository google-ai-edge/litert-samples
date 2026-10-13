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

/// The UI talks to repositories and domain models only: no view or view
/// model imports a service (the live camera's preview type is a domain
/// model, `domain/models/preview_source.dart`).
void main() {
  test('no file under lib/ui imports lib/data/services', () {
    final offenders = [
      for (final file in Directory('lib/ui').listSync(recursive: true))
        if (file is File && file.path.endsWith('.dart'))
          for (final line in file.readAsLinesSync())
            if (RegExp(r'''^import '.*data/services/''').hasMatch(line))
              '${file.path}: $line',
    ];
    expect(offenders, isEmpty);
  });
}
