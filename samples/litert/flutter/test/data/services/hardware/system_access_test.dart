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
import 'package:litert_edge_demos/data/services/hardware/system_access.dart';

void main() {
  test('LocalSystemFiles: sysfs-style names with ":" and links, missing '
      'paths are null/empty', () {
    final dir = Directory.systemTemp.createTempSync('sysfs');
    addTearDown(() => dir.deleteSync(recursive: true));
    final device = Directory('${dir.path}/devices/0000:00:04.0')
      ..createSync(recursive: true);
    File('${device.path}/class').writeAsStringSync('0x030200\n');
    Link('${dir.path}/bus/0000:00:04.0')
        .createSync(device.path, recursive: true);

    const files = LocalSystemFiles();
    expect(files.list('${dir.path}/bus'), ['0000:00:04.0']);
    expect(files.read('${dir.path}/bus/0000:00:04.0/class'), '0x030200\n');
    expect(files.read('${dir.path}/nope'), isNull);
    expect(files.list('${dir.path}/nope'), isEmpty);
  });

  test(
    'LocalProcessRunner: a missing tool is null, output is captured',
    () async {
      const runner = LocalProcessRunner();
      expect(await runner.run('no-such-tool-for-the-probe', const []), isNull);
      final echo = await runner.run('echo', const ['hello']);
      expect(echo!.exitCode, 0);
      expect(echo.stdout.trim(), 'hello');
    },
  );
}
