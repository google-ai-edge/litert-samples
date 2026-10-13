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
import 'package:litert_edge_demos/data/services/model_store/checksum_record.dart';

import '../../../support/test_bytes.dart';

/// The `.sha256` record both stores of verified files write and read.
void main() {
  late Directory dir;

  const name = 'model.litertlm';
  final sha = sha256Hex(testBytes(100, seed: 43));

  setUp(() => dir = Directory.systemTemp.createTempSync('checksum_record'));
  tearDown(() => dir.deleteSync(recursive: true));

  File target() => File('${dir.path}/$name');
  File record() => File('${target().path}.sha256');

  test('is <file>.sha256 holding "<hex>  <name>" and a newline', () async {
    expect(ChecksumRecord.of(target()).path, '${target().path}.sha256');

    await ChecksumRecord.write(target(), sha, name);

    expect(record().readAsStringSync(), '$sha  $name\n');
    expect(await ChecksumRecord.read(target()), sha);
  });

  test('writing replaces an earlier record', () async {
    record().writeAsStringSync('${'0' * 64}  old name and more\n');

    await ChecksumRecord.write(target(), sha, name);

    expect(record().readAsStringSync(), '$sha  $name\n');
  });

  test('reads the first word; null when missing or blank', () async {
    expect(await ChecksumRecord.read(target()), isNull);

    for (final (content, hash) in [
      ('', null),
      (' \n\t', null),
      ('  $sha', sha),
      ('$sha\tother words\n', sha),
    ]) {
      record().writeAsStringSync(content);
      expect(await ChecksumRecord.read(target()), hash, reason: content);
    }
  });
}
