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
import 'package:litert_edge_demos/data/services/model_store/bundled_model_files.dart';
import 'package:litert_edge_demos/data/services/model_store/model_file_ops.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../../support/test_bytes.dart';

/// Counts hashes.
final class _CountingOps extends ModelFileOps {
  int hashes = 0;

  @override
  Future<String> sha256OfFile(
    String path, {
    required ByteProgress onProgress,
    Future<void>? cancel,
  }) {
    hashes++;
    return super.sha256OfFile(path, onProgress: onProgress, cancel: cancel);
  }
}

/// Characterization of the Android extraction's `.sha256` record: what is
/// written, which records are trusted, and that an old record that cannot
/// be deleted fails the extraction (the model store only logs that case).
void main() {
  late Directory tmp;
  late List<String> copies;
  late _CountingOps ops;
  final bytes = testBytes(4096, seed: 23);
  final sha = sha256Hex(bytes);
  final file = BundledFile(
    asset: 'assets/models/tiny.tflite',
    sizeBytes: bytes.length,
    sha256: sha,
  );

  setUp(() {
    tmp = Directory.systemTemp.createTempSync('bundled_record');
    copies = [];
    ops = _CountingOps();
  });
  tearDown(() => tmp.deleteSync(recursive: true));

  BundledModelFiles android() => BundledModelFiles(
    operatingSystem: 'android',
    executable: '/system/bin/app_process64',
    storeRoot: () async => tmp,
    copier: (asset, target) async {
      copies.add(asset);
      File(target).writeAsBytesSync(bytes);
    },
    ops: ops,
  );

  File stored() => File('${tmp.path}/bundled/tiny.tflite');
  File record() => File('${stored().path}.sha256');

  /// The file in place with [content] as its record, as an earlier launch
  /// left it.
  void extractedBefore(String content) {
    stored()
      ..createSync(recursive: true)
      ..writeAsBytesSync(bytes);
    record().writeAsStringSync(content);
  }

  test('the record is "<hex>  <name>" and a newline', () async {
    expect(await android().pathOf(file), isA<Ok<String>>());

    expect(record().readAsStringSync(), '$sha  tiny.tflite\n');
  });

  test('a record of the hash alone, blanks around it, is trusted', () async {
    extractedBefore('  $sha ');

    expect(await android().pathOf(file), isA<Ok<String>>());

    expect(copies, isEmpty);
    expect(ops.hashes, 0);
  });

  for (final (what, content) in [
    ('for other bytes', '${'0' * 64}  tiny.tflite\n'),
    ('empty', ''),
    ('blank', ' \n'),
  ]) {
    test('a record $what (the size right): extracted again, the record '
        'rewritten', () async {
      extractedBefore(content);

      expect(await android().pathOf(file), isA<Ok<String>>());

      expect(copies, hasLength(1));
      expect(record().readAsStringSync(), '$sha  tiny.tflite\n');
    });
  }

  test('no record (the size right): extracted again', () async {
    stored()
      ..createSync(recursive: true)
      ..writeAsBytesSync(bytes);

    expect(await android().pathOf(file), isA<Ok<String>>());

    expect(copies, hasLength(1));
    expect(record().readAsStringSync(), '$sha  tiny.tflite\n');
  });

  test('an old record that cannot be deleted fails the extraction; the file in '
      'place is not replaced, no .extract left', () async {
    expect(await android().pathOf(file), isA<Ok<String>>());
    stored().writeAsBytesSync([1]);
    Process.runSync('chflags', ['uchg', record().path]);
    addTearDown(() => Process.runSync('chflags', ['nouchg', record().path]));

    final result = await android().pathOf(file);

    expect((result as Error<String>).error, isA<FileSystemException>());
    expect(stored().readAsBytesSync(), [1]);
    expect(record().readAsStringSync(), '$sha  tiny.tflite\n');
    expect(File('${stored().path}.extract').existsSync(), isFalse);
  }, skip: Platform.isMacOS ? false : 'chflags uchg is macOS only');
}
