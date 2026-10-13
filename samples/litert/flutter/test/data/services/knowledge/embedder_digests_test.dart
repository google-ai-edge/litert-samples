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

import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/services/knowledge/embedder_digests.dart';
import 'package:litert_edge_demos/data/services/model_store/bundled_model_files.dart';
import 'package:litert_edge_demos/data/services/model_store/model_file_ops.dart';

/// Real hashing, counted.
class CountingOps extends ModelFileOps {
  CountingOps();

  final List<String> hashed = [];

  @override
  Future<String> sha256OfFile(
    String path, {
    required ByteProgress onProgress,
    Future<void>? cancel,
  }) async {
    hashed.add(path);
    return sha256.convert(await File(path).readAsBytes()).toString();
  }
}

void main() {
  late Directory dir;
  late CountingOps ops;

  /// A 5-byte "model" the build knows by name, size and SHA-256.
  final known = BundledFile(
    asset: 'assets/models/model.tflite',
    sizeBytes: 5,
    sha256: sha256.convert(utf8.encode('model')).toString(),
  );

  setUp(() {
    dir = Directory.systemTemp.createTempSync('embedder_digests_test');
    ops = CountingOps();
  });

  tearDown(() => dir.deleteSync(recursive: true));

  FileEmbedderDigests digests({
    String os = 'android',
    String executable = '/system/bin/app_process',
  }) => FileEmbedderDigests(
    cacheDir: () async => Directory('${dir.path}/cache'),
    operatingSystem: os,
    executable: executable,
    ops: ops,
    known: [known],
  );

  File write(String path, String text) => File(path)
    ..parent.createSync(recursive: true)
    ..writeAsStringSync(text);

  test('Android\'s extracted copy: the build\'s SHA-256 from its record, '
      'nothing hashed', () async {
    final model = write('${dir.path}/bundled/model.tflite', 'model');
    write('${model.path}.sha256', '${known.sha256}  model.tflite\n');

    expect(await digests().digestOf(model.path), known.sha256);
    expect(ops.hashed, isEmpty);
  });

  test('a record that does not match the size is not trusted', () async {
    final model = write('${dir.path}/bundled/model.tflite', 'model!');
    write('${model.path}.sha256', '${known.sha256}  model.tflite\n');

    final hex = await digests().digestOf(model.path);

    expect(hex, sha256.convert(utf8.encode('model!')).toString());
    expect(ops.hashed, [model.path]);
  });

  test(
    'the asset in place in a desktop bundle: the build\'s SHA-256',
    () async {
      final app = '${dir.path}/X.app/Contents';
      final model = write(
        '$app/Frameworks/App.framework/Resources/flutter_assets/'
            'assets/models/model.tflite',
        'model',
      );

      final hex = await digests(
        os: 'macos',
        executable: '$app/MacOS/X',
      ).digestOf(model.path);

      expect(hex, known.sha256);
      expect(ops.hashed, isEmpty);
    },
  );

  test('any other file is hashed once, then cached by size and mtime; a '
      'changed file is hashed again', () async {
    final model = write('${dir.path}/dev/model.tflite', 'model');

    final first = await digests().digestOf(model.path);
    final second = await digests().digestOf(model.path);

    expect(first, known.sha256, reason: 'the same bytes, found by hashing');
    expect(second, first);
    expect(ops.hashed, [model.path], reason: 'the second came from the cache');
    expect(
      File('${dir.path}/cache/embedder_digests.json').existsSync(),
      isTrue,
    );

    model.writeAsStringSync('other bytes');
    final third = await digests().digestOf(model.path);
    expect(third, sha256.convert(utf8.encode('other bytes')).toString());
    expect(ops.hashed, [model.path, model.path]);
  });

  test('an unreadable cache is hashed again, not trusted', () async {
    final model = write('${dir.path}/dev/model.tflite', 'model');
    write('${dir.path}/cache/embedder_digests.json', '{not json');

    expect(await digests().digestOf(model.path), known.sha256);
    expect(ops.hashed, [model.path]);
  });

  test('of() digests the model and the tokenizer', () async {
    final model = write('${dir.path}/dev/model.tflite', 'model');
    final tokenizer = write('${dir.path}/dev/sentencepiece.model', 'tok');

    final digest = await digests().of(
      modelPath: model.path,
      tokenizerPath: tokenizer.path,
    );

    expect(digest.model, known.sha256);
    expect(digest.tokenizer, sha256.convert(utf8.encode('tok')).toString());
  });
}
