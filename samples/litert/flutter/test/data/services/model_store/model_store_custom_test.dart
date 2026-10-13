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
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/services/model_store/model_file_ops.dart';
import 'package:litert_edge_demos/data/services/model_store/model_store.dart';
import 'package:litert_edge_demos/domain/models/chat_model.dart';
import 'package:litert_edge_demos/domain/models/provisioning.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../../support/fake_model_server.dart';
import '../../../support/test_bytes.dart';

/// Counts hashes, so a rescan can be shown to hash nothing.
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

void main() {
  late FakeModelServer server;
  late Directory root;
  late Directory picked;
  late _CountingOps ops;
  final stores = <ModelStore>[];

  const name = 'gemma3-1b_ekv1280_sm8750.litertlm';
  final bytes = testBytes(300 * 1024, seed: 11);
  final sha = sha256Hex(bytes);

  setUp(() async {
    server = await FakeModelServer.start();
    root = Directory.systemTemp.createTempSync('custom_store_root');
    picked = Directory.systemTemp.createTempSync('custom_store_picked');
    ops = _CountingOps();
  });

  tearDown(() async {
    for (final store in stores) {
      await store.close();
    }
    stores.clear();
    await server.close();
    root.deleteSync(recursive: true);
    picked.deleteSync(recursive: true);
  });

  ModelStore newStore() {
    final store = ModelStore(
      root: () async => root,
      ops: ops,
      retryDelay: (_) => Duration.zero,
    );
    stores.add(store);
    return store;
  }

  File customFile(String file) => File('${root.path}/custom/$file');

  File pick(String file, Uint8List content) =>
      File('${picked.path}/$file')..writeAsBytesSync(content);

  group('import', () {
    test('stores the file, computes and records its checksum', () async {
      final store = newStore();
      final source = pick(name, bytes);

      final result = await store.importCustomFile(source.path);

      final file = (result as Ok<CustomModelFile>).value;
      expect(file.name, name);
      expect(file.sizeBytes, bytes.length);
      expect(file.sha256, sha);
      expect(file.checksumMatched, isFalse, reason: 'nothing to compare with');
      expect(customFile(name).readAsBytesSync(), bytes);
      expect(customFile('$name.sha256').readAsStringSync(), startsWith(sha));
      expect(customFile('$name.import').existsSync(), isFalse);
      expect(store.customFile.value, isA<StoreFileReady>());
      expect(source.existsSync(), isTrue, reason: "the user's file stays");
    });

    test('a checksum the user entered must match', () async {
      final store = newStore();
      final source = pick(name, bytes);

      final ok = await store.importCustomFile(source.path, expectedSha256: sha);
      expect((ok as Ok<CustomModelFile>).value.checksumMatched, isTrue);

      final other = pick('other.litertlm', testBytes(1024, seed: 3));
      final bad = await store.importCustomFile(other.path, expectedSha256: sha);

      final error = (bad as Error<CustomModelFile>).error;
      expect(error, isA<ChecksumMismatchException>());
      expect(error.toString(), contains('the checksum you entered expects'));
      expect(customFile('other.litertlm').existsSync(), isFalse);
      expect(
        store.customFile.value,
        isA<StoreFileReady>(),
        reason: 'a failed import leaves the previous model in place',
      );
      expect((store.customFile.value as StoreFileReady).path, endsWith(name));
    });

    // A worker that fails the import: model_store_test.dart (wiring) and
    // model_file_ops_test.dart (a crash, an error, no result).

    test('only .litertlm files are taken', () async {
      final store = newStore();
      final source = pick('model.tflite', bytes);

      final result = await store.importCustomFile(source.path);

      final error = (result as Error<CustomModelFile>).error;
      expect(error, isA<CustomModelFileException>());
      expect(error.toString(), contains('.litertlm'));
      expect(Directory('${root.path}/custom').existsSync(), isFalse);
    });

    test("the picker's temporary copy is moved, not copied", () async {
      final store = newStore();
      final inbox = Directory('${picked.path}/Inbox')..createSync();
      final source = File('${inbox.path}/$name')..writeAsBytesSync(bytes);

      final result = await store.importCustomFile(
        source.path,
        moveFrom: inbox.path,
      );

      expect(result, isA<Ok<CustomModelFile>>());
      expect(source.existsSync(), isFalse);
      expect(customFile(name).readAsBytesSync(), bytes);
    });
  });

  group('download', () {
    test('without size or checksum: the size is probed, the checksum '
        'recorded', () async {
      server.files[name] = ServedFile(bytes);
      final store = newStore();

      final result = await store.downloadCustomFile(server.url(name));

      final file = (result as Ok<CustomModelFile>).value;
      expect(file.name, name);
      expect(file.sizeBytes, bytes.length);
      expect(file.sha256, sha);
      expect(file.checksumMatched, isFalse);
      expect(customFile(name).readAsBytesSync(), bytes);
      expect(customFile('$name.sha256').existsSync(), isTrue);
      expect(
        Directory('${root.path}/custom')
            .listSync()
            .where((e) => e.path.endsWith('.part')),
        isEmpty,
      );
    });

    test('a wrong checksum fails and nothing is stored', () async {
      server.files[name] = ServedFile(bytes);
      final store = newStore();

      final result = await store.downloadCustomFile(
        server.url(name),
        sha256: sha256Hex([1, 2, 3]),
        sizeBytes: bytes.length,
      );

      expect((result as Error).error, isA<ChecksumMismatchException>());
      expect(customFile(name).existsSync(), isFalse);
      expect(store.customFile.value, isA<StoreFileMissing>());
    });

    // A dropped connection resumed with a Range request: model_store_test.dart
    // (the default part file) and http_file_downloader_test.dart.

    test('an HTML page instead of the file is named as such', () async {
      server.files[name] = ServedFile(
        Uint8List.fromList('<!doctype html><p>quota</p>'.codeUnits),
        contentType: 'text/html',
      );
      final store = newStore();

      final result = await store.downloadCustomFile(server.url(name));

      expect((result as Error).error, isA<HtmlInsteadOfFileException>());
    });

    test('HTTP 404 is shown', () async {
      final store = newStore();

      final result = await store.downloadCustomFile(server.url('missing'));

      final error = (result as Error<CustomModelFile>).error;
      expect(error, isA<DownloadHttpException>());
      expect(error.toString(), contains('404'));
    });

    test('a link without a file name gets a generic one, distinct per '
        'link (a second Drive link must not land on the file the '
        'engine runs)', () {
      String nameOf(String url) => ModelStore.customFileNameFor(Uri.parse(url));
      const drive =
          'https://drive.usercontent.google.com/download?id=abc&export=download';
      expect(
        nameOf(drive),
        matches(RegExp(r'^custom-model-[0-9a-f]{8}\.litertlm$')),
      );
      expect(nameOf(drive), nameOf(drive), reason: 'the same link resumes');
      expect(nameOf(drive.replaceFirst('abc', 'xyz')), isNot(nameOf(drive)));
      expect(
        ModelStore.customFileNameFor(
          Uri.parse('https://huggingface.co/x/y/resolve/main/$name'),
        ),
        name,
      );
    });
  });

  group('rescan', () {
    test(
      'a stored file is found by size and record, hashing nothing',
      () async {
        final first = newStore();
        final stored = (await first.importCustomFile(
          pick(name, bytes).path,
        ) as Ok<CustomModelFile>).value;
        final hashesBefore = ops.hashes;

        final second = newStore();
        final path = await second.useCustomFile(stored);

        expect((path as Ok<String?>).value, customFile(name).path);
        expect(second.customFile.value, isA<StoreFileReady>());
        expect(ops.hashes, hashesBefore, reason: 'the record is trusted');
      },
    );

    test('a changed file is unverified; a deleted one missing', () async {
      final store = newStore();
      final stored = (await store.importCustomFile(
        pick(name, bytes).path,
      ) as Ok<CustomModelFile>).value;

      customFile(name).writeAsBytesSync([1, 2, 3]);
      expect((await store.useCustomFile(stored) as Ok<String?>).value, isNull);
      expect(store.customFile.value, isA<StoreFileUnverified>());

      customFile(name).deleteSync();
      expect((await store.useCustomFile(stored) as Ok<String?>).value, isNull);
      expect(store.customFile.value, isA<StoreFileMissing>());
    });

    test('prune keeps only the named models and their records', () async {
      final store = newStore();
      await store.importCustomFile(pick('old.litertlm', testBytes(64)).path);
      await store.importCustomFile(
        pick('running.litertlm', testBytes(65)).path,
      );
      await store.importCustomFile(pick(name, bytes).path);

      await store.pruneCustom(keep: {name, 'running.litertlm'});

      final left = Directory(
        '${root.path}/custom',
      ).listSync().map((e) => e.uri.pathSegments.last).toList()..sort();
      expect(left, [
        name,
        '$name.sha256',
        'running.litertlm',
        'running.litertlm.sha256',
      ]);
    });
  });
}
