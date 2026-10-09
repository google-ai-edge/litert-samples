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

import 'package:flutter/services.dart';
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

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  late Directory tmp;
  final bytes = testBytes(4096, seed: 21);
  final file = BundledFile(
    asset: 'assets/models/tiny.tflite',
    sizeBytes: bytes.length,
    sha256: sha256Hex(bytes),
  );

  setUp(() => tmp = Directory.systemTemp.createTempSync('bundled_files'));
  tearDown(() => tmp.deleteSync(recursive: true));

  group('where the assets are, per platform', () {
    test('macOS, iOS, Linux and Windows bundles; Android has none', () {
      expect(
        flutterAssetsDirFor(
          operatingSystem: 'macos',
          executable: '/Apps/X.app/Contents/MacOS/X',
        ),
        '/Apps/X.app/Contents/Frameworks/App.framework/Resources/'
        'flutter_assets',
      );
      expect(
        flutterAssetsDirFor(
          operatingSystem: 'ios',
          executable: '/var/containers/Bundle/Application/U/Runner.app/Runner',
        ),
        '/var/containers/Bundle/Application/U/Runner.app/Frameworks/'
        'App.framework/flutter_assets',
      );
      expect(
        flutterAssetsDirFor(
          operatingSystem: 'linux',
          executable: '/opt/app/litert_edge_demos',
        ),
        '/opt/app/data/flutter_assets',
      );
      expect(
        flutterAssetsDirFor(
          operatingSystem: 'windows',
          executable: r'C:\app\litert_edge_demos.exe',
        ),
        r'C:\app\data\flutter_assets',
      );
      expect(
        flutterAssetsDirFor(
          operatingSystem: 'android',
          executable: '/system/bin/app_process64',
        ),
        isNull,
      );
    });

    test('a store path relative to the documents directory', () {
      expect(
        relativePath(
          '/data/user/0/app/files/models/bundled/x.extract',
          from: '/data/user/0/app/app_flutter',
        ),
        '../files/models/bundled/x.extract',
      );
    });
  });

  group('in place (desktop, iOS)', () {
    BundledModelFiles macOs() => BundledModelFiles(
      operatingSystem: 'macos',
      executable: '${tmp.path}/X.app/Contents/MacOS/X',
      copier: (_, _) => fail('nothing is copied in place'),
    );

    File assetFile() => File(
      '${tmp.path}/X.app/Contents/Frameworks/App.framework/Resources/'
      'flutter_assets/${file.asset}',
    );

    test('the asset file itself is used, checked by size', () async {
      assetFile()
        ..createSync(recursive: true)
        ..writeAsBytesSync(bytes);

      final path = await macOs().pathOf(file);

      expect((path as Ok<String>).value, assetFile().path);
    });

    test('missing or the wrong size is an error, not a download', () async {
      final missing = await macOs().pathOf(file);
      expect((missing as Error<String>).error, isA<BundledFileException>());
      expect(missing.error.toString(), contains('not in this app bundle'));

      assetFile()
        ..createSync(recursive: true)
        ..writeAsBytesSync([1, 2, 3]);
      final wrong = await macOs().pathOf(file);
      expect((wrong as Error<String>).error.toString(), contains('3 bytes'));
    });
  });

  group('Android: extracted once, verified, recorded', () {
    late List<String> copies;
    late _CountingOps ops;
    Uint8List payload = bytes;
    Exception? copyFails;

    setUp(() {
      copies = [];
      ops = _CountingOps();
      payload = bytes;
      copyFails = null;
    });

    Future<void> copier(String asset, String target) async {
      copies.add(asset);
      // Half written, then the failure (a full disk).
      File(target).writeAsBytesSync(payload.sublist(0, payload.length ~/ 2));
      if (copyFails case final e?) throw e;
      File(target).writeAsBytesSync(payload);
    }

    BundledModelFiles android() => BundledModelFiles(
      operatingSystem: 'android',
      executable: '/system/bin/app_process64',
      storeRoot: () async => tmp,
      copier: copier,
      ops: ops,
    );

    File stored() => File('${tmp.path}/bundled/tiny.tflite');

    test('the first launch copies, hashes, renames and records; the next '
        'one skips the copy and the hash', () async {
      final first = await android().pathOf(file);

      expect((first as Ok<String>).value, stored().path);
      expect(stored().readAsBytesSync(), bytes);
      expect(
        File('${stored().path}.sha256').readAsStringSync(),
        startsWith(file.sha256),
      );
      expect(File('${stored().path}.extract').existsSync(), isFalse);
      expect(copies, [file.asset]);
      expect(ops.hashes, 1);

      final second = await android().pathOf(file);

      expect((second as Ok<String>).value, stored().path);
      expect(copies, hasLength(1), reason: 'skipped: extracted before');
      expect(ops.hashes, 1, reason: 'the record is trusted');
    });

    test('an interrupted copy (a .extract left behind) is started again, '
        'never used', () async {
      Directory('${tmp.path}/bundled').createSync();
      File('${stored().path}.extract').writeAsBytesSync([9, 9]);

      final path = await android().pathOf(file);

      expect(path, isA<Ok<String>>());
      expect(copies, hasLength(1));
      expect(stored().readAsBytesSync(), bytes);
    });

    test('a copy whose hash differs is refused and deleted', () async {
      payload = Uint8List.fromList(List.filled(bytes.length, 7));

      final path = await android().pathOf(file);

      final error = (path as Error<String>).error;
      expect(error, isA<BundledFileException>());
      expect(error.toString(), contains('SHA-256'));
      expect(stored().existsSync(), isFalse);
      expect(File('${stored().path}.extract').existsSync(), isFalse);
    });

    test('a short copy is refused by size before hashing', () async {
      payload = Uint8List(10);

      final path = await android().pathOf(file);

      expect((path as Error<String>).error.toString(), contains('10 bytes'));
      expect(ops.hashes, 0);
    });

    test(
      'a failed copy (a full disk) is an error and leaves no .extract',
      () async {
        copyFails = const FileSystemException('No space left on device');

        final path = await android().pathOf(file);

        expect((path as Error<String>).error.toString(), contains('No space'));
        expect(File('${stored().path}.extract').existsSync(), isFalse);
        expect(stored().existsSync(), isFalse);
      },
    );

    test('after an extraction, files of earlier builds are removed; this '
        "build's files, records and copies in progress stay", () async {
      final dir = Directory('${tmp.path}/bundled')..createSync();
      final other = const BundledFile(
        asset: 'assets/models/other.model',
        sizeBytes: 3,
        sha256: 'x',
      );
      for (final name in [
        'embedder_v1.tflite',
        'embedder_v1.tflite.sha256',
        'other.model',
        'other.model.extract',
      ]) {
        File('${dir.path}/$name').writeAsStringSync('old');
      }
      final files = BundledModelFiles(
        operatingSystem: 'android',
        executable: '/system/bin/app_process64',
        storeRoot: () async => tmp,
        copier: copier,
        ops: ops,
        known: [file, other],
      );

      expect(await files.pathOf(file), isA<Ok<String>>());

      final left = dir.listSync().map((e) => e.uri.pathSegments.last).toSet();
      expect(left, {
        'tiny.tflite',
        'tiny.tflite.sha256',
        'other.model',
        'other.model.extract',
      });
    });

    test('a changed file (no matching record) is extracted again', () async {
      await android().pathOf(file);
      stored().writeAsBytesSync([1]);

      final again = await android().pathOf(file);

      expect(again, isA<Ok<String>>());
      expect(copies, hasLength(2));
      expect(stored().readAsBytesSync(), bytes);
    });
  });

  group('largeFileHandlerCopier: the call large_file_handler 0.5.2 gets', () {
    const documents =
        '/data/user/0/com.google.ai.edge.examples.litert_edge_demos/app_flutter';
    const target =
        '/data/user/0/com.google.ai.edge.examples.litert_edge_demos/files/models/bundled/'
        'embeddinggemma.tflite.extract';
    late List<MethodCall> calls;

    setUp(() {
      calls = [];
      final messenger =
          TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger;
      messenger.setMockMethodCallHandler(
        const MethodChannel('plugins.flutter.io/path_provider'),
        (call) async => call.method == 'getApplicationDocumentsDirectory'
            ? documents
            : throw MissingPluginException(call.method),
      );
      messenger.setMockMethodCallHandler(
        const MethodChannel('large_file_handler'),
        (call) async {
          calls.add(call);
          return null;
        },
      );
    });

    tearDown(() {
      final messenger =
          TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger;
      messenger
        ..setMockMethodCallHandler(
          const MethodChannel('plugins.flutter.io/path_provider'),
          null,
        )
        ..setMockMethodCallHandler(
          const MethodChannel('large_file_handler'),
          null,
        );
    });

    test('the asset key reaches Kotlin once prefixed (the plugin adds '
        '"assets/"), and the target joined onto the documents directory '
        'is the absolute target', () async {
      await largeFileHandlerCopier(
        'assets/models/embeddinggemma.tflite',
        target,
      );

      final call = calls.single;
      // No progress channel: the plain copy.
      expect(call.method, 'copyAssetToLocal');
      final args = call.arguments as Map<Object?, Object?>;
      // Kotlin: assetManager.open(getLookupKeyForAsset(assetName)), which
      // maps a Flutter asset key to flutter_assets/<key>.
      expect(args['assetName'], 'assets/models/embeddinggemma.tflite');
      // The plugin joins `<documents>/<targetPath>`
      // (`MethodChannelLargeFileHandler._getLocalFilePath`).
      final joined = args['targetPath']! as String;
      expect(
        joined,
        '$documents/../files/models/bundled/'
        'embeddinggemma.tflite.extract',
      );
      expect(File(joined).absolute.uri.normalizePath().path, target);
    });

    test('a key outside assets/ is refused before any call', () async {
      await expectLater(
        largeFileHandlerCopier('models/x.tflite', target),
        throwsArgumentError,
      );
      expect(calls, isEmpty);
    });
  });

  test('the real assets in the repository match the build constants', () {
    for (final f in [kBundledEmbedderModel, kBundledEmbedderTokenizer]) {
      final local = File(f.asset);
      expect(local.existsSync(), isTrue, reason: f.asset);
      expect(local.lengthSync(), f.sizeBytes, reason: f.asset);
    }
  });
}
