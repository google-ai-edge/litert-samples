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
import 'package:litert_edge_demos/data/services/model_store/models_folder.dart';
import 'package:litert_edge_demos/domain/models/chat_model.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../../support/fake_bundles.dart';

/// The first 32 bytes of a `.litertlm` (the magic, the container version,
/// where its header ends), and nothing after them: what the plugin's reader
/// refuses before it parses anything.
Uint8List preamble({int major = 1, required int headerEnd}) {
  final b = ByteData(32);
  for (final (i, unit) in 'LITERTLM'.codeUnits.indexed) {
    b.setUint8(i, unit);
  }
  b
    ..setUint32(8, major, Endian.little)
    ..setUint32(12, 5, Endian.little)
    ..setUint32(24, headerEnd, Endian.little);
  return b.buffer.asUint8List();
}

/// The [LocalModelException] of a failed [inspectLitertlm].
LocalModelException refusal(Result<BundleReading> result) =>
    (result as Error<BundleReading>).error as LocalModelException;

/// The plugin's reason of an [inspectLitertlm] that could not read the
/// header and goes on (the engine loads such a file).
LiteRtLmBundleException notRead(Result<BundleReading> result) =>
    ((result as Ok<BundleReading>).value as BundleNotRead).reason;

void main() {
  late Directory tmp;

  setUp(() => tmp = Directory.systemTemp.createTempSync('models_folder'));
  tearDown(() => tmp.deleteSync(recursive: true));

  group("inspectLitertlm: the plugin's readLiteRtLmBundleInfo, refused only "
      'where a load refuses', () {
    test('not a .litertlm: not refused (the engine loads it without reading '
        "it), with the plugin's message naming the path", () async {
      final file = File('${tmp.path}/x.litertlm')
        ..writeAsBytesSync(List.filled(64, 1));

      final reason = notRead(await inspectLitertlm(file.path));

      expect(reason.problem, LiteRtLmBundleProblem.notLitertlm);
      expect(
        reason.message,
        '${file.path} is not a .litertlm container (no LITERTLM magic)',
      );
    });

    test('a file that ends inside its own header: incomplete, as the plugin '
        'says it', () async {
      final file = File('${tmp.path}/cut.litertlm')
        ..writeAsBytesSync(preamble(headerEnd: 4096));

      final error = refusal(await inspectLitertlm(file.path));

      expect(error.message, contains('ends inside its own header'));
      expect(error.message, contains(file.path));
    });

    test('a container version the plugin does not read: not refused, the '
        "plugin's reason kept", () async {
      final file = File('${tmp.path}/v2.litertlm')
        ..writeAsBytesSync(preamble(major: 2, headerEnd: 64));

      final reason = notRead(await inspectLitertlm(file.path));

      expect(reason.problem, LiteRtLmBundleProblem.unsupportedVersion);
      expect(reason.message, contains('version 2'));
    });

    test('missing or unreadable: an error naming the path', () async {
      final missing = '${tmp.path}/gone.litertlm';
      final locked = File('${tmp.path}/locked.litertlm')
        ..writeAsBytesSync(List.filled(64, 7));
      Process.runSync('chmod', ['000', locked.path]);
      addTearDown(() => Process.runSync('chmod', ['644', locked.path]));

      expect(
        refusal(await inspectLitertlm(missing)).message,
        '$missing does not exist.',
      );
      final denied = refusal(await inspectLitertlm(locked.path));
      expect(denied.message, contains('cannot be read'));
      expect(denied.message, contains(locked.path));
      expect(denied.permissionDenied, isTrue);
    });

    test('a real file in a folder the app may not search: denied, not '
        '"does not exist" (File.exists says false there)', () async {
      final dir = Directory('${tmp.path}/unsearchable')..createSync();
      final file = File('${dir.path}/model.litertlm')
        ..writeAsBytesSync(List.filled(64, 7));
      Process.runSync('chmod', ['000', dir.path]);
      addTearDown(() => Process.runSync('chmod', ['755', dir.path]));
      expect(file.existsSync(), isFalse, reason: 'the trap');

      final error = refusal(await inspectLitertlm(file.path));

      expect(error.permissionDenied, isTrue);
      expect(error.message, isNot(contains('does not exist')));
    });
  });

  group('inspectLitertlm: what it does with what the plugin reports', () {
    test('a complete file: what the plugin read, as it is', () async {
      final info = bundleInfo(vision: true, audio: true, npu: kTestNpuTarget);

      final result = await inspectLitertlm(
        '/m/model.litertlm',
        read: (_) async => info,
      );

      expect(
        ((result as Ok<BundleReading>).value as BundleRead).bundle,
        same(info),
      );
    });

    test('a file shorter than its sections (isComplete false) is refused at '
        'pick time with the lengths the plugin reports', () async {
      final info = bundleInfo(
        vision: true,
        fileLength: 1000,
        expectedLength: 2538799104,
      );
      expect(info.isComplete, isFalse);

      final error = refusal(
        await inspectLitertlm('/m/half.litertlm', read: (_) async => info),
      );

      expect(
        error.message,
        '/m/half.litertlm is incomplete: it is 1000 bytes and its sections '
        'end at 2538799104 (a copy or download that stopped early). Copy or '
        'download it again.',
      );
    });

    test("the reader's incomplete (a file that ends inside its header) is "
        "refused with the plugin's message, as the load refuses it", () async {
      final error = refusal(
        await inspectLitertlm(
          '/m/cut.litertlm',
          read: (path) async => throw LiteRtLmBundleException(
            LiteRtLmBundleProblem.incomplete,
            '$path is 100 bytes and ends inside its own header (4096 bytes): '
            'the download is incomplete',
            path: path,
          ),
        ),
      );

      expect(
        error.message,
        '/m/cut.litertlm is 100 bytes and ends inside its own header (4096 '
        'bytes): the download is incomplete',
      );
    });

    test(
      'any other LiteRtLmBundleException is no refusal: the engine loads '
      "such a file without reading it, and the plugin's reason is kept",
      () async {
        for (final problem in [
          LiteRtLmBundleProblem.notLitertlm,
          LiteRtLmBundleProblem.unsupportedVersion,
          LiteRtLmBundleProblem.malformed,
        ]) {
          final reason = notRead(
            await inspectLitertlm(
              '/m/bad.litertlm',
              read: (path) async => throw LiteRtLmBundleException(
                problem,
                '$path: the .litertlm header is out of range',
                path: path,
              ),
            ),
          );

          expect(reason.problem, problem);
          expect(
            reason.message,
            '/m/bad.litertlm: the .litertlm header is out of range',
          );
        }
      },
    );

    FileSystemException osError(String path, int errno, String message) =>
        FileSystemException('Cannot open file', path, OSError(message, errno));

    test('ENOENT: does not exist', () async {
      final error = refusal(
        await inspectLitertlm(
          '/x/gone.litertlm',
          read: (path) async =>
              throw osError(path, 2, 'No such file or directory'),
        ),
      );
      expect(error.message, '/x/gone.litertlm does not exist.');
      expect(error.permissionDenied, isFalse);
    });

    test(
      'EACCES: cannot be read, permission denied (the advice can fire)',
      () async {
        final error = refusal(
          await inspectLitertlm(
            '/x/locked.litertlm',
            read: (path) async => throw osError(path, 13, 'Permission denied'),
          ),
        );
        expect(error.message, contains('cannot be read (Permission denied)'));
        expect(error.permissionDenied, isTrue);
      },
    );
  });

  group('ModelsFolder', () {
    test('created on first use; lists only .litertlm files, by name, with '
        'size and date', () async {
      final dir = Directory('${tmp.path}/models');
      final folder = ModelsFolder(directory: () async => dir);

      expect((await folder.list() as Ok).value, isEmpty);
      expect(dir.existsSync(), isTrue);

      writeFakeLitertlm('${dir.path}/b.litertlm', FakeBundle.textOnlyNpu);
      final a = writeFakeLitertlm(
        '${dir.path}/a.LITERTLM',
        FakeBundle.gemma4E2b,
      );
      File('${dir.path}/notes.txt').writeAsStringSync('x');
      File('${dir.path}/.hidden.litertlm').writeAsStringSync('x');
      Directory('${dir.path}/sub.litertlm').createSync();

      final entries = (await folder.list() as Ok<List<LocalModelEntry>>).value;

      expect(entries.map((e) => e.name), ['a.LITERTLM', 'b.litertlm']);
      expect(entries.first.sizeBytes, a.lengthSync());
      expect(entries.first.path, '${dir.path}/a.LITERTLM');
      expect(
        entries.first.modified.difference(DateTime.now()).inMinutes.abs(),
        lessThan(5),
      );
    });

    test('a folder the app does not create and may not read is an error with '
        'the reason, not an exception', () async {
      final parent = Directory('${tmp.path}/locked')..createSync();
      final dir = Directory('${parent.path}/models')..createSync();
      Process.runSync('chmod', ['000', dir.path]);
      addTearDown(() => Process.runSync('chmod', ['755', dir.path]));
      final folder = ModelsFolder(directory: () async => dir, create: false);

      final r = await folder.list();

      expect('${(r as Error).error}', contains('cannot be read'));
    });

    test('a folder that cannot be created is an error, not a crash', () async {
      final blocker = File('${tmp.path}/file')..writeAsStringSync('x');
      final folder = ModelsFolder(
        directory: () async => Directory('${blocker.path}/models'),
      );

      expect(await folder.list(), isA<Error<List<LocalModelEntry>>>());
    });
  });

  group('homeModelsPath: the default folder under the home on Linux and '
      'Windows', () {
    test(r'Windows: %USERPROFILE%\litert-demos\models, not Documents (which '
        'OneDrive may redirect)', () {
      expect(
        homeModelsPath(
          operatingSystem: 'windows',
          environment: const {
            'USERPROFILE': r'C:\Users\me',
            'HOME': '/c/Users/me',
          },
        ),
        r'C:\Users\me\litert-demos\models',
      );
    });

    test('Linux: \$HOME/litert-demos/models', () {
      expect(
        homeModelsPath(
          operatingSystem: 'linux',
          environment: const {'HOME': '/home/me'},
        ),
        '/home/me/litert-demos/models',
      );
    });

    test('the variable unset or empty is an error naming it', () {
      expect(
        () => homeModelsPath(operatingSystem: 'windows', environment: const {}),
        throwsA(
          isA<LocalModelException>().having(
            (e) => e.message,
            'message',
            'USERPROFILE is not set.',
          ),
        ),
      );
      expect(
        () => homeModelsPath(
          operatingSystem: 'linux',
          environment: const {'HOME': ''},
        ),
        throwsA(isA<LocalModelException>()),
      );
    });

    test("macOS, iOS and Android: none (the app's own folder)", () {
      for (final os in ['macos', 'ios', 'android']) {
        expect(
          homeModelsPath(
            operatingSystem: os,
            environment: const {'HOME': '/Users/me'},
          ),
          isNull,
          reason: os,
        );
      }
    });
  });
}
