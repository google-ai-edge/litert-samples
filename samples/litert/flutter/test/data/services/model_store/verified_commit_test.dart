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
import 'dart:isolate';

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/services/model_store/checksum_record.dart';
import 'package:litert_edge_demos/data/services/model_store/model_file_ops.dart';
import 'package:litert_edge_demos/data/services/model_store/store_operation.dart';
import 'package:litert_edge_demos/data/services/model_store/verified_commit.dart';
import 'package:litert_edge_demos/domain/models/provisioning.dart';

import '../../../support/recording_reporter.dart';
import '../../../support/test_bytes.dart';

/// A hash that never ends until its isolate is killed (a cancel).
void _hangingFileWorker((Object, SendPort) args) {
  ReceivePort(); // an open port keeps the isolate alive
}

/// Counts hashes.
final class _CountingOps extends ModelFileOps {
  _CountingOps() : super(progressInterval: Duration.zero);

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

/// [VerifiedCommit] on its own: the hash in a worker, the size and checksum
/// checks, and the order of the commit (old record, rename, new record).
void main() {
  late Directory dir;
  late RecordingReporter reporter;
  late _CountingOps ops;
  late VerifiedCommit commit;

  const name = 'model.litertlm';
  final bytes = testBytes(64 * 1024, seed: 41);
  final sha = sha256Hex(bytes);
  final n = bytes.length;

  setUp(() {
    dir = Directory.systemTemp.createTempSync('verified_commit');
    reporter = RecordingReporter();
    ops = _CountingOps();
    commit = VerifiedCommit(ops);
  });
  tearDown(() => dir.deleteSync(recursive: true));

  File target() => File('${dir.path}/$name');
  File temp() => File('${dir.path}/$name.part');
  File record() => File('${target().path}.sha256');

  ExpectedFile expected({String? sha256, int? sizeBytes}) => ExpectedFile(
    name: name,
    sizeBytes: sizeBytes ?? n,
    sha256: sha256,
    source: 'https://example.com/$name',
  );

  Matcher ready() =>
      isA<StoreFileReady>().having((s) => s.path, 'path', target().path);

  group('hash', () {
    test('reports Verifying at once, then its progress; returns the '
        'SHA-256', () async {
      temp().writeAsBytesSync(bytes);

      final hex = await commit.hash(name, n, temp(), CancelToken(), reporter);

      expect(hex, sha);
      expect(reporter.reported, [
        isA<StoreFileVerifying>()
            .having((s) => s.processed, 'processed', 0)
            .having((s) => s.total, 'total', n),
      ]);
      expect(
        reporter.progressed.last,
        isA<StoreFileVerifying>().having((s) => s.processed, 'processed', n),
      );
    });

    test('a cancel kills the worker', () async {
      temp().writeAsBytesSync(bytes);
      final cancel = CancelToken();
      final hashing = const VerifiedCommit(
        ModelFileOps(worker: _hangingFileWorker),
      ).hash(name, n, temp(), cancel, reporter);

      cancel.cancel();

      await expectLater(hashing, throwsA(isA<OperationCancelledException>()));
    });
  });

  group('verifyAndCommit', () {
    test('hashes, renames onto the target, records it and reports Ready; '
        'returns the hash', () async {
      temp().writeAsBytesSync(bytes);

      final hex = await commit.verifyAndCommit(
        expected(),
        temp(),
        target(),
        CancelToken(),
        reporter,
      );

      expect(hex, sha);
      expect(target().readAsBytesSync(), bytes);
      expect(temp().existsSync(), isFalse);
      expect(record().readAsStringSync(), '$sha  $name\n');
      expect(reporter.reported, [isA<StoreFileVerifying>(), ready()]);
    });

    test('a matching checksum is accepted', () async {
      temp().writeAsBytesSync(bytes);

      expect(
        await commit.verifyAndCommit(
          expected(sha256: sha),
          temp(),
          target(),
          CancelToken(),
          reporter,
        ),
        sha,
      );
    });

    test('a wrong size: a size mismatch naming the source, the temporary '
        'file deleted, nothing hashed', () async {
      temp().writeAsBytesSync(bytes);

      await expectLater(
        commit.verifyAndCommit(
          expected(sizeBytes: n + 1),
          temp(),
          target(),
          CancelToken(),
          reporter,
        ),
        throwsA(
          isA<SizeMismatchException>()
              .having((e) => e.actual, 'actual', n)
              .having((e) => e.source, 'source', 'https://example.com/$name'),
        ),
      );
      expect(temp().existsSync(), isFalse);
      expect(ops.hashes, 0);
      expect(reporter.all, isEmpty);
    });
  });

  group('commit', () {
    /// A file and its record already in place, as an earlier commit left
    /// them.
    final old = testBytes(100, seed: 42);
    void storedBefore() {
      target().writeAsBytesSync(old);
      record().writeAsStringSync('${sha256Hex(old)}  $name\n');
    }

    test('a checksum that does not match: refused, the temporary file '
        'deleted, the file in place and its record untouched', () async {
      storedBefore();
      temp().writeAsBytesSync(bytes);

      await expectLater(
        commit.commit(
          expected(sha256: 'a' * 64),
          temp(),
          target(),
          sha,
          reporter,
        ),
        throwsA(
          isA<ChecksumMismatchException>()
              .having((e) => e.expected, 'expected', 'a' * 64)
              .having((e) => e.actual, 'actual', sha),
        ),
      );
      expect(temp().existsSync(), isFalse);
      expect(target().readAsBytesSync(), old);
      expect(await ChecksumRecord.read(target()), sha256Hex(old));
      expect(reporter.all, isEmpty);
    });

    test('without a checksum the hash is taken as given and recorded; the '
        'file in place is replaced', () async {
      storedBefore();
      temp().writeAsBytesSync(bytes);

      await commit.commit(expected(), temp(), target(), sha, reporter);

      expect(target().readAsBytesSync(), bytes);
      expect(await ChecksumRecord.read(target()), sha);
      expect(reporter.reported, [ready()]);
    });

    test('the old record goes before the rename: a rename that fails leaves '
        'the file unrecorded, never vouched for', () async {
      storedBefore();
      // No temporary file: the rename fails.

      await expectLater(
        commit.commit(expected(), temp(), target(), sha, reporter),
        throwsA(isA<FileSystemException>()),
      );
      expect(record().existsSync(), isFalse);
      expect(target().readAsBytesSync(), old);
      expect(reporter.all, isEmpty);
    });

    test('an old record that cannot be deleted aborts before the rename: a '
        'write error, the temporary file deleted, the old file and its '
        'record kept together', () async {
      storedBefore();
      temp().writeAsBytesSync(bytes);
      Process.runSync('chflags', ['uchg', record().path]);
      addTearDown(() => Process.runSync('chflags', ['nouchg', record().path]));

      await expectLater(
        commit.commit(expected(), temp(), target(), sha, reporter),
        throwsA(
          isA<StoreWriteException>()
              .having((e) => e.fileName, 'fileName', name)
              .having((e) => e.cause.path, 'path', record().path)
              .having(
                (e) => e.message,
                'message',
                startsWith(
                  'Could not write $name: the old checksum record cannot be '
                  'replaced, so the new file was not stored',
                ),
              ),
        ),
      );
      expect(target().readAsBytesSync(), old, reason: 'not renamed');
      expect(await ChecksumRecord.read(target()), sha256Hex(old));
      expect(temp().existsSync(), isFalse);
      expect(reporter.all, isEmpty);
    }, skip: Platform.isMacOS ? false : 'chflags uchg is macOS only');

    test('verifyAndCommit: the same abort for a download, its .part '
        'deleted', () async {
      storedBefore();
      temp().writeAsBytesSync(bytes);
      Process.runSync('chflags', ['uchg', record().path]);
      addTearDown(() => Process.runSync('chflags', ['nouchg', record().path]));

      await expectLater(
        commit.verifyAndCommit(
          expected(sha256: sha),
          temp(),
          target(),
          CancelToken(),
          reporter,
        ),
        throwsA(isA<StoreWriteException>()),
      );
      expect(target().readAsBytesSync(), old);
      expect(await ChecksumRecord.read(target()), sha256Hex(old));
      expect(temp().existsSync(), isFalse);
      expect(reporter.reported, [isA<StoreFileVerifying>()]);
    }, skip: Platform.isMacOS ? false : 'chflags uchg is macOS only');
  });
}
