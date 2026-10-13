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

import 'dart:async';
import 'dart:io';
import 'dart:isolate';

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/services/model_store/checksum_record.dart';
import 'package:litert_edge_demos/data/services/model_store/model_file_ops.dart';
import 'package:litert_edge_demos/data/services/model_store/model_importer.dart';
import 'package:litert_edge_demos/data/services/model_store/store_operation.dart';
import 'package:litert_edge_demos/data/services/model_store/verified_commit.dart';
import 'package:litert_edge_demos/domain/models/provisioning.dart';

import '../../../support/recording_reporter.dart';
import '../../../support/test_bytes.dart';

/// A copy that reports it runs (one progress message: 0 of 1 bytes), then
/// never ends until its isolate is killed (a cancel).
void _hangingFileWorker((Object, SendPort) args) {
  final (_, port) = args;
  ReceivePort(); // an open port keeps the isolate alive
  port.send((0, 1));
}

/// Completes [copying] on the first copy progress: sent by the copy's
/// worker isolate, so the worker runs by then.
final class _CopyStarted implements FileStateReporter {
  final Completer<void> copying = Completer<void>();

  @override
  void report(StoreFileState state) {}

  @override
  void progress(StoreFileState state) {
    if (state is StoreFileCopying && !copying.isCompleted) copying.complete();
  }
}

/// Counts hashes and copies; [clone] false refuses every clone.
final class _Ops extends ModelFileOps {
  _Ops({this.clone = true, super.worker});

  final bool clone;
  int hashes = 0;
  int copies = 0;

  @override
  Future<String> sha256OfFile(
    String path, {
    required ByteProgress onProgress,
    Future<void>? cancel,
  }) {
    hashes++;
    return super.sha256OfFile(path, onProgress: onProgress, cancel: cancel);
  }

  @override
  Future<String> copyAndHash(
    String source,
    String target, {
    required ByteProgress onProgress,
    Future<void>? cancel,
  }) {
    copies++;
    return super.copyAndHash(
      source,
      target,
      onProgress: onProgress,
      cancel: cancel,
    );
  }

  @override
  bool tryClone(String source, String target) =>
      clone && super.tryClone(source, target);
}

/// Refuses to clone, after making [dir] writable again: the move failed on
/// a read-only folder, the copy that follows may drop the picker's file.
final class _UnlockingOps extends _Ops {
  _UnlockingOps(this.dir) : super(clone: false);

  final Directory dir;

  @override
  bool tryClone(String source, String target) {
    Process.runSync('chmod', ['755', dir.path]);
    return false;
  }
}

/// [ModelImporter] on its own: move, clone or copy, links, the picker's
/// temporary copy, the checksum, a stale or cancelled `.import`.
void main() {
  late Directory store;
  late Directory picked;
  late RecordingReporter reporter;

  const name = 'model.litertlm';
  final bytes = testBytes(96 * 1024, seed: 51);
  final sha = sha256Hex(bytes);
  final n = bytes.length;

  setUp(() {
    store = Directory.systemTemp.createTempSync('importer_store');
    picked = Directory.systemTemp.createTempSync('importer_picked');
    reporter = RecordingReporter();
  });
  tearDown(() {
    store.deleteSync(recursive: true);
    picked.deleteSync(recursive: true);
  });

  File target() => File('${store.path}/$name');
  File temp() => File('${target().path}.import');
  File pick([String dir = '']) {
    final folder = Directory('${picked.path}/$dir')
      ..createSync(recursive: true);
    return File('${folder.path}/$name')..writeAsBytesSync(bytes);
  }

  Future<(ImportMethod, String)> import(
    ModelFileOps ops,
    File source, {
    String? expectedSha256,
    String? moveFrom,
    CancelToken? cancel,
  }) => ModelImporter(ops, VerifiedCommit(ops)).importFile(
    source.path,
    target(),
    name: name,
    sizeBytes: n,
    expectedSha256: expectedSha256,
    moveFrom: moveFrom,
    cancel: cancel ?? CancelToken(),
    reporter: reporter,
  );

  Matcher ready() =>
      isA<StoreFileReady>().having((s) => s.path, 'path', target().path);

  void expectStored() {
    expect(target().readAsBytesSync(), bytes);
    expect(File('${target().path}.sha256').readAsStringSync(), '$sha  $name\n');
    expect(temp().existsSync(), isFalse);
  }

  test('a byte copy where nothing clones: Copying at once, hashed on the '
      'way, the mtime kept, the picked file left alone', () async {
    final source = pick();
    final mtime = DateTime(2026, 5, 2, 8, 30);
    source.setLastModifiedSync(mtime);
    final ops = _Ops(clone: false);

    final (method, hex) = await import(ops, source);

    expect(method, ImportMethod.copied);
    expect(hex, sha);
    expectStored();
    expect(ops.copies, 1);
    expect(ops.hashes, 0);
    expect(target().lastModifiedSync(), mtime);
    expect(source.existsSync(), isTrue);
    expect(reporter.reported, [
      isA<StoreFileCopying>()
          .having((s) => s.copied, 'copied', 0)
          .having((s) => s.total, 'total', n),
      ready(),
    ]);
  });

  test(
    'an APFS clone: Verifying at once, hashed once, nothing copied',
    () async {
      final ops = _Ops();

      final (method, hex) = await import(ops, pick());

      expect(method, ImportMethod.cloned);
      expect(hex, sha);
      expectStored();
      expect(ops.hashes, 1);
      expect(ops.copies, 0);
      expect(reporter.reported, [isA<StoreFileVerifying>(), ready()]);
    },
    skip: Platform.isMacOS ? false : 'APFS clones are macOS/iOS only',
  );

  test("the picker's temporary copy is moved, then hashed", () async {
    final source = pick('Inbox');
    final ops = _Ops();

    final (method, _) = await import(
      ops,
      source,
      moveFrom: '${picked.path}/Inbox',
    );

    expect(method, ImportMethod.moved);
    expectStored();
    expect(source.existsSync(), isFalse);
    expect(ops.hashes, 1);
    expect(reporter.reported, [isA<StoreFileVerifying>(), ready()]);
  });

  test(
    "the picker's copy that cannot be moved is copied, then dropped",
    () async {
      final inbox = Directory('${picked.path}/Inbox');
      final source = pick('Inbox');
      Process.runSync('chmod', ['555', inbox.path]);
      addTearDown(() => Process.runSync('chmod', ['755', inbox.path]));

      final (method, _) = await import(
        _UnlockingOps(inbox),
        source,
        moveFrom: inbox.path,
      );

      expect(method, ImportMethod.copied);
      expectStored();
      expect(source.existsSync(), isFalse, reason: 'not kept twice');
    },
  );

  test('a file outside moveFrom is neither moved nor dropped', () async {
    final source = pick('elsewhere');

    final (method, _) = await import(
      _Ops(clone: false),
      source,
      moveFrom: '${picked.path}/Inbox',
    );

    expect(method, ImportMethod.copied);
    expectStored();
    expect(source.existsSync(), isTrue);
  });

  group('the picker folder is compared on canonical paths', () {
    late Directory inbox;

    setUp(() => inbox = Directory('${picked.path}/Inbox')..createSync());

    test('a path that only starts with moveFrom (Inbox/../elsewhere) is '
        'neither moved nor dropped', () async {
      final outside = pick('elsewhere');
      final sneaky = File('${inbox.path}/../elsewhere/$name');

      final (method, _) = await import(
        _Ops(clone: false),
        sneaky,
        moveFrom: inbox.path,
      );

      expect(method, ImportMethod.copied);
      expectStored();
      expect(outside.readAsBytesSync(), bytes, reason: 'left where it was');
    });

    test('a link in the picker folder to a file outside it: the file is '
        'copied, the link not moved, the file not dropped', () async {
      final outside = pick('elsewhere');
      final link = Link('${inbox.path}/$name')..createSync(outside.path);

      final (method, _) = await import(
        _Ops(clone: false),
        File(link.path),
        moveFrom: inbox.path,
      );

      expect(method, ImportMethod.copied);
      expectStored();
      expect(FileSystemEntity.isLinkSync(target().path), isFalse);
      expect(outside.readAsBytesSync(), bytes);
      expect(FileSystemEntity.isLinkSync(link.path), isTrue);
    });

    test("a link to the picker's own copy: the file it points to is copied "
        '(never the link moved), then dropped', () async {
      final copy = File('${inbox.path}/copy.litertlm')..writeAsBytesSync(bytes);
      final link = Link('${inbox.path}/$name')..createSync(copy.path);

      final (method, _) = await import(
        _Ops(clone: false),
        File(link.path),
        moveFrom: inbox.path,
      );

      expect(method, ImportMethod.copied);
      expectStored();
      expect(FileSystemEntity.isLinkSync(target().path), isFalse);
      expect(copy.existsSync(), isFalse, reason: 'not kept twice');
    });

    test('moveFrom given through a link (as /var is /private/var on macOS): '
        "the picker's copy is still moved", () async {
      final source = pick('Inbox');
      final alias = Link('${picked.path}/alias')..createSync(inbox.path);

      final (method, _) = await import(_Ops(), source, moveFrom: alias.path);

      expect(method, ImportMethod.moved);
      expectStored();
      expect(source.existsSync(), isFalse);
    });
  });

  test('a link is followed: the file it points to is stored', () async {
    final real = pick('real');
    final link = Link('${picked.path}/link.litertlm')..createSync(real.path);

    await import(_Ops(clone: false), File(link.path));

    expectStored();
    expect(FileSystemEntity.isLinkSync(target().path), isFalse);
    expect(real.existsSync(), isTrue);
  });

  test('a stale .import from an earlier crash is replaced', () async {
    temp().writeAsBytesSync([9, 9, 9]);

    await import(_Ops(), pick());

    expectStored();
  });

  test('a checksum that matches is accepted', () async {
    final (_, hex) = await import(_Ops(), pick(), expectedSha256: sha);

    expect(hex, sha);
    expectStored();
  });

  test('a checksum that does not match: refused, no .import left, the file '
      'in place untouched', () async {
    final old = testBytes(10, seed: 52);
    target().writeAsBytesSync(old);
    await ChecksumRecord.write(target(), sha256Hex(old), name);

    await expectLater(
      import(_Ops(), pick(), expectedSha256: 'b' * 64),
      throwsA(isA<ChecksumMismatchException>()),
    );

    expect(target().readAsBytesSync(), old);
    expect(await ChecksumRecord.read(target()), sha256Hex(old));
    expect(temp().existsSync(), isFalse);
  });

  test('a cancel during the copy kills it; no .import left', () async {
    final cancel = CancelToken();
    addTearDown(cancel.cancel); // a failed wait leaves no worker behind
    final started = _CopyStarted();
    final ops = _Ops(clone: false, worker: _hangingFileWorker);
    final importing = ModelImporter(ops, VerifiedCommit(ops)).importFile(
      pick().path,
      target(),
      name: name,
      sizeBytes: n,
      expectedSha256: null,
      moveFrom: null,
      cancel: cancel,
      reporter: started,
    );
    await started.copying.future.timeout(
      const Duration(seconds: 10),
      onTimeout: () => fail('the copy worker never reported'),
    );

    cancel.cancel();

    await expectLater(importing, throwsA(isA<OperationCancelledException>()));
    expect(temp().existsSync(), isFalse);
    expect(target().existsSync(), isFalse);
  });
}
