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
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/services/model_store/http_file_downloader.dart'
    show PartOpener, PartSink;
import 'package:litert_edge_demos/data/services/model_store/model_file_ops.dart';
import 'package:litert_edge_demos/data/services/model_store/model_store.dart';
import 'package:litert_edge_demos/domain/models/chat_model.dart';
import 'package:litert_edge_demos/domain/models/provisioning.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../../support/failing_file_workers.dart';
import '../../../support/fake_model_server.dart';
import '../../../support/test_bytes.dart';

/// [ModelFileOps] that counts hashes and copies and can refuse to clone.
final class CountingOps extends ModelFileOps {
  CountingOps({this.clone = true});

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

/// A part sink whose disk fills up after [limit] bytes.
final class FullDiskSink implements PartSink {
  FullDiskSink(this._inner, this.limit);

  final PartSink _inner;
  final int limit;
  int written = 0;

  /// Writes what fits, then fails like a full disk.
  @override
  Future<void> add(List<int> bytes) async {
    if (written + bytes.length > limit) {
      final fits = limit - written;
      await _inner.add(bytes.sublist(0, fits));
      written += fits;
      throw const FileSystemException(
        'Write failed',
        '/models/x.part',
        OSError('No space left on device', 28),
      );
    }
    written += bytes.length;
    await _inner.add(bytes);
  }

  @override
  Future<void> close() => _inner.close();
}

/// A part sink that behaves like `RandomAccessFile`: closing it while a write
/// is pending throws. Each write waits for [release] once it has started
/// (the slow disk), signalling [writing].
final class SlowStrictSink implements PartSink {
  SlowStrictSink(this._inner);

  final PartSink _inner;
  final Completer<void> writing = Completer<void>();
  final Completer<void> release = Completer<void>();
  int _pending = 0;
  bool closed = false;

  @override
  Future<void> add(List<int> bytes) async {
    _pending++;
    try {
      if (!writing.isCompleted) writing.complete();
      await release.future;
      await _inner.add(bytes);
    } finally {
      _pending--;
    }
  }

  @override
  Future<void> close() async {
    if (_pending > 0) {
      throw const FileSystemException(
        'An async operation is currently pending',
      );
    }
    closed = true;
    await _inner.close();
  }
}

/// The store's transfer and import engine, through the custom chat model's
/// download and import (the only files the store holds): HTTP Range resume,
/// reconnects, the stall watchdog, the backpressured body reader, cancel,
/// close, the one-operation lock, a full disk and failing file workers.
void main() {
  late FakeModelServer server;
  late Directory root;
  late CountingOps ops;
  final stores = <ModelStore>[];

  const name = 'gemma-4-E2B-it.litertlm';
  final bytes = testBytes(600 * 1024, seed: 7);
  final sha = sha256Hex(bytes);

  setUp(() async {
    server = await FakeModelServer.start();
    root = Directory.systemTemp.createTempSync('model_store_test');
    ops = CountingOps();
  });

  tearDown(() async {
    for (final store in stores) {
      await store.close();
    }
    stores.clear();
    await server.close();
    root.deleteSync(recursive: true);
  });

  ModelStore newStore({
    int maxAttempts = 3,
    Duration stallTimeout = const Duration(seconds: 30),
    Duration progressInterval = const Duration(milliseconds: 250),
    PartOpener? openPart,
    ModelFileOps? fileOps,
    Duration closeWait = const Duration(seconds: 5),
  }) {
    final store = ModelStore(
      closeWait: closeWait,
      root: () async => root,
      ops: fileOps ?? ops,
      maxAttempts: maxAttempts,
      retryDelay: (_) => Duration.zero,
      stallTimeout: stallTimeout,
      progressInterval: progressInterval,
      openPart:
          openPart ??
          (file, {required append}) async => _FilePartSink(
            await file.open(mode: append ? FileMode.append : FileMode.write),
          ),
    );
    stores.add(store);
    return store;
  }

  void serve({ServedFile? file}) =>
      server.files[name] = file ?? ServedFile(bytes);

  /// Downloads [name] from the server with its size and checksum given (no
  /// size probe: every request is a transfer).
  Future<Result<CustomModelFile>> download(
    ModelStore store, {
    Uri? url,
    String? checksum,
  }) => store.downloadCustomFile(
    url ?? server.url(name),
    sha256: checksum ?? sha,
    sizeBytes: bytes.length,
  );

  File customFile(String file) => File('${root.path}/custom/$file');

  /// The `.part` files in the custom folder (one per URL).
  List<File> parts() {
    final dir = Directory('${root.path}/custom');
    if (!dir.existsSync()) return const [];
    return dir
        .listSync()
        .whereType<File>()
        .where((f) => f.path.endsWith('.part'))
        .toList();
  }

  Exception errorOf(Result<CustomModelFile> result) =>
      (result as Error<CustomModelFile>).error;

  group('download', () {
    test('downloads, verifies, stores atomically and records the checksum; '
        'the state goes Downloading, Verifying, Ready', () async {
      serve();
      final store = newStore();
      final seen = <Type>{};
      store.customFile.addListener(
        () => seen.add(store.customFile.value.runtimeType),
      );

      final result = await download(store);

      expect((result as Ok<CustomModelFile>).value.checksumMatched, isTrue);
      expect(
        seen,
        containsAll([StoreFileDownloading, StoreFileVerifying, StoreFileReady]),
      );
      expect(customFile(name).readAsBytesSync(), bytes);
      expect(parts(), isEmpty);
      expect(customFile('$name.sha256').readAsStringSync(), startsWith(sha));
      expect(
        (store.customFile.value as StoreFileReady).path,
        customFile(name).path,
      );
    });

    test(
      'the default part file (flushed, then closed) downloads and verifies',
      () async {
        serve(file: ServedFile(bytes, dropAfter: 200 * 1024));
        final store = ModelStore(
          root: () async => root,
          ops: ops,
          retryDelay: (_) => Duration.zero,
        );
        stores.add(store);

        expect(await download(store), isA<Ok<CustomModelFile>>());
        expect(customFile(name).readAsBytesSync(), bytes);
        expect(server.requestsFor(name).last.range, 'bytes=${200 * 1024}-');
      },
    );

    // Redirects and the Range header on each hop:
    // http_file_downloader_test.dart.

    test('after the reconnects the error is shown, the part kept, and Retry '
        'resumes', () async {
      serve(file: ServedFile(bytes, dropAfter: 100 * 1024, dropTimes: 2));
      final store = newStore(maxAttempts: 2);

      final failed = await download(store);

      final error = errorOf(failed);
      expect(error, isA<DownloadNetworkException>());
      expect(error.toString(), contains('Network error while downloading'));
      expect(error.toString(), contains('Retry resumes it'));
      expect((error as DownloadNetworkException).partialBytes, 200 * 1024);
      expect(parts().single.lengthSync(), 200 * 1024);
      expect(
        store.customFile.value,
        isA<StoreFileMissing>(),
        reason: 'the previous state, as before the download',
      );

      expect(await download(store), isA<Ok<CustomModelFile>>());
      expect(server.requestsFor(name).last.range, 'bytes=${200 * 1024}-');
      expect(store.customFile.value, isA<StoreFileReady>());
    });

    test('a SHA-256 mismatch is an error and the part is deleted', () async {
      serve();
      final store = newStore();

      final result = await download(store, checksum: 'a' * 64);

      final error = errorOf(result);
      expect(error, isA<ChecksumMismatchException>());
      expect(error.toString(), contains('failed verification'));
      expect(
        error.toString(),
        contains('the checksum you entered expects ${'a' * 64}'),
      );
      expect(parts(), isEmpty);
      expect(customFile(name).existsSync(), isFalse);
    });

    test('a Google Drive interstitial (text/html) names the file', () async {
      serve(
        file: ServedFile(
          Uint8List.fromList(
            '<html><body>Quota exceeded</body></html>'.codeUnits,
          ),
          contentType: 'text/html; charset=utf-8',
        ),
      );
      final store = newStore();

      final error = errorOf(await download(store));

      expect(error, isA<HtmlInsteadOfFileException>());
      expect(error.toString(), contains('a web page instead of $name'));
      expect(parts(), isEmpty);
      // The same failure from a Drive link says what Drive does.
      final driveUrl = Uri.parse(
        'https://drive.usercontent.google.com/download?id=abc&export=download&confirm=t',
      );
      final driveMessage = HtmlInsteadOfFileException(
        name,
        driveUrl,
        drive: true,
      ).message;
      expect(
        driveMessage,
        contains(
          'Google Drive returned a web page instead of $name (its '
          'virus-scan warning or download quota)',
        ),
      );
      // The link is shown without its query (a signed link's token).
      expect(driveMessage, contains('https://drive.usercontent.google.com/'));
      expect(driveMessage, isNot(contains('id=abc')));
    });

    test('an HTML page served as octet-stream is caught by sniffing', () async {
      serve(
        file: ServedFile(
          Uint8List.fromList(
            '\n  <!DOCTYPE html><title>Google Drive - Virus scan warning</title>'
                .codeUnits,
          ),
          reportedLength: bytes.length,
        ),
      );
      final store = newStore();

      final error = errorOf(await download(store));

      expect(error, isA<HtmlInsteadOfFileException>());
      expect(error.toString(), contains('a web page instead of $name'));
    });

    test('a hash worker that crashes fails the download (no uncaught error, '
        'not stuck on Verifying) and frees the store', () async {
      serve();
      final store = newStore(
        fileOps: const ModelFileOps(worker: crashingFileWorker),
      );

      final error = errorOf(await download(store));

      expect(
        error,
        isA<FileOpFailedException>().having(
          (e) => e.message,
          'message',
          contains('file worker crashed'),
        ),
      );
      expect(store.customFile.value, isA<StoreFileMissing>());
      expect(store.busy.value, isFalse);
    });

    test('a full disk is a visible error and keeps what was written', () async {
      serve();
      final store = newStore(
        openPart: (file, {required append}) async => FullDiskSink(
          _FilePartSink(
            await file.open(mode: append ? FileMode.append : FileMode.write),
          ),
          128 * 1024,
        ),
      );

      final error = errorOf(await download(store));

      expect(error, isA<DiskFullException>());
      expect(error.toString(), contains('The disk is full while writing'));
      expect(parts().single.lengthSync(), 128 * 1024);
      expect(server.requestsFor(name), hasLength(1), reason: 'no reconnects');
    });

    test(
      'cancel keeps the partial file and a later download resumes',
      () async {
        serve(
          file: ServedFile(
            bytes,
            chunkBytes: 16 * 1024,
            chunkDelay: const Duration(milliseconds: 20),
          ),
        );
        final store = newStore(progressInterval: Duration.zero);
        final reached = Completer<void>();
        store.customFile.addListener(() {
          if (store.customFile.value case StoreFileDownloading(:final received)
              when received >= 64 * 1024 && !reached.isCompleted) {
            reached.complete();
          }
        });

        final running = download(store);
        await reached.future;
        store.cancel();
        final result = await running;

        expect(errorOf(result), isA<OperationCancelledException>());
        expect(store.customFile.value, isA<StoreFileMissing>());
        final kept = parts().single.lengthSync();
        expect(kept, greaterThan(0));

        expect(await download(store), isA<Ok<CustomModelFile>>());
        expect(server.requestsFor(name).last.range, 'bytes=$kept-');
      },
    );

    for (final how in ['cancel', 'store close']) {
      test('$how during a slow .part write ends as a cancel, the file '
          'closed', () async {
        serve();
        final opened = Completer<SlowStrictSink>();
        final store = newStore(
          openPart: (file, {required append}) async {
            final sink = SlowStrictSink(
              _FilePartSink(
                await file.open(
                  mode: append ? FileMode.append : FileMode.write,
                ),
              ),
            );
            opened.complete(sink);
            return sink;
          },
        );
        final updates = <StoreFileState>[];

        final running = download(store);
        final sink = await opened.future;
        await sink.writing.future;
        store.customFile.addListener(() => updates.add(store.customFile.value));
        // Whether the sink was closed when store.close() returned (null: it
        // has not returned).
        bool? sinkClosedAtClose;
        Future<void>? closing;
        if (how == 'cancel') {
          store.cancel();
        } else {
          closing = store.close().then((_) => sinkClosedAtClose = sink.closed);
        }
        // The write is still pending when the read stops; let it finish.
        await Future<void>.delayed(const Duration(milliseconds: 50));
        if (closing != null) {
          expect(
            sinkClosedAtClose,
            isNull,
            reason: 'close waits for the operation it cancelled',
          );
        }
        sink.release.complete();
        if (closing != null) {
          // The close future itself, not the download's.
          await closing;
          expect(sinkClosedAtClose, isTrue, reason: 'the .part closed first');
        }
        final result = await running;

        expect(errorOf(result), isA<OperationCancelledException>());
        expect(sink.closed, isTrue, reason: 'no leaked handle');
        if (how == 'cancel') {
          expect(store.customFile.value, isA<StoreFileMissing>());
          expect(
            updates.whereType<StoreFileDownloading>(),
            isEmpty,
            reason: 'no progress after the cancel',
          );
          expect(parts().single.existsSync(), isTrue, reason: 'kept');
        }
      });
    }

    test('an operation that never ends holds close only for closeWait '
        '(logged), not forever', () async {
      serve();
      final opened = Completer<SlowStrictSink>();
      final store = newStore(
        closeWait: const Duration(milliseconds: 50),
        openPart: (file, {required append}) async {
          final sink = SlowStrictSink(
            _FilePartSink(
              await file.open(mode: append ? FileMode.append : FileMode.write),
            ),
          );
          opened.complete(sink);
          return sink;
        },
      );
      final running = download(store);
      final sink = await opened.future;
      await sink.writing.future;

      await store.close(); // the write is never released

      sink.release.complete();
      expect(await running, isA<Error<CustomModelFile>>());
    });

    test('a stalled connection is cut by the watchdog and resumed', () async {
      serve(file: ServedFile(bytes, stallAfter: 64 * 1024));
      final store = newStore(stallTimeout: const Duration(milliseconds: 600));
      // The first response stalls; the resume is served in full.
      store.customFile.addListener(() {
        if (store.customFile.value case StoreFileDownloading(attempt: 2)) {
          server.files[name] = ServedFile(bytes);
        }
      });

      expect(await download(store), isA<Ok<CustomModelFile>>());
      expect(server.requestsFor(name).last.range, 'bytes=${64 * 1024}-');
    });

    test('a server that ignores Range restarts the file from zero', () async {
      // An earlier attempt left 1000 bytes.
      serve(file: ServedFile(bytes, dropAfter: 1000));
      final store = newStore(maxAttempts: 1);
      expect(await download(store), isA<Error<CustomModelFile>>());
      expect(parts().single.lengthSync(), 1000);

      serve(file: ServedFile(bytes, ignoreRange: true));
      expect(await download(store), isA<Ok<CustomModelFile>>());
      expect(server.requestsFor(name).last.range, 'bytes=1000-');
      expect(customFile(name).readAsBytesSync(), bytes);
    });

    test('a wrong Content-Length fails before writing anything', () async {
      serve(file: ServedFile(bytes, reportedLength: bytes.length + 1));
      final store = newStore();

      final error = errorOf(await download(store));

      expect(error, isA<SizeMismatchException>());
      expect(error.toString(), contains('not the ${bytes.length} expected'));
      expect(parts(), isEmpty);
    });

    test('HTTP 404 is shown at once, without reconnects', () async {
      serve(file: ServedFile(bytes, status: 404));
      final store = newStore();

      final error = errorOf(await download(store));

      expect(error, isA<DownloadHttpException>());
      expect(error.toString(), contains('HTTP 404'));
      expect(server.requestsFor(name), hasLength(1));
    });

    test('progress is throttled, never one update per chunk', () async {
      serve(file: ServedFile(bytes, chunkBytes: 4 * 1024));
      final store = newStore(progressInterval: const Duration(hours: 1));
      var updates = 0;
      store.customFile.addListener(() => updates++);

      expect(await download(store), isA<Ok<CustomModelFile>>());

      // 150 chunks; only the transitions notify: downloading, verifying,
      // ready.
      expect(updates, lessThanOrEqualTo(3));
    });
  });

  test('one operation at a time: a second download, an import or a rescan '
      'while one runs is refused', () async {
    serve(
      file: ServedFile(bytes, chunkDelay: const Duration(milliseconds: 10)),
    );
    final store = newStore();
    final picked = Directory.systemTemp.createTempSync('model_store_busy');
    addTearDown(() => picked.deleteSync(recursive: true));
    final other = File('${picked.path}/other.litertlm')
      ..writeAsBytesSync(testBytes(64));

    final first = download(store);
    expect(store.busy.value, isTrue);
    expect(errorOf(await download(store)), isA<StoreBusyException>());
    expect(
      errorOf(await store.importCustomFile(other.path)),
      isA<StoreBusyException>(),
    );
    expect(
      (await store.useCustomFile(null) as Error<String?>).error,
      isA<StoreBusyException>(),
    );
    expect(await first, isA<Ok<CustomModelFile>>());
    expect(store.busy.value, isFalse);
  });

  group('import', () {
    late Directory picked;

    setUp(() => picked = Directory.systemTemp.createTempSync('model_import'));
    tearDown(() => picked.deleteSync(recursive: true));

    for (final (what, worker) in [
      ('reports an error', erroringFileWorker),
      ('exits without a result', silentFileWorker),
    ]) {
      test('a worker that $what fails the import, not the store', () async {
        final source = File('${picked.path}/$name')..writeAsBytesSync(bytes);
        final store = newStore(fileOps: ModelFileOps(worker: worker));

        final error = errorOf(await store.importCustomFile(source.path));

        expect(error, isA<FileOpFailedException>());
        expect(store.customFile.value, isA<StoreFileMissing>());
        expect(customFile('$name.import').existsSync(), isFalse);
        expect(store.busy.value, isFalse);
      });
    }

    // How a file is brought in (a byte copy, a clone, a link followed):
    // model_importer_test.dart.
  });

  group('pruneOldModelFolders (after an upgrade)', () {
    void folder(String name) => File('${root.path}/$name/file.bin')
      ..createSync(recursive: true)
      ..writeAsStringSync('x');

    test('removes the folders earlier builds downloaded into (the retired '
        'Gemma download, the recognizers); keeps custom/, bundled/ and the '
        'folder of a file used in place', () async {
      for (final name in [
        kRetiredGemmaFolder,
        'whisperBase',
        'moonshineTiny',
        'custom',
        'bundled',
      ]) {
        folder(name);
      }
      final store = newStore();

      final removed = await store.pruneOldModelFolders(
        keepPaths: {'${root.path}/moonshineTiny/file.bin'},
      );

      expect(removed.map((p) => p.split('/').last).toSet(), {
        kRetiredGemmaFolder,
        'whisperBase',
      });
      for (final kept in ['moonshineTiny', 'custom', 'bundled']) {
        expect(Directory('${root.path}/$kept').existsSync(), isTrue);
      }
    });

    test('nothing is pruned while an operation runs', () async {
      folder(kRetiredGemmaFolder);
      serve(
        file: ServedFile(bytes, chunkDelay: const Duration(milliseconds: 10)),
      );
      final store = newStore();

      final running = download(store);
      expect(await store.pruneOldModelFolders(), isEmpty);
      expect(
        Directory('${root.path}/$kRetiredGemmaFolder').existsSync(),
        isTrue,
      );
      expect(await running, isA<Ok<CustomModelFile>>());
    });
  });
}

final class _FilePartSink implements PartSink {
  _FilePartSink(this._file);

  final RandomAccessFile _file;

  @override
  Future<void> add(List<int> bytes) => _file.writeFrom(bytes);

  @override
  Future<void> close() => _file.close();
}
