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
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/services/model_store/http_file_downloader.dart'
    show PartOpener, PartSink;
import 'package:litert_edge_demos/data/services/model_store/model_file_ops.dart';
import 'package:litert_edge_demos/data/services/model_store/model_store.dart';
import 'package:litert_edge_demos/domain/models/chat_model.dart';
import 'package:litert_edge_demos/domain/models/provisioning.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../../support/fake_model_server.dart';
import '../../../support/scripted_http_server.dart';
import '../../../support/test_bytes.dart';

/// A hash or copy that never ends until its isolate is killed (a cancel).
void _hangingFileWorker((Object, SendPort) args) {
  ReceivePort(); // an open port keeps the isolate alive
}

/// Writes each chunk but its first byte: the disk lost bytes the transfer
/// counted as written.
final class _LossySink implements PartSink {
  _LossySink(this._file);

  final RandomAccessFile _file;

  @override
  Future<void> add(List<int> bytes) => _file.writeFrom(bytes, 1);

  @override
  Future<void> close() => _file.close();
}

/// Characterization of the store's edges the main suites do not reach: a
/// complete earlier `.part`, servers that resume a file of another size, send
/// too much, redirect forever or never answer; the size probe's other
/// answers; bytes lost on the way to disk; a cancel while reconnecting, while
/// waiting for headers or while hashing; imports that cannot write; the store
/// once closed; pruning that cannot delete or list. The downloader's own
/// rules (a `.part` longer than the file, a resume elsewhere or refused, an
/// early end without Content-Length, a refused probe) are in
/// http_file_downloader_test.dart, the picker's copy that cannot be moved in
/// model_importer_test.dart.
void main() {
  late Directory root;
  final stores = <ModelStore>[];
  final closers = <Future<void> Function()>[];

  const name = 'edge.litertlm';
  final bytes = testBytes(200 * 1024, seed: 5);
  final sha = sha256Hex(bytes);
  final n = bytes.length;

  setUp(() {
    root = Directory.systemTemp.createTempSync('model_store_edges');
  });

  tearDown(() async {
    for (final store in stores) {
      await store.close();
    }
    stores.clear();
    for (final close in closers.reversed) {
      await close();
    }
    closers.clear();
    root.deleteSync(recursive: true);
  });

  ModelStore newStore({
    int maxAttempts = 3,
    Duration Function(int attempt)? retryDelay,
    ModelFileOps ops = const ModelFileOps(),
    PartOpener? openPart,
    Future<Directory> Function()? rootOf,
  }) {
    final store = openPart == null
        ? ModelStore(
            root: rootOf ?? () async => root,
            ops: ops,
            maxAttempts: maxAttempts,
            retryDelay: retryDelay ?? (_) => Duration.zero,
          )
        : ModelStore(
            root: rootOf ?? () async => root,
            ops: ops,
            maxAttempts: maxAttempts,
            retryDelay: retryDelay ?? (_) => Duration.zero,
            openPart: openPart,
          );
    stores.add(store);
    return store;
  }

  Future<FakeModelServer> fakeServer() async {
    final server = await FakeModelServer.start();
    closers.add(server.close);
    return server;
  }

  Future<ScriptedHttpServer> scriptedServer(
    Future<void> Function(ScriptedRequest request) handler,
  ) async {
    final server = await ScriptedHttpServer.start(handler);
    closers.add(server.close);
    return server;
  }

  /// A URL nothing listens on.
  Future<Uri> refusedUrl() async {
    final socket = await ServerSocket.bind(InternetAddress.loopbackIPv4, 0);
    final port = socket.port;
    await socket.close();
    return Uri.parse('http://127.0.0.1:$port/$name');
  }

  Directory customDir() => Directory('${root.path}/custom');
  File stored() => File('${customDir().path}/$name');

  /// The `.part` a download from [url] resumes (`<name>.<8 hex of the URL's
  /// SHA-256>.part`).
  File partFor(Uri url) => File(
    '${customDir().path}/$name.'
    '${sha256Hex(url.toString().codeUnits).substring(0, 8)}.part',
  );

  File leavePart(Uri url, List<int> content) => partFor(url)
    ..createSync(recursive: true)
    ..writeAsBytesSync(content);

  Future<Result<CustomModelFile>> download(
    ModelStore store,
    Uri url, {
    bool withSize = true,
  }) => withSize
      ? store.downloadCustomFile(url, sha256: sha, sizeBytes: n)
      : store.downloadCustomFile(url);

  Exception errorOf(Result<Object?> result) => (result as Error).error;

  /// 206 from [from] to the end, or the whole file with 200.
  Future<void> serveFrom(ScriptedRequest r, int from) => from > 0
      ? r.respond(
          206,
          headers: {
            'Content-Range': 'bytes $from-${n - 1}/$n',
            'Content-Length': '${n - from}',
          },
          body: Uint8List.sublistView(bytes, from),
        )
      : r.respond(200, headers: {'Content-Length': '$n'}, body: bytes);

  group('an earlier .part', () {
    test('complete: verified and stored without a request', () async {
      final server = await fakeServer();
      final url = server.url(name);
      leavePart(url, bytes);

      expect(await download(newStore(), url), isA<Ok<CustomModelFile>>());

      expect(server.requests, isEmpty);
      expect(stored().readAsBytesSync(), bytes);
      expect(partFor(url).existsSync(), isFalse);
    });
  });

  group('the server', () {
    test('resumes a file of another size (206): a size mismatch, not '
        'retried, the .part kept', () async {
      final server = await fakeServer();
      server.files[name] = ServedFile(testBytes(n + 100, seed: 9));
      final url = server.url(name);
      leavePart(url, bytes.sublist(0, 1000));

      final error = errorOf(await download(newStore(), url));

      expect(error, isA<SizeMismatchException>());
      expect(
        error.toString(),
        '$name from 127.0.0.1 is ${n + 100} bytes, not the $n expected. It '
        'is not the right file.',
      );
      expect(server.requestsFor(name), hasLength(1));
      expect(partFor(url).lengthSync(), 1000);
    });

    for (final status in [429, 503]) {
      test('answers HTTP $status: retried, then the error names it', () async {
        final server = await fakeServer();
        server.files[name] = ServedFile(bytes, status: status);

        final error = errorOf(
          await download(newStore(maxAttempts: 2), server.url(name)),
        );

        expect(error, isA<DownloadNetworkException>());
        expect(
          error.toString(),
          'Network error while downloading $name: Downloading $name failed: '
          'HTTP $status from 127.0.0.1.. Retry starts it again.',
        );
        expect(server.requestsFor(name), hasLength(2));
      });
    }

    test('sends more than the file without a Content-Length: a size '
        'mismatch, the .part deleted, not retried', () async {
      final server = await scriptedServer(
        (r) => r.respond(200, body: [...bytes, ...testBytes(1000, seed: 4)]),
      );
      final url = server.url(name);

      final error = errorOf(await download(newStore(), url));

      expect(
        error,
        isA<SizeMismatchException>()
            .having((e) => e.expected, 'expected', n)
            .having((e) => e.actual, 'actual', greaterThan(n))
            .having((e) => e.source, 'source', '127.0.0.1'),
      );
      expect(partFor(url).existsSync(), isFalse);
      expect(server.requests, hasLength(1));
    });

    test('ends a body without a Content-Length early, every attempt: the '
        'error says where it closed, the .part kept', () async {
      final server = await scriptedServer(
        (r) => r.respond(200, body: bytes.sublist(0, 50 * 1024)),
      );
      final url = server.url(name);

      final error = errorOf(await download(newStore(maxAttempts: 1), url));

      expect(error, isA<DownloadNetworkException>());
      expect(
        error.toString(),
        'Network error while downloading $name: the connection closed at '
        '0.1 MB of 0.2 MB. The partial file is kept (0.1 MB); Retry resumes '
        'it.',
      );
      expect(partFor(url).lengthSync(), 50 * 1024);
    });

    test('refuses the connection: a network error after the attempts, '
        'nothing kept', () async {
      final url = await refusedUrl();

      final error = errorOf(await download(newStore(), url));

      expect(
        error,
        isA<DownloadNetworkException>().having(
          (e) => e.partialBytes,
          'partialBytes',
          0,
        ),
      );
      expect(error.toString(), startsWith('Network error while downloading'));
      expect(error.toString(), endsWith('Retry starts it again.'));
    });

    test('redirects forever: given up after ten hops (HTTP 508), not '
        'retried', () async {
      final server = await fakeServer();
      server.files[name] = ServedFile(bytes);

      final error = errorOf(
        await download(newStore(), server.url('${'redirect/' * 10}$name')),
      );

      expect(error, isA<DownloadHttpException>());
      expect(
        error.toString(),
        'Downloading $name failed: HTTP 508 from 127.0.0.1.',
      );
      expect(server.requests, hasLength(10));
      expect(server.requestsFor(name), isEmpty);
    });

    test('sends a page behind a byte-order mark as octet-stream: caught by '
        'sniffing', () async {
      final server = await fakeServer();
      server.files[name] = ServedFile(
        Uint8List.fromList([
          0xEF,
          0xBB,
          0xBF,
          ...'\t<HTML><body>Sign in</body></HTML>'.codeUnits,
        ]),
        reportedLength: n,
      );

      final error = errorOf(await download(newStore(), server.url(name)));

      expect(error, isA<HtmlInsteadOfFileException>());
    });
  });

  group('cancel', () {
    test('while waiting to reconnect: a cancel, the .part kept', () async {
      final server = await fakeServer();
      server.files[name] = ServedFile(bytes, dropAfter: 64 * 1024);
      final url = server.url(name);
      final store = newStore(retryDelay: (_) => const Duration(hours: 1));
      store.customFile.addListener(() {
        if (store.customFile.value case StoreFileDownloading(attempt: 2)) {
          store.cancel();
        }
      });

      final error = errorOf(await download(store, url));

      expect(error, isA<OperationCancelledException>());
      expect(store.customFile.value, isA<StoreFileMissing>());
      expect(partFor(url).lengthSync(), 64 * 1024);
      expect(server.requestsFor(name), hasLength(1));
    });

    test('while the request waits for its headers: a cancel', () async {
      final server = await scriptedServer((_) async {}); // never answers
      final arrived = server.arrivals.first;
      final store = newStore();

      final running = download(store, server.url(name));
      await arrived;
      store.cancel();

      expect(errorOf(await running), isA<OperationCancelledException>());
      expect(store.busy.value, isFalse);
    });

    test('while the download is hashed: a cancel, the .part kept', () async {
      final server = await fakeServer();
      server.files[name] = ServedFile(bytes);
      final url = server.url(name);
      final store = newStore(
        ops: const ModelFileOps(worker: _hangingFileWorker),
      );
      store.customFile.addListener(() {
        if (store.customFile.value is StoreFileVerifying) {
          Timer(const Duration(milliseconds: 50), store.cancel);
        }
      });

      final error = errorOf(await download(store, url));

      expect(error, isA<OperationCancelledException>());
      expect(store.customFile.value, isA<StoreFileMissing>());
      expect(partFor(url).lengthSync(), n);
      expect(store.busy.value, isFalse);
    });

    test('while an import is cloned and hashed or copied: a cancel, no '
        '.import left', () async {
      final picked = Directory.systemTemp.createTempSync('edges_picked');
      addTearDown(() => picked.deleteSync(recursive: true));
      final source = File('${picked.path}/$name')..writeAsBytesSync(bytes);
      final store = newStore(
        ops: const ModelFileOps(worker: _hangingFileWorker),
      );
      store.customFile.addListener(() {
        if (store.customFile.value
            case StoreFileVerifying() || StoreFileCopying()) {
          Timer(const Duration(milliseconds: 50), store.cancel);
        }
      });

      final error = errorOf(await store.importCustomFile(source.path));

      expect(error, isA<OperationCancelledException>());
      expect(store.customFile.value, isA<StoreFileMissing>());
      expect(File('${stored().path}.import').existsSync(), isFalse);
      expect(source.existsSync(), isTrue);
    });
  });

  group('the size probe (no size given)', () {
    test('a 206 to the one-byte probe: the size from Content-Range', () async {
      final server = await scriptedServer(
        (r) => r.range == 'bytes=0-0'
            ? r.respond(
                206,
                headers: {
                  'Content-Range': 'bytes 0-0/$n',
                  'Content-Length': '1',
                },
                body: [bytes[0]],
              )
            : serveFrom(r, 0),
      );

      final result = await download(
        newStore(),
        server.url(name),
        withSize: false,
      );

      expect((result as Ok<CustomModelFile>).value.sizeBytes, n);
      expect(server.requests.map((r) => r.range), ['bytes=0-0', null]);
    });

    for (final (what, answer) in [
      ('a 200 without Content-Length', (ScriptedRequest r) => r.respond(200)),
      (
        'a 206 without the total',
        (ScriptedRequest r) => r.respond(
          206,
          headers: {'Content-Range': 'bytes 0-0/*', 'Content-Length': '1'},
          body: [1],
        ),
      ),
    ]) {
      test('$what: the error asks for the size', () async {
        final server = await scriptedServer(answer);
        final url = server.url(name);

        final error = errorOf(await download(newStore(), url, withSize: false));

        expect(error, isA<UnknownSizeException>());
        expect(
          error.toString(),
          '127.0.0.1 does not report the file size: enter the size in bytes '
          'and download again.',
        );
        expect(server.requests, hasLength(1));
      });
    }

    test('a probe hung up on before its headers: a network error', () async {
      final server = await scriptedServer((r) async => r.hangUp());

      final error = errorOf(
        await download(newStore(), server.url(name), withSize: false),
      );

      expect(error, isA<DownloadNetworkException>());
      expect(server.requests, hasLength(1), reason: 'the probe has no retry');
    });
  });

  group('TLS that fails (an https URL to a plain HTTP server)', () {
    for (final withSize in [true, false]) {
      test(
        withSize
            ? 'the transfer: a network error after the attempts'
            : 'the probe: a network error',
        () async {
          final server = await fakeServer();
          server.files[name] = ServedFile(bytes);
          final url = server.url(name).replace(scheme: 'https');

          final error = errorOf(
            await download(newStore(), url, withSize: withSize),
          );

          expect(
            error,
            isA<DownloadNetworkException>().having(
              (e) => e.partialBytes,
              'partialBytes',
              0,
            ),
          );
          expect(server.requests, isEmpty, reason: 'no request got through');
        },
      );
    }
  });

  test('bytes lost on the way to disk: the size check before the hash '
      'fails, the .part deleted', () async {
    final server = await fakeServer();
    server.files[name] = ServedFile(bytes, chunkBytes: 16 * 1024);
    final url = server.url(name);
    final store = newStore(
      openPart: (file, {required append}) async => _LossySink(
        await file.open(mode: append ? FileMode.append : FileMode.write),
      ),
    );

    final error = errorOf(await download(store, url));

    expect(
      error,
      isA<SizeMismatchException>()
          .having((e) => e.expected, 'expected', n)
          .having((e) => e.actual, 'actual', lessThan(n))
          .having((e) => e.source, 'source', url.toString()),
    );
    expect(partFor(url).existsSync(), isFalse);
    expect(stored().existsSync(), isFalse);
  });

  group('import', () {
    late Directory picked;

    setUp(() => picked = Directory.systemTemp.createTempSync('edges_import'));
    tearDown(() => picked.deleteSync(recursive: true));

    test('an old record that cannot be deleted aborts the import before the '
        'rename (as the Android extraction does): the old file and its record '
        'stay together, so the next scan cannot vouch for other bytes of the '
        'same size', () async {
      final store = newStore();
      final first = File('${picked.path}/$name')..writeAsBytesSync(bytes);
      final imported = await store.importCustomFile(first.path);
      final kept = (imported as Ok<CustomModelFile>).value;
      final record = File('${stored().path}.sha256');
      Process.runSync('chflags', ['uchg', record.path]);
      addTearDown(() => Process.runSync('chflags', ['nouchg', record.path]));
      // Other bytes of the same size: only the record tells them apart.
      final other = testBytes(n, seed: 6);
      final second = Directory('${picked.path}/second')..createSync();
      final next = File('${second.path}/$name')..writeAsBytesSync(other);

      final error = errorOf(await store.importCustomFile(next.path));

      expect(
        error,
        isA<StoreWriteException>().having(
          (e) => e.message,
          'message',
          contains('the old checksum record cannot be replaced'),
        ),
      );
      expect(stored().readAsBytesSync(), bytes, reason: 'not renamed');
      expect(record.readAsStringSync(), '$sha  $name\n');
      expect(
        store.customFile.value,
        isA<StoreFileReady>(),
        reason: 'the state from before the import',
      );
      expect(File('${stored().path}.import').existsSync(), isFalse);
      expect(
        next.existsSync(),
        isTrue,
        reason: 'the picked file is the user\'s',
      );

      // The next launch's scan: the kept file still matches its record.
      expect(
        await newStore().useCustomFile(kept),
        isA<Ok<String?>>().having((r) => r.value, 'path', stored().path),
      );
    }, skip: Platform.isMacOS ? false : 'chflags uchg is macOS only');

    test('a store folder that cannot be written: a write error (not a '
        'full disk), the state kept', () async {
      final source = File('${picked.path}/$name')..writeAsBytesSync(bytes);
      customDir().createSync(recursive: true);
      Process.runSync('chmod', ['555', customDir().path]);
      addTearDown(() => Process.runSync('chmod', ['755', customDir().path]));
      final store = newStore();

      final error = errorOf(await store.importCustomFile(source.path));

      expect(error, isA<StoreWriteException>());
      expect(error.toString(), startsWith('Could not write the custom model:'));
      expect(store.customFile.value, isA<StoreFileMissing>());
      expect(store.busy.value, isFalse);
    });
  });

  group('the store', () {
    test('once closed refuses every operation; pruning does nothing', () async {
      final store = newStore();
      final kept = File('${customDir().path}/old.litertlm')
        ..createSync(recursive: true);
      Directory('${root.path}/whisperBase').createSync();
      await store.close();

      for (final result in [
        await store.useCustomFile(null),
        await store.importCustomFile(kept.path),
        await store.downloadCustomFile(Uri.parse('http://127.0.0.1:9/x')),
      ]) {
        expect(
          errorOf(result),
          isA<UnexpectedError>().having(
            (e) => '$e',
            'message',
            'Bad state: store closed',
          ),
        );
      }
      expect(await store.pruneOldModelFolders(), isEmpty);
      await store.pruneCustom(keep: const {});
      expect(kept.existsSync(), isTrue);
      expect(Directory('${root.path}/whisperBase').existsSync(), isTrue);
    });

    test('a second close is the first one', () async {
      final store = newStore();

      final first = store.close();
      final second = store.close();

      expect(identical(first, second), isTrue);
      await second;
    });

    test('useCustomFile(null) clears the slot', () async {
      final picked = Directory.systemTemp.createTempSync('edges_clear');
      addTearDown(() => picked.deleteSync(recursive: true));
      final store = newStore();
      final source = File('${picked.path}/$name')..writeAsBytesSync(bytes);
      expect(
        await store.importCustomFile(source.path),
        isA<Ok<CustomModelFile>>(),
      );
      expect(store.customFile.value, isA<StoreFileReady>());

      expect(await store.useCustomFile(null), isA<Ok<String?>>());
      expect(store.customFile.value, isA<StoreFileMissing>());
    });

    test('a scan that fails is an error; the state is kept', () async {
      final store = newStore(
        rootOf: () async =>
            throw const FileSystemException('no app support', '/x'),
      );

      final result = await store.useCustomFile(
        CustomModelFile(
          name: name,
          sizeBytes: n,
          sha256: sha,
          checksumMatched: false,
        ),
      );

      expect(errorOf(result), isA<FileSystemException>());
      expect(store.customFile.value, isA<StoreFileMissing>());
    });

    test('pruneOldModelFolders: a folder that cannot be deleted is logged, not '
        'thrown', () async {
      final locked = Directory('${root.path}/whisperBase/locked')
        ..createSync(recursive: true);
      File('${locked.path}/file.bin').writeAsStringSync('x');
      Process.runSync('chmod', ['555', locked.path]);
      addTearDown(() => Process.runSync('chmod', ['755', locked.path]));

      expect(await newStore().pruneOldModelFolders(), isEmpty);
      expect(locked.existsSync(), isTrue);
    });

    test('pruneCustom: folders are skipped, a file that cannot be deleted is '
        'logged, not thrown', () async {
      final sub = Directory('${customDir().path}/sub')
        ..createSync(recursive: true);
      final old = File('${customDir().path}/old.litertlm')..createSync();
      Process.runSync('chmod', ['555', customDir().path]);
      addTearDown(() => Process.runSync('chmod', ['755', customDir().path]));

      await newStore().pruneCustom(keep: const {});

      expect(old.existsSync(), isTrue);
      expect(sub.existsSync(), isTrue);
    });

    test('pruneCustom: a custom folder that cannot be listed is logged, not '
        'thrown', () async {
      customDir().createSync(recursive: true);
      Process.runSync('chmod', ['311', customDir().path]);
      addTearDown(() => Process.runSync('chmod', ['755', customDir().path]));

      await newStore().pruneCustom(keep: const {});
    });

    test(
      'pruneCustom: nothing is deleted while a custom operation runs',
      () async {
        final server = await fakeServer();
        server.files[name] = ServedFile(
          bytes,
          chunkDelay: const Duration(milliseconds: 10),
        );
        final old = File('${customDir().path}/old.litertlm')
          ..createSync(recursive: true);
        final store = newStore();

        final running = download(store, server.url(name));
        await store.pruneCustom(keep: const {});

        expect(old.existsSync(), isTrue);
        expect(await running, isA<Ok<CustomModelFile>>());
      },
    );
  });
}
