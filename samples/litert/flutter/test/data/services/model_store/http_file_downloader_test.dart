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

import 'package:fake_async/fake_async.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/services/model_store/http_file_downloader.dart';
import 'package:litert_edge_demos/data/services/model_store/store_operation.dart';
import 'package:litert_edge_demos/domain/models/provisioning.dart';

import '../../../support/fake_model_server.dart';
import '../../../support/recording_reporter.dart';
import '../../../support/scripted_http_server.dart';
import '../../../support/test_bytes.dart';

/// A part sink that counts writes in flight at once (backpressure keeps it
/// at one) and takes [delay] per write (a slow disk).
final class _SlowSink implements PartSink {
  _SlowSink(this._file, this.delay);

  final RandomAccessFile _file;
  final Duration delay;
  int _inFlight = 0;
  int maxInFlight = 0;

  @override
  Future<void> add(List<int> bytes) async {
    _inFlight++;
    if (_inFlight > maxInFlight) maxInFlight = _inFlight;
    try {
      await Future<void>.delayed(delay);
      await _file.writeFrom(bytes);
    } finally {
      _inFlight--;
    }
  }

  @override
  Future<void> close() => _file.close();
}

/// [HttpFileDownloader] on its own: resume, reconnects, the stall watchdog
/// on the injected clock, the Range and size checks, the backpressured
/// reader, cancel, the size probe and the client's lifecycle.
void main() {
  late Directory dir;
  late RecordingReporter reporter;
  late CancelToken cancel;
  final downloaders = <HttpFileDownloader>[];
  final closers = <Future<void> Function()>[];

  const name = 'file.bin';
  final bytes = testBytes(160 * 1024, seed: 31);
  final n = bytes.length;

  setUp(() {
    dir = Directory.systemTemp.createTempSync('http_file_downloader');
    reporter = RecordingReporter();
    cancel = CancelToken();
  });

  tearDown(() async {
    for (final downloader in downloaders) {
      downloader.close();
    }
    downloaders.clear();
    for (final close in closers.reversed) {
      await close();
    }
    closers.clear();
    dir.deleteSync(recursive: true);
  });

  HttpFileDownloader newDownloader({
    int maxAttempts = 3,
    Duration Function(int attempt)? retryDelay,
    Duration stallTimeout = const Duration(seconds: 30),
    PartOpener openPart = openRandomAccessPart,
    Duration Function()? clock,
    HttpClient Function()? httpClient,
  }) {
    final downloader = HttpFileDownloader(
      httpClient: httpClient,
      maxAttempts: maxAttempts,
      retryDelay: retryDelay ?? (_) => Duration.zero,
      stallTimeout: stallTimeout,
      openPart: openPart,
      clock: clock,
    );
    downloaders.add(downloader);
    return downloader;
  }

  Future<FakeModelServer> fakeServer([ServedFile? file]) async {
    final server = await FakeModelServer.start();
    closers.add(server.close);
    server.files[name] = file ?? ServedFile(bytes);
    return server;
  }

  Future<ScriptedHttpServer> scriptedServer(
    Future<void> Function(ScriptedRequest request) handler,
  ) async {
    final server = await ScriptedHttpServer.start(handler);
    closers.add(server.close);
    return server;
  }

  File part() => File('${dir.path}/$name.part');

  File leavePart(List<int> content) => part()..writeAsBytesSync(content);

  RemoteFile remote(Uri url) => RemoteFile(name: name, url: url, sizeBytes: n);

  Future<void> download(HttpFileDownloader downloader, Uri url) =>
      downloader.download(remote(url), part(), cancel, reporter);

  Matcher downloading(int received, {int attempt = 1}) =>
      isA<StoreFileDownloading>()
          .having((s) => s.received, 'received', received)
          .having((s) => s.total, 'total', n)
          .having((s) => s.attempt, 'attempt', attempt);

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

  group('download', () {
    test('fetches the file into the .part: the attempt reported at once, '
        'each chunk as progress', () async {
      final server = await fakeServer(ServedFile(bytes, chunkBytes: 32 * 1024));

      await download(newDownloader(), server.url(name));

      expect(part().readAsBytesSync(), bytes);
      expect(reporter.reported, [downloading(0)]);
      expect(reporter.progressed.last, downloading(n));
      final received = [
        for (final s in reporter.progressed)
          (s as StoreFileDownloading).received,
      ];
      expect(received, orderedEquals([...received]..sort()));
      expect(server.requestsFor(name).single.range, isNull);
    });

    test('resumes what the .part holds with a Range request', () async {
      final server = await fakeServer();
      leavePart(bytes.sublist(0, 50000));

      await download(newDownloader(), server.url(name));

      expect(part().readAsBytesSync(), bytes);
      expect(server.requestsFor(name).single.range, 'bytes=50000-');
      expect(reporter.reported, [downloading(50000)]);
    });

    test('a complete .part: no request, nothing reported', () async {
      final server = await fakeServer();
      leavePart(bytes);

      await download(newDownloader(), server.url(name));

      expect(server.requests, isEmpty);
      expect(reporter.all, isEmpty);
    });

    test('a .part longer than the file is deleted and the file fetched from '
        '0', () async {
      final server = await fakeServer();
      leavePart(testBytes(n + 1));

      await download(newDownloader(), server.url(name));

      expect(part().readAsBytesSync(), bytes);
      expect(server.requestsFor(name).single.range, isNull);
    });

    test('a dropped connection: the reconnect reported at once, after the '
        'retry delay for that attempt, resuming', () async {
      final server = await fakeServer(ServedFile(bytes, dropAfter: 64 * 1024));
      final delays = <int>[];
      final downloader = newDownloader(
        retryDelay: (attempt) {
          delays.add(attempt);
          return Duration.zero;
        },
      );

      await download(downloader, server.url(name));

      expect(part().readAsBytesSync(), bytes);
      expect(delays, [1]);
      expect(reporter.reported, [
        downloading(0),
        downloading(64 * 1024, attempt: 2), // the reconnect is scheduled
        downloading(64 * 1024, attempt: 2), // the attempt starts
      ]);
      expect(server.requestsFor(name).last.range, 'bytes=${64 * 1024}-');
    });

    test('after the last attempt: a network error, the bytes kept', () async {
      final server = await fakeServer(
        ServedFile(bytes, dropAfter: 64 * 1024, dropTimes: 2),
      );

      final error = await download(
        newDownloader(maxAttempts: 2),
        server.url(name),
      ).then<Object?>((_) => null, onError: (Object e) => e);

      expect(
        error,
        isA<DownloadNetworkException>().having(
          (e) => e.partialBytes,
          'partialBytes',
          128 * 1024,
        ),
      );
      expect(part().lengthSync(), 128 * 1024);
    });

    group('the stall watchdog (fake time, real I/O)', () {
      // Checks every 100 ms.
      const stall = Duration(milliseconds: 600);
      const check = Duration(milliseconds: 100);
      late FakeAsync async;
      late Duration jumped;

      setUp(() {
        async = FakeAsync();
        jumped = Duration.zero;
      });

      /// The downloader on fake time: the watchdog's timer is fake, its
      /// clock is fake time plus [jumped].
      HttpFileDownloader onFakeTime() => newDownloader(
        stallTimeout: stall,
        clock: () => async.elapsed + jumped,
      );

      /// Lets real I/O run, then the fake zone's microtasks and due timers
      /// (the I/O callbacks land there), until [condition] holds. Fake time
      /// does not move.
      Future<void> pump(bool Function() condition, String what) async {
        final watch = Stopwatch()..start();
        while (true) {
          async.flushMicrotasks();
          async.elapse(Duration.zero);
          if (condition()) return;
          if (watch.elapsed > const Duration(seconds: 10)) fail('No $what');
          await Future<void>.delayed(const Duration(milliseconds: 5));
        }
      }

      /// [pump] for a short real while: whatever was going to happen has.
      Future<void> settle() async {
        final watch = Stopwatch()..start();
        await pump(
          () => watch.elapsed > const Duration(milliseconds: 100),
          'time',
        );
      }

      test('silence on the injected clock: a stall after the timeout, and '
          'the reconnect resumes', () async {
        final server = await fakeServer(
          ServedFile(bytes, stallAfter: 64 * 1024),
        );
        final downloader = onFakeTime();
        Object? failure;
        var done = false;
        async.run(
          (_) => unawaited(
            download(downloader, server.url(name)).then<void>(
              (_) {
                done = true;
              },
              onError: (Object e) {
                failure = e;
              },
            ),
          ),
        );
        await pump(
          () => part().existsSync() && part().lengthSync() == 64 * 1024,
          'the first 64 KB',
        );
        // The last chunk's activity is recorded at fake time zero.
        await settle();

        async.elapse(stall - check);
        await settle();
        expect(server.requests, hasLength(1), reason: 'not silent long enough');

        server.files[name] = ServedFile(bytes);
        async.elapse(check);
        await pump(() => done || failure != null, 'the end of the download');

        expect(failure, isNull);
        expect(part().readAsBytesSync(), bytes);
        expect(server.requestsFor(name).map((r) => r.range), [
          null,
          'bytes=${64 * 1024}-',
        ]);
      });

      test('a clock jump with no check in between (the app suspended, e.g. '
          'the phone backgrounded) starts a fresh window: no retry used up; '
          'the timeout of silence after it is a stall', () async {
        final server = await fakeServer(
          ServedFile(bytes, stallAfter: 64 * 1024),
        );
        final downloader = onFakeTime();
        Object? failure;
        var done = false;
        async.run(
          (_) => unawaited(
            download(downloader, server.url(name)).then<void>(
              (_) {
                done = true;
              },
              onError: (Object e) {
                failure = e;
              },
            ),
          ),
        );
        await pump(
          () => part().existsSync() && part().lengthSync() == 64 * 1024,
          'the first 64 KB',
        );
        // The last chunk's activity is recorded at fake time zero.
        await settle();
        async.elapse(check * 2);

        // Suspended for an hour: the clock moves, no timer runs; the next
        // check runs late.
        jumped += const Duration(hours: 1);
        async.elapse(check);
        await settle();
        expect(
          server.requests,
          hasLength(1),
          reason: 'the suspension is not a stall',
        );

        // Silence in the fresh window, just short of the timeout.
        async.elapse(stall - check);
        await settle();
        expect(server.requests, hasLength(1));
        expect(reporter.reported, [downloading(0)], reason: 'no reconnect');

        server.files[name] = ServedFile(bytes);
        async.elapse(check);
        await pump(() => done || failure != null, 'the end of the download');

        expect(failure, isNull);
        expect(part().readAsBytesSync(), bytes);
        expect(reporter.reported, [
          downloading(0),
          downloading(64 * 1024, attempt: 2),
          downloading(64 * 1024, attempt: 2),
        ], reason: 'one reconnect: the real stall\'s');
      });
    });

    test('a 206 from another offset than asked: the .part deleted, the file '
        'fetched from 0', () async {
      final server = await scriptedServer(
        (r) => r.range == null
            ? serveFrom(r, 0)
            : r.respond(
                206,
                headers: {
                  'Content-Range': 'bytes 0-${n - 1}/$n',
                  'Content-Length': '$n',
                },
                body: bytes,
              ),
      );
      leavePart(bytes.sublist(0, 1000));

      await download(newDownloader(), server.url(name));

      expect(part().readAsBytesSync(), bytes);
      expect(server.requests.map((r) => r.range), ['bytes=1000-', null]);
    });

    test('a 416 for the resume range: the .part deleted, the file fetched '
        'from 0', () async {
      final server = await scriptedServer(
        (r) => r.range == null
            ? serveFrom(r, 0)
            : r.respond(416, headers: {'Content-Length': '0'}),
      );
      leavePart(bytes.sublist(0, 1000));

      await download(newDownloader(), server.url(name));

      expect(part().readAsBytesSync(), bytes);
      expect(server.requests.map((r) => r.range), ['bytes=1000-', null]);
    });

    test('a 206 for another total: a size mismatch, not retried, the .part '
        'kept', () async {
      final server = await fakeServer(ServedFile(testBytes(n + 5)));
      leavePart(bytes.sublist(0, 1000));

      await expectLater(
        download(newDownloader(), server.url(name)),
        throwsA(
          isA<SizeMismatchException>()
              .having((e) => e.actual, 'actual', n + 5)
              .having((e) => e.source, 'source', '127.0.0.1'),
        ),
      );
      expect(server.requestsFor(name), hasLength(1));
      expect(part().lengthSync(), 1000);
    });

    test('a Content-Length that does not fit: a size mismatch before '
        'anything is written', () async {
      final server = await fakeServer(ServedFile(bytes, reportedLength: n - 1));

      await expectLater(
        download(newDownloader(), server.url(name)),
        throwsA(isA<SizeMismatchException>()),
      );
      expect(part().existsSync(), isFalse);
    });

    test('HTTP 404 is not retried', () async {
      final server = await fakeServer(ServedFile(bytes, status: 404));

      await expectLater(
        download(newDownloader(), server.url(name)),
        throwsA(
          isA<DownloadHttpException>().having(
            (e) => e.statusCode,
            'statusCode',
            404,
          ),
        ),
      );
      expect(server.requestsFor(name), hasLength(1));
    });

    test('HTTP 503 is retried, then a network error naming it', () async {
      final server = await fakeServer(ServedFile(bytes, status: 503));

      await expectLater(
        download(newDownloader(), server.url(name)),
        throwsA(
          isA<DownloadNetworkException>().having(
            (e) => e.cause,
            'cause',
            isA<DownloadHttpException>(),
          ),
        ),
      );
      expect(server.requestsFor(name), hasLength(3));
    });

    test(
      'text/html: a web page instead of the file, nothing written',
      () async {
        final server = await fakeServer(
          ServedFile(
            Uint8List.fromList('<html></html>'.codeUnits),
            contentType: 'text/html',
          ),
        );

        await expectLater(
          download(newDownloader(), server.url(name)),
          throwsA(isA<HtmlInsteadOfFileException>()),
        );
        expect(part().existsSync(), isFalse);
      },
    );

    test('a page in the first chunk of an octet-stream: sniffed', () async {
      final server = await fakeServer(
        ServedFile(
          Uint8List.fromList(' <!doctype html><p>no</p>'.codeUnits),
          reportedLength: n,
        ),
      );

      await expectLater(
        download(newDownloader(), server.url(name)),
        throwsA(isA<HtmlInsteadOfFileException>()),
      );
    });

    test('more bytes than the file (no Content-Length): a size mismatch, '
        'the .part deleted, not retried', () async {
      final server = await scriptedServer(
        (r) => r.respond(200, body: [...bytes, 1, 2, 3]),
      );

      await expectLater(
        download(newDownloader(), server.url(name)),
        throwsA(
          isA<SizeMismatchException>().having(
            (e) => e.actual,
            'actual',
            greaterThan(n),
          ),
        ),
      );
      expect(part().existsSync(), isFalse);
      expect(server.requests, hasLength(1));
    });

    test('a body without Content-Length that ends early is resumed', () async {
      final server = await scriptedServer(
        (r) => r.range == null
            ? r.respond(200, body: bytes.sublist(0, 4096))
            : serveFrom(r, 4096),
      );

      await download(newDownloader(), server.url(name));

      expect(part().readAsBytesSync(), bytes);
      expect(server.requests.map((r) => r.range), [null, 'bytes=4096-']);
    });

    test('redirects keep the Range header; ten hops is the limit', () async {
      final server = await fakeServer(ServedFile(bytes, dropAfter: 1000));

      await download(newDownloader(), server.url('redirect/$name'));

      expect(part().readAsBytesSync(), bytes);
      expect(server.requests.map((r) => r.range), [
        null,
        null,
        'bytes=1000-',
        'bytes=1000-',
      ]);

      part().deleteSync();
      await expectLater(
        download(newDownloader(), server.url('${'redirect/' * 10}$name')),
        throwsA(
          isA<DownloadHttpException>().having(
            (e) => e.statusCode,
            'statusCode',
            HttpStatus.loopDetected,
          ),
        ),
      );
    });

    group('redirect checks', () {
      /// A client that sends every request, HTTPS ones too, as plain HTTP
      /// to [server]: an `https://` URL then reaches the local server, and
      /// its redirects are the downloader's to judge.
      HttpClient Function() routedTo(ScriptedHttpServer server) =>
          () => HttpClient()
            ..findProxy = ((_) => 'DIRECT')
            ..connectionFactory = (uri, proxyHost, proxyPort) =>
                Socket.startConnect(
                  InternetAddress.loopbackIPv4,
                  server.url('').port,
                );

      Future<ScriptedHttpServer> redirectingTo(String location) =>
          scriptedServer(
            (r) => r.path == '/$name'
                ? r.respond(302, headers: {'Location': location})
                : serveFrom(r, 0),
          );

      TypeMatcher<UnsafeRedirectException> refused(String from, String to) =>
          isA<UnsafeRedirectException>()
              .having((e) => e.fileName, 'fileName', name)
              .having((e) => e.from.toString(), 'from', from)
              .having((e) => e.to.toString(), 'to', to);

      test('to a scheme other than HTTP(S): refused before any request to '
          'it, the download and the size probe alike', () async {
        final server = await redirectingTo('file:///etc/passwd');
        final url = server.url(name);

        await expectLater(
          download(newDownloader(), url),
          throwsA(
            refused('$url', 'file:///etc/passwd').having(
              (e) => e.message,
              'message',
              'Downloading $name stopped: ${url.host} redirected it to a '
                  'file: link, not HTTP(S).',
            ),
          ),
        );
        await expectLater(
          newDownloader().probeSize(name, url, cancel),
          throwsA(refused('$url', 'file:///etc/passwd')),
        );
        expect(server.requests.map((r) => r.path), ['/$name', '/$name']);
        expect(part().existsSync(), isFalse);
      });

      test('HTTPS down to plain HTTP: refused, the bytes never '
          'fetched', () async {
        final server = await redirectingTo('http://cdn.test/cdn/$name');
        final url = Uri.parse('https://models.test/$name');

        await expectLater(
          download(newDownloader(httpClient: routedTo(server)), url),
          throwsA(
            refused('$url', 'http://cdn.test/cdn/$name').having(
              (e) => e.message,
              'message',
              'Downloading $name stopped: models.test redirected it from '
                  'HTTPS to plain HTTP (http://cdn.test), where the bytes '
                  'could be changed on the way.',
            ),
          ),
        );
        expect(server.requests.map((r) => r.path), ['/$name']);
        expect(part().existsSync(), isFalse);
      });

      test('HTTPS to HTTPS on another host (a CDN) is followed', () async {
        final server = await redirectingTo('https://cdn.test/cdn/$name');
        final url = Uri.parse('https://models.test/$name');

        await download(newDownloader(httpClient: routedTo(server)), url);

        expect(part().readAsBytesSync(), bytes);
        expect(server.requests.map((r) => r.path), ['/$name', '/cdn/$name']);
      });

      test('plain HTTP up to HTTPS is followed', () async {
        final server = await redirectingTo('https://cdn.test/cdn/$name');

        await download(
          newDownloader(httpClient: routedTo(server)),
          Uri.parse('http://models.test/$name'),
        );

        expect(part().readAsBytesSync(), bytes);
      });
    });

    test('backpressure: one write at a time on a slow disk', () async {
      final server = await fakeServer(ServedFile(bytes, chunkBytes: 8 * 1024));
      _SlowSink? sink;
      final downloader = newDownloader(
        openPart: (file, {required append}) async => sink = _SlowSink(
          await file.open(mode: append ? FileMode.append : FileMode.write),
          const Duration(milliseconds: 5),
        ),
      );

      await download(downloader, server.url(name));

      expect(part().readAsBytesSync(), bytes);
      expect(sink!.maxInFlight, 1);
    });
  });

  group('cancel', () {
    test('before the request: a cancel, no request made', () async {
      final server = await fakeServer();
      cancel.cancel();

      await expectLater(
        download(newDownloader(), server.url(name)),
        throwsA(isA<OperationCancelledException>()),
      );
      expect(server.requests, isEmpty);
    });

    test('during the body: a cancel, the .part kept, no progress after '
        'it', () async {
      final server = await fakeServer(
        ServedFile(
          bytes,
          chunkBytes: 8 * 1024,
          chunkDelay: const Duration(milliseconds: 20),
        ),
      );
      final watching = _CancelAt(reporter, 32 * 1024, cancel);

      await expectLater(
        newDownloader().download(
          remote(server.url(name)),
          part(),
          cancel,
          watching,
        ),
        throwsA(isA<OperationCancelledException>()),
      );
      expect(cancel.cancelled, isTrue);
      expect(watching.progressAfterCancel, 0);
      expect(part().lengthSync(), inInclusiveRange(32 * 1024, n - 1));
    });

    test('while waiting to reconnect: a cancel', () async {
      final server = await fakeServer(ServedFile(bytes, dropAfter: 1000));
      final downloader = newDownloader(
        retryDelay: (_) {
          scheduleMicrotask(cancel.cancel);
          return const Duration(hours: 1);
        },
      );

      await expectLater(
        download(downloader, server.url(name)),
        throwsA(isA<OperationCancelledException>()),
      );
      expect(part().lengthSync(), 1000);
    });
  });

  group('probeSize', () {
    Future<int> probe(HttpFileDownloader downloader, Uri url) =>
        downloader.probeSize(name, url, cancel);

    test('asks for one byte; a 206 gives the total', () async {
      final server = await scriptedServer(
        (r) => r.respond(
          206,
          headers: {'Content-Range': 'bytes 0-0/$n', 'Content-Length': '1'},
          body: [0],
        ),
      );

      expect(await probe(newDownloader(), server.url(name)), n);
      expect(server.requests.single.range, 'bytes=0-0');
    });

    test('a 200 gives its Content-Length', () async {
      final server = await fakeServer();

      expect(await probe(newDownloader(), server.url(name)), n);
    });

    test('text/html: a web page instead of the file', () async {
      final server = await fakeServer(
        ServedFile(Uint8List(10), contentType: 'text/html'),
      );

      await expectLater(
        probe(newDownloader(), server.url(name)),
        throwsA(isA<HtmlInsteadOfFileException>()),
      );
    });

    test('HTTP 404', () async {
      final server = await fakeServer();

      await expectLater(
        probe(newDownloader(), server.url('missing')),
        throwsA(isA<DownloadHttpException>()),
      );
    });

    test('no size reported', () async {
      final server = await scriptedServer((r) => r.respond(200, body: [1]));

      await expectLater(
        probe(newDownloader(), server.url(name)),
        throwsA(isA<UnknownSizeException>()),
      );
    });

    test('refused: a network error with nothing kept', () async {
      final socket = await ServerSocket.bind(InternetAddress.loopbackIPv4, 0);
      final url = Uri.parse('http://127.0.0.1:${socket.port}/$name');
      await socket.close();

      await expectLater(
        probe(newDownloader(), url),
        throwsA(
          isA<DownloadNetworkException>().having(
            (e) => e.partialBytes,
            'partialBytes',
            0,
          ),
        ),
      );
    });

    test('cancelled before: no request', () async {
      final server = await fakeServer();
      cancel.cancel();

      await expectLater(
        probe(newDownloader(), server.url(name)),
        throwsA(isA<OperationCancelledException>()),
      );
      expect(server.requests, isEmpty);
    });
  });

  test('one client, made on first use, shared by the probe and the '
      'transfer, force-closed by close()', () async {
    final server = await fakeServer();
    final made = <HttpClient>[];
    final downloader = newDownloader(
      httpClient: () {
        final client = HttpClient();
        made.add(client);
        return client;
      },
    );
    expect(made, isEmpty, reason: 'nothing downloaded yet');

    final url = server.url(name);
    expect(await downloader.probeSize(name, url, cancel), n);
    await download(downloader, url);
    expect(made, hasLength(1));

    downloader.close();
    await expectLater(
      Future.sync(() => made.single.getUrl(url)),
      throwsStateError,
    );
  });

  group('names', () {
    test('urlTag: eight hex digits of the URL, distinct per URL', () {
      final a = Uri.parse('https://example.com/a.litertlm');
      final b = Uri.parse('https://example.com/b.litertlm');

      expect(urlTag(a), sha256Hex(a.toString().codeUnits).substring(0, 8));
      expect(urlTag(a), matches(RegExp(r'^[0-9a-f]{8}$')));
      expect(urlTag(a), isNot(urlTag(b)));
    });

    test('partFileFor: <target>.<urlTag>.part', () {
      final url = Uri.parse('https://example.com/m.litertlm');

      expect(
        HttpFileDownloader.partFileFor(File('/s/custom/m.litertlm'), url).path,
        '/s/custom/m.litertlm.${urlTag(url)}.part',
      );
    });
  });

  test('openRandomAccessPart truncates or appends; the bytes are on disk '
      'once closed', () async {
    final file = File('${dir.path}/p.part');
    for (final (append, chunk) in [
      (false, [1, 2]),
      (true, [3]),
      (false, [4]),
    ]) {
      final sink = await openRandomAccessPart(file, append: append);
      await sink.add(chunk);
      await sink.close();
    }
    expect(file.readAsBytesSync(), [4]);

    final sink = await openRandomAccessPart(file, append: true);
    await sink.add([5, 6]);
    await sink.close();
    expect(file.readAsBytesSync(), [4, 5, 6]);
  });
}

/// Cancels [cancel] once [reporter] has seen [atBytes] received, then
/// counts the progress that still arrives.
final class _CancelAt implements FileStateReporter {
  _CancelAt(this._inner, this.atBytes, this.cancel);

  final RecordingReporter _inner;
  final int atBytes;
  final CancelToken cancel;
  int progressAfterCancel = 0;

  @override
  void report(StoreFileState state) => _inner.report(state);

  @override
  void progress(StoreFileState state) {
    if (cancel.cancelled) progressAfterCancel++;
    _inner.progress(state);
    if (state case StoreFileDownloading(:final received)
        when received >= atBytes) {
      cancel.cancel();
    }
  }
}
