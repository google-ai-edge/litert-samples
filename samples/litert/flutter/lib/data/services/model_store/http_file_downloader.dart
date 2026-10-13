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

import 'package:crypto/crypto.dart' as crypto;
import 'package:flutter/foundation.dart';

import '../../../domain/models/provisioning.dart';
import '../../../utils/stall_watchdog.dart';
import 'store_operation.dart';

String _mb(int bytes) => '${(bytes / 1e6).toStringAsFixed(1)} MB';

/// Where a download writes; [add] completes once the bytes are written.
abstract interface class PartSink {
  Future<void> add(List<int> bytes);
  Future<void> close();
}

/// Opens the `.part` file of a download; tests replace it to play a full
/// disk.
typedef PartOpener = Future<PartSink> Function(
  File file, {
  required bool append,
});

/// The default [PartOpener]: a `RandomAccessFile`, appended to or truncated.
Future<PartSink> openRandomAccessPart(
  File file, {
  required bool append,
}) async => _RandomAccessPartSink(
  await file.open(mode: append ? FileMode.append : FileMode.write),
);

final class _RandomAccessPartSink implements PartSink {
  _RandomAccessPartSink(this._file);

  final RandomAccessFile _file;

  @override
  Future<void> add(List<int> bytes) => _file.writeFrom(bytes);

  /// Flushed first: the file is verified and renamed into place next, and
  /// its checksum record is written with `flush: true`, so the bytes it
  /// vouches for must be on disk too.
  @override
  Future<void> close() async {
    try {
      await _file.flush();
    } finally {
      await _file.close();
    }
  }
}

HttpClient _defaultClient() => HttpClient()
  ..connectionTimeout = const Duration(seconds: 30)
  ..autoUncompress = false
  ..userAgent = 'litert_edge_demos model download';

/// The default wait before reconnect [attempt] + 1: 2 s, 4 s, 8 s, …
Duration defaultRetryDelay(int attempt) => Duration(seconds: 1 << attempt);

Duration Function() _stopwatchClock() {
  final watch = Stopwatch()..start();
  return () => watch.elapsed;
}

/// Eight hex digits of [url]'s SHA-256: a `.part` resumes only the download
/// it was started for.
String urlTag(Uri url) =>
    crypto.sha256.convert(url.toString().codeUnits).toString().substring(0, 8);

/// Errors after which the same request may succeed: the connection dropped,
/// stalled, or the server had a transient failure.
final class _Retryable implements Exception {
  const _Retryable(this.cause);

  final Object cause;

  @override
  String toString() => '$cause';
}

final class _Stalled implements Exception {
  const _Stalled(this.after);

  final Duration after;

  @override
  String toString() => 'no data for ${after.inSeconds} s';
}

final class _ClosedEarly implements Exception {
  const _ClosedEarly(this.received, this.expected);

  final int received;
  final int expected;

  @override
  String toString() =>
      'the connection closed at ${_mb(received)} of ${_mb(expected)}';
}

/// The server sent more bytes than the expected size.
final class _Overflow implements Exception {
  const _Overflow(this.atLeast);

  final int atLeast;
}

/// What a download must produce: [sizeBytes] bytes from [url]; [name] is
/// the file the messages name.
final class const RemoteFile({
  required final String name,
  required final Uri url,
  required final int sizeBytes,
});

/// HTTP downloads of large files into a `.part` file, as the model store
/// runs them: HTTP Range resume of what the `.part` holds, up to
/// [maxAttempts] attempts ([retryDelay] apart, each resuming), a stall
/// watchdog ([StallWatchdog]: no data for the stall timeout, measured on the
/// injected clock; an app suspension is not a stall), the client's connect
/// timeout, redirects followed by hand so the
/// Range header survives each hop (HTTP(S) only, never from HTTPS down to
/// HTTP), and a backpressured body reader. What
/// can be checked before the bytes are hashed is: the status, Content-Range,
/// Content-Length, a web page instead of the file, a body longer than the
/// file. Hashes nothing; the caller verifies the finished `.part`.
///
/// State goes to a [FileStateReporter]: [StoreFileDownloading] at once when
/// an attempt starts or a reconnect is scheduled, and as progress after each
/// chunk written. Logs as `[ModelStore]` (the store's transfer log).
final class HttpFileDownloader {
  HttpFileDownloader({
    HttpClient Function()? httpClient,
    this._stallTimeout = defaultStallTimeout,
    this._maxAttempts = defaultMaxAttempts,
    this._retryDelay = defaultRetryDelay,
    this._openPart = openRandomAccessPart,
    Duration Function()? clock,
  }) : _newClient = httpClient ?? _defaultClient,
       _now = clock ?? _stopwatchClock();

  static const defaultStallTimeout = Duration(seconds: 30);
  static const defaultMaxAttempts = 3;

  final HttpClient Function() _newClient;
  final Duration _stallTimeout;

  /// Attempts per file before a network error is shown (each resumes).
  final int _maxAttempts;
  final Duration Function(int attempt) _retryDelay;
  final PartOpener _openPart;

  /// Monotonic time, for the stall watchdog and the transfer log.
  final Duration Function() _now;

  HttpClient? _client;

  HttpClient get _http => _client ??= _newClient();

  /// The `.part` a download of [url] into [target] resumes:
  /// `<target>.<urlTag>.part`.
  static File partFileFor(File target, Uri url) =>
      File('${target.path}.${urlTag(url)}.part');

  /// Force-closes the HTTP client: a request or body read in flight fails.
  /// The store cancels its operation first, so that ends as a cancel.
  void close() => _client?.close(force: true);

  /// The total size of [url]: a one-byte Range request (the 206's
  /// Content-Range), else a 200's Content-Length. No reconnects.
  Future<int> probeSize(String name, Uri url, CancelToken cancel) async {
    final HttpClientResponse response;
    try {
      response = await _open(name, url, 0, cancel, range: 'bytes=0-0');
    } on SocketException catch (e) {
      throw DownloadNetworkException(name, e.message, 0);
    } on HttpException catch (e) {
      throw DownloadNetworkException(name, e.message, 0);
    } on TlsException catch (e) {
      throw DownloadNetworkException(name, e.message, 0);
    } finally {
      cancel.interruptPhase = null;
    }
    try {
      if (response.headers.contentType?.mimeType == ContentType.html.mimeType) {
        throw HtmlInsteadOfFileException(name, url, drive: _isGoogleDrive(url));
      }
      switch (response.statusCode) {
        case HttpStatus.partialContent:
          final total = RegExp(r'/(\d+)$').firstMatch(
            response.headers.value(HttpHeaders.contentRangeHeader) ?? '',
          );
          final bytes = int.tryParse(total?.group(1) ?? '');
          if (bytes != null && bytes > 0) return bytes;
        case HttpStatus.ok:
          if (response.contentLength > 0) return response.contentLength;
        case final status:
          throw DownloadHttpException(name, url, status);
      }
      throw UnknownSizeException(url);
    } finally {
      await _discard(response);
    }
  }

  /// Downloads the rest of [file] into [part] with up to [_maxAttempts]
  /// attempts, each resuming [part]. Returns once [part] holds
  /// `file.sizeBytes` bytes; throws [DownloadNetworkException] after the
  /// last attempt (the `.part` kept), [OperationCancelledException] on a
  /// cancel, the provisioning error a response proved, or the
  /// [FileSystemException] of opening or writing the `.part` (a full or
  /// read-only disk; not retried).
  Future<void> download(
    RemoteFile file,
    File part,
    CancelToken cancel,
    FileStateReporter reporter,
  ) async {
    for (var attempt = 1; ; attempt++) {
      try {
        await _transfer(file, part, cancel, reporter, attempt);
        return;
      } on _Retryable catch (e) {
        if (cancel.cancelled) throw const OperationCancelledException();
        final have = await _lengthOf(part);
        if (attempt >= _maxAttempts) {
          throw DownloadNetworkException(file.name, e.cause, have);
        }
        final wait = _retryDelay(attempt);
        debugPrint(
          '[ModelStore] ${file.name}: ${e.cause}; reconnecting in '
          '${wait.inMilliseconds} ms (attempt ${attempt + 1}/$_maxAttempts, '
          'resume at $have)',
        );
        reporter.report(
          StoreFileDownloading(
            received: have,
            total: file.sizeBytes,
            attempt: attempt + 1,
          ),
        );
        await Future.any([Future<void>.delayed(wait), cancel.onCancel]);
        if (cancel.cancelled) throw const OperationCancelledException();
      }
    }
  }

  /// One HTTP request for the rest of [file], appended to [part].
  Future<void> _transfer(
    RemoteFile file,
    File part,
    CancelToken cancel,
    FileStateReporter reporter,
    int attempt,
  ) async {
    var have = await _lengthOf(part);
    if (have > file.sizeBytes) {
      await deleteQuietly(part);
      have = 0;
    }
    if (have == file.sizeBytes) return; // complete; verify it
    reporter.report(
      StoreFileDownloading(
        received: have,
        total: file.sizeBytes,
        attempt: attempt,
      ),
    );
    // A check that runs late (the app was suspended: a backgrounded phone)
    // starts a fresh window rather than counting the suspension as silence
    // and using up an attempt.
    final watchdog = StallWatchdog(
      interval: _stallTimeout ~/ 6,
      clockMicros: () => _now().inMicroseconds,
    );
    watchdog.arm(
      timeout: () => _stallTimeout,
      onStall: (_) => cancel.interrupt(_Stalled(_stallTimeout)),
    );
    try {
      final response = await _open(file.name, file.url, have, cancel);
      watchdog.activity(_now().inMicroseconds);
      var append = have > 0;
      switch (response.statusCode) {
        case HttpStatus.partialContent:
          final range = RegExp(r'^bytes (\d+)-(\d+)/(\d+|\*)$').firstMatch(
            response.headers.value(HttpHeaders.contentRangeHeader) ?? '',
          );
          final start = int.tryParse(range?.group(1) ?? '');
          final total = int.tryParse(range?.group(3) ?? '');
          if (total != null && total != file.sizeBytes) {
            await _discard(response);
            throw SizeMismatchException(
              file.name,
              expected: file.sizeBytes,
              actual: total,
              source: file.url.host,
            );
          }
          if (start != have) {
            // A range we did not ask for: start over.
            await _discard(response);
            await deleteQuietly(part);
            throw _Retryable('the server resumed at $start, not at $have');
          }
        case HttpStatus.ok:
          if (have > 0) {
            debugPrint(
              '[ModelStore] ${file.name}: the server ignored the range; '
              'restarting from 0',
            );
          }
          have = 0;
          append = false;
        case HttpStatus.requestedRangeNotSatisfiable:
          await _discard(response);
          await deleteQuietly(part);
          throw const _Retryable('the server rejected the resume range');
        case final status:
          await _discard(response);
          final error = DownloadHttpException(file.name, file.url, status);
          // 429 and 5xx are worth a reconnect; any other 4xx is not.
          if (status == 429 || status >= 500) throw _Retryable(error);
          throw error;
      }
      if (response.headers.contentType?.mimeType == ContentType.html.mimeType) {
        await _discard(response);
        throw HtmlInsteadOfFileException(
          file.name,
          file.url,
          drive: _isGoogleDrive(file.url),
        );
      }
      final length = response.contentLength;
      if (length >= 0 && length != file.sizeBytes - have) {
        await _discard(response);
        throw SizeMismatchException(
          file.name,
          expected: file.sizeBytes,
          actual: length + have,
          source: file.url.host,
        );
      }
      if (have > 0) {
        debugPrint('[ModelStore] ${file.name}: resuming at $have bytes');
      }
      final sink = await _openPart(part, append: append);
      var received = have;
      var first = true;
      _Overflow? overflow;
      final started = _now();
      try {
        await _readBody(
          response,
          cancel,
          (chunk) async {
            if (first) {
              first = false;
              if (_looksLikeHtml(chunk)) {
                throw HtmlInsteadOfFileException(
                  file.name,
                  file.url,
                  drive: _isGoogleDrive(file.url),
                );
              }
            }
            if (received + chunk.length > file.sizeBytes) {
              throw _Overflow(received + chunk.length);
            }
            await sink.add(chunk);
            received += chunk.length;
            watchdog.activity(_now().inMicroseconds);
          },
          afterWrite: () {
            reporter.progress(
              StoreFileDownloading(
                received: received,
                total: file.sizeBytes,
                attempt: attempt,
              ),
            );
          },
        );
      } on _Overflow catch (e) {
        overflow = e;
      } finally {
        await sink.close();
      }
      if (overflow != null) {
        await deleteQuietly(part);
        throw SizeMismatchException(
          file.name,
          expected: file.sizeBytes,
          actual: overflow.atLeast,
          source: file.url.host,
        );
      }
      final seconds = (_now() - started).inMilliseconds / 1000;
      debugPrint(
        '[ModelStore] ${file.name}: received ${received - have} bytes in '
        '${seconds.toStringAsFixed(1)} s',
      );
      if (received < file.sizeBytes) {
        throw _Retryable(_ClosedEarly(received, file.sizeBytes));
      }
    } on SocketException catch (e) {
      throw _Retryable(e.message);
    } on HttpException catch (e) {
      throw _Retryable(e.message);
    } on TlsException catch (e) {
      throw _Retryable(e.message);
    } on TimeoutException catch (e) {
      throw _Retryable('timed out ($e)');
    } on _Stalled catch (e) {
      throw _Retryable(e);
    } finally {
      watchdog.disarm();
      cancel.interruptPhase = null;
    }
  }

  /// Reads [response] chunk by chunk through [write], one at a time (the
  /// subscription pauses while a chunk is written: backpressure), calling
  /// [afterWrite] after each chunk while the read goes on. A failing [write]
  /// or [CancelToken.interrupt] cancels the subscription, which closes the
  /// connection, and fails the returned future with that error, but only
  /// once a write still in flight has finished: the caller closes the file
  /// next, and closing a `RandomAccessFile` under a pending `writeFrom`
  /// throws ("An async operation is currently pending"), which would hide
  /// the cancel and leak the handle.
  static Future<void> _readBody(
    HttpClientResponse response,
    CancelToken cancel,
    Future<void> Function(List<int> chunk) write, {
    required void Function() afterWrite,
  }) async {
    final done = Completer<void>();
    Future<void>? inFlight;
    late final StreamSubscription<List<int>> subscription;
    void fail(Object error, [StackTrace? stack]) {
      if (done.isCompleted) return;
      done.completeError(error, stack);
      unawaited(subscription.cancel());
    }

    subscription = response.listen(
      (chunk) {
        subscription.pause();
        final writing = inFlight = write(chunk);
        unawaited(
          writing.then((_) {
            // Stopped meanwhile: no progress after the cancel, no resume.
            if (done.isCompleted) return;
            afterWrite();
            subscription.resume();
          }, onError: fail),
        );
      },
      onError: fail,
      onDone: () {
        if (!done.isCompleted) done.complete();
      },
      cancelOnError: true,
    );
    cancel.interruptPhase = fail;
    if (cancel.cancelled) fail(const OperationCancelledException());
    try {
      await done.future;
    } finally {
      // Its own error, if any, already reached `fail` (or lost to an earlier
      // one); here it only has to be over.
      await inFlight?.then<void>((_) {}, onError: (Object _) {});
    }
  }

  /// Drops a response without reading it (closing its connection): a body
  /// shorter than its Content-Length would make `drain` throw.
  static Future<void> _discard(HttpClientResponse response) async {
    try {
      await response.listen(null).cancel();
    } on Exception catch (e) {
      debugPrint('[ModelStore] discarding a response failed: $e');
    }
  }

  /// Opens a GET for [url] from byte [from], following redirects by hand
  /// so the Range header survives each hop (Hugging Face and Drive redirect
  /// to their CDNs). [range] overrides the Range header (the size probe).
  /// Each hop is checked before it is requested: a redirect to a link that
  /// is not HTTP(S), or from HTTPS down to HTTP, throws
  /// [UnsafeRedirectException]; another host is fine (the CDNs).
  Future<HttpClientResponse> _open(
    String name,
    Uri url,
    int from,
    CancelToken cancel, {
    String? range,
  }) async {
    final client = _http;
    var uri = url;
    for (var hop = 0; hop < 10; hop++) {
      if (cancel.cancelled) throw const OperationCancelledException();
      final request = await client.getUrl(uri);
      cancel.interruptPhase = request.abort;
      if (cancel.cancelled) request.abort(const OperationCancelledException());
      request
        ..followRedirects = false
        ..headers.set(HttpHeaders.acceptEncodingHeader, 'identity');
      if (range != null) {
        request.headers.set(HttpHeaders.rangeHeader, range);
      } else if (from > 0) {
        request.headers.set(HttpHeaders.rangeHeader, 'bytes=$from-');
      }
      final response = await request.close();
      final location = response.headers.value(HttpHeaders.locationHeader);
      if (response.isRedirect && location != null) {
        await _discard(response);
        final next = uri.resolve(location);
        _checkRedirect(name, uri, next);
        uri = next;
        continue;
      }
      if (hop > 0) {
        debugPrint('[ModelStore] $name: served by ${uri.host}');
      }
      return response;
    }
    throw DownloadHttpException(name, url, HttpStatus.loopDetected);
  }

  /// Throws [UnsafeRedirectException] unless [to] is HTTP(S) and keeps
  /// HTTPS when [from] had it.
  static void _checkRedirect(String name, Uri from, Uri to) {
    final web = to.isScheme('https') || to.isScheme('http');
    if (web && !(from.isScheme('https') && to.isScheme('http'))) return;
    debugPrint(
      '[ModelStore] $name: refused the redirect from ${from.scheme}://'
      '${from.host} to ${to.scheme}://${to.host}',
    );
    throw UnsafeRedirectException(name, from: from, to: to);
  }

  static bool _isGoogleDrive(Uri url) =>
      url.host == 'drive.google.com' ||
      url.host == 'drive.usercontent.google.com' ||
      url.host.endsWith('.googleusercontent.com');

  /// A page, not a model: `<!doctype html` or `<html` after optional BOM
  /// and whitespace. Model files are binary; tokenizers start with `{`.
  static bool _looksLikeHtml(List<int> chunk) {
    var i = 0;
    if (chunk.length >= 3 &&
        chunk[0] == 0xEF &&
        chunk[1] == 0xBB &&
        chunk[2] == 0xBF) {
      i = 3;
    }
    while (i < chunk.length &&
        (chunk[i] == 0x20 || (chunk[i] >= 0x09 && chunk[i] <= 0x0D))) {
      i++;
    }
    final end = i + 14 > chunk.length ? chunk.length : i + 14;
    final head = String.fromCharCodes(chunk.sublist(i, end)).toLowerCase();
    return head.startsWith('<!doctype html') || head.startsWith('<html');
  }

  static Future<int> _lengthOf(File file) async =>
      await file.exists() ? await file.length() : 0;
}
