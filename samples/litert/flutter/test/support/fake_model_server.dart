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
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

/// How the fake server answers one path.
final class ServedFile {
  ServedFile(
    this.bytes, {
    this.dropAfter,
    this.dropTimes = 1,
    this.ignoreRange = false,
    this.contentType = 'application/octet-stream',
    this.chunkDelay = Duration.zero,
    this.chunkBytes = 64 * 1024,
    this.stallAfter,
    this.status = 200,
    this.reportedLength,
  });

  final Uint8List bytes;

  /// Drop the connection after sending this many body bytes of a response,
  /// for the first [dropTimes] responses.
  final int? dropAfter;
  int dropTimes;

  /// Answer every request with the whole file (200), as a server without
  /// Range support does.
  final bool ignoreRange;
  final String contentType;
  final Duration chunkDelay;
  final int chunkBytes;

  /// Send this many body bytes, then nothing (the connection stays open).
  final int? stallAfter;
  final int status;

  /// A Content-Length that differs from the bytes (a wrong file).
  final int? reportedLength;
}

/// One request the server saw.
final class SeenRequest {
  SeenRequest(this.path, this.range);

  final String path;
  final String? range;

  @override
  String toString() => '$path range=$range';
}

/// A local HTTP server for the model store tests: Range requests (206),
/// redirects (`/redirect/<path>` answers 302 to `/<path>`), dropped and
/// stalled connections, HTML pages.
final class FakeModelServer {
  FakeModelServer._(this._server) {
    _server.listen(_handle);
  }

  static Future<FakeModelServer> start() async =>
      FakeModelServer._(await HttpServer.bind(InternetAddress.loopbackIPv4, 0));

  final HttpServer _server;
  final Map<String, ServedFile> files = {};
  final List<SeenRequest> requests = [];
  final List<Socket> _stalled = [];

  /// Responses the client closed before the server finished writing.
  int clientHangUps = 0;

  Uri url(String path) => Uri.parse('http://127.0.0.1:${_server.port}/$path');

  List<SeenRequest> requestsFor(String path) =>
      requests.where((r) => r.path == '/$path').toList();

  Future<void> _handle(HttpRequest request) async {
    final path = request.uri.path;
    final range = request.headers.value(HttpHeaders.rangeHeader);
    requests.add(SeenRequest(path, range));
    if (path.startsWith('/redirect/')) {
      request.response
        ..statusCode = HttpStatus.found
        ..headers.set(
          HttpHeaders.locationHeader,
          path.substring('/redirect'.length),
        );
      await request.response.close();
      return;
    }
    final served = files[path.substring(1)];
    if (served == null) {
      request.response.statusCode = HttpStatus.notFound;
      await request.response.close();
      return;
    }
    if (served.status != 200) {
      request.response.statusCode = served.status;
      await request.response.close();
      return;
    }
    var start = 0;
    final match = RegExp(r'^bytes=(\d+)-$').firstMatch(range ?? '');
    if (match != null && !served.ignoreRange) {
      start = int.parse(match.group(1)!);
    }
    final body = Uint8List.sublistView(served.bytes, start);
    final partial = start > 0;
    final drop = served.dropAfter != null && served.dropTimes > 0;
    if (drop) served.dropTimes--;

    // Raw socket: full control over a dropped or stalled connection.
    final socket = await request.response.detachSocket(writeHeaders: false);
    final total = served.bytes.length;
    final length = served.reportedLength ?? body.length;
    final head = StringBuffer()
      ..write(
        partial ? 'HTTP/1.1 206 Partial Content\r\n' : 'HTTP/1.1 200 OK\r\n',
      )
      ..write('Content-Type: ${served.contentType}\r\n')
      ..write('Content-Length: $length\r\n')
      ..write('Accept-Ranges: bytes\r\n')
      ..write('Connection: close\r\n');
    if (partial) {
      head.write('Content-Range: bytes $start-${total - 1}/$total\r\n');
    }
    head.write('\r\n');
    socket.add(ascii.encode(head.toString()));
    final limit = drop
        ? served.dropAfter!.clamp(0, body.length)
        : served.stallAfter?.clamp(0, body.length) ?? body.length;
    try {
      for (var offset = 0; offset < limit; offset += served.chunkBytes) {
        final end = (offset + served.chunkBytes).clamp(0, limit);
        socket.add(Uint8List.sublistView(body, offset, end));
        await socket.flush();
        if (served.chunkDelay > Duration.zero) {
          await Future<void>.delayed(served.chunkDelay);
        }
      }
      if (served.stallAfter != null && !drop) {
        _stalled.add(socket); // keep it open, send nothing more
        return;
      }
      if (drop) {
        socket.destroy();
        return;
      }
      await socket.flush();
      await socket.close();
    } on SocketException {
      // The client hung up on purpose (cancel, a rejected response).
      clientHangUps++;
      socket.destroy();
    }
  }

  Future<void> close() async {
    for (final socket in _stalled) {
      socket.destroy();
    }
    await _server.close(force: true);
  }
}
