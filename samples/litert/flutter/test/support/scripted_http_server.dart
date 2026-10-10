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

import 'fake_model_server.dart' show SeenRequest;

/// One request to a [ScriptedHttpServer], answered by hand.
final class ScriptedRequest {
  ScriptedRequest._(this.index, this.path, this.range, this._socket);

  /// 0 for the server's first request, 1 for the next, …
  final int index;
  final String path;
  final String? range;
  final Socket _socket;

  /// Writes the status line, [headers] and [body] as given (nothing is added:
  /// no Content-Length unless [headers] has one), then closes the
  /// connection, which ends a body without a Content-Length.
  Future<void> respond(
    int status, {
    Map<String, String> headers = const {},
    List<int> body = const [],
  }) async {
    final head = StringBuffer('HTTP/1.1 $status Scripted\r\n');
    headers.forEach((name, value) => head.write('$name: $value\r\n'));
    head.write('Connection: close\r\n\r\n');
    try {
      _socket.add(ascii.encode(head.toString()));
      if (body.isNotEmpty) _socket.add(body);
      await _socket.flush();
      await _socket.close();
    } on SocketException {
      _socket.destroy(); // the client hung up first
    }
  }

  /// Closes the connection without a word (before any header).
  void hangUp() => _socket.destroy();
}

/// A local HTTP server whose every answer the test writes itself: what
/// [FakeModelServer] cannot play (a resume at another offset, a body without
/// Content-Length, a probe answered with 206, a request never answered).
/// A handler that returns without calling [ScriptedRequest.respond] leaves
/// the request unanswered until [close].
final class ScriptedHttpServer {
  ScriptedHttpServer._(this._server, this._handler) {
    _server.listen(_handle);
  }

  static Future<ScriptedHttpServer> start(
    Future<void> Function(ScriptedRequest request) handler,
  ) async => ScriptedHttpServer._(
    await HttpServer.bind(InternetAddress.loopbackIPv4, 0),
    handler,
  );

  final HttpServer _server;
  final Future<void> Function(ScriptedRequest request) _handler;
  final List<SeenRequest> requests = [];
  final List<Socket> _open = [];

  /// Each request as it arrives (a test waits for one).
  final StreamController<ScriptedRequest> _arrivals =
      StreamController.broadcast();
  Stream<ScriptedRequest> get arrivals => _arrivals.stream;

  Uri url(String path) => Uri.parse('http://127.0.0.1:${_server.port}/$path');

  Future<void> _handle(HttpRequest request) async {
    final range = request.headers.value(HttpHeaders.rangeHeader);
    final index = requests.length;
    requests.add(SeenRequest(request.uri.path, range));
    final socket = await request.response.detachSocket(writeHeaders: false);
    _open.add(socket);
    final scripted = ScriptedRequest._(index, request.uri.path, range, socket);
    _arrivals.add(scripted);
    await _handler(scripted);
  }

  Future<void> close() async {
    for (final socket in _open) {
      socket.destroy();
    }
    await _arrivals.close();
    await _server.close(force: true);
  }
}
