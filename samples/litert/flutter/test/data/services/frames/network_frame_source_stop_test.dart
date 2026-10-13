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

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/services/frames/jpeg_decoder.dart';
import 'package:litert_edge_demos/data/services/frames/network_frame_source.dart';
import 'package:litert_edge_demos/domain/models/frame_source_info.dart';
import 'package:litert_edge_demos/utils/result.dart';

/// A stop during each phase of the network camera's start, on a fake HTTP
/// client the test steps through: start fails at once with "Stopped while
/// connecting", opens nothing after the stop, and closes what it opened.
/// network_frame_source_test covers the phases against a real server.

const _stopped = 'Stopped while connecting to Network camera · cam.test:8080';

/// Generous: a loaded machine is slower, never different.
const _patience = Duration(seconds: 15);

String _errorOf(Result<FrameSourceInfo> result) => switch (result) {
  Ok() => fail('expected an error, got $result'),
  Error(:final error) => error.toString(),
};

Future<void> _until(bool Function() condition, String what) async {
  final watch = Stopwatch()..start();
  while (!condition()) {
    if (watch.elapsed > _patience) fail('no $what within $_patience');
    await Future<void>.delayed(const Duration(milliseconds: 5));
  }
}

final class _Headers implements HttpHeaders {
  _Headers([this._contentType]);

  final String? _contentType;

  @override
  String? value(String name) =>
      name == HttpHeaders.contentTypeHeader ? _contentType : null;

  @override
  void set(String name, Object value, {bool preserveHeaderCase = false}) {}

  @override
  Object? noSuchMethod(Invocation invocation) =>
      throw UnimplementedError('_Headers: ${invocation.memberName}');
}

/// An MJPEG answer whose body the test never fills.
final class _Response extends Stream<List<int>> implements HttpClientResponse {
  final StreamController<List<int>> _body = StreamController();
  bool listened = false;
  bool cancelled = false;

  @override
  int get statusCode => 200;

  @override
  String get reasonPhrase => 'OK';

  @override
  HttpHeaders get headers =>
      _Headers('multipart/x-mixed-replace; boundary=frame');

  @override
  StreamSubscription<List<int>> listen(
    void Function(List<int> event)? onData, {
    Function? onError,
    void Function()? onDone,
    bool? cancelOnError,
  }) {
    listened = true;
    _body.onCancel = () {
      cancelled = true;
      return _body.close();
    };
    return _body.stream.listen(
      onData,
      onError: onError,
      onDone: onDone,
      cancelOnError: cancelOnError,
    );
  }

  @override
  Object? noSuchMethod(Invocation invocation) =>
      throw UnimplementedError('_Response: ${invocation.memberName}');
}

/// The GET; its answer is the client's [_Client.answer].
final class _Request implements HttpClientRequest {
  _Request(this._client);

  final _Client _client;

  @override
  HttpHeaders get headers => _Headers();

  @override
  Future<HttpClientResponse> close() {
    _client.sent = true;
    return _client.answer.future;
  }

  @override
  Object? noSuchMethod(Invocation invocation) =>
      throw UnimplementedError('_Request: ${invocation.memberName}');
}

/// One connection whose GET is answered when the test completes [answer];
/// a forced close fails a GET still waiting, as dart:io's does.
final class _Client implements HttpClient {
  final Completer<HttpClientResponse> answer = Completer();
  bool sent = false;
  bool closed = false;

  @override
  Duration? connectionTimeout;

  @override
  set findProxy(String Function(Uri url)? f) {}

  @override
  Future<HttpClientRequest> getUrl(Uri url) async => _Request(this);

  @override
  void close({bool force = false}) {
    closed = true;
    if (force && !answer.isCompleted) {
      answer.completeError(
        const HttpException('Connection closed before full header'),
      );
    }
  }

  @override
  Object? noSuchMethod(Invocation invocation) =>
      throw UnimplementedError('_Client: ${invocation.memberName}');
}

void main() {
  late _Client client;
  late NetworkFrameSource source;

  setUp(() {
    client = _Client();
    source = NetworkFrameSource(
      url: Uri.parse('http://cam.test:8080/video'),
      httpClient: () => client,
      decoder: () async => const EngineJpegDecoder(),
      // Far longer than "at once": a start that waits for either fails.
      stallTimeout: const Duration(seconds: 10),
      connectTimeout: const Duration(seconds: 10),
    );
  });

  tearDown(() => source.stop());

  test('stop while the GET waits for its answer', () async {
    final starting = source.start((_) {});
    await _until(() => client.sent, 'GET');
    final watch = Stopwatch()..start();

    await source.stop();

    expect(_errorOf(await starting.timeout(_patience)), _stopped);
    expect(watch.elapsed, lessThan(const Duration(seconds: 5)));
    expect(client.closed, isTrue);
  });

  test('stop queued right behind the answer\'s headers (between the '
      'phases): the body is never read', () async {
    final response = _Response();
    final starting = source.start((_) {});
    await _until(() => client.sent, 'GET');
    final watch = Stopwatch()..start();

    client.answer.complete(response);
    scheduleMicrotask(() => unawaited(source.stop()));

    expect(_errorOf(await starting.timeout(_patience)), _stopped);
    expect(watch.elapsed, lessThan(const Duration(seconds: 5)));
    expect(response.listened, isFalse, reason: 'nothing opened after stop');
    expect(client.closed, isTrue);
  });

  test('stop while waiting for the first frame', () async {
    final response = _Response();
    final starting = source.start((_) {});
    await _until(() => client.sent, 'GET');
    client.answer.complete(response);
    await _until(() => response.listened, 'body read');
    final watch = Stopwatch()..start();

    await source.stop();

    expect(_errorOf(await starting.timeout(_patience)), _stopped);
    expect(watch.elapsed, lessThan(const Duration(seconds: 5)));
    expect(response.cancelled, isTrue);
    expect(client.closed, isTrue);
  });
}
