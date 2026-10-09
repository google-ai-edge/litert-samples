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
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/services/frames/engine_image_decoder.dart';
import 'package:litert_edge_demos/data/services/frames/jpeg_decoder.dart';
import 'package:litert_edge_demos/data/services/frames/network_frame_source.dart';
import 'package:litert_edge_demos/data/services/frames/turbojpeg.dart';
import 'package:litert_edge_demos/domain/models/frame_source_info.dart';
import 'package:litert_edge_demos/domain/models/preview_source.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../../support/mjpeg.dart';

Uri urlOf(HttpServer server, [String path = '/video']) =>
    Uri.parse('http://127.0.0.1:${server.port}$path');

String errorOf(Result<FrameSourceInfo> result) => switch (result) {
  Ok() => fail('expected an error, got $result'),
  Error(:final error) => error.toString(),
};

/// How long a test waits for something that must happen (frames, an error,
/// the server seeing the connection go). Generous on purpose: a loaded
/// machine is slower, never different. The tests wait for the event itself,
/// never a fixed time, except to show that something does *not* happen.
const _patience = Duration(seconds: 15);

/// What a source reports, collected; a test awaits a count of it instead of
/// sleeping and hoping that many arrived.
final class _Collected<T> {
  final List<T> values = [];
  final List<(int, Completer<void>)> _waiters = [];

  void add(T value) {
    values.add(value);
    _waiters.removeWhere((waiter) {
      if (values.length < waiter.$1) return false;
      waiter.$2.complete();
      return true;
    });
  }

  /// Completes once [n] values have arrived; fails the test after
  /// [_patience].
  Future<void> atLeast(int n) {
    if (values.length >= n) return Future.value();
    final reached = Completer<void>();
    _waiters.add((n, reached));
    return reached.future.timeout(
      _patience,
      onTimeout: () => fail('${values.length} of $n arrived in $_patience'),
    );
  }
}

/// Polls [condition] (state with no event of its own) until it holds;
/// fails the test after [_patience].
Future<void> _until(bool Function() condition, String what) async {
  final watch = Stopwatch()..start();
  while (!condition()) {
    if (watch.elapsed > _patience) fail('no $what within $_patience');
    await Future<void>.delayed(const Duration(milliseconds: 5));
  }
}

/// TurboJPEG on this machine; its tests are skipped without it.
final String? _noTurboJpeg =
    TurboJpegLibrary.open(turboJpegCandidates()) == null
    ? 'libturbojpeg not found'
    : null;

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  // flutter_test answers every HttpClient with 400; these tests talk to a
  // real local server.
  HttpOverrides.global = null;

  final cats = catsJpeg();
  final servers = <HttpServer>[];
  final sources = <NetworkFrameSource>[];

  Future<HttpServer> server(Future<void> Function(HttpRequest) handler) async {
    final s = await serve(handler);
    servers.add(s);
    return s;
  }

  /// The engine decoder unless a test picks another: these tests are about
  /// the network (the TurboJPEG path has its own group below). The
  /// timeouts are generous unless a test is about one: a first frame that
  /// takes a loaded machine over 2 s is not a failure of these tests.
  NetworkFrameSource source(
    Uri url, {
    JpegDecoderFactory? decoder,
    Duration stall = const Duration(seconds: 10),
    Duration connect = const Duration(seconds: 10),
  }) {
    final s = NetworkFrameSource(
      url: url,
      decoder: decoder ?? () async => const EngineJpegDecoder(),
      stallTimeout: stall,
      connectTimeout: connect,
      watchInterval: const Duration(milliseconds: 100),
    );
    sources.add(s);
    return s;
  }

  tearDown(() async {
    for (final s in sources) {
      await s.stop();
    }
    sources.clear();
    for (final s in servers) {
      await s.close(force: true);
    }
    servers.clear();
  });

  testWidgets('streams the cats MJPEG: label, size, RGBA frames that match '
      'the JPEG, a live preview; stop ends frames and the connection', (
    tester,
  ) async {
    await tester.runAsync(() async {
      final s = await server((request) => streamMjpeg(request, [cats]));
      final expected = await decodeEncodedImage(cats, maxSide: 1280);
      expected.image.dispose();
      final src = source(urlOf(s));
      final frames = _Collected<String>();
      final previews = <ui.Image?>[];
      final image = (src.preview as ImagePreviewSource).image;
      image.addListener(() => previews.add(image.value));

      final started = await src.start(
        (f) => frames.add(
          '${f.width}x${f.height} ${f.format.name} '
          '${f.planes.single.bytesPerRow} '
          '${f.planes.single.bytes.take(8).join(',')}',
        ),
      );
      final info = (started as Ok<FrameSourceInfo>).value;
      expect(info.label, 'Network camera · 127.0.0.1:${s.port}');
      expect(
        (info.width, info.height, info.format, info.mirrored),
        (640, 480, FramePixelFormat.rgba8888, false),
      );
      expect(info.stallTimeout, greaterThan(const Duration(seconds: 2)));

      // Five frames, however long a loaded machine takes to decode them
      // (a fixed 600 ms window here failed under full-suite load).
      await frames.atLeast(5);
      expect(
        frames.values.first,
        '640x480 rgba8888 2560 ${expected.rgba.take(8).join(',')}',
      );
      expect(previews.whereType<ui.Image>(), isNotEmpty);

      await src.stop();
      final count = frames.values.length;
      expect(previews.last, isNull, reason: 'preview cleared on stop');
      // The server sees the connection go. Polled: dart:io gives the
      // handler no event for it (its writes after the close neither fail
      // nor block).
      await _until(
        () => s.connectionsInfo().total == 0,
        'drop of the server-side connection',
      );
      // A window in which nothing may arrive (a decode in flight at stop
      // must be dropped); a slow machine only widens it.
      await Future<void>.delayed(const Duration(milliseconds: 200));
      expect(frames.values.length, count, reason: 'nothing after stop');
    });
  });

  testWidgets('small chunks and no Content-Length still give whole frames', (
    tester,
  ) async {
    await tester.runAsync(() async {
      final s = await server(
        (request) =>
            streamMjpeg(request, [cats], contentLength: false, chunkSize: 1500),
      );
      final src = source(urlOf(s));
      final frames = _Collected<int>();
      final started = await src.start((f) => frames.add(f.width));
      expect(started, isA<Ok<FrameSourceInfo>>());
      await frames.atLeast(3);
      expect(frames.values.toSet(), {640});
    });
  });

  testWidgets('latest frame wins: while one decodes only the newest waits; '
      'the rest are dropped', (tester) async {
    await tester.runAsync(() async {
      final jpegs = [
        for (var i = 0; i < 6; i++) Uint8List.fromList([...cats, i]),
      ];
      final held = Completer<void>();
      final gate = Completer<void>();
      final decodedTags = <int>[];
      final gated = _GatedDecoder((jpeg) async {
        decodedTags.add(jpeg.last);
        if (decodedTags.length == 2) {
          held.complete();
          await gate.future; // hold the 2nd
        }
      });
      // The test sends each part when it wants it: no pacing to race.
      final parts = StreamController<Uint8List>();
      final s = await server((request) async {
        final response = request.response
          ..headers.set(
            'Content-Type',
            'multipart/x-mixed-replace; boundary=frame',
          )
          ..bufferOutput = false;
        response.add(const [13, 10]);
        await response.flush();
        await for (final jpeg in parts.stream) {
          response.add(mjpegPart('frame', jpeg));
          await response.flush();
        }
        await response.close();
      });
      final src = source(urlOf(s), decoder: () async => gated);
      final delivered = _Collected<int>();

      parts.add(jpegs[0]);
      final started = await src.start((f) => delivered.add(f.width));
      expect(started, isA<Ok<FrameSourceInfo>>());
      parts.add(jpegs[1]);
      await held.future.timeout(_patience);
      // Frames 2..5 arrive while frame 1 (the second decode) is held.
      for (final jpeg in jpegs.skip(2)) {
        parts.add(jpeg);
      }
      await _until(() => src.receivedFrames == 6, 'arrival of all 6 frames');
      gate.complete();
      await delivered.atLeast(3); // 0, the held 1, and the newest waiting
      await pumpEventQueue();

      expect(decodedTags, [0, 1, 5], reason: '2, 3 and 4 were dropped');
      expect(delivered.values, hasLength(3));
      await parts.close();
    });
  });

  testWidgets('TurboJPEG on a worker isolate: frames, the decoder named in '
      'the info, a preview made from the pixels; the decoder closed on stop', (
    tester,
  ) async {
    await tester.runAsync(() async {
      final s = await server((request) => streamMjpeg(request, [cats]));
      JpegFrameDecoder? opened;
      final src = source(
        urlOf(s),
        decoder: () async =>
            opened = await openJpegDecoder(platform: TargetPlatform.linux),
      );
      final frames = _Collected<String>();
      final previews = <ui.Image?>[];
      final image = (src.preview as ImagePreviewSource).image;
      image.addListener(() => previews.add(image.value));
      final started = await src.start(
        (f) => frames.add('${f.width}x${f.height}'),
      );
      final info = (started as Ok<FrameSourceInfo>).value;
      expect(info.decoder, startsWith('TurboJPEG (worker isolate'));
      expect(info.label, 'Network camera · 127.0.0.1:${s.port}');
      await frames.atLeast(5);
      expect(frames.values.toSet(), {'640x480'});
      await _until(
        () => previews.whereType<ui.Image>().isNotEmpty,
        'preview made from the pixels',
      );
      expect(src.averageDecodeTime, isNotNull);
      debugPrint(
        'NETCAM_UNIT decoder="${info.decoder}" '
        'decode_avg=${src.averageDecodeTime!.inMicroseconds / 1000}ms',
      );

      await src.stop();
      await expectLater(
        opened!.decode(cats, maxSide: 1280),
        throwsA(anything),
        reason: 'the worker is closed with the source',
      );
    });
  }, skip: _noTurboJpeg != null);

  testWidgets('stop after the first frame decoded, before start resumes: its '
      'image is disposed once, by the preview that owns it', (tester) async {
    await tester.runAsync(() async {
      final s = await server((request) => streamMjpeg(request, [cats]));
      final src = source(urlOf(s));
      final image = (src.preview as ImagePreviewSource).image;
      Future<void>? stopping;
      // The engine decoder's first image reaches the preview synchronously,
      // just before start's wait completes; a microtask from here lands
      // after that completion and before start resumes.
      image.addListener(() {
        if (image.value == null || stopping != null) return;
        scheduleMicrotask(() => stopping = src.stop());
      });

      final started = await src.start((_) {});

      expect(errorOf(started), contains('Stopped while connecting'));
      expect(stopping, isNotNull, reason: 'stop ran inside the window');
      // A second dispose of the image trips ui.Image's debug assert, in
      // start or here.
      await stopping;
      expect(image.value, isNull);
    });
  });

  testWidgets('stop while the decoder is still opening: the decoder that '
      'arrives afterwards is closed', (tester) async {
    await tester.runAsync(() async {
      final opening = Completer<JpegFrameDecoder>();
      final decoder = _CountingDecoder();
      final src = source(
        Uri.parse('http://127.0.0.1:9/video'),
        decoder: () => opening.future,
      );
      final started = src.start((_) {});
      await Future<void>.delayed(Duration.zero);
      await src.stop();
      opening.complete(decoder);
      expect(errorOf(await started), contains('Stopped while connecting'));
      expect(decoder.closes, 1);
    });
  });

  testWidgets('a decoder that dies fails the source with its own reason, not '
      '"the camera\'s frames do not decode"', (tester) async {
    await tester.runAsync(() async {
      final s = await server(
        (request) => streamMjpeg(request, [
          cats,
        ], interval: const Duration(milliseconds: 20)),
      );
      final decoder = _CountingDecoder();
      final errors = _Collected<Exception>();
      final src = source(urlOf(s), decoder: () async => decoder);
      expect(
        await src.start((_) {}, onError: errors.add),
        isA<Ok<FrameSourceInfo>>(),
      );
      decoder.dead = 'the decoder isolate exited';
      await errors.atLeast(1);
      await pumpEventQueue();
      final error = errors.values.single.toString();
      expect(error, contains('The JPEG decoder stopped'));
      expect(error, contains('isolate exited'));
    });
  });

  testWidgets('no libturbojpeg on Linux: the engine codec, and the label says '
      '"slow JPEG decoder"', (tester) async {
    await tester.runAsync(() async {
      final s = await server((request) => streamMjpeg(request, [cats]));
      final src = source(
        urlOf(s),
        decoder: () => openJpegDecoder(
          platform: TargetPlatform.linux,
          candidates: const ['/nonexistent/libturbojpeg.so.0'],
        ),
      );
      final started = await src.start((_) {});
      final info = (started as Ok<FrameSourceInfo>).value;
      expect(
        info.label,
        'Network camera · 127.0.0.1:${s.port} · slow JPEG decoder',
      );
      expect(info.decoder, 'engine codec');
    });
  });

  testWidgets("ffmpeg's -listen server: application/octet-stream whose body "
      'starts with the boundary line', (tester) async {
    await tester.runAsync(() async {
      final s = await server((request) async {
        request.response
          ..headers.set('Content-Type', 'application/octet-stream')
          ..bufferOutput = false;
        for (var i = 0; i < 100; i++) {
          request.response.add(
            mjpegPart('ffmpeg', cats, lengthHeader: 'Content-length'),
          );
          await request.response.flush();
          await Future<void>.delayed(const Duration(milliseconds: 40));
        }
      });
      final frames = _Collected<int>();
      final started = await source(urlOf(s)).start((f) => frames.add(f.width));
      expect(started, isA<Ok<FrameSourceInfo>>());
      await frames.atLeast(1);
      expect(frames.values.first, 640);
    });
  });

  testWidgets('credentials in the URL go out as Basic auth', (tester) async {
    await tester.runAsync(() async {
      String? auth;
      final s = await server((request) async {
        auth = request.headers.value('authorization');
        await streamMjpeg(request, [cats]);
      });
      final url = Uri.parse('http://user:secret@127.0.0.1:${s.port}/video');
      final src = source(url);
      final started = await src.start((_) {});
      expect(started, isA<Ok<FrameSourceInfo>>());
      expect(auth, 'Basic ${base64.encode(utf8.encode('user:secret'))}');
      final info = (started as Ok<FrameSourceInfo>).value;
      expect(info.label, isNot(contains('secret')));
    });
  });

  group('fails fast with a clear error', () {
    testWidgets('unreachable host: connection refused', (tester) async {
      await tester.runAsync(() async {
        final socket = await ServerSocket.bind(InternetAddress.loopbackIPv4, 0);
        final port = socket.port;
        await socket.close();
        final src = source(Uri.parse('http://127.0.0.1:$port/video'));
        final error = errorOf(await src.start((_) {}));
        expect(error, contains('connection refused'));
        expect(error, contains('127.0.0.1:$port'));
      });
    });

    testWidgets('an HTML page is not MJPEG', (tester) async {
      await tester.runAsync(() async {
        final s = await server((request) async {
          request.response
            ..headers.contentType = ContentType.html
            ..write('<html><body>IP Webcam</body></html>');
          await request.response.close();
        });
        final error = errorOf(await source(urlOf(s, '/')).start((_) {}));
        expect(error, contains('web page (text/html)'));
        expect(error, contains('/video'));
      });
    });

    testWidgets('a single JPEG is not a stream', (tester) async {
      await tester.runAsync(() async {
        final s = await server((request) async {
          request.response
            ..headers.contentType = ContentType('image', 'jpeg')
            ..add(cats);
          await request.response.close();
        });
        final error = errorOf(
          await source(urlOf(s, '/shot.jpg')).start((_) {}),
        );
        expect(error, contains('single JPEG'));
      });
    });

    testWidgets('a login is required (HTTP 401)', (tester) async {
      await tester.runAsync(() async {
        final s = await server((request) async {
          request.response
            ..statusCode = HttpStatus.unauthorized
            ..headers.set('WWW-Authenticate', 'Basic realm="IP Webcam"');
          await request.response.close();
        });
        final error = errorOf(await source(urlOf(s)).start((_) {}));
        expect(error, contains('asks for a login (HTTP 401'));
        expect(error, contains('http://user:password@host:port/video'));
      });
    });

    testWidgets('a wrong path (HTTP 404)', (tester) async {
      await tester.runAsync(() async {
        final s = await server((request) async {
          request.response.statusCode = HttpStatus.notFound;
          await request.response.close();
        });
        final error = errorOf(await source(urlOf(s, '/vid')).start((_) {}));
        expect(error, contains('HTTP 404'));
      });
    });

    testWidgets('connected, but no frame within the stall timeout', (
      tester,
    ) async {
      await tester.runAsync(() async {
        final s = await server(
          (request) => streamMjpeg(request, [cats], count: 0),
        );
        final watch = Stopwatch()..start();
        final error = errorOf(
          await source(
            urlOf(s),
            stall: const Duration(milliseconds: 400),
          ).start((_) {}),
        );
        expect(error, contains('no JPEG frame arrived within'));
        // Its own 400 ms, not the helper's 10 s (an upper bound only).
        expect(watch.elapsed, lessThan(const Duration(seconds: 8)));
      });
    });

    testWidgets('a stream that stalls after starting reports it once', (
      tester,
    ) async {
      await tester.runAsync(() async {
        final s = await server(
          (request) => streamMjpeg(request, [cats], count: 3),
        );
        final errors = _Collected<Exception>();
        final src = source(urlOf(s), stall: const Duration(milliseconds: 500));
        final started = await src.start((_) {}, onError: errors.add);
        expect(started, isA<Ok<FrameSourceInfo>>());
        await errors.atLeast(1);
        // Once: three more watchdog checks report nothing more (a window in
        // which something must NOT happen; a slow machine only widens it).
        await Future<void>.delayed(const Duration(milliseconds: 300));
        final error = errors.values.single.toString();
        expect(error, contains('stopped sending frames'));
        expect(error, contains('Reconnect'));
      });
    });

    testWidgets('the server closing the stream is reported', (tester) async {
      await tester.runAsync(() async {
        final s = await server(
          (request) => streamMjpeg(request, [cats], count: 3, closeAfter: true),
        );
        final errors = _Collected<Exception>();
        final started = await source(urlOf(s))
            .start((_) {}, onError: errors.add);
        expect(started, isA<Ok<FrameSourceInfo>>());
        await errors.atLeast(1);
        await pumpEventQueue();
        expect(errors.values.single.toString(), contains('closed the stream'));
      });
    });

    testWidgets('frames that never decode fail the source', (tester) async {
      await tester.runAsync(() async {
        final s = await server(
          (request) => streamMjpeg(request, [
            Uint8List.fromList(utf8.encode('not a jpeg')),
          ], interval: const Duration(milliseconds: 5)),
        );
        final error = errorOf(await source(urlOf(s)).start((_) {}));
        expect(error, contains('do not decode as JPEG'));
      });
    });
  });

  test('error texts never carry the URI (it may hold a password)', () {
    final uri = Uri.parse('http://user:secret@192.168.1.23:8080/video');
    expect(
      networkErrorText(
        HttpException('Connection closed while receiving data', uri: uri),
      ),
      'Connection closed while receiving data',
    );
    expect(
      networkCameraFailure(
        HttpException('Connection closed before full header', uri: uri),
        uri,
      ).toString(),
      isNot(contains('secret')),
    );
  });

  test('mjpegBoundaryOf: the IP Webcam answer gives its boundary', () {
    final url = Uri.parse('http://192.168.1.23:8080/video');
    expect(
      mjpegBoundaryOf(
        url: url,
        status: 200,
        contentType: 'multipart/x-mixed-replace;boundary=Ba4oTvQMY8ew04N8dcnM',
      ),
      isA<Ok<String>>().having((r) => r.value, 'value', 'Ba4oTvQMY8ew04N8dcnM'),
    );
    final missing = mjpegBoundaryOf(
      url: url,
      status: 200,
      contentType: 'multipart/x-mixed-replace',
    );
    expect(
      (missing as Error<String>).error.toString(),
      contains('without a boundary'),
    );
  });
}

/// The engine decoder, with a hook before each decode.
final class _GatedDecoder implements JpegFrameDecoder {
  _GatedDecoder(this._before);

  final Future<void> Function(Uint8List jpeg) _before;

  @override
  String get description => 'gated engine codec';

  @override
  bool get slow => false;

  @override
  String? get fallbackReason => null;

  @override
  String? get failure => null;

  @override
  Future<RgbaFrame> decode(Uint8List jpeg, {required int maxSide}) async {
    await _before(jpeg);
    return const EngineJpegDecoder().decode(jpeg, maxSide: maxSide);
  }

  @override
  Future<void> close() async {}
}

/// The engine decoder that counts its closes and can play a dead worker.
final class _CountingDecoder implements JpegFrameDecoder {
  int closes = 0;

  /// Set: every decode fails, and [failure] says why.
  String? dead;

  @override
  String get description => 'counting engine codec';

  @override
  bool get slow => false;

  @override
  String? get fallbackReason => null;

  @override
  String? get failure => dead;

  @override
  Future<RgbaFrame> decode(Uint8List jpeg, {required int maxSide}) {
    if (dead case final reason?) {
      return Future.error(TurboJpegException('not running ($reason)'));
    }
    return const EngineJpegDecoder().decode(jpeg, maxSide: maxSide);
  }

  @override
  Future<void> close() async => closes++;
}
