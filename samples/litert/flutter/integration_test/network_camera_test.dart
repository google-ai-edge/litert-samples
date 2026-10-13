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

// Demo 3 with a network camera (MJPEG over HTTP) on the real app (macOS).
// The camera is an MJPEG server on this Mac: by default one inside the test
// (127.0.0.1, multipart/x-mixed-replace, test_assets/cats.jpg at ~30 fps, the
// way the IP Webcam app serves /video), or any external one given with
// NETWORK_CAMERA_URL (e.g. ffmpeg's mpjpeg muxer). YOLO26n on the GPU,
// moonshine / Inflect on the CPU, Gemma 4 E2B on the GPU.
//
//   flutter test integration_test/network_camera_test.dart -d macos \
//     --dart-define=GEMMA_MODEL_PATH=<path/to/gemma-4-E2B-it.litertlm>
//
//   # Against ffmpeg (steps 1-3 only; the test cannot stall or kill it):
//   ffmpeg -re -loop 1 -i test_assets/cats.jpg -vf scale=1280:960 -r 30 \
//     -q:v 5 -f mpjpeg -listen 1 http://127.0.0.1:8090/video &
//   flutter test integration_test/network_camera_test.dart -d macos \
//     --dart-define=GEMMA_MODEL_PATH=<path/to/gemma-4-E2B-it.litertlm> \
//     --dart-define=NETWORK_CAMERA_URL=http://127.0.0.1:8090/video
//
// 1. setup → home → Demo 3 on the network camera; ≥ 30 frames detected, the
//    cats among the boxes, the camera's frames at src_fps > 10.
//    `NETCAM_TEST src_fps=… det_fps=… size=… lat_p50=… decoder="…" …` (the
//    JPEG decoder: TurboJPEG on a worker isolate, or the engine codec) and a
//    screenshot path.
// 2. A detailed question ("Describe the scene.", fixture mic): the network
//    frame goes to Gemma; the answer mentions a cat. `CAMQ route=detailed …`
// 3. The chip names the source with its size and rate.
// 4. (in-test server) The stream stalls: Demo 3 fails within ~6 s with
//    "stopped sending frames", the button says Reconnect; Reconnect brings
//    the frames back. `NETCAM_STALL fail_ms=… reconnect_ms=…`
// 5. (in-test server) The server goes away: "closed the stream"; Reconnect
//    while it is down says "connection refused"; up again, Reconnect works.
// 6. Leaving Demo 3 closes the HTTP connection.

import 'dart:convert';
import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/app.dart';
import 'package:litert_edge_demos/config/demos.dart';
import 'package:litert_edge_demos/config/dependencies.dart';
import 'package:litert_edge_demos/config/env.dart';
import 'package:litert_edge_demos/domain/models/frame_source_spec.dart';
import 'package:litert_edge_demos/domain/models/live_state.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/models/route_decision.dart';
import 'package:litert_edge_demos/domain/models/voice.dart';
import 'package:litert_edge_demos/domain/vision/coco_vocabulary.dart';
import 'package:litert_edge_demos/ui/features/home/views/home_screen.dart';
import 'package:litert_edge_demos/ui/features/live_camera/view_models/live_camera_view_model.dart';
import 'package:litert_edge_demos/ui/features/live_camera/views/live_camera_screen.dart';
import 'package:litert_edge_demos/utils/pcm.dart';
import 'package:litert_edge_demos/utils/redact_url.dart';
import 'package:litert_edge_demos/utils/result.dart';
import 'package:provider/provider.dart';

import 'support/app_window.dart';
import 'support/demo3_settings.dart';
import 'support/fixtures.dart';
import 'support/pump.dart';
import 'support/wav.dart';

final _screenKey = GlobalKey();

Future<String> saveScreenshot(String name) async {
  final boundary =
      _screenKey.currentContext?.findRenderObject() as RenderRepaintBoundary?;
  if (boundary == null) fail('No RepaintBoundary to capture');
  final image = await boundary.toImage(pixelRatio: 2);
  final png = await image.toByteData(format: ui.ImageByteFormat.png);
  image.dispose();
  if (png == null) fail('PNG encoding failed');
  final file = File('${Directory.systemTemp.path}/$name.png');
  await file.writeAsBytes(png.buffer.asUint8List());
  return file.path;
}

String ms(Duration? d) => d == null ? '–' : '${d.inMilliseconds}';
String f1(double? v) => v?.toStringAsFixed(1) ?? '–';

Future<Uint8List> asset(String path) async =>
    (await rootBundle.load(path)).buffer.asUint8List();

/// An MJPEG camera like IP Webcam's /video: `multipart/x-mixed-replace`,
/// one JPEG part with Content-Length every [interval]. [stalled] keeps the
/// connection open without sending; [close] drops the server.
final class MjpegTestServer {
  MjpegTestServer._(this._server, this._jpeg, this._interval) {
    _server.listen(_serve);
  }

  static Future<MjpegTestServer> start(
    Uint8List jpeg, {
    int port = 0,
    Duration interval = const Duration(milliseconds: 33),
  }) async => MjpegTestServer._(
    await HttpServer.bind(InternetAddress.loopbackIPv4, port),
    jpeg,
    interval,
  );

  final HttpServer _server;
  final Uint8List _jpeg;
  final Duration _interval;
  bool stalled = false;
  bool _closed = false;
  int partsSent = 0;

  int get port => _server.port;
  int get connections => _server.connectionsInfo().total;

  Future<void> _serve(HttpRequest request) async {
    const boundary = 'Ba4oTvQMY8ew04N8dcnM';
    final response = request.response
      ..headers.set(
        'Content-Type',
        'multipart/x-mixed-replace;boundary=$boundary',
      )
      ..bufferOutput = false;
    final head = latin1.encode(
      '--$boundary\r\nContent-Type: image/jpeg\r\n'
      'Content-Length: ${_jpeg.length}\r\n\r\n',
    );
    try {
      while (!_closed) {
        if (!stalled) {
          response
            ..add(head)
            ..add(_jpeg)
            ..add(const [13, 10]);
          await response.flush();
          partsSent++;
        }
        await Future<void>.delayed(_interval);
      }
    } on Object {
      // The app closed the connection.
    } finally {
      try {
        await response.close();
      } on Object {
        // Already gone.
      }
    }
  }

  Future<void> close() async {
    _closed = true;
    await _server.close(force: true);
  }
}

void main() {
  initIntegrationTest();

  testWidgets('network camera: live boxes on MJPEG frames, a detailed '
      'question about a network frame, stall and loss with Reconnect', (
    tester,
  ) async {
    if (kGemmaModelPath.isEmpty) fail('Pass GEMMA_MODEL_PATH');
    final external = kNetworkCameraUrl.isNotEmpty;
    final cats = await asset('test_assets/cats.jpg');
    final qDescribe = pcm16FromWav(await asset('test_assets/q_describe.wav'));

    MjpegTestServer? server;
    final Uri url;
    if (external) {
      url = Uri.parse(kNetworkCameraUrl);
    } else {
      server = await MjpegTestServer.start(cats);
      url = Uri.parse('http://127.0.0.1:${server.port}/video');
    }
    addTearDown(() async => server?.close());
    final label = 'Network camera · ${url.host}:${url.port}';
    debugPrint('NETCAM_TEST camera=${redactUrl(url)} external=$external');

    final mic = FixtureMicService(qDescribe);
    await resetDemo3Settings();
    final deps = await AppDependencies.create(
      mic: mic,
      frameSource: Result.ok(NetworkSourceSpec(url)),
    );
    // Also on a failed expectation: the models and the source are released.
    addTearDown(deps.dispose);
    await tester.pumpWidget(
      RepaintBoundary(
        key: _screenKey,
        child: App(dependencies: deps),
      ),
    );

    await pumpUntil(
      tester,
      () => find.byType(HomeScreen).evaluate().isNotEmpty,
      timeout: const Duration(minutes: 8),
      reason: 'model setup',
    );
    for (final id in [
      ModelId.chat,
      ModelId.inflectNano,
      ModelId.yolo26n,
      ModelId.moonshineTiny,
    ]) {
      expect(deps.models.states.value[id], isA<ModelReady>(), reason: '$id');
    }
    await tester.pump(const Duration(milliseconds: 400));

    // 1. Demo 3 on the network camera.
    await tester.tap(find.byKey(HomeKeys.tile(Demo.liveCamera)));
    await pumpUntil(
      tester,
      () => find.byType(LiveCameraScreen).evaluate().isNotEmpty,
      timeout: const Duration(seconds: 5),
      reason: 'Demo 3 to open',
    );
    final vm = Provider.of<LiveCameraViewModel>(
      tester.element(find.byType(LiveCameraScreen)),
      listen: false,
    );
    expect(vm.sourceLock, contains('FRAME_SOURCE=network'));

    Future<int> waitLive(String what, {int frames = 30}) async {
      var detections = 0;
      void onFrame() {
        if (deps.live.frames.value != null) detections++;
      }

      final watch = Stopwatch()..start();
      deps.live.frames.addListener(onFrame);
      try {
        await pumpUntil(
          tester,
          () => deps.live.state.value is LiveRunning && detections >= frames,
          timeout: const Duration(seconds: 30),
          reason: '$what to run live',
          describe: () =>
              'state=${deps.live.state.value} start=${vm.startError} '
              'detections=$detections',
        );
      } finally {
        deps.live.frames.removeListener(onFrame);
      }
      return watch.elapsedMilliseconds;
    }

    await pumpUntil(
      tester,
      () => vm.chatReady,
      timeout: const Duration(seconds: 15),
      reason: 'the camera chat to open',
    );
    final toLive = await waitLive('the network camera');
    // A second of steady state for the rates.
    await pumpFor(tester, const Duration(seconds: 2));
    final state = deps.live.state.value;
    expect(state, isA<LiveRunning>());
    // "… · slow JPEG decoder" when the engine codec stands in for TurboJPEG.
    expect((state as LiveRunning).source, startsWith(label));
    final stats = deps.live.stats.value;
    final decoder = deps.live.sourceInfo?.decoder ?? '–';
    final frame = deps.live.frames.value!;
    final boxes = [
      for (var i = 0; i < frame.count; i++)
        '${cocoName(frame.classId(i))} ${frame.score(i).toStringAsFixed(2)}',
    ];
    debugPrint(
      'NETCAM_TEST source="${state.source}" '
      'size=${stats.sourceWidth}x${stats.sourceHeight} '
      'src_fps=${f1(stats.sourceFps)} det_fps=${f1(stats.fps)} '
      'lat_p50=${f1(stats.latencyMs)}ms copy_p50=${f1(stats.copyMs)}ms '
      'pre/run/post=${f1(stats.preMs)}/${f1(stats.runMs)}/${f1(stats.postMs)}ms '
      'drop_busy=${stats.droppedBusy} drop_rate=${stats.droppedRate} '
      'to_live=${toLive}ms detector=${vm.detectorLabel} '
      'decoder="$decoder" boxes=[${boxes.join(', ')}]',
    );
    expect(stats.sourceWidth, greaterThan(0));
    // The bar on every platform: the camera's frames reach the detector at
    // more than 10 fps (the in-test server sends ~20-30, IP Webcam 15-30).
    expect(
      stats.sourceFps,
      greaterThan(10),
      reason: 'src_fps > 10 (decoder: $decoder)',
    );
    expect(stats.fps, greaterThan(10), reason: 'detector ≥ 10 fps');
    expect([
      for (var i = 0; i < frame.count; i++) cocoName(frame.classId(i)),
    ], contains('cat'));
    // 3. The chip: source, size and rate.
    expect(
      find.textContaining(
        '${state.source} · ${stats.sourceWidth}×${stats.sourceHeight} · ',
      ),
      findsOneWidget,
    );
    debugPrint('SCREENSHOT ${await saveScreenshot('netcam_live')}');

    // 2. A detailed question: the network frame goes to Gemma.
    mic.pcm = qDescribe;
    final gesture = await tester.startGesture(
      tester.getCenter(find.byKey(LiveCameraKeys.mic)),
    );
    await pumpUntil(
      tester,
      () => vm.isListening,
      timeout: const Duration(seconds: 3),
      reason: 'the mic to open',
    );
    await pumpFor(
      tester,
      pcm16Duration(qDescribe.length, 16000) +
          const Duration(milliseconds: 550),
    );
    final release = Stopwatch()..start();
    await gesture.up();
    await pumpUntil(
      tester,
      () => vm.phase == TurnPhase.idle && vm.exchange.answer != null,
      timeout: const Duration(seconds: 90),
      reason: 'the detailed answer',
      describe: () =>
          'phase=${vm.phase} q=${vm.exchange.question} '
          'a=${vm.exchange.answer} notice=${vm.exchange.notice}',
    );
    final camera = deps.diagnostics.latest.lastCameraTurn!;
    final generation = deps.diagnostics.latest.lastGeneration;
    final sent = vm.lastSentImage;
    final answer = vm.exchange.answer!;
    // release_to_idle is the test's wall clock around pumps: a hidden window
    // stretches it (pumps wait for frames). turn_total is the app's own
    // release-to-drained figure.
    debugPrint(
      'CAMQ route=detailed source=network rule=${camera.route.rule} '
      'frame=${sent?.frameId} png=${sent?.width}x${sent?.height} '
      'png_bytes=${sent?.png.length} snap=${ms(camera.snapshotLatency)}ms '
      'ttft=${ms(generation?.timeToFirstToken)}ms '
      'release_to_idle=${release.elapsedMilliseconds}ms '
      'turn_total=${ms(deps.diagnostics.latest.lastVoiceTurn?.total)}ms '
      'transcript="${vm.exchange.question}" answer="$answer"',
    );
    expect(camera.route, isA<DetailedRoute>());
    expect(sent, isNotNull, reason: 'the network frame went to Gemma');
    expect(answer.toLowerCase(), contains('cat'));
    await waitLive('the network camera after the answer', frames: 10);

    if (server != null) {
      // 4. The stream stalls (phone asleep, Wi-Fi gone, connection kept).
      final stall = Stopwatch()..start();
      server.stalled = true;
      await pumpUntil(
        tester,
        () => deps.live.state.value is LiveFailed,
        timeout: const Duration(seconds: 12),
        reason: 'the stall to fail Demo 3',
      );
      final failMs = stall.elapsedMilliseconds;
      final message = (deps.live.state.value as LiveFailed).message;
      expect(message, contains('stopped sending frames'));
      await tester.pump();
      final retry = find.byKey(LiveCameraKeys.retryLive);
      expect(
        find.descendant(of: retry, matching: find.text('Reconnect')),
        findsOneWidget,
      );
      debugPrint('SCREENSHOT ${await saveScreenshot('netcam_stalled')}');
      server.stalled = false;
      final reconnect = Stopwatch()..start();
      await tester.tap(retry);
      await waitLive('Reconnect after the stall', frames: 10);
      debugPrint(
        'NETCAM_STALL fail_ms=$failMs reconnect_ms='
        '${reconnect.elapsedMilliseconds} message="$message"',
      );

      // 5. The server goes away, then comes back on the same port.
      final port = server.port;
      await server.close();
      await pumpUntil(
        tester,
        () => deps.live.state.value is LiveFailed,
        timeout: const Duration(seconds: 12),
        reason: 'the lost server to fail Demo 3',
      );
      final lost = (deps.live.state.value as LiveFailed).message;
      expect(lost, anyOf(contains('closed the stream'), contains('Lost')));
      await tester.tap(find.byKey(LiveCameraKeys.retryLive));
      await pumpUntil(
        tester,
        () => vm.startError != null,
        timeout: const Duration(seconds: 12),
        reason: 'Reconnect with the server down to fail',
      );
      final refused = vm.startError!;
      expect(refused, contains('connection refused'));
      server = await MjpegTestServer.start(cats, port: port);
      await tester.tap(find.byKey(LiveCameraKeys.retryLive));
      await waitLive('Reconnect after the server came back', frames: 10);
      debugPrint('NETCAM_LOSS lost="$lost" down="$refused" reconnected=true');
    }

    // 6. Leaving Demo 3 closes the connection.
    await tester.pageBack();
    await pumpUntil(
      tester,
      () =>
          find.byType(LiveCameraScreen).evaluate().isEmpty &&
          deps.live.state.value is LiveStopped,
      timeout: const Duration(seconds: 5),
      reason: 'Demo 3 to stop',
    );
    if (server case final s?) {
      await pumpUntil(
        tester,
        () => s.connections == 0,
        timeout: const Duration(seconds: 3),
        reason: 'the HTTP connection to close',
      );
      debugPrint('NETCAM_TEST parts_sent=${s.partsSent} closed=true');
    }
  });
}
