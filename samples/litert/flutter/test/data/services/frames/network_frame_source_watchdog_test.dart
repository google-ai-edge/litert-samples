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

import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/services/frames/jpeg_decoder.dart';
import 'package:litert_edge_demos/data/services/frames/network_frame_source.dart';
import 'package:litert_edge_demos/domain/models/frame_source_info.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../../support/mjpeg.dart';

/// The network camera's own stall watchdog when the app itself stops
/// running for a while (a debugger pause, system sleep): the time it was
/// suspended must not count as a silent camera. network_frame_source_test
/// covers a real stall.
///
/// Real time (a local server): the suspension is the main isolate blocked
/// in a busy loop, so no timer, socket read or server write runs during it.
/// The server holds its parts just before and during it, and what it had
/// already sent arrives first: no part is pending when the isolate stops, so
/// the overdue watchdog check (queued during the block) runs before the
/// next part can arrive, and only the late check keeps the suspension from
/// counting as silence.
void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  // flutter_test answers every HttpClient with 400; this test talks to a
  // real local server.
  HttpOverrides.global = null;

  const stall = Duration(milliseconds: 800);
  const watchInterval = Duration(milliseconds: 100);
  const partInterval = Duration(milliseconds: 20);
  const patience = Duration(seconds: 15);

  final servers = <HttpServer>[];
  final sources = <NetworkFrameSource>[];
  var serving = true;

  tearDown(() async {
    serving = false;
    for (final s in sources) {
      await s.stop();
    }
    sources.clear();
    for (final s in servers) {
      await s.close(force: true);
    }
    servers.clear();
  });

  /// Polls [condition] until it holds; fails the test after [patience].
  Future<void> until(bool Function() condition, String what) async {
    final watch = Stopwatch()..start();
    while (!condition()) {
      if (watch.elapsed > patience) fail('no $what within $patience');
      await Future<void>.delayed(const Duration(milliseconds: 5));
    }
  }

  testWidgets('a check that runs late because the app was suspended gives a '
      'fresh window instead of reporting a stall; frames carry on', (
    tester,
  ) async {
    await tester.runAsync(() async {
      final cats = catsJpeg();
      var hold = false;
      final server = await serve((request) async {
        final response = request.response
          ..headers.set(
            'Content-Type',
            'multipart/x-mixed-replace; boundary=frame',
          )
          ..bufferOutput = false;
        response.add(const [13, 10]);
        await response.flush();
        while (serving) {
          await Future<void>.delayed(partInterval);
          if (hold) continue;
          response.add(mjpegPart('frame', cats));
          await response.flush();
        }
        await response.close();
      });
      servers.add(server);
      final src = NetworkFrameSource(
        url: Uri.parse('http://127.0.0.1:${server.port}/video'),
        decoder: () async => const EngineJpegDecoder(),
        stallTimeout: stall,
        connectTimeout: const Duration(seconds: 10),
        watchInterval: watchInterval,
      );
      sources.add(src);
      final errors = <Exception>[];
      final started = await src.start((_) {}, onError: errors.add);
      expect(started, isA<Ok<FrameSourceInfo>>());
      await until(() => src.receivedFrames >= 3, 'first frames');

      // Nothing new from the server; what it already sent arrives.
      hold = true;
      await Future<void>.delayed(const Duration(milliseconds: 200));
      final held = src.receivedFrames;
      // Suspended for half as long again as the stall timeout: with the
      // hold, 1.4 s without a part.
      final suspended = Stopwatch()..start();
      while (suspended.elapsed < stall * 1.5) {}
      hold = false;

      // Longer than the stall timeout again, with frames flowing.
      await Future<void>.delayed(stall + watchInterval * 2);
      expect(errors, isEmpty, reason: 'the suspension is not a stall');
      expect(src.receivedFrames, greaterThan(held), reason: 'frames resumed');
    });
  });
}
