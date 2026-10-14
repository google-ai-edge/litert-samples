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

// Demo 3's visual memory in the real app (macOS, with its window): the
// fixture sequence plays in the live view, the memory fills while the demo
// is idle, and memory questions are asked aloud (spoken by the app's own
// Inflect at start-up, heard through the fixture microphone by moonshine).
//
//   flutter test integration_test/visual_memory_app_test.dart -d macos \
//     --dart-define=GEMMA_MODEL_PATH=<path/to/gemma-4-E2B-it.litertlm> \
//     --dart-define=MEMORY_FIXTURE_DIR=<a copy of test_assets/showcase>
//
// The sandboxed macOS debug build reads files only from its container and
// ~/Downloads: pass copies there. Needs the screen unlocked and the test
// window visible (support/app_window.dart). The pipeline itself (and the
// acceptance constants) is checked without a window by
// integration_test/visual_memory_test.dart.
//
// 1. setup → home → Demo 3 over three scenes (the Wi-Fi password note, the
//    street sign, the pirate mug; the app's fixture source holds each 3 s
//    and loops): the memory chip counts at least one frame of each.
// 2. "Where was the sign about the wifi password?" aloud: routed to the
//    memory, the card shows the Wi-Fi note's frame, and Gemma answers about
//    it (the chat model has images). `MEMQ …`
// 3. "Have you seen a giraffe?" aloud: "I haven't seen that in the last …",
//    with no LLM turn.
// 4. A new source (the cats alone) clears the memory: the chip says
//    "Memory: nothing yet".

import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/app.dart';
import 'package:litert_edge_demos/config/bootstrap.dart';
import 'package:litert_edge_demos/config/demos.dart';
import 'package:litert_edge_demos/config/dependencies.dart';
import 'package:litert_edge_demos/config/env.dart';
import 'package:litert_edge_demos/config/live_camera_config.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/services/detector/memory_shot.dart';
import 'package:litert_edge_demos/data/services/frames/engine_image_decoder.dart';
import 'package:litert_edge_demos/domain/models/frame_source_info.dart';
import 'package:litert_edge_demos/domain/models/frame_source_spec.dart';
import 'package:litert_edge_demos/domain/models/live_state.dart';
import 'package:litert_edge_demos/domain/models/route_decision.dart';
import 'package:litert_edge_demos/domain/models/scene_snapshot.dart';
import 'package:litert_edge_demos/domain/models/visual_memory.dart';
import 'package:litert_edge_demos/domain/models/voice.dart';
import 'package:litert_edge_demos/ui/features/home/views/home_screen.dart';
import 'package:litert_edge_demos/ui/features/live_camera/view_models/live_camera_view_model.dart';
import 'package:litert_edge_demos/ui/features/live_camera/views/live_camera_screen.dart';
import 'package:litert_edge_demos/utils/result.dart';
import 'package:provider/provider.dart';

import 'support/app_window.dart';
import 'support/demo3_settings.dart';
import 'support/fixtures.dart';
import 'support/pump.dart';
import 'support/question_audio.dart';

const kMemoryFixtureDir = String.fromEnvironment('MEMORY_FIXTURE_DIR');

const _scenes = [
  ('wifi', 'sign_wifi.jpg'),
  ('street_sign', 'sign_do_not_feed.jpg'),
  ('pirate_mug', 'coco_2592_pirate_mug.jpg'),
];

void main() {
  initIntegrationTest();

  testWidgets('visual memory in the app: the right frame on the card, an '
      'unseen object rejected, cleared on a new source', (tester) async {
    if (kGemmaModelPath.isEmpty || kMemoryFixtureDir.isEmpty) {
      fail('Pass GEMMA_MODEL_PATH and MEMORY_FIXTURE_DIR');
    }
    // The scenes where the app's fixture source can read them, in order.
    final dir = Directory('${Directory.systemTemp.path}/visual_memory_app');
    if (dir.existsSync()) dir.deleteSync(recursive: true);
    dir.createSync(recursive: true);
    final sceneGrey = <String, Uint8List>{};
    for (final (i, (name, file)) in _scenes.indexed) {
      final bytes = await File('$kMemoryFixtureDir/$file').readAsBytes();
      await File('${dir.path}/${i}_$name.jpg').writeAsBytes(bytes);
      final decoded = await decodeEncodedImage(bytes, maxSide: kFixtureMaxSide);
      final w = decoded.image.width;
      final h = decoded.image.height;
      decoded.image.dispose();
      final (tw, th) = fitLongSide(w, h, kMemoryThumbMaxSide);
      final rgb = downscaleRgb(
        RgbaPixels(width: w, height: h, bytes: decoded.rgba),
        tw,
        th,
      );
      sceneGrey[name] = greyThumbnail(
        rgb,
        tw,
        th,
        kMemoryGreySide * kMemoryCellPixels,
      );
    }
    String sceneOf(MemoryEntry e) => sceneGrey.entries
        .reduce(
          (a, b) =>
              greyDifference(a.value, e.grey) <= greyDifference(b.value, e.grey)
              ? a
              : b,
        )
        .key;
    final catsPath = '${Directory.systemTemp.path}/visual_memory_cats.jpg';
    final cats = await rootBundle.load('test_assets/cats.jpg');
    await File(catsPath).writeAsBytes(cats.buffer.asUint8List());

    // The questions, spoken by the app's own TTS (moonshine's 5 s window).
    await initEdgeAi();
    final tts = await loadAppTts();
    final window = kSttConfigs[Demo.liveCamera.stt]!.window;
    final wifiQuestion = await speakQuestion(
      tts,
      'Where was the sign about the wifi password?',
      window: window,
    );
    final giraffeQuestion = await speakQuestion(
      tts,
      'Have you seen a giraffe?',
      window: window,
    );
    await tts.close();

    final mic = FixtureMicService(wifiQuestion.pcm);
    await resetDemo3Settings();
    final deps = await AppDependencies.create(
      mic: mic,
      frameSource: Result.ok(FixtureSourceSpec([dir.path])),
    );
    await tester.pumpWidget(App(dependencies: deps));
    await pumpUntil(
      tester,
      () => find.byType(HomeScreen).evaluate().isNotEmpty,
      timeout: const Duration(minutes: 8),
      reason: 'model setup',
    );
    await tester.tap(find.byKey(HomeKeys.tile(Demo.liveCamera)));
    await pumpUntil(
      tester,
      () => find.byType(LiveCameraScreen).evaluate().isNotEmpty,
      timeout: const Duration(seconds: 5),
      reason: 'the camera screen',
    );
    final vm = Provider.of<LiveCameraViewModel>(
      tester.element(find.byType(LiveCameraScreen)),
      listen: false,
    );
    await pumpUntil(
      tester,
      () => vm.chatReady && deps.live.state.value is LiveRunning,
      timeout: const Duration(seconds: 20),
      reason: 'the camera chat open and the fixtures live',
    );

    // 1. The memory fills (the fixture loops: every scene comes round).
    await pumpUntil(
      tester,
      () =>
          {for (final e in deps.memory.entries) sceneOf(e)}
              .containsAll([for (final (name, _) in _scenes) name]),
      timeout: const Duration(seconds: 60),
      reason: 'a frame of every scene in the memory',
      describe: () =>
          '${deps.memory.status.value.frames} frames, '
          'off=${deps.memory.status.value.unavailable}',
    );
    expect(find.byKey(LiveCameraKeys.memoryChip), findsOneWidget);
    expect(find.textContaining('Memory: '), findsOneWidget);

    Future<void> askAloud(Uint8List pcm) async {
      mic.pcm = pcm;
      final gesture = await tester.startGesture(
        tester.getCenter(find.byKey(LiveCameraKeys.mic)),
      );
      await pumpUntil(
        tester,
        () => vm.isListening,
        timeout: const Duration(seconds: 10),
        reason: 'Listening',
      );
      await pumpUntil(
        tester,
        () => mic.played,
        timeout: const Duration(seconds: 10),
        reason: 'the question played',
      );
      await pumpFor(tester, const Duration(milliseconds: 300));
      await gesture.up();
      await pumpUntil(
        tester,
        () => vm.phase == TurnPhase.idle && vm.exchange.answer != null,
        timeout: const Duration(seconds: 60),
        reason: 'the answer',
        describe: () =>
            'phase=${vm.phase} q=${vm.exchange.question} '
            'notice=${vm.exchange.notice}',
      );
    }

    // 2. The Wi-Fi note, found by meaning.
    await askAloud(wifiQuestion.pcm);
    final turn = deps.diagnostics.latest.lastCameraTurn!;
    expect(turn.route, isA<MemoryRoute>(), reason: vm.exchange.question);
    final found = turn.memory!.recall;
    expect(found, isA<MemoryHit>(), reason: '${turn.memory!.query}: $found');
    final hit = found as MemoryHit;
    expect(sceneOf(hit.entry), 'wifi');
    expect(find.byKey(LiveCameraKeys.memoryCard), findsOneWidget);
    debugPrint(
      'MEMQ heard="${vm.exchange.question}" query="${turn.memory!.query}" '
      'cos=${hit.cosine?.toStringAsFixed(3)} '
      'rival=${hit.rival?.toStringAsFixed(3)} age=${hit.age.inSeconds}s '
      'search=${turn.memory!.elapsed.inMilliseconds}ms '
      'ttft=${turn.timeToFirstToken?.inMilliseconds}ms '
      'answer="${vm.exchange.answer}"',
    );

    // 3. Something never shown: rejected, no LLM turn.
    final llmBefore = deps.diagnostics.latest.llmTurns;
    await askAloud(giraffeQuestion.pcm);
    expect(vm.exchange.answer, startsWith("I haven't seen that in the last"));
    expect(deps.diagnostics.latest.llmTurns, llmBefore);
    expect(find.byKey(LiveCameraKeys.memoryCard), findsNothing);

    // 4. A new source session forgets everything.
    final started = await deps.live.start(
      FixtureSourceSpec([catsPath]),
      owner: vm,
    );
    expect(started, isA<Ok<FrameSourceInfo>>());
    expect(deps.memory.entries, isEmpty);
    await tester.pump();
    expect(find.text('Memory: nothing yet'), findsOneWidget);

    await tester.pageBack();
    await pumpFor(tester, const Duration(milliseconds: 500));
    expect(deps.memory.status.value.active, isFalse);
    await deps.dispose();
  });
}
