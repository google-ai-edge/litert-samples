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

// Showcase (macOS): a scripted run through both demos that saves
// screenshots of them. Not a regression test: it fails only when a step
// cannot be reached, and every shot is best effort.
//
//   flutter test integration_test/showcase_test.dart -d macos \
//     --dart-define=GEMMA_MODEL_PATH=<path/to/gemma-4-E2B-it.litertlm> \
//     --dart-define=SHOWCASE_DIR=<path/to/test_assets>/showcase
//
// The detector and EmbeddingGemma are the ones built into the app.
//
// SHOWCASE_DIR is test_assets/showcase (its parent's cats.jpg is used too).
// The sandboxed macOS debug build reads files only from its container and
// ~/Downloads: pass a copy of test_assets/ under ~/Downloads.
//
// Fixtures (test_assets/showcase/): a COCO val2017 scene picked with the
// detector (a pirate mug next to a knife), two signs drawn by
// tool/make_showcase_signs.py, and the spoken questions (16 kHz WAV).
// Shots go to the app container's tmp/ as `showcase_*.png` (printed as
// `SHOT …`). The diagnostics overlay is off except in the `*_nerd` shots.
//
// Keep the window visible: a covered macOS window stops Flutter frames.

import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/app.dart';
import 'package:litert_edge_demos/config/demos.dart';
import 'package:litert_edge_demos/config/dependencies.dart';
import 'package:litert_edge_demos/config/env.dart';
import 'package:litert_edge_demos/data/services/skills/skill_store_service.dart';
import 'package:litert_edge_demos/domain/models/chat_entry.dart';
import 'package:litert_edge_demos/domain/models/frame_source_info.dart';
import 'package:litert_edge_demos/domain/models/frame_source_spec.dart';
import 'package:litert_edge_demos/domain/models/live_state.dart';
import 'package:litert_edge_demos/domain/models/voice.dart';
import 'package:litert_edge_demos/ui/core/debug_overlay.dart';
import 'package:litert_edge_demos/ui/features/home/views/home_screen.dart';
import 'package:litert_edge_demos/ui/features/live_camera/view_models/live_camera_view_model.dart';
import 'package:litert_edge_demos/ui/features/live_camera/views/live_camera_screen.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/view_models/voice_chat_view_model.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/views/chat_keys.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/views/citation_chips.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/views/skills_sheet.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/views/voice_chat_screen.dart';
import 'package:litert_edge_demos/utils/pcm.dart';
import 'package:litert_edge_demos/utils/result.dart';
import 'package:provider/provider.dart';

import 'support/app_window.dart';
import 'support/demo3_settings.dart';
import 'support/fixtures.dart';
import 'support/pump.dart';
import 'support/skill_fixtures.dart';
import 'support/wav.dart';

const kShowcaseDir = String.fromEnvironment('SHOWCASE_DIR');

final _screenKey = GlobalKey();

void main() {
  initIntegrationTest();

  testWidgets('showcase: screenshots of both demos', (tester) async {
    if (kGemmaModelPath.isEmpty || kShowcaseDir.isEmpty) {
      fail('Pass GEMMA_MODEL_PATH and SHOWCASE_DIR');
    }
    final dir = kShowcaseDir;
    final shots = <String>[];
    final skipped = <String>[];

    Future<void> shot(String name) async {
      await pumpFor(tester, const Duration(milliseconds: 500));
      final boundary =
          _screenKey.currentContext?.findRenderObject()
              as RenderRepaintBoundary?;
      if (boundary == null) {
        skipped.add('$name (no boundary)');
        return;
      }
      final image = await boundary.toImage(pixelRatio: 2);
      final png = await image.toByteData(format: ui.ImageByteFormat.png);
      image.dispose();
      if (png == null) {
        skipped.add('$name (encode)');
        return;
      }
      final file = File('${Directory.systemTemp.path}/showcase_$name.png');
      await file.writeAsBytes(png.buffer.asUint8List());
      shots.add(file.path);
      debugPrint('SHOT ${file.path}');
    }

    /// Runs one best-effort step: a failure is printed and skipped.
    Future<void> step(String name, Future<void> Function() body) async {
      try {
        await body();
      } catch (e) {
        skipped.add('$name: $e');
        debugPrint('SHOT skipped $name: $e');
      }
    }

    void overlay(Type screen, {required bool on}) {
      final visible = DebugOverlayHost.visibilityOf(
        tester.element(find.byType(screen)),
      );
      visible?.value = on;
    }

    // The kid-clock skill is added at run time (Skills → Reload).
    final skillsDir = await defaultSkillsDirectory();
    final kidDir = Directory('${skillsDir.path}/kid-clock');
    if (await kidDir.exists()) await kidDir.delete(recursive: true);
    addTearDown(() async {
      if (await kidDir.exists()) await kidDir.delete(recursive: true);
    });

    final mug = await File('$dir/coco_2592_pirate_mug.jpg').readAsBytes();
    final mic = FixtureMicService(Uint8List(0));
    await resetDemo3Settings();
    final deps = await AppDependencies.create(
      mic: mic,
      imageInput: FixtureImageInputService(mug),
      // Demo 3's camera: one clean scene with a mug.
      frameSource: Result.ok(
        FixtureSourceSpec(['$dir/coco_2592_pirate_mug.jpg']),
      ),
    );
    await tester.pumpWidget(
      RepaintBoundary(
        key: _screenKey,
        child: App(dependencies: deps),
      ),
    );

    // 1. Home, with both tiles ready and the knowledge base indexed.
    await pumpUntil(
      tester,
      () => find.byType(HomeScreen).evaluate().isNotEmpty,
      timeout: const Duration(minutes: 10),
      reason: 'model setup',
    );
    await pumpFor(tester, const Duration(seconds: 3));
    overlay(HomeScreen, on: false);
    await step('home', () => shot('home'));

    // 2. Demo 1.
    await tester.tap(find.byKey(HomeKeys.tile(Demo.voiceChat)));
    await pumpUntil(
      tester,
      () => find.byType(VoiceChatScreen).evaluate().isNotEmpty,
      timeout: const Duration(seconds: 5),
      reason: 'Demo 1 to open',
    );
    final vm = Provider.of<VoiceChatViewModel>(
      tester.element(find.byType(VoiceChatScreen)),
      listen: false,
    );
    await pumpUntil(
      tester,
      () => vm.isReady && vm.hasSkills,
      timeout: const Duration(seconds: 30),
      reason: 'the agent chat to open',
    );
    overlay(VoiceChatScreen, on: false);

    Future<ChatEntry> ask(String prompt) async {
      final before = vm.entries.length;
      await tester.tap(find.byKey(ChatKeys.input));
      await tester.pump();
      await tester.enterText(find.byKey(ChatKeys.input), prompt);
      await tester.pump();
      await tester.tap(find.byKey(ChatKeys.send));
      await pumpUntil(
        tester,
        () => vm.entries.length > before,
        timeout: const Duration(seconds: 10),
        reason: 'the turn to start',
      );
      await pumpUntil(
        tester,
        () => !vm.send.running && vm.phase == TurnPhase.idle,
        timeout: const Duration(seconds: 120),
        reason: 'the reply to "$prompt"',
      );
      await pumpFor(tester, const Duration(milliseconds: 300));
      final reply = vm.entries.lastWhere(
        (e) => e.role == ChatRole.assistant || e.role == ChatRole.error,
      );
      debugPrint(
        'SHOWCASE q="$prompt" steps=${reply.steps} '
        'reply="${reply.text.replaceAll('\n', ' ')}"',
      );
      return reply;
    }

    Future<void> fresh() async {
      await vm.newConversation.execute();
      await pumpFor(tester, const Duration(milliseconds: 300));
    }

    await step('image', () async {
      await tester.tap(find.byKey(ChatKeys.attachGallery));
      await pumpUntil(
        tester,
        () => vm.attachment != null,
        timeout: const Duration(seconds: 10),
        reason: 'the picture to attach',
      );
      await ask("What's funny about this picture?");
      await shot('demo1_image_qa');
    });

    await step('rag', () async {
      await fresh();
      final reply = await ask(
        'What is the input size of the YOLO 26 nano detector?',
      );
      debugPrint('SHOWCASE rag_tool_steps=${reply.steps.length}');
      await shot('demo1_rag_citations');
      await tester.tap(find.byKey(KnowledgeChipKeys.citation(1)));
      await pumpFor(tester, const Duration(milliseconds: 600));
      await shot('demo1_rag_excerpt');
      await tester.tap(find.text('Close'));
      await pumpFor(tester, const Duration(milliseconds: 400));
    });

    await step('time', () async {
      await fresh();
      await ask('Roughly how late is it right now?');
      await shot('demo1_skill_steps_time');
    });

    await step('device', () async {
      await ask('Which accelerator is running right now?');
      await shot('demo1_device_info');
      overlay(VoiceChatScreen, on: true);
      await pumpFor(tester, const Duration(milliseconds: 800));
      await shot('demo1_nerd');
      overlay(VoiceChatScreen, on: false);
    });

    await step('skills sheet', () async {
      await kidDir.create(recursive: true);
      await File('${kidDir.path}/SKILL.md').writeAsString(kidClockSkillMd);
      await tester.tap(find.byKey(ChatKeys.more));
      await tester.pumpAndSettle();
      await tester.tap(find.byKey(ChatKeys.skills));
      await tester.pumpAndSettle();
      await tester.tap(find.byKey(SkillsSheetKeys.reload));
      await pumpUntil(
        tester,
        () =>
            !vm.reloadSkills.running &&
            !vm.applySkills.running &&
            vm.skillCatalog!.skills.any((s) => s.skill.name == 'kid-clock'),
        timeout: const Duration(seconds: 30),
        reason: 'the kid-clock skill to load',
      );
      await tester.pumpAndSettle();
      await shot('demo1_skills_sheet');
      await tester.tapAt(const Offset(10, 10));
      await tester.pumpAndSettle();
      await ask('Tell my kid what time it is.');
      await shot('demo1_kid_clock');
    });

    await step('barge-in', () async {
      await fresh();
      final before = vm.entries.length;
      await tester.tap(find.byKey(ChatKeys.input));
      await tester.enterText(
        find.byKey(ChatKeys.input),
        'Tell me a long story about a robot who learns to paint.',
      );
      await tester.tap(find.byKey(ChatKeys.send));
      await pumpUntil(
        tester,
        () => vm.partialReply.value.length > 120,
        timeout: const Duration(seconds: 30),
        reason: 'the story to stream',
      );
      await tester.tap(find.byKey(ChatKeys.stop));
      await pumpUntil(
        tester,
        () =>
            vm.entries.length > before + 1 &&
            vm.phase == TurnPhase.idle &&
            !vm.send.running,
        timeout: const Duration(seconds: 20),
        reason: 'the stop',
      );
      await shot('demo1_interrupted');
    });

    await tester.pageBack();
    await pumpUntil(
      tester,
      () =>
          find.byType(HomeScreen).evaluate().isNotEmpty &&
          find.byType(VoiceChatScreen).evaluate().isEmpty,
      timeout: const Duration(seconds: 5),
      reason: 'back home',
    );
    await pumpFor(tester, const Duration(milliseconds: 500));

    // 3. Demo 3.
    await tester.tap(find.byKey(HomeKeys.tile(Demo.liveCamera)));
    await pumpUntil(
      tester,
      () => find.byType(LiveCameraScreen).evaluate().isNotEmpty,
      timeout: const Duration(seconds: 5),
      reason: 'Demo 3 to open',
    );
    final cam = Provider.of<LiveCameraViewModel>(
      tester.element(find.byType(LiveCameraScreen)),
      listen: false,
    );
    await pumpUntil(
      tester,
      () => cam.chatReady,
      timeout: const Duration(seconds: 20),
      reason: 'the camera chat',
    );
    overlay(LiveCameraScreen, on: false);

    Future<void> scene(String file) async {
      final started = await deps.live.start(
        FixtureSourceSpec(['$dir/$file']),
        owner: cam,
      );
      if (started is! Ok<FrameSourceInfo>) fail('fixture $file: $started');
      var frames = 0;
      void onFrame() => frames++;
      deps.live.frames.addListener(onFrame);
      try {
        await pumpUntil(
          tester,
          () => deps.live.state.value is LiveRunning && frames >= 10,
          timeout: const Duration(seconds: 20),
          reason: '$file to run live',
        );
      } finally {
        deps.live.frames.removeListener(onFrame);
      }
    }

    Future<void> askAloud(String wav) async {
      mic.pcm = pcm16FromWav(await File('$dir/$wav').readAsBytes());
      final gesture = await tester.startGesture(
        tester.getCenter(find.byKey(LiveCameraKeys.mic)),
      );
      await pumpUntil(
        tester,
        () => cam.isListening,
        timeout: const Duration(seconds: 3),
        reason: 'the mic to open',
      );
      await pumpFor(
        tester,
        pcm16Duration(mic.pcm.length, 16000) +
            const Duration(milliseconds: 550),
      );
      await gesture.up();
    }

    Future<void> answered() async {
      await pumpUntil(
        tester,
        () => cam.phase == TurnPhase.idle && cam.exchange.answer != null,
        timeout: const Duration(seconds: 60),
        reason: 'the answer',
      );
      await pumpFor(tester, const Duration(milliseconds: 400));
      debugPrint(
        'SHOWCASE camq q="${cam.exchange.question}" '
        'a="${cam.exchange.answer}" route="${cam.exchange.route}"',
      );
    }

    /// A detailed question: the shot is taken while the frame is still
    /// frozen and the answer has been generated (it is being spoken).
    Future<void> detailed(String wav, String name) async {
      await askAloud(wav);
      await pumpUntil(
        tester,
        () => cam.frozen.value != null && deps.conversation.isGenerating.value,
        timeout: const Duration(seconds: 20),
        reason: '$name: the frozen frame',
      );
      await pumpUntil(
        tester,
        () =>
            !deps.conversation.isGenerating.value &&
            cam.partialReply.value.isNotEmpty,
        timeout: const Duration(seconds: 40),
        reason: '$name: the generated answer',
      );
      if (cam.frozen.value != null) {
        await shot(name);
      } else {
        skipped.add('$name (unfroze before the shot)');
      }
      await answered();
    }

    await step('cats count', () async {
      await scene('../cats.jpg');
      await askAloud('q_cats.wav');
      await answered();
      await shot('demo3_fast_count_cats');
      overlay(LiveCameraScreen, on: true);
      await pumpFor(tester, const Duration(milliseconds: 800));
      await shot('demo3_nerd');
      overlay(LiveCameraScreen, on: false);
    });

    await step('pirate mug', () async {
      await scene('coco_2592_pirate_mug.jpg');
      await askAloud('q_see.wav');
      await answered();
      await shot('demo3_inventory_mug');
      await detailed('q_describe.wav', 'demo3_detailed_mug');
    });

    await step('sign: do not feed', () async {
      await scene('sign_do_not_feed.jpg');
      await detailed('q_sign.wav', 'demo3_sign_do_not_feed');
    });

    await step('sign: wifi', () async {
      await scene('sign_wifi.jpg');
      await detailed('q_sign.wav', 'demo3_sign_wifi');
    });

    debugPrint('SHOWCASE shots=${shots.length} skipped=$skipped');
    await tester.pumpWidget(const SizedBox.shrink());
    await deps.dispose();
  }, timeout: const Timeout(Duration(minutes: 30)));
}
