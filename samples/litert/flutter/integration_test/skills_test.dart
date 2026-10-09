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

// Demo 1 end to end (macOS): agent skills from Markdown.
//
//   flutter test integration_test/skills_test.dart -d macos \
//     --dart-define=GEMMA_MODEL_PATH=<path/to/gemma-4-E2B-it.litertlm>
//
// 1. "What time is it?" → current_time run directly by the app:
//    the app's clock, spoken.
// 2. "Which backends are you running on?" → device_info run directly: the
//    hardware report verbatim — the chip, then the chat model's real backend
//    and how it is known.
// 3. A time question the router leaves to the model: Gemma loads the
//    current-time skill and calls current_time.
// 4. A kid-clock SKILL.md dropped into the skills folder works after
//    Reload, without a rebuild: "Tell my kid what time it is." → current_time.
// 5. A malformed SKILL.md is listed with its parse error in the Skills sheet.
// 6. Device questions call device_info even when retrieval brings GPU docs.
// Prints `SKILLS time_ttfa=…ms` and one `SKILLS q=…` line per turn. Removes
// what it wrote to the folder.

import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/app.dart';
import 'package:litert_edge_demos/config/demos.dart';
import 'package:litert_edge_demos/config/dependencies.dart';
import 'package:litert_edge_demos/config/env.dart';
import 'package:litert_edge_demos/data/services/skills/skill_store_service.dart';
import 'package:litert_edge_demos/domain/models/chat_entry.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/models/skill_step.dart';
import 'package:litert_edge_demos/ui/features/home/views/home_screen.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/view_models/voice_chat_view_model.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/views/chat_keys.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/views/skills_sheet.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/views/voice_chat_screen.dart';
import 'package:provider/provider.dart';

import 'support/app_window.dart';
import 'support/pump.dart';
import 'support/skill_fixtures.dart';

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

String describeChat(VoiceChatViewModel vm) =>
    'phase=${vm.phase.name} error=${vm.error} canSend=${vm.canSend} '
    'ready=${vm.isReady} send=${vm.send.running} '
    'apply=${vm.applySkills.running} reload=${vm.reloadSkills.running} '
    'stt=${vm.selectStt.running} pending=${vm.skillsPending} '
    'sheet=${find.byKey(SkillsSheetKeys.sheet).evaluate().length} entries='
    '${vm.entries.map((e) => '${e.role.name}:"${e.text}" ${e.steps}').join(' | ')}';

String ms(Duration? d) => d == null ? '–' : '${d.inMilliseconds}';

void main() {
  initIntegrationTest();

  testWidgets('Demo 1 skills from Markdown: time and device info, a skill '
      'added at run time, a broken file listed', (tester) async {
    if (kGemmaModelPath.isEmpty) fail('Pass GEMMA_MODEL_PATH');

    // The folder the app scans; leftovers of an earlier run would change the
    // first scan.
    final skillsDir = await defaultSkillsDirectory();
    final kidDir = Directory('${skillsDir.path}/kid-clock');
    final brokenDir = Directory('${skillsDir.path}/broken-skill');
    Future<void> cleanUp() async {
      for (final dir in [kidDir, brokenDir]) {
        if (await dir.exists()) await dir.delete(recursive: true);
      }
    }

    await cleanUp();
    addTearDown(cleanUp);

    final deps = await AppDependencies.create();
    await tester.pumpWidget(
      RepaintBoundary(
        key: _screenKey,
        child: App(dependencies: deps),
      ),
    );

    // Setup: the required models (the knowledge base is optional here).
    await pumpUntil(
      tester,
      () => find.byType(HomeScreen).evaluate().isNotEmpty,
      timeout: const Duration(minutes: 12),
      reason: 'model setup',
      describe: () => '${deps.models.states.value}',
    );
    for (final MapEntry(:key, :value) in deps.models.states.value.entries) {
      if (key.spec.required) {
        expect(value, isA<ModelReady>(), reason: key.spec.displayName);
      }
    }
    await tester.pump(const Duration(milliseconds: 400));
    final tile = find.byKey(HomeKeys.tile(Demo.voiceChat));
    await pumpUntil(
      tester,
      () => tester.widget<ListTile>(tile).enabled,
      timeout: const Duration(seconds: 10),
      reason: 'the Demo 1 tile to be enabled',
    );
    await tester.tap(tile);
    await pumpUntil(
      tester,
      () => find.byType(VoiceChatScreen).evaluate().length == 1,
      // Entering Demo 1 switches the recognizer to Whisper; its load and
      // warm-up hold frames for a few seconds on macOS.
      timeout: const Duration(seconds: 20),
      reason: 'the chat screen to open',
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
      describe: () => describeChat(vm),
    );
    final catalog = vm.skillCatalog!;
    debugPrint(
      'SKILLS dir=${catalog.directory} skills='
      '${catalog.skills.map((s) => s.skill.name).toList()} '
      'errors=${catalog.errors.map((e) => e.path).toList()}',
    );
    final names = catalog.skills.map((s) => s.skill.name);
    expect(names, containsAll(['device-info', 'current-time']));
    // A user folder seeded by an older build may still hold the removed
    // skills: they are listed but name intents the app no longer has.
    for (final removed in ['timer', 'camera-watch']) {
      if (names.contains(removed)) debugPrint('SKILLS leftover: $removed');
    }
    expect(vm.speakReplies, isTrue);
    await tester.pump(const Duration(milliseconds: 400));

    /// Types [prompt], taps Send, waits for the turn and its audio. Returns
    /// the reply (or the error entry of a failed turn).
    Future<ChatEntry> ask(String prompt) async {
      final before = vm.entries.length;
      // Focus the field first: after a modal sheet closed, the text input
      // client is re-attached only once the field has focus again.
      await tester.tap(find.byKey(ChatKeys.input));
      await tester.pump();
      await tester.enterText(find.byKey(ChatKeys.input), prompt);
      await tester.pump();
      final field = tester.widget<TextField>(find.byKey(ChatKeys.input));
      expect(
        field.controller?.text,
        prompt,
        reason: 'the prompt reached the text field',
      );
      await tester.tap(find.byKey(ChatKeys.send));
      await pumpUntil(
        tester,
        () => vm.entries.length > before,
        timeout: const Duration(seconds: 10),
        reason: 'the turn to start',
        describe: () => describeChat(vm),
      );
      await pumpUntil(
        tester,
        () => !vm.send.running,
        timeout: const Duration(seconds: 120),
        reason: 'the reply to "$prompt"',
        describe: () => describeChat(vm),
      );
      await tester.pump(const Duration(milliseconds: 100));
      final reply = vm.entries.lastWhere(
        (e) => e.role == ChatRole.assistant || e.role == ChatRole.error,
      );
      final g = deps.diagnostics.latest.lastGeneration;
      final r = reply.knowledge?.retrieval;
      debugPrint(
        'SKILLS q="$prompt" steps=${reply.steps} tools=${g?.toolRounds} '
        'ttft=${ms(g?.timeToFirstToken)}ms '
        'first_audio=${ms(deps.diagnostics.latest.lastVoiceTurn?.firstAudio)}ms '
        'rag=${r?.outcome.name} top=${r?.topSimilarity?.toStringAsFixed(3)} '
        'reply="${reply.text.replaceAll('\n', ' ')}"',
      );
      return reply;
    }

    /// The call of [intent], also when the model spelled it as the skill
    /// name the executor normalizes ("current-time" runs current_time).
    IntentCalled? called(ChatEntry reply, String intent) {
      for (final step in reply.steps) {
        if (step case IntentCalled(intent: final name)
            when name.toLowerCase().replaceAll('-', '_') == intent) {
          return step;
        }
      }
      return null;
    }

    // 1. The time runs current_time directly: the app's clock, spoken.
    final timeReply = await ask('What time is it?');
    expect(timeReply.role, ChatRole.assistant, reason: describeChat(vm));
    expect(timeReply.steps.whereType<SkillLoaded>(), isEmpty);
    expect(called(timeReply, 'current_time'), isNotNull);
    expect(timeReply.text, startsWith('It is '));
    expect(
      timeReply.text,
      timeReply.steps.whereType<IntentSucceeded>().single.result,
    );
    final timeTtfa = deps.diagnostics.latest.lastVoiceTurn?.firstAudio;
    expect(timeTtfa, isNotNull, reason: 'the reply was spoken');

    // 2. Device info: the hardware report, verbatim (no rephrasing).
    final deviceReply = await ask('Which backends are you running on?');
    expect(
      called(deviceReply, 'device_info'),
      isNotNull,
      reason: describeChat(vm),
    );
    expect(deviceReply.role, ChatRole.assistant, reason: describeChat(vm));
    expect(
      deviceReply.text,
      deviceReply.steps.whereType<IntentSucceeded>().single.result,
    );
    expect(
      deviceReply.text,
      startsWith('This Mac has an Apple'),
      reason: 'names the chip',
    );
    // The loaded chat model leads the second sentence: Gemma 4 E2B, or the
    // user's own model when one is chosen.
    final chatName = deps.conversation.capabilities.modelName;
    expect(
      deviceReply.text,
      contains('. $chatName '),
      reason: 'the chat model ($chatName) first',
    );
    expect(
      deviceReply.text,
      contains('on the GPU (confirmed)'),
      reason: 'names the real backend and how it is known',
    );
    await tester.pump(const Duration(milliseconds: 400));
    debugPrint('SCREENSHOT ${await saveScreenshot('skills_device_info')}');

    // 3. A time question the router leaves to the model (not one of the
    //    direct phrasings): Gemma loads the skill and calls current_time.
    final modelTime = await ask('Roughly how late is it right now?');
    expect(
      called(modelTime, 'current_time'),
      isNotNull,
      reason: describeChat(vm),
    );

    // 4–5. A skill and a broken file dropped into the folder, then Reload.
    await kidDir.create(recursive: true);
    await File('${kidDir.path}/SKILL.md').writeAsString(kidClockSkillMd);
    await brokenDir.create(recursive: true);
    await File('${brokenDir.path}/SKILL.md').writeAsString(brokenSkillMd);
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
          !vm.skillsPending &&
          vm.skillCatalog!.skills.any((s) => s.skill.name == 'kid-clock'),
      timeout: const Duration(seconds: 30),
      reason: 'the reload to apply the kid-clock skill',
      describe: () => describeChat(vm),
    );
    expect(vm.lastReload, SkillsReload.applied);
    await tester.pumpAndSettle();
    final brokenTile = find.byKey(
      SkillsSheetKeys.error('broken-skill/SKILL.md'),
    );
    await tester.scrollUntilVisible(
      brokenTile,
      120,
      // The sheet's list (its folder path is a SelectableText, which has a
      // Scrollable of its own further down).
      scrollable: find
          .descendant(
            of: find.byKey(SkillsSheetKeys.sheet),
            matching: find.byType(Scrollable),
          )
          .first,
    );
    expect(brokenTile, findsOneWidget);
    final brokenError = vm.skillCatalog!.errors.singleWhere(
      (e) => e.path == 'broken-skill/SKILL.md',
    );
    expect(brokenError.message, contains("expected a '---' fenced"));
    await tester.pump(const Duration(milliseconds: 400));
    debugPrint('SCREENSHOT ${await saveScreenshot('skills_sheet')}');
    // Close the sheet.
    await tester.tapAt(const Offset(10, 10));
    await tester.pumpAndSettle();
    expect(
      vm.entries.last.text,
      contains('started over'),
      reason: 'the chat says the history was dropped',
    );

    final kidReply = await ask('Tell my kid what time it is.');
    // The model may call the skill itself as the intent; the app
    // then runs its only call (current_time).
    expect(
      called(kidReply, 'current_time') ?? called(kidReply, 'kid-clock'),
      isNotNull,
      reason: describeChat(vm),
    );
    expect(
      kidReply.steps.whereType<IntentSucceeded>().last.result,
      startsWith('It is '),
    );

    // 6. Device questions that clear the knowledge-base gate.
    for (final prompt in const [
      'Which accelerator is running?',
      'What device am I on?',
    ]) {
      final reply = await ask(prompt);
      expect(
        called(reply, 'device_info'),
        isNotNull,
        reason: 'answered from the docs instead: ${reply.text}',
      );
    }

    debugPrint('SKILLS time_ttfa=${ms(timeTtfa)}ms');

    await tester.pumpWidget(const SizedBox.shrink());
    await deps.dispose();
  }, timeout: const Timeout(Duration(minutes: 20)));
}
