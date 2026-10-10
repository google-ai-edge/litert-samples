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

// Demo 1 with images on the real app (macOS): Gemma 4 E2B on the GPU
// (vision encoder on the CPU), Whisper base and Inflect-nano-v2 on the CPU,
// soloud playback.
//
//   flutter test integration_test/image_chat_test.dart -d macos \
//     --dart-define=GEMMA_MODEL_PATH=<path/to/gemma-4-E2B-it.litertlm>
//
// The picker is a fixture returning test_assets/cats.jpg (COCO 39769: two
// cats on a pink sofa, 640×480 JPEG); normalization to PNG is the real one.
// The mic is a fixture playing test_assets/q_cats.wav ("How many cats do you
// see?").
//
// 1. setup → home → Demo 1; the gallery button attaches the photo.
// 2. Typed "What animal is this?" → /cat/, the image is sent.
// 3. Typed "How many are there?" → /two|2/, the image is NOT sent again (it
//    is still in the live chat's context).
// 4. Typed "Describe this photo in great detail." and, while it is spoken
//    and Gemma still generates, the mic is pressed (barge-in): the reply is
//    stopped, LiteRT-LM rebuilds the conversation from text only, and the
//    voice follow-up "How many cats do you see?" must re-send the image:
//    /two|2/, `[Conversation] image resent (lost: stop)` in the log.
// Prints `IMG …` per turn, `WARMUP …`, `BARGE …` and a screenshot path.

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
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/domain/models/assistant_event.dart';
import 'package:litert_edge_demos/domain/models/chat_entry.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/models/voice.dart';
import 'package:litert_edge_demos/ui/core/debug_overlay.dart';
import 'package:litert_edge_demos/ui/features/home/views/home_screen.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/view_models/voice_chat_view_model.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/views/chat_keys.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/views/voice_chat_screen.dart';
import 'package:provider/provider.dart';

import 'support/app_window.dart';
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

VoiceChatViewModel chatViewModel(WidgetTester tester) =>
    Provider.of(tester.element(find.byType(VoiceChatScreen)), listen: false);

String describeChat(VoiceChatViewModel vm) =>
    'phase=${vm.phase.name} error=${vm.error} canSend=${vm.canSend} '
    'send.running=${vm.send.running} ready=${vm.isReady} '
    'stopButton=${find.byKey(ChatKeys.stop).evaluate().length} '
    'input="${find.byKey(ChatKeys.input).evaluate().isEmpty ? '-' : (find.byKey(ChatKeys.input).evaluate().single.widget as TextField).controller?.text}" entries='
    '${vm.entries.map((e) => '${e.role.name}:"${e.text}"${e.image == null ? '' : '[img]'}${e.interrupted ? '(i)' : ''}').join(' | ')}';

String ms(Duration? d) => d == null ? '–' : '${d.inMilliseconds}';

final _two = RegExp(r'\btwo\b|\b2\b', caseSensitive: false);

void main() {
  initIntegrationTest();

  testWidgets('image chat: answers about the photo, keeps it for follow-ups, '
      're-sends it after a barge-in', (tester) async {
    if (kGemmaModelPath.isEmpty) {
      fail('Pass --dart-define=GEMMA_MODEL_PATH=<path to the .litertlm>');
    }
    // The log is part of the acceptance ("image resent").
    final log = <String>[];
    final originalDebugPrint = debugPrint;
    debugPrint = (message, {wrapWidth}) {
      log.add(message ?? '');
      originalDebugPrint(message, wrapWidth: wrapWidth);
    };
    addTearDown(() => debugPrint = originalDebugPrint);

    final cats = (await rootBundle.load('test_assets/cats.jpg')).buffer
        .asUint8List();
    expect(cats.length, 173131, reason: 'the fixture from the spec');
    final question = pcm16FromWav(
      (await rootBundle.load('test_assets/q_cats.wav')).buffer.asUint8List(),
    );
    final mic = FixtureMicService(question);
    final picker = FixtureImageInputService(cats);
    final deps = await AppDependencies.create(mic: mic, imageInput: picker);
    await tester.pumpWidget(
      RepaintBoundary(
        key: _screenKey,
        child: App(dependencies: deps),
      ),
    );

    // 1. Setup, home, Demo 1.
    await pumpUntil(
      tester,
      () => find.byType(HomeScreen).evaluate().isNotEmpty,
      timeout: const Duration(minutes: 12),
      reason: 'model setup',
      onPoll: () {
        for (final MapEntry(:key, :value) in deps.models.states.value.entries) {
          if (value case ModelFailed(:final message) when key.spec.required) {
            fail('${key.spec.displayName} failed: $message');
          }
        }
      },
    );
    final llm = deps.models.states.value[ModelId.chat];
    final llmInfo = (llm! as ModelReady).info;
    expect(llmInfo.backend, 'gpu');
    final warmUpLog = log.lastWhere(
      (l) => l.startsWith('[LlmService] warm-up'),
      orElse: () => '',
    );
    expect(warmUpLog, contains('B PNG'), reason: 'warm-up sent an image');
    debugPrint(
      'WARMUP llm load=${llmInfo.loadTime.inMilliseconds}ms '
      'warm=${llmInfo.warmUpTime.inMilliseconds}ms ($warmUpLog)',
    );
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
      timeout: const Duration(seconds: 5),
      reason: 'the chat screen to open',
    );
    final vm = chatViewModel(tester);
    await pumpUntil(
      tester,
      () => !vm.open.running && vm.isReady,
      timeout: const Duration(seconds: 30),
      reason: 'the chat to open',
      describe: () => describeChat(vm),
    );
    await tester.pump(const Duration(milliseconds: 400));

    // Attach through the real gallery button.
    expect(find.byKey(ChatKeys.attachCamera), findsNothing, reason: 'macOS');
    await tester.tap(find.byKey(ChatKeys.attachGallery));
    await pumpUntil(
      tester,
      () => vm.attachment != null,
      timeout: const Duration(seconds: 10),
      reason: 'the picture to attach',
      describe: () => describeChat(vm),
    );
    await tester.pump();
    final attached = vm.attachment!;
    expect(find.byKey(ChatKeys.attachmentThumbnail), findsOneWidget);
    expect((attached.width, attached.height), (640, 480));
    debugPrint(
      'ATTACH png=${attached.width}x${attached.height} '
      '${attached.png.length}B from=${attached.sourceBytes}B '
      'normalize=${attached.normalizeTime.inMilliseconds}ms',
    );

    GenerationMetrics lastGen() => deps.diagnostics.latest.lastGeneration!;
    void printTurn(String name, GenerationMetrics g, String reply) =>
        debugPrint(
          'IMG turn=$name ttft=${ms(g.timeToFirstToken)}ms '
          'tokens=${g.imageTokens ?? '–'} prefill=${g.prefillTokens ?? '–'} '
          'prompt=${g.promptTokens ?? '–'} ctx=${g.contextTokens ?? '–'} '
          'sent=${g.imageSent} resent=${g.imageResent?.name ?? '–'} '
          'reset=${g.contextReset} '
          'tokps=${g.tokensPerSecond?.toStringAsFixed(1)} '
          'reply="${reply.replaceAll('\n', ' ')}"',
        );

    /// Types [prompt], taps Send, waits for the turn (and its audio) to end.
    Future<ChatEntry> ask(String prompt) async {
      final before = vm.entries.length;
      await tester.pump(const Duration(milliseconds: 50));
      // Focus first: the touch gesture of a barge-in unfocuses the field.
      await tester.tap(find.byKey(ChatKeys.input));
      await tester.pump();
      await tester.enterText(find.byKey(ChatKeys.input), prompt);
      await tester.pump();
      expect(
        tester.widget<TextField>(find.byKey(ChatKeys.input)).controller?.text,
        prompt,
        reason: 'the prompt did not reach the input',
      );
      await tester.tap(find.byKey(ChatKeys.send));
      await pumpUntil(
        tester,
        () => vm.entries.length > before,
        timeout: const Duration(seconds: 5),
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
      expect(vm.error, isNull, reason: describeChat(vm));
      expect(vm.entries.length, before + 2, reason: describeChat(vm));
      expect(vm.entries[before].image, same(attached.png));
      return vm.entries.last;
    }

    // 2. The first question sends the photo.
    final animal = await ask('What animal is this?');
    final first = lastGen();
    printTurn('animal', first, animal.text);
    expect(animal.text.toLowerCase(), contains('cat'));
    expect(first.imageSent, isTrue);
    expect(first.imageResent, isNull);
    expect(deps.conversation.imageInContext, same(attached.png));
    expect(find.byKey(ChatKeys.entryImage), findsWidgets);

    // 3. The follow-up does not send it again.
    final count = await ask('How many are there?');
    final second = lastGen();
    printTurn('count', second, count.text);
    expect(count.text, matches(_two));
    expect(second.imageAttached, isTrue);
    expect(second.imageSent, isFalse, reason: 'still in the live context');
    await tester.pump(const Duration(milliseconds: 400));
    debugPrint('SCREENSHOT ${await saveScreenshot('image_chat')}');

    // 4. Barge in on a long description while Gemma still generates.
    await tester.enterText(
      find.byKey(ChatKeys.input),
      'Describe this photo in great detail.',
    );
    await tester.pump();
    await tester.tap(find.byKey(ChatKeys.send));
    await pumpUntil(
      tester,
      () => vm.phase == TurnPhase.speaking,
      timeout: const Duration(seconds: 60),
      reason: 'the description to start speaking',
      describe: () => describeChat(vm),
    );
    await pumpFor(tester, const Duration(milliseconds: 300));
    expect(
      deps.conversation.isGenerating.value,
      isTrue,
      reason: 'the barge-in must stop a running generation',
    );
    final entriesBefore = vm.entries.length;
    final gesture = await tester.startGesture(
      tester.getCenter(find.byKey(ChatKeys.mic)),
    );
    // The user-visible cue ("Listening… release to send"); that the capture
    // runs behind it is checked, not waited for.
    await pumpUntil(
      tester,
      () => vm.isListening,
      timeout: const Duration(seconds: 3),
      reason: 'the barge-in to open the mic',
      describe: () => describeChat(vm),
    );
    expect(
      mic.starts,
      greaterThanOrEqualTo(1),
      reason: 'Listening is shown only once the capture runs',
    );
    expect(vm.entries[entriesBefore].interrupted, isTrue);
    await pumpFor(tester, const Duration(milliseconds: 2000));
    await gesture.up();
    await pumpUntil(
      tester,
      () => deps.diagnostics.latest.lastBargeIn?.interruptDone != null,
      timeout: const Duration(seconds: 12),
      reason: 'the interrupted turn to drain',
    );
    final barge = deps.diagnostics.latest.lastBargeIn!;
    await pumpUntil(
      tester,
      () => vm.phase == TurnPhase.idle || vm.phase == TurnPhase.error,
      timeout: const Duration(seconds: 90),
      reason: 'the voice follow-up',
      describe: () => describeChat(vm),
    );
    expect(vm.phase, TurnPhase.idle, reason: describeChat(vm));
    final followUser = vm.entries.lastWhere((e) => e.role == ChatRole.user);
    final followReply = vm.entries.last;
    final third = lastGen();
    printTurn('after-barge-in', third, followReply.text);
    expect(followUser.text.toLowerCase(), contains('cat'));
    expect(followUser.image, same(attached.png));
    expect(followReply.role, ChatRole.assistant);
    expect(followReply.text, matches(_two), reason: describeChat(vm));
    expect(third.imageSent, isTrue);
    expect(third.imageResent, ImageLoss.stop);
    expect(log, contains('[Conversation] image resent (lost: stop)'));
    expect(deps.conversation.imageInContext, same(attached.png));
    expect(
      debugOverlayLines(deps.diagnostics.latest),
      contains(startsWith('img resent (lost: stop)')),
    );
    debugPrint(
      'BARGE silenced=${ms(barge.silenced)}ms drain=${ms(barge.interruptDone)}ms '
      'transcript="${followUser.text}" '
      'log="${log.firstWhere((l) => l.contains('image resent'))}"',
    );
    await tester.pump(const Duration(milliseconds: 400));
    debugPrint(
      'SCREENSHOT ${await saveScreenshot('image_chat_after_barge_in')}',
    );

    // 5. Two more typed turns on the rebuilt conversation: the re-sent image
    //    stays in context, and the budget guard's "used" must include the
    //    history LiteRT-LM replayed after the stop.
    final colour = await ask('What colour is the blanket?');
    final fourth = lastGen();
    printTurn('colour', fourth, colour.text);
    expect(fourth.imageSent, isFalse);
    expect(colour.text.toLowerCase(), contains('pink'));
    final asleep = await ask('Are they asleep?');
    final fifth = lastGen();
    printTurn('asleep', fifth, asleep.text);
    expect(fifth.imageSent, isFalse);
    final budgets = log
        // A plain chat logs "budget used=", an agent chat (Demo 1 with
        // skills) "agent budget used=".
        .where(
          (l) => RegExp(r'^\[Conversation\] (agent )?budget used=').hasMatch(l),
        )
        .map((l) => int.parse(RegExp(r'used=(\d+)').firstMatch(l)![1]!))
        .toList();
    debugPrint(
      'TOKENS ctx: animal=${first.contextTokens} count=${second.contextTokens} '
      'after_barge_in=${third.contextTokens} colour=${fourth.contextTokens} '
      'asleep=${fifth.contextTokens} | budget used per turn=$budgets',
    );
    // The turn after the rebuild starts from what the rebuilt conversation
    // reported, not from zero.
    expect(budgets[4], third.contextTokens);
    expect(budgets[4], greaterThan(second.contextTokens!));

    // Memory after the turns (maxTokens sizes the KV cache).
    final mem = deps.diagnostics.latest;
    debugPrint(
      'MEM maxTokens=${kLlmConfig.maxTokens} '
      'rss=${(mem.rssBytes ?? 0) >> 20}MB peak=${(mem.peakRssBytes ?? 0) >> 20}MB',
    );
    await tester.pumpWidget(const SizedBox.shrink());
    await deps.dispose();
  }, timeout: const Timeout(Duration(minutes: 20)));
}
