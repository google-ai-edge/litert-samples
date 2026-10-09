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

// End-to-end on the real app (flutter_edge_ai, Gemma 4 E2B on the GPU):
// setup → home → Demo 1; one streamed reply; a stopped reply and the turn
// after it; then back to home in the middle of a reply and into Demo 1 again,
// which must wait for the drain and come up with a fresh chat.
//
//   flutter test integration_test/chat_smoke_test.dart -d macos \
//     --dart-define=GEMMA_MODEL_PATH=<path/to/gemma-4-E2B-it.litertlm>
//
// Prints `CHAT backend=<b> ttft=<ms>ms tokps=<x>`,
// `STOP latency=<ms>ms ui=<ms>ms`, `AFTER_STOP …` and `SWITCH …`.

import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/app.dart';
import 'package:litert_edge_demos/config/demos.dart';
import 'package:litert_edge_demos/config/dependencies.dart';
import 'package:litert_edge_demos/config/env.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/domain/models/chat_entry.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/ui/features/home/views/home_screen.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/view_models/voice_chat_view_model.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/views/chat_keys.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/views/voice_chat_screen.dart';
import 'package:provider/provider.dart';

import 'support/app_window.dart';
import 'support/pump.dart';

VoiceChatViewModel chatViewModel(WidgetTester tester) =>
    Provider.of(tester.element(find.byType(VoiceChatScreen)), listen: false);

String describeChat(WidgetTester tester) {
  final vm = chatViewModel(tester);
  final send = find.byKey(ChatKeys.send);
  final input = find.byKey(ChatKeys.input);
  return 'canSend=${vm.canSend} isReady=${vm.isReady} '
      'send.running=${vm.send.running} stop.running=${vm.stop.running} '
      'new.running=${vm.newConversation.running} entries=${vm.entries.length} '
      'sendButton=${send.evaluate().length} '
      'stopButton=${find.byKey(ChatKeys.stop).evaluate().length} '
      'input="${input.evaluate().isEmpty ? '-' : tester.widget<TextField>(input).controller?.text}"';
}

final _screenKey = GlobalKey();

/// Saves the app as a PNG in the sandbox temp directory; returns its path.
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

/// Taps the Demo 1 tile on home and waits until its chat is open.
Future<void> openDemo1(WidgetTester tester) async {
  final tile = find.byKey(HomeKeys.tile(Demo.voiceChat));
  await pumpUntil(
    tester,
    () => tile.evaluate().isNotEmpty && tester.widget<ListTile>(tile).enabled,
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
  // This view model's own open must have finished (it waits for a draining
  // turn), not just "a chat exists", which the previous demo's chat satisfies.
  await pumpUntil(
    tester,
    () => !vm.open.running && vm.isReady,
    timeout: const Duration(seconds: 30),
    reason: 'the chat to open',
    describe: () => 'open.running=${vm.open.running} error: ${vm.error}',
  );
  await tester.pump(const Duration(milliseconds: 400)); // route transition
}

/// Focuses the input, types [prompt], checks it arrived, taps Send.
Future<void> typeAndSend(WidgetTester tester, String prompt) async {
  final input = find.byKey(ChatKeys.input);
  await tester.tap(input);
  await tester.pump();
  await tester.enterText(input, prompt);
  await tester.pump();
  expect(
    tester.widget<TextField>(input).controller?.text,
    prompt,
    reason: 'the prompt did not reach the input. ${describeChat(tester)}',
  );
  await tester.tap(find.byKey(ChatKeys.send));
}

/// Types [prompt], taps Send, waits for the turn to end; returns the last
/// committed entry.
Future<ChatEntry> ask(WidgetTester tester, String prompt) async {
  final vm = chatViewModel(tester);
  final before = vm.entries.length;
  // Let the UI catch up with the view model (the Send button replaces Stop
  // one frame after a turn ends).
  await tester.pump(const Duration(milliseconds: 50));
  await typeAndSend(tester, prompt);
  await pumpUntil(
    tester,
    () => vm.entries.length > before,
    timeout: const Duration(seconds: 5),
    reason: 'the turn to start',
    describe: () => describeChat(tester),
  );
  await pumpUntil(
    tester,
    () => !vm.send.running,
    timeout: const Duration(seconds: 120),
    reason: 'the reply to "$prompt"',
    describe: () => describeChat(tester),
  );
  expect(vm.error, isNull, reason: 'turn failed: ${vm.error}');
  expect(vm.entries.length, before + 2, reason: 'user + assistant entries');
  return vm.entries.last;
}

void main() {
  initIntegrationTest();

  testWidgets('chat streams a GPU reply, stops mid-reply, and continues', (
    tester,
  ) async {
    if (kGemmaModelPath.isEmpty) {
      fail('Pass --dart-define=GEMMA_MODEL_PATH=<path to the .litertlm>');
    }
    final deps = await AppDependencies.create();
    await tester.pumpWidget(
      RepaintBoundary(
        key: _screenKey,
        child: App(dependencies: deps),
      ),
    );

    // Setup: install, load on the GPU, warm up; then home.
    await pumpUntil(
      tester,
      () => find.byType(HomeScreen).evaluate().isNotEmpty,
      timeout: const Duration(minutes: 6),
      reason: 'model setup',
      onPoll: () {
        final state = deps.models.states.value[ModelId.chat];
        if (state case ModelFailed(:final message)) {
          fail('Model setup failed: $message');
        }
      },
    );
    await tester.pump(const Duration(milliseconds: 400)); // route transition
    // The detector is built into the app: setup loads it from the asset
    // bundle like every other built-in model, so the Live camera tile opens.
    expect(
      deps.models.states.value[ModelId.yolo26n],
      isA<ModelReady>(),
      reason: 'the built-in detector',
    );
    expect(
      find.descendant(
        of: find.byKey(HomeKeys.tile(Demo.liveCamera)),
        matching: find.textContaining(RegExp('^Ready')),
      ),
      findsOneWidget,
    );
    await openDemo1(tester);
    final vm = chatViewModel(tester);
    await pumpUntil(
      tester,
      () => vm.isReady,
      timeout: const Duration(seconds: 30),
      reason: 'the chat to open (error: ${vm.error})',
    );

    final llm = deps.diagnostics.latest.models[ModelId.chat];
    expect(llm, isNotNull);
    expect(llm!.backend, 'gpu', reason: 'blocking error expected otherwise');

    // 1. One factual turn.
    final reply = await ask(
      tester,
      'What is the capital of France? Answer with only the city name.',
    );
    expect(reply.role, ChatRole.assistant);
    expect(reply.text.trim(), isNotEmpty);
    expect(reply.text.toLowerCase(), contains('paris'));
    final first = deps.diagnostics.latest.lastGeneration;
    expect(first, isNotNull);
    expect(first!.timeToFirstToken, isNotNull);
    debugPrint(
      'CHAT backend=${llm.backend} '
      'ttft=${first.timeToFirstToken!.inMilliseconds}ms '
      'tokps=${first.tokensPerSecond?.toStringAsFixed(1)} '
      '(${first.tokensPerSecondSource.name}) chunks=${first.chunks} '
      'load=${llm.loadTime.inMilliseconds}ms '
      'warmup=${llm.warmUpTime.inMilliseconds}ms '
      'reply="${reply.text.trim()}"',
    );

    // Let the coalesced overlay catch up, then keep a picture of it.
    await tester.pump(const Duration(milliseconds: 400));
    debugPrint('SCREENSHOT ${await saveScreenshot('chat_smoke')}');

    // 2. Stop a long reply once text is streaming.
    await typeAndSend(
      tester,
      'Write a 400-word story about a lighthouse keeper and a storm.',
    );
    await pumpUntil(
      tester,
      () => vm.partialReply.value.length > 40,
      timeout: const Duration(seconds: 60),
      reason: 'the long reply to start streaming',
    );
    final stopWatch = Stopwatch()..start();
    await tester.tap(find.byKey(ChatKeys.stop));
    await pumpUntil(
      tester,
      () => !vm.send.running,
      timeout: const Duration(seconds: 10),
      reason: 'the stopped turn to end',
    );
    final uiStopMs = stopWatch.elapsedMilliseconds;
    final stopped = deps.diagnostics.latest.lastGeneration;
    expect(stopped?.stopped, isTrue);
    expect(vm.entries.last.interrupted, isTrue);
    final stopMs = stopped!.stopLatency!.inMilliseconds;
    debugPrint(
      'STOP latency=${stopMs}ms ui=${uiStopMs}ms chunks=${stopped.chunks}',
    );
    // The acceptance bound is on the app's own figure: the stop request
    // until the turn's last event (GenerationMetrics.stopLatency). The test's
    // stopwatch (`ui`) adds the tap, the frame pumps and the poll of a
    // loaded test machine, so it is logged, not bounded.
    expect(stopMs, lessThanOrEqualTo(500), reason: 'Stop within 500 ms');

    // 3. The next turn works after the stop.
    final after = await ask(
      tester,
      'What is 2 plus 3? Answer with only the number.',
    );
    expect(after.text, contains('5'));
    final third = deps.diagnostics.latest.lastGeneration!;
    debugPrint(
      'AFTER_STOP ttft=${third.timeToFirstToken?.inMilliseconds}ms '
      'tokps=${third.tokensPerSecond?.toStringAsFixed(1)} '
      'reply="${after.text.trim()}"',
    );

    // 4. Back to home in the middle of a long reply, straight into Demo 1
    //    again: the leaving view model stops without waiting, and the new
    //    one's open() must wait for that turn to drain, then give a fresh chat.
    await typeAndSend(
      tester,
      'Write a 400-word story about a lighthouse keeper and a storm.',
    );
    await pumpUntil(
      tester,
      () => vm.partialReply.value.length > 40,
      timeout: const Duration(seconds: 60),
      reason: 'the long reply to start streaming',
    );
    await tester.pageBack();
    await pumpUntil(
      tester,
      () => find.byType(VoiceChatScreen).evaluate().isEmpty,
      timeout: const Duration(seconds: 5),
      reason: 'the chat route to pop',
    );
    final switchWatch = Stopwatch()..start();
    await openDemo1(tester);
    final again = chatViewModel(tester);
    final openMs = switchWatch.elapsedMilliseconds;
    expect(identical(again, vm), isFalse, reason: 'a new view model');
    expect(again.entries, isEmpty, reason: 'a fresh chat on screen');
    expect(deps.conversation.profile, same(kVoiceChatProfile));
    final rome = await ask(
      tester,
      'What is the capital of Italy? Answer with only the city name.',
    );
    expect(rome.text.toLowerCase(), contains('rome'));
    expect(again.entries, hasLength(2), reason: 'no history carried over');
    debugPrint(
      'SWITCH back_mid_reply=true open=${openMs}ms '
      'entries_after_reentry=0 reply="${rome.text.trim()}"',
    );

    await tester.pumpWidget(const SizedBox.shrink());
    await deps.dispose();
  }, timeout: const Timeout(Duration(minutes: 10)));
}
