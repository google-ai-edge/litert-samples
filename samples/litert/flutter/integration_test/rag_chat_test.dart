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

// The knowledge base end to end in Demo 1 (macOS): Gemma 4 E2B on the GPU,
// EmbeddingGemma on the CPU, the app's own sqlite-vec index over assets/kb.
//
//   flutter test integration_test/rag_chat_test.dart -d macos \
//     --dart-define=GEMMA_MODEL_PATH=<path/to/gemma-4-E2B-it.litertlm>
//
// EmbeddingGemma is the one built into the app, loaded by setup.
//
// 1. setup → the knowledge base installs the prebuilt index (first launch,
//    ~0.1 s; `[Knowledge] prebuilt index installed`), indexes on the device
//    when the prebuilt one does not match (~50 s; the reason is logged), or
//    reuses the index from an earlier launch (`[Knowledge] index reused`).
//    Run twice to see both; the run prints
//    `KB launch=prebuilt|indexed|reused …`.
// 2. Demo 1, typed on-topic question from test_assets/kb_golden.json: the
//    retrieval is used, the reply carries citation chips, the top excerpt is
//    from the expected document, and the reply states the excerpt's fact
//    (640, the detector's input size).
// 3. An off-topic golden question: below the gate, no new chips.
// 4. Regression: on-topic golden questions, each on a fresh
//    conversation, answer from the excerpts with no tool step. Step 2's
//    question counts too. Prints `RAG no_tool=k/5`.
// Prints `RAG q=… retrieve=…ms top=… ttft=… cited=… steps=…` per turn.

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
import 'package:litert_edge_demos/domain/models/knowledge.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/ui/features/home/views/home_screen.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/view_models/voice_chat_view_model.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/views/chat_keys.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/views/citation_chips.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/views/voice_chat_screen.dart';
import 'package:provider/provider.dart';

import 'support/app_window.dart';
import 'support/pump.dart';

/// From test_assets/kb_golden.json: on-topic (yolo26n-object-detector.md,
/// "640 × 640 images") and off-topic.
const _onTopic = 'What is the input size of the YOLO 26 nano detector?';
const _onTopicDoc = 'yolo26n-object-detector.md';
const _offTopic = 'How long should I boil an egg for a soft yolk?';

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
    'phase=${vm.phase.name} error=${vm.error} entries='
    '${vm.entries.map((e) => '${e.role.name}:"${e.text}"').join(' | ')}';

String ms(Duration? d) => d == null ? '–' : '${d.inMilliseconds}';

void main() {
  initIntegrationTest();

  testWidgets('RAG in Demo 1: an on-topic question is answered from cited '
      'excerpts, an off-topic one without', (tester) async {
    if (kGemmaModelPath.isEmpty) fail('Pass GEMMA_MODEL_PATH');
    final log = <String>[];
    final originalDebugPrint = debugPrint;
    debugPrint = (message, {wrapWidth}) {
      log.add(message ?? '');
      originalDebugPrint(message, wrapWidth: wrapWidth);
    };
    addTearDown(() => debugPrint = originalDebugPrint);

    final deps = await AppDependencies.create();
    await tester.pumpWidget(
      RepaintBoundary(
        key: _screenKey,
        child: App(dependencies: deps),
      ),
    );

    // 1. Setup, then the knowledge base (indexes without blocking setup).
    await pumpUntil(
      tester,
      () => find.byType(HomeScreen).evaluate().isNotEmpty,
      timeout: const Duration(minutes: 12),
      reason: 'model setup',
      onPoll: () {
        for (final MapEntry(:key, :value) in deps.models.states.value.entries) {
          if (value case ModelFailed(:final message)
              when key.spec.required || key == ModelId.embeddingGemma) {
            fail('${key.spec.displayName} failed: $message');
          }
        }
      },
    );
    await pumpUntil(
      tester,
      () => deps.knowledge.status.value is KnowledgeReady,
      timeout: const Duration(minutes: 4),
      reason: 'the knowledge base to be ready',
      onPoll: () {
        if (deps.knowledge.status.value case KnowledgeFailed(:final message)) {
          fail('KB failed: $message');
        }
        if (deps.knowledge.status.value case KnowledgeUnavailable(
          :final reason,
        )) {
          fail('KB unavailable: $reason');
        }
      },
    );
    final ready = deps.knowledge.status.value as KnowledgeReady;
    final (launch, prefix) = switch ((ready.reused, ready.origin)) {
      (true, _) => ('reused', '[Knowledge] index reused'),
      (false, KnowledgeOrigin.prebuilt) => (
        'prebuilt',
        '[Knowledge] prebuilt index installed',
      ),
      (false, KnowledgeOrigin.device) => ('indexed', '[Knowledge] indexed'),
    };
    final launchLine = log.where((l) => l.startsWith(prefix));
    expect(launchLine, isNotEmpty, reason: 'the log says which');
    debugPrint(
      'KB launch=$launch origin=${ready.origin.name} chunks=${ready.chunks} '
      'elapsed=${ready.elapsed.inMilliseconds}ms '
      'prebuilt_skipped=${ready.prebuiltSkipped ?? '-'} '
      'log="${launchLine.first}"',
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

    /// Types [prompt], taps Send, waits for the turn (and its audio).
    Future<ChatEntry> ask(String prompt) async {
      final before = vm.entries.length;
      // After New conversation: wait until Send is enabled, and focus the
      // field first so the text input client is attached.
      await pumpUntil(
        tester,
        () => vm.canSend,
        timeout: const Duration(seconds: 10),
        reason: 'Send to be enabled',
        describe: () => describeChat(vm),
      );
      await tester.tap(find.byKey(ChatKeys.input));
      await tester.pump(const Duration(milliseconds: 50));
      await tester.enterText(find.byKey(ChatKeys.input), prompt);
      await tester.pump();
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
      expect(vm.error, isNull, reason: describeChat(vm));
      await tester.pump(const Duration(milliseconds: 100));
      return vm.entries.last;
    }

    void printTurn(String q, ChatEntry reply) {
      final r = reply.knowledge?.retrieval;
      final g = deps.diagnostics.latest.lastGeneration;
      debugPrint(
        'RAG q="$q" outcome=${r?.outcome.name} retrieve=${ms(r?.latency)}ms '
        'top=${r?.topSimilarity?.toStringAsFixed(3)} '
        'excerpts=${r?.passages.length} '
        'docs=${r?.passages.map((p) => p.doc).toList()} '
        'cited=${reply.knowledge?.cited.toList()} '
        'ttft=${ms(g?.timeToFirstToken)}ms prompt=${g?.promptTokens} '
        'ctx=${g?.contextTokens} steps=${reply.steps} '
        'reply="${reply.text.replaceAll('\n', ' ')}"',
      );
    }

    // 2. On-topic: excerpts in the prompt, chips under the reply.
    final onReply = await ask(_onTopic);
    printTurn(_onTopic, onReply);
    expect(onReply.role, ChatRole.assistant);
    final onKnowledge = onReply.knowledge;
    expect(onKnowledge, isNotNull);
    final retrieval = onKnowledge!.retrieval;
    expect(retrieval.outcome, RetrievalOutcome.used);
    expect(retrieval.passages, isNotEmpty);
    expect(retrieval.passages.first.doc, _onTopicDoc);
    expect(find.byKey(KnowledgeChipKeys.citation(1)), findsOneWidget);
    // Grounded: the reply states the excerpt's fact.
    expect(retrieval.passages.first.content, contains('640'));
    expect(onReply.text, contains('640'), reason: onReply.text);
    // A knowledge question is answered without a tool round.
    final noToolMisses = <String>[
      if (onReply.steps.isNotEmpty) '$_onTopic: ${onReply.steps}',
    ];
    final chipsAfterOnTopic = find
        .byKey(KnowledgeChipKeys.citation(1))
        .evaluate()
        .length;
    await tester.pump(const Duration(milliseconds: 400));
    debugPrint('SCREENSHOT ${await saveScreenshot('rag_chat')}');

    // 3. Off-topic: below the gate, the plain question, no chips.
    final offReply = await ask(_offTopic);
    printTurn(_offTopic, offReply);
    expect(offReply.role, ChatRole.assistant);
    expect(offReply.knowledge?.retrieval.outcome, RetrievalOutcome.belowGate);
    expect(offReply.knowledge?.retrieval.passages, isEmpty);
    expect(offReply.text, isNot(matches(RegExp(r'\[\d'))));
    expect(
      find.byKey(KnowledgeChipKeys.citation(1)).evaluate().length,
      lessThanOrEqualTo(chipsAfterOnTopic),
      reason: 'no chips under the off-topic reply',
    );
    expect(find.byKey(KnowledgeChipKeys.status), findsNothing);

    // 4. Regression: more on-topic questions, each on a fresh chat,
    //    answered from the excerpts with no tool step.
    for (final q in const [
      'What does the E in Gemma 4 E2B stand for?',
      'How many dimensions does EmbeddingGemma produce, and can I make the '
          'vectors smaller?',
      'Which entitlements does an iPhone app need to load a big model?',
      "With the compiled model API, how can I tell if my model really ran on "
          "the GPU and didn't just silently fall back to the CPU?",
    ]) {
      await vm.newConversation.execute();
      await tester.pump(const Duration(milliseconds: 200));
      final reply = await ask(q);
      printTurn(q, reply);
      expect(reply.role, ChatRole.assistant, reason: describeChat(vm));
      expect(
        reply.knowledge?.retrieval.outcome,
        RetrievalOutcome.used,
        reason: q,
      );
      if (reply.steps.isNotEmpty) noToolMisses.add('$q: ${reply.steps}');
    }
    debugPrint('RAG no_tool=${5 - noToolMisses.length}/5 misses=$noToolMisses');
    expect(noToolMisses, isEmpty, reason: 'knowledge questions took a tool');

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
