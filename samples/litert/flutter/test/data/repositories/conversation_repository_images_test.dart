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

import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/conversation_repository_edge_ai.dart';
import 'package:litert_edge_demos/data/services/llm/llm_service.dart';
import 'package:litert_edge_demos/domain/models/assistant_event.dart';
import 'package:litert_edge_demos/domain/models/knowledge.dart';
import 'package:litert_edge_demos/domain/use_cases/prompt_builder.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../fakes/fake_inference_model.dart';

/// Lets queued microtasks and zero-length timers run.
Future<void> settle() => Future<void>.delayed(Duration.zero);

/// One `ask` stream, collected.
final class Turn {
  Turn(Stream<AssistantEvent> stream) {
    done = stream.listen(events.add).asFuture<void>();
  }

  final List<AssistantEvent> events = [];
  late final Future<void> done;

  GenerationMetrics get metrics => switch (events.last) {
    AssistantDone(:final metrics) => metrics,
    final other => throw StateError('last event is $other'),
  };
}

/// The image re-send rule and the context budget guard on the real
/// repository over the fake engine session.
void main() {
  late FakeInferenceModel model;
  late LlmService llm;
  late EdgeAiConversationRepository repo;
  late List<String> log;
  late DebugPrintCallback originalDebugPrint;

  final cats = Uint8List.fromList([0x89, 0x50, 0x4E, 0x47, 1]);
  final dog = Uint8List.fromList([0x89, 0x50, 0x4E, 0x47, 2]);

  FakeInferenceSession session() => model.lastSession;

  /// Asks and lets the fake model answer [reply] to the end.
  Future<Turn> answer(
    String prompt, {
    Uint8List? image,
    String reply = 'Ok.',
  }) async {
    final turn = Turn(repo.ask(prompt, image: image));
    await settle();
    session()
      ..emit(reply)
      ..finish();
    await turn.done;
    return turn;
  }

  /// Asks, streams one chunk, stops, and lets native end the stream.
  Future<Turn> stopped(String prompt, {Uint8List? image}) async {
    final turn = Turn(repo.ask(prompt, image: image));
    await settle();
    session().emit('Once');
    await settle();
    await repo.stop();
    await turn.done;
    return turn;
  }

  setUp(() async {
    log = [];
    originalDebugPrint = debugPrint;
    debugPrint = (message, {wrapWidth}) => log.add(message ?? '');
    model = FakeInferenceModel();
    llm = LlmService(loadModel: (_) async => model);
    expect(await llm.load(kDefineChatModel), isA<Ok<LlmInfo>>());
    repo = EdgeAiConversationRepository(
      llm: llm,
      stopTimeout: const Duration(seconds: 2),
    );
    expect(await repo.open(kVoiceChatProfile), isA<Ok<void>>());
  });

  tearDown(() async {
    await repo.close();
    debugPrint = originalDebugPrint;
  });

  // The rule itself (identity, the loss reasons, a new image superseding
  // the one in context): image_context_tracker_test.dart and
  // live_chat_test.dart. These keep the repository's part: the query, the
  // metrics and the log. An image on a chat without image support: the turn
  // paths test (preconditions).
  group('re-send rule', () {
    test('the first ask sends the image; a normal end makes it the image in '
        'context', () async {
      final turn = await answer('What animal is this?', image: cats);

      final query = session().queries.single;
      expect(query.text, 'What animal is this?');
      expect(query.imageBytes, same(cats));
      expect(turn.metrics.imageAttached, isTrue);
      expect(turn.metrics.imageSent, isTrue);
      expect(turn.metrics.imageResent, isNull);
      expect(repo.imageInContext, same(cats));
    });

    test('the same image object again is not re-sent', () async {
      await answer('What animal is this?', image: cats);

      final follow = await answer('How many are there?', image: cats);

      expect(session().queries.last.text, 'How many are there?');
      expect(session().queries.last.hasImage, isFalse);
      expect(follow.metrics.imageAttached, isTrue);
      expect(follow.metrics.imageSent, isFalse);
      expect(repo.imageInContext, same(cats));
    });

    test(
      'a stop loses the image: the next ask re-sends it and says so',
      () async {
        await answer('What animal is this?', image: cats);
        final stop = await stopped('Describe it in detail.', image: cats);
        expect(stop.metrics.stopped, isTrue);
        expect(repo.imageInContext, isNull);

        final follow = await answer('What colour is the sofa?', image: cats);

        expect(session().queries.last.imageBytes, same(cats));
        expect(follow.metrics.imageSent, isTrue);
        expect(follow.metrics.imageResent, ImageLoss.stop);
        expect(log, contains('[Conversation] image resent (lost: stop)'));
        expect(repo.imageInContext, same(cats));
      },
    );

    test('a stop in the turn that first sent the image still re-sends it '
        '(barge-in during the first answer)', () async {
      await stopped('What animal is this?', image: cats);

      final follow = await answer('How many?', image: cats);

      expect(follow.metrics.imageSent, isTrue);
      expect(follow.metrics.imageResent, ImageLoss.stop);
    });

    test('a failed turn loses the image', () async {
      await answer('What animal is this?', image: cats);
      final failing = Turn(repo.ask('Again?', image: cats));
      await settle();
      session().failWith(Exception('Stream error: boom'));
      await failing.done;
      expect(failing.events.last, isA<AssistantFailed>());
      expect(repo.imageInContext, isNull);

      final follow = await answer('How many?', image: cats);

      expect(follow.metrics.imageResent, ImageLoss.failure);
    });

    test('cancelling the ask subscription mid-reply loses the '
        'image in context (native rebuilds from text only)', () async {
      await answer('What animal is this?', image: cats);
      final events = <AssistantEvent>[];
      final sub = repo.ask('Describe it.', image: cats).listen(events.add);
      await settle();
      session().emit('Once');
      await settle();
      expect(events.whereType<AssistantTextDelta>(), isNotEmpty);

      // A caller that stops listening (Demo 3's post-turn reset may). An
      // async* body only sees the cancel at its next yield, i.e. the next
      // native token.
      final cancelling = sub.cancel();
      await settle();
      session().emit(' upon');
      await cancelling;
      await settle();

      expect(repo.imageInContext, isNull);
      expect(repo.isGenerating.value, isFalse);
      final follow = await answer('How many?', image: cats);
      expect(follow.metrics.imageSent, isTrue);
      expect(follow.metrics.imageResent, ImageLoss.stop);
      expect(follow.metrics.prefillTokens, isNull, reason: 'native rebuilt');
    });

    test('cancelling the turn that first sent the image re-sends '
        'it next time', () async {
      final sub = repo.ask('What animal is this?', image: cats).listen((_) {});
      await settle();
      session().emit('A');
      await settle();

      final cancelling = sub.cancel();
      await settle();
      session().emit(' cat'); // the next native token delivers the cancel
      await cancelling;
      await settle();

      expect(repo.imageInContext, isNull);
      final follow = await answer('How many?', image: cats);
      expect(follow.metrics.imageSent, isTrue);
      expect(follow.metrics.imageResent, ImageLoss.stop);
    });
  });

  // The budget arithmetic: context_budget_test.dart. A prompt too long for
  // any chat and a stop during the reset: the turn paths test.
  group('budget guard', () {
    test(
      'a turn that fits runs on the same chat and reports the context',
      () async {
        session()
          ..nativeInputTokens = 100
          ..nativeOutputTokens = 20;
        final turn = Turn(repo.ask('What animal is this?', image: cats));
        await settle();
        session()
          ..emit('A cat.')
          ..nativeInputTokens = 100 + 290
          ..nativeOutputTokens = 20 + 4
          ..finish();
        await turn.done;

        final m = turn.metrics;
        expect(model.chatsCreated, 1);
        expect(m.contextReset, isFalse);
        expect(m.prefillTokens, 290);
        expect(m.promptTokens, 5, reason: 'the fake counts length / 4');
        expect(m.imageTokens, 285);
        expect(m.contextTokens, 414);
      },
    );

    test(
      "on a chat's first turn the image estimate leaves out the system "
      'instruction, which LiteRT-LM prefills with the first message',
      () async {
        final systemTokens = (kVoiceChatProfile.systemInstruction.length / 4)
            .ceil();
        final turn = Turn(repo.ask('What animal is this?', image: cats));
        await settle();
        session()
          ..emit('A cat.')
          ..nativeInputTokens = systemTokens + 5 + 275
          ..finish();
        await turn.done;

        expect(turn.metrics.promptTokens, 5 + systemTokens);
        expect(turn.metrics.imageTokens, 275);
      },
    );

    test('a turn that would overflow recreates the chat first, drops the '
        'image from context and re-sends it', () async {
      await answer('What animal is this?', image: cats);
      final full = session()..nativeInputTokens = 3500;

      final turn = await answer('How many are there?', image: cats);

      expect(model.chatsCreated, 2);
      expect(full.closed, isTrue, reason: 'the full conversation is gone');
      expect(session(), isNot(same(full)));
      expect(session().queries.single.imageBytes, same(cats));
      expect(turn.metrics.contextReset, isTrue);
      expect(turn.metrics.imageResent, ImageLoss.budget);
      expect(repo.profile, same(kVoiceChatProfile));
      expect(
        log.any((l) => l.startsWith('[Conversation] context reset (budget)')),
        isTrue,
      );
    });

    test('the reset is announced as its own event, before any '
        'text', () async {
      session().nativeInputTokens = 3600;

      final turn = await answer('How many?', image: cats, reply: 'Two.');

      expect(turn.events.first, isA<AssistantContextReset>());
      expect(turn.events.whereType<AssistantContextReset>(), hasLength(1));
      expect(turn.events[1], isA<AssistantTextDelta>());
      expect(turn.metrics.contextReset, isTrue);
    });

    test('a turn that fails after the reset has still announced '
        'it', () async {
      session().nativeInputTokens = 3600;
      final turn = Turn(repo.ask('How many?', image: cats));
      await settle();
      session().failWith(Exception('Stream error: boom'));
      await turn.done;

      expect(turn.events.first, isA<AssistantContextReset>());
      expect(turn.events.last, isA<AssistantFailed>());
    });

    test('a turn that fits announces no reset', () async {
      final turn = await answer('Hi');

      expect(turn.events.whereType<AssistantContextReset>(), isEmpty);
    });

    test(
      'the image already in context costs nothing extra; a new one does',
      () async {
        await answer('What animal is this?', image: cats);
        // used + prompt + 384 + 32 = 3840 exactly: fits without an image.
        final prompt = 'abcd' * 4; // 4 tokens
        session().nativeInputTokens = 3840 - 4 - 384 - 32;

        final kept = await answer(prompt, image: cats);
        expect(kept.metrics.contextReset, isFalse);

        session().nativeInputTokens = 3840 - 4 - 384 - 32;
        final fresh = await answer(prompt, image: dog);
        expect(fresh.metrics.contextReset, isTrue);
      },
    );

    test(
      "flutter_edge_ai's own count is used when it is higher than native",
      () async {
        await answer('Hi', reply: 'x' * 4 * 3500); // ~3500 tokens of reply
        expect(model.chat!.currentTokens, greaterThan(3400));

        final turn = await answer('And?');

        expect(turn.metrics.contextReset, isTrue);
      },
    );

    test('RAG turns: ~900-token excerpt prompts with the sticky '
        'image accumulate until the guard recreates the chat before the '
        'turn that would overflow, and the image goes again', () async {
      // Three excerpts of ~300 tokens each (the fake counts length / 4),
      // built by the real PromptBuilder: ~950 prompt tokens per turn.
      Passage passage(int n) => Passage(
        id: 'doc#$n',
        doc: 'doc.md',
        title: 'Doc',
        section: 'Section $n',
        content: 'Doc › Section $n\n\n${'lorem ipsum dolor sit ' * 55}',
        similarity: 0.6,
      );
      String rag(String q) =>
          PromptBuilder.build(q, [passage(1), passage(2), passage(3)]);
      final ragTokens = (rag('What is this?').length / 4).ceil();
      expect(ragTokens, inInclusiveRange(900, 1000));

      /// One RAG turn whose prefill and reply the fake engine records.
      Future<GenerationMetrics> ragTurn() async {
        final turn = Turn(repo.ask(rag('What is this?'), image: cats));
        await settle();
        // After the ask started: a budget reset replaces the session.
        final imageSent = session().queries.last.hasImage;
        session()
          ..emit('An answer [1].')
          ..nativeInputTokens += ragTokens + (imageSent ? 270 : 0)
          ..nativeOutputTokens += 120
          ..finish();
        await turn.done;
        return turn.metrics;
      }

      // The guard's invariant, turn by turn: a turn runs on the same chat
      // while what the chat holds plus what it needs fits under the limit,
      // and the first turn that would overflow starts a fresh chat.
      const limit = 4096 - kContextHeadroomTokens;
      const need = kTurnOverheadTokens; // + prompt + reply allowance below
      final perTurn = ragTokens + kVoiceChatProfile.maxOutputTokens + need;
      final metrics = <GenerationMetrics>[];
      var used = 0;
      for (var i = 0; i < 6; i++) {
        final m = await ragTurn();
        metrics.add(m);
        if (m.contextReset) {
          expect(used + perTurn, greaterThan(limit), reason: 'turn ${i + 1}');
          break;
        }
        expect(used + perTurn, lessThanOrEqualTo(limit), reason: 'turn $i');
        used = m.contextTokens!;
      }
      final resetAt = metrics.indexWhere((m) => m.contextReset);
      expect(resetAt, greaterThanOrEqualTo(2), reason: 'two RAG turns fit');
      expect(metrics.first.imageSent, isTrue);
      expect(metrics[1].imageSent, isFalse, reason: 'image still in context');
      final reset = metrics[resetAt];
      expect(reset.imageSent, isTrue);
      expect(reset.imageResent, ImageLoss.budget);
      expect(model.chatsCreated, 2);
      expect(session().queries.single.imageBytes, same(cats));
      expect(session().queries.single.text, startsWith('Excerpts from'));
    });

    test('open() for another demo during the budget reset waits for the turn '
        'and then builds its own chat', () async {
      session().nativeInputTokens = 3600;
      final gate = model.createGate = Completer<void>();
      final turn = Turn(repo.ask('How many?', image: cats));
      await settle();

      final opening = repo.open(kCameraProfile);
      await settle();
      gate.complete();

      expect(await opening, isA<Ok<void>>());
      await turn.done;
      expect(turn.metrics.stopped, isTrue);
      expect(repo.profile, same(kCameraProfile));
      expect(model.chatsCreated, 3, reason: 'setUp + budget + camera');
      expect(repo.imageInContext, isNull);
    });

    test(
      'close() during the budget reset closes the chat it produced',
      () async {
        session().nativeInputTokens = 3600;
        final gate = model.createGate = Completer<void>();
        final turn = Turn(repo.ask('How many?', image: cats));
        await settle();

        final closing = repo.close();
        await settle();
        gate.complete();
        await closing;
        await turn.done;

        expect(model.lastSession.closed, isTrue, reason: 'no orphaned chat');
        expect(repo.isOpen, isFalse);
      },
    );
  });
}
