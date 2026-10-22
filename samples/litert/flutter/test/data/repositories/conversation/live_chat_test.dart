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

import 'package:flutter/foundation.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart' hide ModelSpec;
import 'package:flutter_edge_ai_agent/flutter_edge_ai_agent.dart'
    show Skill, parseSkillMd;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/conversation/live_chat.dart';
import 'package:litert_edge_demos/data/repositories/conversation/native_chat.dart';
import 'package:litert_edge_demos/data/repositories/conversation_repository.dart';
import 'package:litert_edge_demos/data/services/llm/llm_service.dart';
import 'package:litert_edge_demos/domain/models/assistant_event.dart';
import 'package:litert_edge_demos/domain/models/chat_model_config.dart';
import 'package:litert_edge_demos/domain/models/conversation_history.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../../fakes/fake_tool_model.dart';

Skill _bundled(String name) =>
    parseSkillMd(File('assets/skills/$name/SKILL.md').readAsStringSync());

void main() {
  late FakeToolModel model;
  late LiveChat live;
  final cats = Uint8List.fromList([1, 2, 3]);
  final skills = [_bundled('current-time')];

  setUp(() async {
    model = FakeToolModel();
    final llm = LlmService(loadModel: (_) async => model);
    expect(await llm.load(kDefineChatModel), isA<Ok<LlmInfo>>());
    live = LiveChat(
      ChatFactory(
        llm: llm,
        sampler: kSampler,
        executors: const [],
        maxIterations: kAgentMaxIterations,
        agentTools: kAgentTools,
      ),
      stopTimeout: const Duration(seconds: 1),
    );
  });

  tearDown(() => live.close());

  OpenChat open() => switch (live.precheck(null)) {
    Ok(:final value) => value,
    Error(:final error) => throw StateError('not open: $error'),
  };

  String failure(Result<OpenChat> result) => switch (result) {
    Ok() => 'ok',
    Error(:final error) => '$error',
  };

  /// The sent turn ends: [stopped], [responded] (the model produced
  /// something).
  void settle({
    required bool stopped,
    required Uint8List? sentImage,
    bool responded = true,
  }) => live.settleTurn(
    stopped: stopped,
    sentImage: sentImage,
    responded: responded,
  );

  /// The budget guard's "used" for a one-word turn on [chat].
  Future<int> used(InferenceChat chat) async {
    final log = <String>[];
    final original = debugPrint;
    debugPrint = (message, {wrapWidth}) => log.add(message ?? '');
    try {
      await live.checkBudget(
        chat,
        'Q',
        kVoiceChatProfile,
        null,
        () => Duration.zero,
      );
    } finally {
      debugPrint = original;
    }
    final line = log.firstWhere((l) => l.contains('budget used='));
    return int.parse(RegExp(r'used=(\d+)').firstMatch(line)![1]!);
  }

  test('nothing is open at first: a turn cannot start', () {
    expect(live.isOpen, isFalse);
    expect(live.profile, isNull);
    expect(failure(live.precheck(null)), 'The chat is not open');
  });

  test('a rebuild opens the profile; a plain profile has no agent', () async {
    expect(await live.rebuild(kCameraProfile, skills), isA<Ok<void>>());

    expect(live.isOpen, isTrue);
    expect(live.profile, kCameraProfile);
    expect(live.agent, isNull);
    expect(open().profile, kCameraProfile);
  });

  test('a turn: busy from begin to end; the next one may start then', () async {
    await live.rebuild(kCameraProfile, const []);

    final done = live.beginTurn();

    expect(live.isGenerating.value, isTrue);
    expect(failure(live.precheck(null)), 'A reply is already being generated');
    live.endTurn(done);
    expect(done.isCompleted, isTrue);
    expect(live.isGenerating.value, isFalse);
    expect(failure(live.precheck(null)), 'ok');
  });

  test('stop marks the turn and waits for its end; idle it does '
      'nothing', () async {
    await live.rebuild(kCameraProfile, const []);
    await live.stop();
    expect(live.stopRequested, isFalse);

    final done = live.beginTurn();
    final stopping = live.stop();
    expect(live.stopRequested, isTrue);
    expect(live.stopLatency, isNotNull);
    live.endTurn(done);
    await stopping;

    final again = live.beginTurn();
    expect(live.stopRequested, isFalse, reason: 'a new turn starts clean');
    expect(live.stopLatency, isNull);
    live.endTurn(again);
  });

  group('the native conversation', () {
    setUp(() => live.rebuild(kVoiceChatProfile, const []));

    test('an image is sent until a turn ends normally with it; then it is in '
        'context and not sent again', () {
      final chat = open().chat;
      final first = live.prepareSend(chat, cats);
      expect(first.image, same(cats));
      expect(first.resent, isNull);

      settle(stopped: false, sentImage: cats);

      expect(live.imageInContext, same(cats));
      expect(live.prepareSend(chat, cats).image, isNull);
    });

    test('a stopped turn keeps the image (the GPU rewinds it in place) and '
        'makes the next prefill unmeasurable', () {
      final chat = open().chat;
      settle(stopped: false, sentImage: cats);

      settle(stopped: true, sentImage: null);

      final next = live.prepareSend(chat, cats);
      expect(next.image, isNull, reason: 'the engine still holds it');
      expect(next.resent, isNull);
      expect(next.rebuildPending, isTrue);
      expect(live.imageInContext, same(cats));
      settle(stopped: false, sentImage: null);
      expect(live.prepareSend(chat, null).rebuildPending, isFalse);
    });

    test('a stop in the turn that sent the image keeps it', () {
      final chat = open().chat;
      live.prepareSend(chat, cats);

      settle(stopped: true, sentImage: cats);

      expect(live.imageInContext, same(cats));
      expect(live.prepareSend(chat, cats).image, isNull);
    });

    test('a stopped turn that began right after another stop and produced '
        'nothing loses the image it sent (the plugin may have skipped the '
        'send); one that produced something keeps it, and one that sent none '
        'keeps the image in context', () {
      final chat = open().chat;
      final dogs = Uint8List.fromList([4, 5, 6]);
      settle(stopped: false, sentImage: cats);
      settle(stopped: true, sentImage: null);

      live.prepareSend(chat, null);
      settle(stopped: true, sentImage: null, responded: false);
      expect(live.imageInContext, same(cats), reason: 'nothing was sent');

      expect(live.prepareSend(chat, dogs).image, same(dogs));
      settle(stopped: true, sentImage: dogs, responded: false);
      final again = live.prepareSend(chat, dogs);
      expect(again.image, same(dogs));
      expect(again.resent, ImageLoss.stop);

      settle(stopped: true, sentImage: dogs);
      expect(live.imageInContext, same(dogs), reason: 'it answered');
    });

    test('a failed turn loses the image; the prefill stays measurable unless '
        'a stop was requested', () async {
      final chat = open().chat;
      settle(stopped: false, sentImage: cats);

      live.failTurn(sentImage: null);

      expect(live.prepareSend(chat, cats).resent, ImageLoss.failure);
      expect(live.prepareSend(chat, null).rebuildPending, isFalse);

      final done = live.beginTurn();
      final stopping = live.stop();
      live.failTurn(sentImage: cats);
      live.endTurn(done);
      await stopping;
      expect(live.prepareSend(chat, null).rebuildPending, isTrue);
    });

    test('a cancelled turn counts as a stop: the image stays, the next '
        'prefill is unmeasurable', () {
      final chat = open().chat;

      live.cancelTurn(sentImage: cats);

      final next = live.prepareSend(chat, cats);
      expect(next.image, isNull);
      expect(next.resent, isNull);
      expect(next.rebuildPending, isTrue);
    });

    test('a stopped audio turn rebuilds nothing from the mirror: the engine '
        'keeps its recording', () {
      live.mirror.begin(user: kSpokenQuestionPlaceholder, audio: true);

      settle(stopped: true, sentImage: null);

      expect(live.mirrorRebuildDue, isFalse);
    });

    test(
      'after a stop rewound in place the guard takes LiteRT-LM\'s count as '
      'it is: it holds the stopped exchange twice, so the guard errs high',
      () async {
        final chat = open().chat;
        void count(int input, int output) => model.lastSession
          ..nativeInputTokens = input
          ..nativeOutputTokens = output;
        count(300, 10);
        live.prepareSend(chat, null);
        count(315, 22); // stopped after 12 tokens
        settle(stopped: true, sentImage: null);
        expect(await used(chat), 337);

        // The next turn's send runs the rewind: its 27 are prefilled again.
        live.prepareSend(chat, null);
        count(315 + 27 + 1 + 6, 25);
        settle(stopped: false, sentImage: null);

        expect(await used(chat), 349 + 25, reason: 'the 27 count twice');
      },
    );

    test(
      'a rebuild from the mirror (a transcript for a replayed placeholder) '
      'clears the pending recovery and loses the image (rebuild); a stopped '
      'turn after it that produced nothing keeps the image it sent',
      () async {
        final chat = open().chat;
        settle(stopped: false, sentImage: cats);
        final slot = live.mirror.begin(
          user: kSpokenQuestionPlaceholder,
          turn: 7,
          audio: true,
        );
        live.mirror
          ..reply(slot, 'Paris.')
          ..replayed(live.mirror.turns);
        expect(live.mirror.setUser(7, 'Capital of France?'), isTrue);
        settle(stopped: true, sentImage: null);
        expect(live.mirrorRebuildDue, isTrue);

        final plan = await live.planReplay(chat, (full: 2000, afterReset: 900));
        expect(plan!.clear, isTrue);
        await live.applyReplay(chat, plan);

        expect(live.mirrorRebuildDue, isFalse);
        final next = live.prepareSend(chat, cats);
        expect(next.rebuildPending, isFalse);
        expect(next.resent, ImageLoss.rebuild);
        settle(stopped: true, sentImage: cats, responded: false);
        expect(live.imageInContext, same(cats), reason: 'no stop was pending');
      },
    );

    test('the metrics before the turn are the live session\'s', () {
      final chat = open().chat;
      model.lastSession.nativeInputTokens = 64;

      expect(live.prepareSend(chat, null).before?.inputTokens, 64);
    });

    test('a budget recreate loses the image (budget) and clears the pending '
        'recovery: a stopped turn on the fresh chat that produced nothing keeps '
        'the image it sent', () async {
      final chat = open().chat;
      settle(stopped: false, sentImage: cats);
      settle(stopped: true, sentImage: null);
      final chats = model.chats.length;

      final fresh = await live.recreatePlain(kVoiceChatProfile);

      expect(fresh, isNot(same(chat)));
      expect(model.chats.length, chats + 1);
      final next = live.prepareSend(fresh, cats);
      expect(next.resent, ImageLoss.budget);
      expect(next.rebuildPending, isFalse);
      settle(stopped: true, sentImage: cats, responded: false);
      expect(live.imageInContext, same(cats));
    });

    test('a rebuild loses the image (reset)', () async {
      settle(stopped: false, sentImage: cats);

      await live.rebuild(kVoiceChatProfile, const []);

      expect(live.prepareSend(open().chat, cats).resent, ImageLoss.reset);
    });
  });

  group(
    'the native conversation on the NPU (rebuilt from text after a stop)',
    () {
      setUp(() async {
        await live.close();
        model = FakeToolModel(activeBackend: PreferredBackend.npu);
        final llm = LlmService(loadModel: (_) async => model);
        expect(await llm.load(kDefineChatModel), isA<Ok<LlmInfo>>());
        live = LiveChat(
          ChatFactory(
            llm: llm,
            sampler: kSampler,
            executors: const [],
            maxIterations: kAgentMaxIterations,
            agentTools: kAgentTools,
          ),
          stopTimeout: const Duration(seconds: 1),
        );
        await live.rebuild(kVoiceChatProfile, const []);
      });

      test('a stopped turn loses the image and makes the next prefill '
          'unmeasurable', () {
        final chat = open().chat;
        settle(stopped: false, sentImage: cats);

        settle(stopped: true, sentImage: null);

        final next = live.prepareSend(chat, cats);
        expect(next.image, same(cats));
        expect(next.resent, ImageLoss.stop);
        expect(next.rebuildPending, isTrue);
        expect(live.imageInContext, isNull);
        settle(stopped: false, sentImage: cats);
        expect(live.prepareSend(chat, null).rebuildPending, isFalse);
      });

      test('a stop in the turn that first sent the image loses it', () {
        settle(stopped: true, sentImage: cats);

        expect(live.prepareSend(open().chat, cats).resent, ImageLoss.stop);
      });

      test('a cancelled turn counts as a stop', () {
        final chat = open().chat;

        live.cancelTurn(sentImage: cats);

        final next = live.prepareSend(chat, cats);
        expect(next.resent, ImageLoss.stop);
        expect(next.rebuildPending, isTrue);
      });

      test('a stopped audio turn makes the next turn rebuild from the '
          'mirror, which clears the pending recovery', () async {
        final chat = open().chat;
        live.mirror.begin(user: kSpokenQuestionPlaceholder, audio: true);

        settle(stopped: true, sentImage: null);

        expect(live.mirrorRebuildDue, isTrue);
        final plan = await live.planReplay(chat, (full: 2000, afterReset: 900));
        expect(plan!.clear, isTrue);
        expect(plan.why, ContextResetReason.rebuildAfterStop);
        await live.applyReplay(chat, plan);
        expect(live.mirrorRebuildDue, isFalse);
        expect(live.prepareSend(chat, null).rebuildPending, isFalse);
      });
    },
  );

  group('an agent chat', () {
    setUp(() => live.rebuild(kVoiceChatProfile, skills));

    test(
      'staged input is remembered until the agent chat is recreated',
      () async {
        final agent = live.agent!;
        expect(live.staleToolTail, isFalse);

        live.markStaleToolTail();
        expect(live.staleToolTail, isTrue);
        final (_, fresh) = await live.recreateAgent(
          kVoiceChatProfile,
          agent,
          ContextResetReason.interruptedSkill,
        );

        expect(live.staleToolTail, isFalse);
        expect(live.agent, same(fresh));
        expect(fresh, isNot(same(agent)));
        expect(fresh.skills, agent.skills);
      },
    );
  });

  test('an image on a chat built without image support is refused', () async {
    final llm = LlmService(loadModel: (_) async => model);
    await llm.load(
      const ChatModelConfig(
        name: 'No images',
        modelType: ModelType.gemma4,
        llm: LlmConfig(
          maxTokens: 4096,
          backend: PreferredBackend.gpu,
          maxNumImages: 1,
        ),
        tools: true,
      ),
    );
    final noImages = LiveChat(
      ChatFactory(
        llm: llm,
        sampler: kSampler,
        executors: const [],
        maxIterations: kAgentMaxIterations,
        agentTools: kAgentTools,
      ),
      stopTimeout: const Duration(seconds: 1),
    );
    addTearDown(noImages.close);
    await noImages.rebuild(kCameraProfile, const []);

    expect(switch (noImages.precheck(cats)) {
      Ok() => null,
      Error(:final error) => error,
    }, isA<ConversationImageUnsupportedException>());
    expect(noImages.precheck(null), isA<Ok<OpenChat>>());
  });

  test('close refuses turns and rebuilds, and closes the chat', () async {
    await live.rebuild(kCameraProfile, const []);
    final session = model.lastSession;

    await live.close();

    expect(live.closed, isTrue);
    expect(live.isOpen, isFalse);
    expect(session.closed, isTrue);
    expect(failure(live.precheck(null)), 'The chat is not open');
    expect(await live.rebuild(kCameraProfile, const []), isA<Error<void>>());
  });

  test('release closes and forgets the chat but stays usable', () async {
    await live.rebuild(kVoiceChatProfile, skills);
    live.markStaleToolTail();
    final session = model.lastSession;

    await live.release();

    expect(session.closed, isTrue);
    expect((live.isOpen, live.agent, live.staleToolTail), (false, null, false));
    expect(await live.rebuild(kCameraProfile, const []), isA<Ok<void>>());
  });
}
