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

import 'dart:typed_data';

import 'package:flutter_edge_ai_agent/flutter_edge_ai_agent.dart'
    show ErrorResult, Skill, SkillResult, SkillType, TextResult;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/domain/models/assistant_event.dart';
import 'package:litert_edge_demos/domain/models/chat_capabilities.dart';
import 'package:litert_edge_demos/domain/models/chat_side_event.dart';
import 'package:litert_edge_demos/domain/models/knowledge.dart';
import 'package:litert_edge_demos/domain/models/skill_step.dart';
import 'package:litert_edge_demos/domain/use_cases/chat_turn_responder.dart';
import 'package:litert_edge_demos/domain/use_cases/prompt_builder.dart';
import 'package:litert_edge_demos/domain/use_cases/turn_responder_factory.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../fakes/fake_conversation_repository.dart';
import '../../fakes/fake_knowledge.dart';

const _time = Skill(
  name: 'current-time',
  description: 'The time.',
  instructions: 'Call the `run_intent` tool with intent `current_time`.',
  type: SkillType.intent,
);

/// The responder forwards skill steps as side events, and on an
/// agent chat frames excerpts so they never replace a skill.
void main() {
  late FakeConversationRepository conversation;
  late FakeRetriever retriever;
  late List<ChatSideEvent> side;

  setUp(() {
    conversation = FakeConversationRepository();
    retriever = FakeRetriever(
      Result.ok(
        Retrieval(
          outcome: RetrievalOutcome.used,
          passages: [passage(1, similarity: 0.5)],
          gate: 0.4,
        ),
      ),
    );
    side = [];
  });

  tearDown(() => conversation.close());

  Stream<String> respond(String question) =>
      ChatTurnResponder(conversation: conversation, retriever: retriever)
          .prepare(const TurnRequest(typed: true))
          .responder(side.add)
          .respond(question);

  test(
    'skill steps reach the UI as ChatSkillStep side events, in order',
    () async {
      await conversation.open(kVoiceChatProfile, skills: const [_time]);
      final reply = respond('How late is it?').toList();
      await pumpEventQueue();

      const loaded = SkillLoaded(
        'current-time',
        found: true,
        at: Duration(milliseconds: 800),
      );
      const called = IntentCalled(
        'current_time',
        '{}',
        at: Duration(milliseconds: 1600),
      );
      conversation
        ..emitStep(loaded)
        ..emitStep(called)
        ..emit('Done.');
      await conversation.finish();
      expect(await reply, ['Done.']);

      expect(side.whereType<ChatSkillStep>().map((e) => e.step), [
        loaded,
        called,
      ]);
    },
  );

  test('a context reset keeps its reason on the way to the UI', () async {
    await conversation.open(kVoiceChatProfile, skills: const [_time]);
    final reply = respond('Hi').toList();
    await pumpEventQueue();
    conversation
      ..emitContextReset(reason: ContextResetReason.interruptedSkill)
      ..emit('Hello.');
    await conversation.finish();
    await reply;

    expect(
      side.whereType<ChatContextReset>().single.reason,
      ContextResetReason.interruptedSkill,
    );
  });

  test('on an agent chat the excerpts are reference only and skills take '
      'precedence; a plain chat keeps the plain RAG prompt', () async {
    // A documentation question (a live device question skips retrieval on
    // an agent chat; see below).
    await conversation.open(kVoiceChatProfile, skills: const [_time]);
    final agentReply = respond('What GPU does LiteRT use?').toList();
    await pumpEventQueue();
    await conversation.finish();
    await agentReply;

    await conversation.open(kVoiceChatProfile);
    final plainReply = respond('What GPU does LiteRT use?').toList();
    await pumpEventQueue();
    await conversation.finish();
    await plainReply;

    final [agentPrompt, plainPrompt] = conversation.prompts;
    expect(agentPrompt, contains(PromptBuilder.noToolCall));
    expect(agentPrompt, contains('Reference excerpts'));
    expect(plainPrompt, isNot(contains(PromptBuilder.noToolCall)));
    expect(plainPrompt, contains('Answer the question.'));
  });

  group('live skill questions skip retrieval on an agent chat', () {
    Future<void> ask(String question) async {
      final reply = respond(question).toList();
      await pumpEventQueue();
      await conversation.finish();
      await reply;
    }

    test('a live device question: no search, the bare question, and a '
        'skipped retrieval for the overlay', () async {
      await conversation.open(kVoiceChatProfile, skills: const [_time]);
      await ask('Which accelerator is running right now?');

      expect(retriever.questions, isEmpty, reason: 'no knowledge search');
      expect(
        conversation.prompts.single,
        'Which accelerator is running right now?',
      );
      final retrieval = side.whereType<ChatRetrieval>().single.retrieval;
      expect(retrieval.outcome, RetrievalOutcome.skipped);
      expect(retrieval.passages, isEmpty);
      expect(retrieval.detail, contains('deviceFacts'));
    });

    test('time questions skip it too', () async {
      await conversation.open(kVoiceChatProfile, skills: const [_time]);
      for (final q in const ['What time is it?', "What's today's date?"]) {
        await ask(q);
      }
      expect(retriever.questions, isEmpty);
      expect(conversation.prompts, [
        'What time is it?',
        "What's today's date?",
      ]);
    });

    test('timer and watch requests (the app has no such skills) retrieve '
        'like any other question', () async {
      await conversation.open(kVoiceChatProfile, skills: const [_time]);
      await ask('Set a timer for 10 seconds');
      await ask('Tell me when you see a cup');
      expect(retriever.questions, [
        'Set a timer for 10 seconds',
        'Tell me when you see a cup',
      ]);
    });

    test(
      'a knowledge-base question on the same topic still retrieves',
      () async {
        await conversation.open(kVoiceChatProfile, skills: const [_time]);
        await ask('What GPU does LiteRT use?');

        expect(retriever.questions, ['What GPU does LiteRT use?']);
        expect(conversation.prompts.single, contains('Reference excerpts'));
        expect(
          side.whereType<ChatRetrieval>().single.retrieval.outcome,
          RetrievalOutcome.used,
        );
      },
    );

    test('a plain chat (no skills to route to) keeps retrieving', () async {
      await conversation.open(kVoiceChatProfile);
      await ask('Which accelerator is running right now?');

      expect(retriever.questions, ['Which accelerator is running right now?']);
      expect(conversation.prompts.single, contains('Excerpts'));
    });
  });

  group('an empty final reply after an intent that ran', () {
    const timeTold = IntentSucceeded(
      'current_time',
      'It is 2:03 PM on Friday, October 2, 2026.',
      elapsed: Duration(milliseconds: 300),
      at: Duration(milliseconds: 1800),
    );

    test('the intent result becomes the reply (spoken and shown), not a '
        'failed turn', () async {
      await conversation.open(kVoiceChatProfile, skills: const [_time]);
      final reply = respond('Tell me the time').toList();
      await pumpEventQueue();
      conversation.emitStep(timeTold);
      await conversation.finish();

      expect(await reply, ['It is 2:03 PM on Friday, October 2, 2026.']);
    });

    test(
      'whitespace counts as empty; the last successful intent wins',
      () async {
        await conversation.open(kVoiceChatProfile, skills: const [_time]);
        final reply = respond('Tell me about this device and the time')
            .toList();
        await pumpEventQueue();
        conversation
          ..emitStep(
            const IntentSucceeded(
              'device_info',
              'Gemma 4 E2B runs on the GPU (confirmed).',
              elapsed: Duration(milliseconds: 5),
              at: Duration(milliseconds: 900),
            ),
          )
          ..emitStep(timeTold)
          ..emit('  \n');
        await conversation.finish();

        expect(
          (await reply).join().trim(),
          'It is 2:03 PM on Friday, October 2, 2026.',
        );
      },
    );

    test('a model reply is kept as it is', () async {
      await conversation.open(kVoiceChatProfile, skills: const [_time]);
      final reply = respond('Tell me the time').toList();
      await pumpEventQueue();
      conversation
        ..emitStep(timeTold)
        ..emit('It is just after two.');
      await conversation.finish();

      expect(await reply, ['It is just after two.']);
    });

    test('no successful intent (only a failed one): still empty, so the turn '
        'fails visibly', () async {
      await conversation.open(kVoiceChatProfile, skills: const [_time]);
      final reply = respond('How late is it?').toList();
      await pumpEventQueue();
      conversation.emitStep(
        const IntentFailed(
          'current_time',
          'current_time timed out after 3 seconds.',
          at: Duration(milliseconds: 1800),
        ),
      );
      await conversation.finish();

      expect(await reply, isEmpty);
    });

    test('a stopped turn says nothing, even after an intent ran', () async {
      await conversation.open(kVoiceChatProfile, skills: const [_time]);
      final responder = ChatTurnResponder(
        conversation: conversation,
        retriever: retriever,
      ).prepare(const TurnRequest(typed: true)).responder(side.add);
      final reply = responder.respond('Tell me the time').toList();
      await pumpEventQueue();
      conversation.emitStep(timeTold);
      await responder.stop();

      expect(await reply, isEmpty);
    });
  });

  group('an image question that is not a skill question', () {
    Future<void> ask(String question, {Uint8List? image}) async {
      final reply =
          ChatTurnResponder(conversation: conversation, retriever: retriever)
              .prepare(TurnRequest(typed: true, image: image))
              .responder(side.add)
              .respond(question)
              .toList();
      await pumpEventQueue();
      await conversation.finish();
      await reply;
    }

    final png = Uint8List.fromList([137, 80, 78, 71]);

    test('carries the direct-answer hint (Gemma called runIntent for photo '
        'questions in 5 of 6 first turns)', () async {
      await conversation.open(kVoiceChatProfile, skills: const [_time]);
      await ask('What animal is this?', image: png);
      expect(
        conversation.prompts.single,
        endsWith(PromptBuilder.directAnswerHint),
      );
    });

    test(
      'not on a routed skill question, a text-only turn, or a plain chat',
      () async {
        await conversation.open(kVoiceChatProfile, skills: const [_time]);
        await ask('What time is it?', image: png);
        await ask('What animal is a tabby?');
        await conversation.open(kVoiceChatProfile);
        await ask('What animal is this?', image: png);
        for (final prompt in conversation.prompts) {
          expect(prompt, isNot(contains(PromptBuilder.directAnswerHint)));
        }
      },
    );
  });

  group('a high-confidence action runs its intent directly', () {
    late List<(String, String)> runs;
    late SkillResult result;

    Future<List<String>> run(String question) async {
      final responder = ChatTurnResponder(
        conversation: conversation,
        retriever: retriever,
        direct: (intent, params) async {
          runs.add((intent, params));
          return result;
        },
      ).prepare(const TurnRequest(typed: true)).responder(side.add);
      return responder.respond(question).toList();
    }

    setUp(() {
      runs = [];
      result = const TextResult('It is 2:03 PM on Friday, October 2, 2026.');
    });

    test('time: current_time runs at once, the steps show it, the result is '
        'the reply, and the model is never asked', () async {
      await conversation.open(kVoiceChatProfile, skills: const [_time]);
      final reply = await run('What time is it?');

      expect(runs, [('current_time', '{}')]);
      expect(reply.join(), 'It is 2:03 PM on Friday, October 2, 2026.');
      expect(conversation.prompts, isEmpty, reason: 'no LLM turn');
      final steps = side.whereType<ChatSkillStep>().map((e) => e.step);
      expect(steps.first, isA<IntentCalled>());
      expect((steps.first as IntentCalled).intent, 'current_time');
      expect(steps.last, isA<IntentSucceeded>());
    });

    test('a failed intent: a failed step and the user-facing part of the '
        'message (not the instructions meant for the model)', () async {
      result = const ErrorResult(
        'current_time failed: Bad state: no clock. Tell the user it did not '
        'work.',
      );
      await conversation.open(kVoiceChatProfile, skills: const [_time]);
      final reply = (await run('What time is it?')).join();

      expect(reply, contains('no clock'));
      expect(reply, isNot(contains('Tell the user')));
      expect(side.whereType<ChatSkillStep>().last.step, isA<IntentFailed>());
      expect(conversation.prompts, isEmpty);
    });

    test('a device question: device_info runs, and its exact result is the '
        'reply (no model rephrasing: it got the facts wrong)', () async {
      result = const TextResult(
        'Gemma 4 E2B runs on the GPU; the YOLO26n detector on the GPU in '
        'float32, fully accelerated.',
      );
      await conversation.open(kVoiceChatProfile, skills: const [_time]);
      final reply = await run('What device am I on?');
      expect(runs, [('device_info', '{}')]);
      expect(reply.join(), (result as TextResult).text);
      expect(conversation.prompts, isEmpty, reason: 'no model round');
      final steps = side.whereType<ChatSkillStep>().map((e) => e.step);
      expect(steps.first, isA<IntentCalled>());
      expect(steps.last, isA<IntentSucceeded>());
    });

    test('a chat model with tools off: the skills are '
        'not in the chat, but the direct intents still run', () async {
      conversation.capabilities = const ChatCapabilities(
        modelName: 'Gemma 3 NPU',
        images: false,
        tools: false,
      );
      await conversation.open(kVoiceChatProfile, skills: const [_time]);
      expect(conversation.hasSkills, isFalse);
      expect(conversation.skillsNeedTools, isTrue);

      final reply = await run('What time is it?');

      expect(runs, [('current_time', '{}')]);
      expect(reply.join(), 'It is 2:03 PM on Friday, October 2, 2026.');
      expect(conversation.prompts, isEmpty);
    });

    test('anything else, or a plain chat, still goes to the model', () async {
      await conversation.open(kVoiceChatProfile, skills: const [_time]);
      final reply = run('Set a timer for 10 seconds');
      await pumpEventQueue();
      conversation.emit('Started.');
      await conversation.finish();
      await reply;
      expect(runs, isEmpty);

      await conversation.open(kVoiceChatProfile);
      final plain = run('What time is it?');
      await pumpEventQueue();
      conversation.emit('I cannot.');
      await conversation.finish();
      await plain;
      expect(runs, isEmpty, reason: 'no skills, no actions');
    });
  });
}
