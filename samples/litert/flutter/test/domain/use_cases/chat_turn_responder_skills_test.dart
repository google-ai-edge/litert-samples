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
import 'dart:typed_data';

import 'package:flutter_edge_ai_agent/flutter_edge_ai_agent.dart'
    show ErrorResult, Skill, SkillResult, SkillType, TextResult;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/conversation_repository.dart'
    show EmptyAnswerAfterSkillException;
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

const _device = Skill(
  name: 'device-info',
  description: 'Which models and backends run.',
  instructions: 'Call the `run_intent` tool with intent `device_info`.',
  type: SkillType.intent,
);

/// A skill the app does not know, added at run time; it names no intent.
const _recipes = Skill(
  name: 'recipe-finder',
  description: 'Suggest a recipe from what is in the fridge.',
  instructions: 'Ask what is in the fridge, then suggest one recipe.',
  type: SkillType.textOnly,
);

/// The responder forwards skill steps as side events, and on an
/// agent chat frames excerpts so they never replace a skill.
void main() {
  late FakeConversationRepository conversation;
  late FakeRetriever retriever;
  late List<ChatSideEvent> side;

  /// The installed skills (the Skills sheet's catalog). Null: the ones the
  /// chat was last opened with, as the app opens it with the catalog.
  List<Skill>? installed;
  List<Skill> installedNow() =>
      installed ?? conversation.openedSkills.lastOrNull ?? const [];

  ChatTurnResponder responders({DirectIntentRunner? direct}) =>
      ChatTurnResponder(
        conversation: conversation,
        retriever: retriever,
        direct: direct,
        installedSkills: installedNow,
      );

  setUp(() {
    installed = null;
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

  Stream<String> respond(String question) => responders()
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

  test('no answer after a loaded skill fails the turn with the typed '
      'exception: an earlier intent result is not spoken in its place, the '
      'steps still reach the UI', () async {
    await conversation.open(kVoiceChatProfile, skills: const [_time]);
    final spoken = <String>[];
    Object? failure;
    final done = Completer<void>();
    respond('How late is it?').listen(
      spoken.add,
      onError: (Object e) => failure = e,
      onDone: done.complete,
    );
    await pumpEventQueue();
    conversation
      ..emitStep(
        const SkillLoaded(
          'kid-clock',
          found: true,
          at: Duration(milliseconds: 500),
        ),
      )
      ..emitStep(
        const IntentCalled(
          'current_time',
          '{}',
          at: Duration(milliseconds: 700),
        ),
      )
      ..emitStep(
        const IntentSucceeded(
          'current_time',
          'It is 2:03 PM.',
          elapsed: Duration(milliseconds: 2),
          at: Duration(milliseconds: 702),
        ),
      )
      ..emitStep(
        const SkillLoaded(
          'current-time',
          found: true,
          at: Duration(milliseconds: 800),
        ),
      )
      ..emitStep(
        const EmptyAnswerRetried(
          'current-time',
          attempt: 1,
          nudge: 'Go on.',
          at: Duration(milliseconds: 1200),
        ),
      );
    await conversation.fail(
      const EmptyAnswerAfterSkillException('current-time'),
    );
    await done.future;

    expect(
      failure,
      isA<EmptyAnswerAfterSkillException>().having(
        (e) => e.skillName,
        'skillName',
        'current-time',
      ),
    );
    expect(spoken, isEmpty, reason: "the intent's result is not the reply");
    expect(side.whereType<ChatSkillStep>().map((e) => e.step.runtimeType), [
      SkillLoaded,
      IntentCalled,
      IntentSucceeded,
      SkillLoaded,
      EmptyAnswerRetried,
    ]);
  });

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
    expect(
      agentPrompt,
      contains(PromptBuilder.excerptSkills(const ['current-time'])),
    );
    expect(agentPrompt, contains('Reference excerpts'));
    expect(plainPrompt, isNot(contains('skills')));
    expect(plainPrompt, contains('Answer the question.'));
  });

  test('a skill added at run time is named in the hints like a bundled one, '
      'from the chat\'s own skill list', () async {
    await conversation.open(kVoiceChatProfile, skills: const [_recipes]);
    final reply = respond('What can I cook tonight?').toList();
    await pumpEventQueue();
    await conversation.finish();
    await reply;

    expect(
      conversation.prompts.single,
      contains(PromptBuilder.excerptSkills(const ['recipe-finder'])),
    );
  });

  group('live skill questions skip retrieval while an installed intent skill '
      'names their intent', () {
    Future<void> ask(String question) async {
      final reply = respond(question).toList();
      await pumpEventQueue();
      await conversation.finish();
      await reply;
    }

    test('a live device question: no search, the bare question, and a '
        'skipped retrieval for the overlay', () async {
      await conversation.open(
        kVoiceChatProfile,
        skills: const [_time, _device],
      );
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

    test('no loaded skill names the intent: the question retrieves like any '
        'other, and the excerpts leave the chat\'s skills first', () async {
      // device-info deleted: only current-time is left.
      await conversation.open(kVoiceChatProfile, skills: const [_time]);
      await ask('Which accelerator is running right now?');
      // Only a run-time skill: neither rule applies.
      await conversation.open(kVoiceChatProfile, skills: const [_recipes]);
      await ask('What time is it?');

      expect(retriever.questions, [
        'Which accelerator is running right now?',
        'What time is it?',
      ]);
      expect(
        side.whereType<ChatRetrieval>().map((e) => e.retrieval.outcome),
        everyElement(RetrievalOutcome.used),
      );
      expect(
        conversation.prompts.last,
        contains(PromptBuilder.excerptSkills(const ['recipe-finder'])),
      );
    });

    test('a chat model with tools off: the installed skills still gate it, '
        'so the question skips retrieval (the docs would answer a device '
        'question wrong)', () async {
      conversation.capabilities = const ChatCapabilities(
        modelName: 'Gemma 3 NPU',
        images: false,
        tools: false,
      );
      await conversation.open(
        kVoiceChatProfile,
        skills: const [_time, _device],
      );
      expect(conversation.hasSkills, isFalse);
      await ask('Which accelerator is running right now?');

      expect(retriever.questions, isEmpty);
      expect(
        conversation.prompts.single,
        'Which accelerator is running right now?',
      );
    });

    test('an instruction-only skill that merely quotes an intent does not '
        'count', () async {
      installed = const [
        Skill(
          name: 'time-notes',
          description: 'Notes on telling the time.',
          instructions: 'Never call intent `current_time` for this.',
          type: SkillType.textOnly,
        ),
      ];
      await conversation.open(kVoiceChatProfile, skills: installed!);
      await ask('What time is it?');

      expect(retriever.questions, ['What time is it?']);
    });
  });

  // The agent package ends such a turn with a plain DoneEvent (it asks again
  // only after loadSkill): the reply is what the model said, and an empty
  // one fails the turn in VoiceAssistant like any empty reply.
  group('an empty final reply after an intent that ran', () {
    const timeTold = IntentSucceeded(
      'current_time',
      'It is 2:03 PM on Friday, October 2, 2026.',
      elapsed: Duration(milliseconds: 300),
      at: Duration(milliseconds: 1800),
    );

    test('stays empty: the intent result is not spoken in its place, the '
        'step still reaches the UI', () async {
      await conversation.open(kVoiceChatProfile, skills: const [_time]);
      final reply = respond('Tell me the time').toList();
      await pumpEventQueue();
      conversation.emitStep(timeTold);
      await conversation.finish();

      expect(await reply, isEmpty);
      expect(side.whereType<ChatSkillStep>().map((e) => e.step), [timeTold]);
    });

    test('whitespace is passed on as it is, with nothing added', () async {
      await conversation.open(kVoiceChatProfile, skills: const [_time]);
      final reply = respond('Tell me the time').toList();
      await pumpEventQueue();
      conversation
        ..emitStep(timeTold)
        ..emit('  \n');
      await conversation.finish();

      expect(await reply, ['  \n']);
    });

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

    test('a stopped turn says nothing, even after an intent ran', () async {
      await conversation.open(kVoiceChatProfile, skills: const [_time]);
      final responder = responders()
          .prepare(const TurnRequest(typed: true))
          .responder(side.add);
      final reply = responder.respond('Tell me the time').toList();
      await pumpEventQueue();
      conversation.emitStep(timeTold);
      await responder.stop();

      expect(await reply, isEmpty);
    });
  });

  group('an image question that is not a skill question', () {
    Future<void> ask(String question, {Uint8List? image}) async {
      final reply = responders()
          .prepare(TurnRequest(typed: true, image: image))
          .responder(side.add)
          .respond(question)
          .toList();
      await pumpEventQueue();
      await conversation.finish();
      await reply;
    }

    final png = Uint8List.fromList([137, 80, 78, 71]);

    test(
      'carries the direct-answer hint (Gemma called runIntent for photo '
      'questions in 5 of 6 first turns), naming the chat\'s skills',
      () async {
        await conversation.open(
          kVoiceChatProfile,
          skills: const [_time, _recipes],
        );
        await ask('What animal is this?', image: png);
        expect(
          conversation.prompts.single,
          endsWith(
            PromptBuilder.directAnswerHint(const [
              'current-time',
              'recipe-finder',
            ]),
          ),
        );
      },
    );

    test(
      'not on a routed skill question, a text-only turn, or a plain chat',
      () async {
        await conversation.open(kVoiceChatProfile, skills: const [_time]);
        await ask('What time is it?', image: png);
        await ask('What animal is a tabby?');
        await conversation.open(kVoiceChatProfile);
        await ask('What animal is this?', image: png);
        for (final prompt in conversation.prompts) {
          expect(prompt, isNot(contains('A picture is attached')));
        }
      },
    );
  });

  group('a high-confidence action runs its intent directly', () {
    late List<(String, String)> runs;
    late SkillResult result;

    Future<List<String>> run(String question) async {
      final responder = responders(
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

    test('a failed intent: a failed step, and the executor\'s message is '
        'the reply', () async {
      result = const ErrorResult(
        'Intent "current_time" failed: Bad state: no clock',
      );
      await conversation.open(kVoiceChatProfile, skills: const [_time]);
      final reply = (await run('What time is it?')).join();

      expect(reply, 'Intent "current_time" failed: Bad state: no clock');
      expect(side.whereType<ChatSkillStep>().last.step, isA<IntentFailed>());
      expect(conversation.prompts, isEmpty);
    });

    test('a device question: device_info runs, and its exact result is the '
        'reply (no model rephrasing: it got the facts wrong)', () async {
      result = const TextResult(
        'Gemma 4 E2B runs on the GPU; the YOLO26n detector on the GPU in '
        'float32, fully accelerated.',
      );
      await conversation.open(
        kVoiceChatProfile,
        skills: const [_time, _device],
      );
      final reply = await run('What device am I on?');
      expect(runs, [('device_info', '{}')]);
      expect(reply.join(), (result as TextResult).text);
      expect(conversation.prompts, isEmpty, reason: 'no model round');
      final steps = side.whereType<ChatSkillStep>().map((e) => e.step);
      expect(steps.first, isA<IntentCalled>());
      expect(steps.last, isA<IntentSucceeded>());
    });

    /// Plays the model's answer to the one turn [question] starts.
    Future<void> toModel(String question) async {
      final reply = run(question);
      await pumpEventQueue();
      conversation.emit('From the model.');
      await conversation.finish();
      expect(await reply, ['From the model.']);
    }

    test('a chat model with tools off (an NPU build): the model cannot call '
        'the skills, but the installed ones still turn the shortcuts on, so '
        'the time is never invented', () async {
      conversation.capabilities = const ChatCapabilities(
        modelName: 'Gemma 3 NPU',
        images: false,
        tools: false,
      );
      await conversation.open(
        kVoiceChatProfile,
        skills: const [_time, _device],
      );
      expect(conversation.hasSkills, isFalse);
      expect(conversation.skillsNeedTools, isTrue);

      final time = await run('What time is it?');
      await run('What device am I on?');

      expect(runs, [('current_time', '{}'), ('device_info', '{}')]);
      expect(time.join(), 'It is 2:03 PM on Friday, October 2, 2026.');
      expect(conversation.prompts, isEmpty, reason: 'no model round');
    });

    test(
      'tools off with the skill deleted: the request goes to the model',
      () async {
        conversation.capabilities = const ChatCapabilities(
          modelName: 'Gemma 3 NPU',
          images: false,
          tools: false,
        );
        await conversation.open(kVoiceChatProfile, skills: const [_device]);

        await toModel('What time is it?');

        expect(runs, isEmpty);
      },
    );

    test('the installed catalog decides, not the open chat: a skill added or '
        'deleted since the chat opened counts at once', () async {
      await conversation.open(kVoiceChatProfile, skills: const [_time]);
      installed = const [_time, _device];
      await run('What device am I on?');
      installed = const [_device];
      await toModel('What time is it?');

      expect(runs, [('device_info', '{}')]);
    });

    test('an instruction-only skill that merely quotes an intent turns '
        'nothing on', () async {
      await conversation.open(
        kVoiceChatProfile,
        skills: const [
          Skill(
            name: 'time-notes',
            description: 'Notes on telling the time.',
            instructions: 'Never call intent `current_time` for this.',
            type: SkillType.textOnly,
          ),
        ],
      );

      await toModel('What time is it?');

      expect(runs, isEmpty);
    });

    test('the shortcut goes with the skill that names its intent: deleted '
        'or replaced, the request goes to the model', () async {
      // device-info deleted.
      await conversation.open(kVoiceChatProfile, skills: const [_time]);
      await toModel('What device am I on?');
      // current-time replaced by one that names no intent.
      await conversation.open(
        kVoiceChatProfile,
        skills: const [
          Skill(
            name: 'current-time',
            description: 'Say that the time is unknown.',
            instructions: 'Tell the user you cannot read the clock.',
            type: SkillType.textOnly,
          ),
          _device,
        ],
      );
      await toModel('What time is it?');
      // Only a skill the app does not know.
      await conversation.open(kVoiceChatProfile, skills: const [_recipes]);
      await toModel('What time is it?');

      expect(runs, isEmpty);
      expect(side.whereType<ChatSkillStep>(), isEmpty);
    });

    test('any loaded skill that names the intent turns it on, in any of the '
        'forms a SKILL.md names it', () async {
      for (final instructions in const [
        'Call the `run_intent` tool with intent `current_time`.',
        'Call the `run_intent` tool with:\n- intent: current_time',
      ]) {
        await conversation.open(
          kVoiceChatProfile,
          skills: [
            Skill(
              name: 'kid-clock',
              description: 'The time, for a child.',
              instructions: instructions,
              type: SkillType.intent,
            ),
          ],
        );
        await run('What time is it?');
      }

      expect(runs, List.filled(2, ('current_time', '{}')));
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
