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
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_edge_ai/flutter_edge_ai.dart' hide ModelSpec;
import 'package:flutter_edge_ai_agent/flutter_edge_ai_agent.dart'
    show
        AgentToolNames,
        NativeIntentExecutor,
        defaultEmptyAfterSkillNudge,
        Skill,
        TextSkillExecutor,
        loadSkillTool,
        parseSkillMd,
        runIntentTool;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/conversation_repository.dart';
import 'package:litert_edge_demos/data/repositories/conversation_repository_edge_ai.dart';
import 'package:litert_edge_demos/data/services/llm/llm_service.dart';
import 'package:litert_edge_demos/domain/models/assistant_event.dart';
import 'package:litert_edge_demos/domain/models/chat_model_config.dart';
import 'package:litert_edge_demos/domain/models/skill_step.dart';
import 'package:litert_edge_demos/domain/skills/app_intent_handlers.dart';
import 'package:litert_edge_demos/domain/skills/app_intents.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../fakes/fake_tool_model.dart';
import '../../support/package_skill_fixtures.dart';

Skill bundled(String name) =>
    parseSkillMd(File('assets/skills/$name/SKILL.md').readAsStringSync());

/// One `ask` stream with its steps, collected.
final class AgentTurn {
  AgentTurn(ConversationRepository repo, String prompt, {Uint8List? image}) {
    _sub = repo
        .ask(prompt, image: image, onStep: steps.add)
        .listen(events.add, onDone: () => _done.complete());
  }

  late final StreamSubscription<AssistantEvent> _sub;
  final _done = Completer<void>();
  final List<AssistantEvent> events = [];
  final List<SkillStep> steps = [];

  Future<void> get done => _done.future;

  Future<void> cancel() => _sub.cancel();

  List<String> get deltas => [
    for (final e in events)
      if (e case AssistantTextDelta(:final text)) text,
  ];

  AssistantEvent get last => events.last;

  GenerationMetrics get metrics => switch (events.last) {
    AssistantDone(:final metrics) => metrics,
    final other => throw StateError('last event is $other'),
  };
}

/// Lets queued microtasks and zero timers run (the scripted session awaits
/// one zero timer per token).
Future<void> settle([int rounds = 20]) async {
  for (var i = 0; i < rounds; i++) {
    await Future<void>.delayed(Duration.zero);
  }
}

void main() {
  late FakeToolModel model;
  late int deviceCalls;
  late Completer<void>? deviceGate;
  late EdgeAiConversationRepository repo;
  final skills = [bundled('current-time'), bundled('device-info')];

  FakeToolSession session() => model.lastSession;

  setUp(() async {
    model = FakeToolModel();
    deviceCalls = 0;
    deviceGate = null;
    final intents = buildAppIntents(
      deviceFacts: () => 'Gemma 4 E2B runs on the GPU (confirmed).',
      now: () => DateTime(2026, 10, 2, 14, 3),
    );
    final deviceInfo = intents[AppIntent.deviceInfo]!;
    final executor = NativeIntentExecutor(
      handlers: {
        ...intents,
        AppIntent.deviceInfo: (params) async {
          // A test can hold the executor (a tool still running at Stop).
          await deviceGate?.future;
          deviceCalls++;
          return await deviceInfo(params);
        },
      },
    );
    final llm = LlmService(loadModel: (_) async => model);
    expect(await llm.load(kDefineChatModel), isA<Ok<LlmInfo>>());
    repo = EdgeAiConversationRepository(
      llm: llm,
      stopTimeout: const Duration(seconds: 2),
      executors: [executor, TextSkillExecutor()],
    );
  });

  tearDown(() async {
    await repo.close();
  });

  // Which profiles get an agent chat (Demo 3, or no skills: plain):
  // native_chat_test.dart.
  group('open', () {
    test('Demo 1 with skills: an agent chat with only loadSkill and '
        'runIntent, Gemma 4 tool format, images, and every skill listed in '
        'the system prompt', () async {
      expect(
        await repo.open(kVoiceChatProfile, skills: skills),
        isA<Ok<void>>(),
      );

      final args = model.chats.single;
      expect(args.tools.map((t) => t.name), [
        AgentToolNames.loadSkill,
        AgentToolNames.runIntent,
      ]);
      // The package's own declarations, its wording included.
      expect(args.tools.first, same(loadSkillTool));
      expect(args.tools.last, same(runIntentTool));
      expect(args.supportsFunctionCalls, isTrue);
      // The chat inherits the installed type (gemma4 for Gemma 4 E2B);
      // a chat-only type would disagree with the native session's.
      expect(args.modelType, isNull);
      expect(args.toolChoice, ToolChoice.auto);
      expect(args.supportImage, isTrue);
      // The shared sampler, never left to the engine (the bundle's).
      expect(args.temperature, kSampler.temperature);
      expect(args.topK, kSampler.topK);
      final prompt = args.systemInstruction!;
      expect(prompt, contains('- current-time: Get the current local time'));
      expect(prompt, contains('- device-info: Say which models'));
      expect(prompt, isNot(contains('__SKILLS__')));
      expect(prompt, contains('never from knowledge-base excerpts'));
      expect(repo.hasSkills, isTrue);
    });

    test(
      'a chat model with tools off: Demo 1 with skills gets a plain chat '
      'with no tool declarations at all, and says the skills need tools',
      () async {
        final llm = LlmService(loadModel: (_) async => model);
        await llm.load(
          const ChatModelConfig(
            name: 'Gemma 3 NPU',
            modelType: ModelType.gemmaIt,
            llm: LlmConfig(
              maxTokens: 1280,
              backend: PreferredBackend.gpu,
              maxNumImages: 1,
            ),
            tools: false,
          ),
        );
        final plain = EdgeAiConversationRepository(llm: llm);
        addTearDown(plain.close);

        expect(
          await plain.open(kVoiceChatProfile, skills: skills),
          isA<Ok<void>>(),
        );

        final args = model.chats.last;
        expect(args.tools, isEmpty, reason: 'no tools_json, no Dart prompt');
        expect(args.supportsFunctionCalls, isNot(isTrue));
        expect(args.supportImage, isFalse);
        expect(args.systemInstruction, kVoiceChatProfile.systemInstruction);
        expect(plain.hasSkills, isFalse);
        expect(plain.skillsNeedTools, isTrue);
        expect(plain.capabilities.modelName, 'Gemma 3 NPU');
        expect(plain.capabilities.tools, isFalse);
        expect(plain.capabilities.images, isFalse);
      },
    );

    test(
      'release closes the chat and the next open builds a new one',
      () async {
        await repo.open(kVoiceChatProfile, skills: skills);
        final before = model.chats.length;

        await repo.release();

        expect(repo.isOpen, isFalse);
        expect(repo.hasSkills, isFalse);
        expect(await repo.open(kCameraProfile), isA<Ok<void>>());
        expect(model.chats.length, before + 1);
        expect(repo.profile, kCameraProfile);
      },
    );

    test('a reset keeps the skills; a new open with a new skill set lists '
        'it', () async {
      await repo.open(kVoiceChatProfile, skills: skills);
      expect(await repo.reset(ifCurrent: kVoiceChatProfile), isA<Ok<void>>());
      expect(model.chats, hasLength(2));
      expect(model.chats.last.tools, hasLength(2));
      expect(model.chats.last.systemInstruction, contains('- device-info:'));

      await repo.open(kVoiceChatProfile, skills: [bundled('current-time')]);
      expect(model.chats.last.systemInstruction, contains('- current-time:'));
      expect(
        model.chats.last.systemInstruction,
        isNot(contains('- device-info:')),
      );
    });
  });

  // An image on an agent turn, sent once: the turn paths test.
  group('a skill turn', () {
    setUp(() => repo.open(kVoiceChatProfile, skills: skills));

    test('loadSkill → runIntent(current_time) → answer: steps in order, '
        'nothing spoken before the tools ran, the result on the step, '
        'metrics count the tool rounds', () async {
      model.script.addAll([
        const ToolCallTurn('loadSkill', {'skillName': 'current-time'}),
        const ToolCallTurn('runIntent', {
          'intent': 'current_time',
          'parameters': '{}',
        }),
        const TextTurn(['It is ', 'just after two.']),
      ]);

      final turn = AgentTurn(repo, 'How late is it?');
      await turn.done;

      expect(turn.deltas.join(), 'It is just after two.');
      expect(turn.steps, hasLength(3));
      expect(
        turn.steps[0],
        isA<SkillLoaded>()
            .having((s) => s.name, 'name', 'current-time')
            .having((s) => s.found, 'found', isTrue),
      );
      expect(
        turn.steps[1],
        isA<IntentCalled>()
            .having((s) => s.intent, 'intent', 'current_time')
            .having((s) => s.parameters, 'parameters', '{}'),
      );
      expect(
        turn.steps[2],
        isA<IntentSucceeded>().having(
          (s) => s.result,
          'result',
          'It is 2:03 PM on Friday, October 2, 2026.',
        ),
      );
      final metrics = turn.metrics;
      expect(metrics.stopped, isFalse);
      expect(metrics.toolRounds, 2);
      expect(metrics.skillSteps, turn.steps);
      expect(metrics.timeToFirstToken, greaterThan(turn.steps.last.at));
      // The skill's instructions and the intent's result went back to the
      // model as tool responses.
      final responses = session().toolResponses;
      expect(responses, hasLength(2));
      expect(responses.first.text, contains('current_time'));
      expect(responses.last.text, contains('It is 2:03 PM'));
      expect(repo.isGenerating.value, isFalse);
    });

    test('a final generation that comes back empty after an intent: the turn '
        'ends with no text (the agent package ends it with a plain '
        'DoneEvent), and the text history keeps that empty reply, not the '
        'intent\'s result', () async {
      model.script.addAll([
        const ToolCallTurn('loadSkill', {'skillName': 'current-time'}),
        const ToolCallTurn('runIntent', {
          'intent': 'current_time',
          'parameters': '{}',
        }),
        const TextTurn([]),
      ]);

      final turn = AgentTurn(repo, 'What time is it?');
      await turn.done;

      expect(turn.deltas, isEmpty);
      // Empty after an intent's result: no retry (the package asks again
      // only after loadSkill), a plain end.
      expect(turn.steps.whereType<EmptyAnswerRetried>(), isEmpty);
      expect(session().responseRequests, 3);
      expect(turn.last, isA<AssistantDone>());
      expect(turn.metrics.stopped, isFalse);
      expect(turn.steps.last, isA<IntentSucceeded>(), reason: 'the tool ran');
      final kept = repo.history.single;
      expect(kept.assistant, isEmpty);
      expect(kept.replayable, isFalse);
    });

    test('an empty answer right after loadSkill is asked for again once '
        '(the package default): a retry step, the nudge in the chat, then '
        'the answer', () async {
      model.script.addAll([
        const ToolCallTurn('loadSkill', {'skillName': 'current-time'}),
        const TextTurn([]),
        const ToolCallTurn('runIntent', {
          'intent': 'current_time',
          'parameters': '{}',
        }),
        const TextTurn(['It is just after two.']),
      ]);

      final turn = AgentTurn(repo, 'What time is it?');
      await turn.done;

      expect(turn.steps.map((s) => s.runtimeType), [
        SkillLoaded,
        EmptyAnswerRetried,
        IntentCalled,
        IntentSucceeded,
      ]);
      expect(
        turn.steps[1],
        isA<EmptyAnswerRetried>()
            .having((s) => s.skill, 'skill', 'current-time')
            .having((s) => s.attempt, 'attempt', 1)
            .having((s) => s.nudge, 'nudge', defaultEmptyAfterSkillNudge),
      );
      expect(turn.deltas.join(), 'It is just after two.');
      expect(turn.last, isA<AssistantDone>());
      expect(turn.metrics.skillSteps, turn.steps);
      final users = [
        for (final m in session().queries)
          if (m.isUser && m.type != MessageType.toolResponse) m.text,
      ];
      expect(users, ['What time is it?', defaultEmptyAfterSkillNudge]);
      expect(
        repo.history.single.user,
        'What time is it?',
        reason: 'the text mirror keeps the question, not the nudge',
      );
    });

    test('still empty after the retry: the turn fails with an '
        'EmptyAnswerAfterSkillException naming the skill, nothing is '
        'spoken, and the chat is kept', () async {
      model.script.addAll([
        const ToolCallTurn('loadSkill', {'skillName': 'current-time'}),
        const TextTurn([]),
        const TextTurn([]),
      ]);

      final turn = AgentTurn(repo, 'What time is it?');
      await turn.done;

      expect(
        turn.last,
        isA<AssistantFailed>().having(
          (e) => e.error,
          'error',
          isA<EmptyAnswerAfterSkillException>().having(
            (e) => e.skillName,
            'skillName',
            'current-time',
          ),
        ),
      );
      expect(
        '${(turn.last as AssistantFailed).error}',
        contains('empty reply'),
      );
      expect(turn.events.whereType<AssistantDone>(), isEmpty);
      expect(turn.deltas, isEmpty);
      expect(turn.steps.map((s) => s.runtimeType), [
        SkillLoaded,
        EmptyAnswerRetried,
      ]);
      expect(session().responseRequests, 3);
      expect(repo.isGenerating.value, isFalse);
      expect(repo.history.single.assistant, isEmpty);

      // The last generation consumed what was staged: no rebuild.
      final chatsBefore = model.chats.length;
      model.script.add(const TextTurn(['Hello.']));
      final next = AgentTurn(repo, 'Hi');
      await next.done;
      expect(model.chats.length, chatsBefore);
      expect(next.deltas.join(), 'Hello.');
    });

    test('a call that follows prose: core 2.4.0 takes the call chunk out of '
        'the text, so the prose is spoken, the JSON is not, the call runs, '
        'and no failed step is reported', () async {
      model.script.addAll([
        const ProseThenToolCallTurn(
          ['Let me check. '],
          'loadSkill',
          {'skillName': 'current-time'},
        ),
        const ToolCallTurn('runIntent', {
          'intent': 'current_time',
          'parameters': '{}',
        }),
        const TextTurn(['It is just after two.']),
      ]);

      final turn = AgentTurn(repo, 'What time is it?');
      await turn.done;

      expect(turn.deltas.join(), 'Let me check. It is just after two.');
      expect(turn.deltas.join(), isNot(contains('tool_calls')));
      expect(turn.steps.map((s) => s.runtimeType), [
        SkillLoaded,
        IntentCalled,
        IntentSucceeded,
      ]);
      expect(turn.last, isA<AssistantDone>());
    });

    test('a skill written for the agent package: its get-current-time loads '
        'and its intent, a built-in of the package executor, runs', () async {
      await repo.open(
        kVoiceChatProfile,
        skills: [parseSkillMd(packageGetCurrentTimeSkillMd)],
      );
      model.script.addAll([
        const ToolCallTurn('loadSkill', {'skillName': 'get-current-time'}),
        const ToolCallTurn('runIntent', {
          'intent': packageGetCurrentTimeIntent,
          'parameters': '{}',
        }),
        const TextTurn(['It is Friday.']),
      ]);

      final turn = AgentTurn(repo, 'What day is it?');
      await turn.done;

      expect(
        turn.steps[0],
        isA<SkillLoaded>()
            .having((s) => s.name, 'name', 'get-current-time')
            .having((s) => s.found, 'found', isTrue),
      );
      expect(
        session().toolResponses.first.text,
        contains(packageGetCurrentTimeIntent),
        reason: 'loadSkill returned its instructions',
      );
      expect(turn.steps.whereType<IntentFailed>(), isEmpty);
      // The package's own format: `yyyy-MM-ddTHH:mm:ss Weekday`.
      expect(
        turn.steps.whereType<IntentSucceeded>().single.result,
        matches(
          RegExp(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2} [A-Z][a-z]+day$'),
        ),
      );
      expect(turn.deltas.join(), 'It is Friday.');
    });

    test('an ErrorResult is one failed step, not two (AgentLoop._runExecutor), '
        'and the model can still answer', () async {
      model.script.addAll([
        const ToolCallTurn('runIntent', {
          'intent': 'make_coffee',
          'parameters': '{}',
        }),
        const TextTurn(['I cannot make coffee.']),
      ]);

      final turn = AgentTurn(repo, 'Make me a coffee');
      await turn.done;

      final failed = turn.steps.whereType<IntentFailed>().toList();
      expect(failed, hasLength(1));
      expect(failed.single.message, contains('Unknown intent "make_coffee"'));
      expect(
        failed.single.message,
        allOf(
          contains('Allowed intents:'),
          contains(AppIntent.currentTime),
          contains(AppIntent.deviceInfo),
        ),
      );
      expect(turn.metrics.stopped, isFalse);
      expect(turn.deltas.join(), 'I cannot make coffee.');
    });

    test('a tool call the SDK could not parse comes back as text: it is '
        'never spoken, only reported as a failed step', () async {
      model.script.add(
        const UnparsedToolCallTurn(
          '{"role":"assistant","tool_calls":[{"type":"function","function":'
          '{"name":"loadSkill","arguments":{"skillName":"ma',
        ),
      );

      final turn = AgentTurn(repo, 'What time is it');
      await turn.done;

      expect(turn.deltas, isEmpty, reason: 'raw JSON would go to TTS');
      expect(
        turn.steps.single,
        isA<IntentFailed>().having(
          (s) => s.message,
          'message',
          contains('could not be read'),
        ),
      );
      expect(turn.last, isA<AssistantDone>());
    });

    test('an unknown skill is a not-found load and a failed step', () async {
      model.script.addAll([
        const ToolCallTurn('loadSkill', {'skillName': 'weather'}),
        const TextTurn(['No weather skill.']),
      ]);

      final turn = AgentTurn(repo, 'Weather?');
      await turn.done;

      expect(
        turn.steps.first,
        isA<SkillLoaded>().having((s) => s.found, 'found', isFalse),
      );
      expect(turn.steps.whereType<IntentFailed>(), hasLength(1));
    });

    test('too many tool rounds end the turn as a SkillLoopException', () async {
      for (var i = 0; i < kAgentMaxIterations; i++) {
        model.script.add(
          const ToolCallTurn('runIntent', {
            'intent': 'current_time',
            'parameters': '{}',
          }),
        );
      }

      final turn = AgentTurn(repo, 'Loop');
      await turn.done;

      expect(
        turn.last,
        isA<AssistantFailed>().having(
          (e) => e.error,
          'error',
          isA<SkillLoopException>().having(
            (e) => e.iterations,
            'iterations',
            kAgentMaxIterations,
          ),
        ),
      );
      expect(turn.events.whereType<AssistantDone>(), isEmpty);
      expect(repo.isGenerating.value, isFalse);
      expect(
        repo.history.single.assistant,
        isEmpty,
        reason: 'a failed turn: its intents\' results were never the answer',
      );
    });
  });

  group('stop and cancel', () {
    setUp(() => repo.open(kVoiceChatProfile, skills: skills));

    test('a cancelled subscription stops native generation and holds '
        'isGenerating until the agent stream has drained', () async {
      final gate = Completer<void>();
      model.script.add(
        ToolCallTurn('loadSkill', const {
          'skillName': 'current-time',
        }, gate: gate),
      );
      final turn = AgentTurn(repo, 'Tell me the time');
      await settle();
      session().closeOnCancel = false; // native ends the turn a bit later
      expect(session().streaming, isTrue);

      await turn.cancel();
      await settle();

      expect(session().stopCalls, greaterThanOrEqualTo(1));
      expect(session().nativeCancels, 1);
      expect(
        repo.isGenerating.value,
        isTrue,
        reason: 'the native turn has not ended yet',
      );
      final busy = await repo.ask('Next').toList();
      expect(
        (busy.single as AssistantFailed).error,
        isA<ConversationNotReadyException>(),
      );

      session().finish(); // native CANCELLED arrives
      await settle();
      expect(repo.isGenerating.value, isFalse);
    });

    test('Stop while the intent runs: the tool finishes, the turn ends '
        'stopped, and the next ask rebuilds the chat first (stale tool '
        'tail)', () async {
      deviceGate = Completer<void>();
      model.script.addAll([
        const ToolCallTurn('loadSkill', {'skillName': 'device-info'}),
        const ToolCallTurn('runIntent', {
          'intent': 'device_info',
          'parameters': '{}',
        }),
      ]);
      final turn = AgentTurn(repo, 'Which hardware is this?');
      await settle();
      expect(turn.steps.last, isA<IntentCalled>(), reason: 'tool running');

      final stopping = repo.stop();
      await settle();
      deviceGate!.complete();
      await stopping;
      await turn.done;

      expect(turn.metrics.stopped, isTrue);
      expect(turn.deltas, isEmpty);
      expect(turn.steps.last, isA<IntentSucceeded>(), reason: 'the tool ran');
      expect(deviceCalls, 1);
      expect(repo.isGenerating.value, isFalse);

      // The next ask rebuilds before sending.
      final chatsBefore = model.chats.length;
      model.script.add(const TextTurn(['Hello.']));
      final next = AgentTurn(repo, 'Hi');
      await next.done;
      expect(model.chats.length, chatsBefore + 1);
      expect(model.chats.last.tools, hasLength(2));
      expect(next.deltas.join(), 'Hello.');
      // The model forgot the history: the UI is told why.
      expect(
        next.events.first,
        isA<AssistantContextReset>().having(
          (e) => e.reason,
          'reason',
          ContextResetReason.interruptedSkill,
        ),
      );
      expect(next.events.whereType<AssistantContextReset>(), hasLength(1));
      expect(next.metrics.contextReset, isTrue);
      expect(
        next.metrics.contextResetReason,
        ContextResetReason.interruptedSkill,
      );
      final users = [
        for (final m in session().queries)
          if (m.isUser) m.text,
      ];
      expect(users, ['Hi'], reason: 'no stale tool response before it');
    });

    test('Stop during the generation right after loadSkill: the stop flag '
        "is up before native stops (the package's stop rule), so the stopped "
        'empty generation is not taken for an empty answer: no retry, no '
        'nudge, the turn ends stopped; the next turn rebuilds the chat '
        '(0.3.0 ends the cut run with no event)', () async {
      final gate = Completer<void>();
      model.script.addAll([
        const ToolCallTurn('loadSkill', {'skillName': 'current-time'}),
        TextTurn(const ['It is noon.'], gate: gate),
        const TextTurn(['A retry would read this.']),
      ]);
      final turn = AgentTurn(repo, 'What time is it');
      // A native stop that returns after the cut reply ended: with the flag
      // set after it, the loop would see an empty answer and retry.
      session().stopReturnsAfterTheEnd = true;
      // The loadSkill round ran; the next generation streams, still empty.
      for (
        var i = 0;
        i < 200 && !(session().responseRequests == 2 && session().streaming);
        i++
      ) {
        await Future<void>.delayed(Duration.zero);
      }
      expect(session().responseRequests, 2);
      expect(session().streaming, isTrue);

      await repo.stop();
      await turn.done;
      gate.complete();
      await settle();

      expect(session().nativeCancels, 1);
      expect(turn.steps.whereType<EmptyAnswerRetried>(), isEmpty);
      expect(session().responseRequests, 2, reason: 'no generation after it');
      expect(
        [
          for (final m in session().queries)
            if (m.isUser && m.type != MessageType.toolResponse) m.text,
        ],
        ['What time is it'],
        reason: 'no nudge was sent',
      );
      expect(turn.deltas, isEmpty);
      expect(turn.last, isA<AssistantDone>());
      expect(turn.metrics.stopped, isTrue);

      // flutter_edge_ai_agent 0.3.0 ends this cut run with no event, as it
      // ends a cancel seen at a poll with input staged: the app cannot tell
      // the two apart, so the next turn rebuilds and says the history went.
      model.script
        ..clear()
        ..add(const TextTurn(['Hello.']));
      final chatsBefore = model.chats.length;
      final next = AgentTurn(repo, 'Hi');
      await next.done;
      expect(model.chats.length, chatsBefore + 1);
      expect(
        next.events.first,
        isA<AssistantContextReset>().having(
          (e) => e.reason,
          'reason',
          ContextResetReason.interruptedSkill,
        ),
      );
      expect(next.deltas.join(), 'Hello.');
    });

    test('Stop mid tool-call generation: the partial JSON core surfaces as '
        'text is dropped; nothing is spoken', () async {
      final midGate = Completer<void>();
      model.script.add(
        ToolCallTurn('loadSkill', const {
          'skillName': 'current-time',
        }, midGate: midGate),
      );
      final turn = AgentTurn(repo, 'What time is it');
      await settle();

      await repo.stop();
      await turn.done;

      expect(session().nativeCancels, 1);
      expect(turn.deltas, isEmpty, reason: 'no "{"role":… leak');
      expect(turn.metrics.stopped, isTrue);

      // The cut generation consumed the staged prompt and staged nothing:
      // the next ask keeps the chat (and its history).
      final chatsBefore = model.chats.length;
      model.script.add(const TextTurn(['Ok.']));
      await AgentTurn(repo, 'Never mind').done;
      expect(model.chats.length, chatsBefore);
    });

    test('Stop after the prompt is staged but before the first generation: '
        'core returns at its first isCancelled poll, leaving the prompt '
        'staged, so the next ask rebuilds first', () async {
      session().queryGate = Completer<void>();
      final turn = AgentTurn(repo, 'What time is it');
      await settle();

      final stopping = repo.stop();
      await settle();
      session().queryGate!.complete();
      await stopping;
      await turn.done;

      expect(turn.metrics.stopped, isTrue);
      expect(session().responseRequests, 0, reason: 'never generated');
      final chatsBefore = model.chats.length;
      model.script.add(const TextTurn(['Hello.']));
      await AgentTurn(repo, 'Hi').done;
      expect(model.chats.length, chatsBefore + 1);
      expect(
        [
          for (final m in session().queries)
            if (m.isUser) m.text,
        ],
        ['Hi'],
        reason: '"What time is it" is not glued onto it',
      );
    });

    test('a failed agent turn is followed by a rebuild the next turn '
        'announces as an interrupted skill call', () async {
      model.script.addAll([
        const ToolCallTurn('loadSkill', {'skillName': 'current-time'}),
        FailTurn(StateError('decode failed')),
      ]);
      final failed = AgentTurn(repo, 'What time is it');
      await failed.done;
      expect(failed.last, isA<AssistantFailed>());
      final chatsBefore = model.chats.length;
      model.script.add(const TextTurn(['Hello.']));

      final next = AgentTurn(repo, 'Hi');
      await next.done;

      expect(model.chats.length, chatsBefore + 1);
      expect(
        next.events.first,
        isA<AssistantContextReset>().having(
          (e) => e.reason,
          'reason',
          ContextResetReason.interruptedSkill,
        ),
      );
    });

    test('a rebuild after an interrupted skill call, then a question too '
        'long for any chat: the reset is announced as not replayed (no '
        'counts claimed), and the next question replays the history', () async {
      model.script.add(const TextTurn(['Hello there.']));
      await AgentTurn(repo, 'Hello').done;
      model.script.addAll([
        const ToolCallTurn('loadSkill', {'skillName': 'current-time'}),
        FailTurn(StateError('decode failed')),
      ]);
      await AgentTurn(repo, 'What time is it').done;
      final chatsBefore = model.chats.length;

      final tooLong = AgentTurn(repo, 'word ' * 5000);
      await tooLong.done;

      expect(model.chats.length, chatsBefore + 1, reason: 'rebuilt first');
      final reset = tooLong.events.whereType<AssistantContextReset>().single;
      expect(reset.reason, ContextResetReason.interruptedSkill);
      expect(reset.replayed, isFalse);
      expect(
        (tooLong.last as AssistantFailed).error,
        isA<ConversationTooLongException>(),
      );

      model.script.add(const TextTurn(['Hi again.']));
      final next = AgentTurn(repo, 'Hi');
      await next.done;

      expect(next.deltas.join(), 'Hi again.');
      expect(
        next.events.whereType<AssistantContextReset>(),
        isEmpty,
        reason: 'every earlier turn fit the replay: nothing more to say',
      );
      expect(model.chats.length, chatsBefore + 1, reason: 'no second rebuild');
      final users = [
        for (final m in session().queries)
          if (m.isUser) m.text,
      ];
      expect(
        users.join('\n'),
        contains('Hello there.'),
        reason: 'the replay went out with this question',
      );
    });

    test('after too many tool rounds the last tool responses are staged: '
        'the next ask rebuilds first', () async {
      for (var i = 0; i < kAgentMaxIterations; i++) {
        model.script.add(
          const ToolCallTurn('runIntent', {
            'intent': 'current_time',
            'parameters': '{}',
          }),
        );
      }
      await AgentTurn(repo, 'Loop').done;
      final chatsBefore = model.chats.length;
      model.script.add(const TextTurn(['Ok.']));

      await AgentTurn(repo, 'Hi').done;

      expect(model.chats.length, chatsBefore + 1);
    });

    test('Stop during the answer: the text so far, stopped, no rebuild '
        'needed (the tool response was consumed)', () async {
      final gate = Completer<void>();
      model.script.addAll([
        const ToolCallTurn('runIntent', {
          'intent': 'current_time',
          'parameters': '{}',
        }),
        TextTurn(const ['It is ', 'noon.'], gate: gate),
      ]);
      final turn = AgentTurn(repo, 'Time?');
      await settle();
      gate.complete();
      // Let the first token through, then stop.
      while (turn.deltas.isEmpty) {
        await Future<void>.delayed(Duration.zero);
      }
      await repo.stop();
      await turn.done;

      expect(turn.metrics.stopped, isTrue);
      expect(turn.deltas.first, 'It is ');
      final chatsBefore = model.chats.length;
      model.script.add(const TextTurn(['Ok.']));
      await AgentTurn(repo, 'Thanks').done;
      expect(model.chats.length, chatsBefore, reason: 'no rebuild');
    });
  });

  group('context budget', () {
    setUp(() => repo.open(kVoiceChatProfile, skills: skills));

    test('the guard reserves the tool rounds and the largest skill body: a '
        'chat that would fit a plain turn is started over first', () async {
      model.script.add(const TextTurn(['Hi.']));
      await AgentTurn(repo, 'Hello').done;
      // Plain need: prompt + 384 reply + 32 overhead ≈ 420. Leave room for
      // that but not for the tool reserve (2 × 192 + the largest skill).
      final limit = model.maxTokens - kContextHeadroomTokens;
      session().nativeInputTokens = limit - 600;
      session().nativeOutputTokens = 0;
      final chatsBefore = model.chats.length;
      model.script.add(const TextTurn(['Fresh.']));

      final turn = AgentTurn(repo, 'Tell me the time');
      await turn.done;

      expect(
        turn.events.first,
        isA<AssistantContextReset>().having(
          (e) => e.reason,
          'reason',
          ContextResetReason.budget,
        ),
      );
      expect(turn.metrics.contextResetReason, ContextResetReason.budget);
      expect(turn.metrics.contextReset, isTrue);
      expect(model.chats.length, chatsBefore + 1);
      expect(model.chats.last.tools, hasLength(2), reason: 'still an agent');
    });
  });
}
