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
import 'package:litert_edge_demos/domain/skills/app_intent_executor.dart';
import 'package:litert_edge_demos/domain/skills/app_intent_handlers.dart';
import 'package:litert_edge_demos/domain/skills/app_intents.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../fakes/fake_tool_model.dart';

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
    final executor = AppIntentExecutor({
      ...intents,
      AppIntent.deviceInfo: AppIntentSpec(
        usage: deviceInfo.usage,
        handler: (params) async {
          // A test can hold the executor (a tool still running at Stop).
          await deviceGate?.future;
          deviceCalls++;
          return await deviceInfo.handler(params);
        },
      ),
    });
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
      // Our wording, the package's names and parameters.
      expect(args.tools.last.description, contains('loaded skill'));
      for (final (ours, theirs) in [
        (args.tools.first, loadSkillTool),
        (args.tools.last, runIntentTool),
      ]) {
        expect(ours.parameters['required'], theirs.parameters['required']);
        expect(
          (ours.parameters['properties'] as Map).keys,
          (theirs.parameters['properties'] as Map).keys,
        );
      }
      expect(args.supportsFunctionCalls, isTrue);
      // The chat inherits the installed type (gemma4 for Gemma 4 E2B);
      // a chat-only type would disagree with the native session's.
      expect(args.modelType, isNull);
      expect(args.toolChoice, ToolChoice.auto);
      expect(args.supportImage, isTrue);
      expect(args.temperature, kSampler.temperature, reason: 'not .8');
      expect(args.topK, kSampler.topK, reason: 'not fromModel\'s 1');
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
              supportImage: false,
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

    test('an ErrorResult is one failed step, not two (agent_loop.dart:379), '
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
      expect(failed.single.message, contains(AppIntent.listed));
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
