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

// Characterisation of EdgeAiConversationRepository's two turn paths, ahead
// of merging them: the plain chat (`askTurn`'s own body in
// conversation/turn.dart) and the agent chat (`_runAgentTurn` in
// conversation/agent_turn.dart). Each scenario runs on both, over the same
// fake engine (FakeToolModel: a plain chat is the same model without tools),
// and pins what a caller sees: the events, the metrics fields, the failures.
// Where the paths differ today, the test says so per path.
import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart' hide ModelSpec;
import 'package:flutter_edge_ai_agent/flutter_edge_ai_agent.dart'
    show Skill, parseSkillMd;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/conversation_repository.dart';
import 'package:litert_edge_demos/data/repositories/conversation_repository_edge_ai.dart';
import 'package:litert_edge_demos/data/services/llm/llm_service.dart';
import 'package:litert_edge_demos/domain/models/assistant_event.dart';
import 'package:litert_edge_demos/domain/models/chat_model_config.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../fakes/fake_tool_model.dart';

Skill _bundled(String name) =>
    parseSkillMd(File('assets/skills/$name/SKILL.md').readAsStringSync());

final _skills = [_bundled('current-time'), _bundled('device-info')];

enum _Path { plain, agent }

/// [FakeToolModel] whose session creation can be held ([createGate]): a
/// chat rebuild (budget guard, interrupted skill) then stays in progress.
class _GatedToolModel extends FakeToolModel {
  Completer<void>? createGate;

  @override
  Future<InferenceModelSession> createSession({
    double temperature = .8,
    int randomSeed = 1,
    int topK = 1,
    double? topP,
    String? loraPath,
    bool? enableVisionModality,
    bool? enableAudioModality,
    String? systemInstruction,
    bool enableThinking = false,
    List<Tool> tools = const [],
    int? maxOutputTokens,
  }) async {
    await createGate?.future;
    return super.createSession(
      temperature: temperature,
      randomSeed: randomSeed,
      topK: topK,
      topP: topP,
      loraPath: loraPath,
      enableVisionModality: enableVisionModality,
      enableAudioModality: enableAudioModality,
      systemInstruction: systemInstruction,
      enableThinking: enableThinking,
      tools: tools,
      maxOutputTokens: maxOutputTokens,
    );
  }
}

/// One `ask` stream, collected.
final class _Turn {
  _Turn(ConversationRepository repo, String prompt, {Uint8List? image}) {
    done = repo.ask(prompt, image: image).listen(events.add).asFuture<void>();
  }

  final List<AssistantEvent> events = [];
  late final Future<void> done;

  List<String> get deltas => [
    for (final e in events)
      if (e case AssistantTextDelta(:final text)) text,
  ];

  GenerationMetrics get metrics => switch (events.last) {
    AssistantDone(:final metrics) => metrics,
    final other => throw StateError('last event is $other'),
  };
}

/// The metrics fields that do not depend on wall-clock time.
Map<String, Object?> _fields(GenerationMetrics m) => {
  'chunks': m.chunks,
  'stopped': m.stopped,
  'tokensPerSecondSource': m.tokensPerSecondSource,
  'imageAttached': m.imageAttached,
  'imageSent': m.imageSent,
  'imageResent': m.imageResent,
  'contextReset': m.contextReset,
  'contextResetReason': m.contextResetReason,
  'prefillTokens': m.prefillTokens,
  'promptTokens': m.promptTokens,
  'toolRounds': m.toolRounds,
  'skillSteps': m.skillSteps.length,
};

/// [n] tokens for the fake tokenizer (length / 4).
String _tokens(int n) => 'abcd' * n;

/// Lets queued microtasks and the scripted session's zero timers run.
Future<void> settle([int rounds = 20]) async {
  for (var i = 0; i < rounds; i++) {
    await Future<void>.delayed(Duration.zero);
  }
}

void main() {
  late _GatedToolModel model;
  late EdgeAiConversationRepository repo;

  FakeToolSession session() => model.lastSession;
  final limit = 4096 - kContextHeadroomTokens; // FakeToolModel.maxTokens
  final cats = Uint8List.fromList([0x89, 0x50, 0x4E, 0x47, 1]);

  Future<EdgeAiConversationRepository> repoOn(ChatModelConfig config) async {
    final llm = LlmService(loadModel: (_) async => model);
    expect(await llm.load(config), isA<Ok<LlmInfo>>());
    return EdgeAiConversationRepository(
      llm: llm,
      stopTimeout: const Duration(seconds: 2),
    );
  }

  Future<void> open(EdgeAiConversationRepository r, _Path path) async {
    final skills = path == _Path.agent ? _skills : const <Skill>[];
    expect(await r.open(kVoiceChatProfile, skills: skills), isA<Ok<void>>());
    expect(r.hasSkills, path == _Path.agent);
  }

  setUp(() async {
    model = _GatedToolModel();
    repo = await repoOn(kDefineChatModel);
  });

  tearDown(() => repo.close());

  for (final path in _Path.values) {
    group('${path.name} path', () {
      setUp(() => open(repo, path));

      test('an answer: deltas in order, then exactly one AssistantDone; the '
          'turn ends idle', () async {
        model.script.add(const TextTurn(['Hel', 'lo.']));

        final turn = _Turn(repo, _tokens(5));
        await turn.done;

        expect(turn.deltas, ['Hel', 'lo.']);
        expect(turn.events, hasLength(3));
        expect(turn.events.last, isA<AssistantDone>());
        expect(_fields(turn.metrics), {
          'chunks': 2,
          'stopped': false,
          'tokensPerSecondSource': TokenRateSource.chunks,
          'imageAttached': false,
          'imageSent': false,
          'imageResent': null,
          'contextReset': false,
          'contextResetReason': ContextResetReason.budget,
          'prefillTokens': null,
          'promptTokens': 5,
          'toolRounds': 0,
          'skillSteps': 0,
        });
        expect(turn.metrics.timeToFirstToken, isNotNull);
        expect(turn.metrics.stopLatency, isNull);
        expect(turn.metrics.contextTokens, isNotNull);
        expect(repo.isGenerating.value, isFalse);
        expect(
          session().queries.where((m) => m.isUser).single.text,
          _tokens(5),
        );
      });

      test('an image: sent with the user message, then in context and not '
          'sent again; the first image turn also counts the system '
          'instruction in promptTokens (on the agent path its system prompt '
          'and the tool declarations)', () async {
        model.script.addAll([
          const TextTurn(['A cat.']),
          const TextTurn(['Two.']),
        ]);

        final first = _Turn(repo, _tokens(5), image: cats);
        await first.done;
        final second = _Turn(repo, _tokens(3), image: cats);
        await second.done;

        final users = session().queries.where((m) => m.isUser).toList();
        expect(users.first.imageBytes, same(cats));
        expect(users.last.imageBytes, isNull, reason: 'already in context');
        expect(repo.imageInContext, same(cats));
        final m1 = first.metrics;
        expect(
          (m1.imageAttached, m1.imageSent, m1.imageResent),
          (true, true, null),
        );
        final systemTokens = switch (path) {
          _Path.plain =>
            (kVoiceChatProfile.systemInstruction.length / 4).ceil(),
          _Path.agent =>
            (model.chats.last.systemInstruction!.length / 4).ceil() +
                kToolDeclarationTokens,
        };
        expect(m1.promptTokens, 5 + systemTokens);
        final m2 = second.metrics;
        expect((m2.imageAttached, m2.imageSent), (true, false));
        expect(m2.promptTokens, 3);
      });

      group('preconditions: one AssistantFailed, nothing reaches the '
          'engine, idle', () {
        test('the conversation is closed', () async {
          await repo.close();

          final events = await repo.ask('Hi').toList();

          expect(
            events.single,
            isA<AssistantFailed>().having(
              (e) => e.error,
              'error',
              isA<ConversationNotReadyException>().having(
                (e) => '$e',
                'message',
                contains('The chat is not open'),
              ),
            ),
          );
          expect(session().queries, isEmpty);
        });

        test('a reply is already being generated', () async {
          model.script.add(TextTurn(const ['One.'], gate: Completer<void>()));
          final first = _Turn(repo, 'one');
          await settle();
          expect(repo.isGenerating.value, isTrue);

          final events = await repo.ask('two').toList();

          expect(
            events.single,
            isA<AssistantFailed>().having(
              (e) => '${e.error}',
              'error',
              contains('A reply is already being generated'),
            ),
          );
          expect(
            [
              for (final m in session().queries)
                if (m.isUser) m.text,
            ],
            ['one'],
          );
          expect(repo.isGenerating.value, isTrue, reason: 'the first goes on');
          await repo.stop();
          await first.done;
        });

        test('while the budget guard rebuilds the chat a second ask is told '
            'a reply is already being generated, not that the chat is not '
            'open', () async {
          model.script.add(const TextTurn(['Fresh.']));
          session().nativeInputTokens = limit - 100;
          final gate = model.createGate = Completer<void>();
          final first = _Turn(repo, _tokens(5));
          await settle();
          expect(repo.isOpen, isFalse, reason: 'the old chat is closed');

          final events = await repo.ask('two').toList();

          expect(
            (events.single as AssistantFailed).error,
            isA<ConversationNotReadyException>().having(
              (e) => '$e',
              'message',
              contains('A reply is already being generated'),
            ),
          );
          gate.complete();
          await first.done;
          expect(first.deltas, ['Fresh.']);
          expect(first.metrics.contextReset, isTrue);
        });

        test('an image on a chat built without image support', () async {
          final noImages = await repoOn(
            const ChatModelConfig(
              name: 'No images',
              modelType: ModelType.gemma4,
              llm: LlmConfig(
                maxTokens: 4096,
                backend: PreferredBackend.gpu,
                supportImage: false,
                maxNumImages: 1,
              ),
              tools: true,
            ),
          );
          addTearDown(noImages.close);
          await open(noImages, path);

          final events = await noImages
              .ask('What is this?', image: cats)
              .toList();

          expect(
            (events.single as AssistantFailed).error,
            isA<ConversationImageUnsupportedException>(),
          );
          expect(session().queries, isEmpty);
          expect(noImages.isGenerating.value, isFalse);
        });
      });

      test('a prompt that fits a fresh chat only without its image fails as '
          'too long with it: no rebuild, nothing sent; without the image the '
          'same prompt runs', () async {
        // What a fresh chat must hold besides the prompt: the reply, the
        // turn overhead and, on an agent chat, the tool rounds (with the
        // largest skill body) and the system prompt with the tools.
        var reserve = kVoiceChatProfile.maxOutputTokens + kTurnOverheadTokens;
        if (path == _Path.agent) {
          final largestSkill = _skills
              .map(
                (s) =>
                    ('${s.name}\n${s.description}\n${s.instructions}'.length /
                            4)
                        .ceil(),
              )
              .reduce((a, b) => a > b ? a : b);
          final system =
              (model.chats.last.systemInstruction!.length / 4).ceil() +
              kToolDeclarationTokens;
          reserve +=
              kAgentToolRounds * (kToolCallTokens + kToolResultTokens) +
              largestSkill +
              system;
        }
        // Half the image allowance below the limit without the image.
        final prompt = _tokens(limit - reserve - kImageTokenAllowance ~/ 2);
        final chats = model.chats.length;

        final refused = await repo.ask(prompt, image: cats).toList();

        expect(
          (refused.single as AssistantFailed).error,
          isA<ConversationTooLongException>(),
        );
        expect(model.chats.length, chats, reason: 'no rebuild');
        expect(session().queries, isEmpty);
        expect(repo.isGenerating.value, isFalse);

        model.script.add(const TextTurn(['Ok.']));
        final accepted = _Turn(repo, prompt);
        await accepted.done;
        expect(accepted.deltas, ['Ok.']);
        expect(accepted.metrics.contextReset, isFalse);
      });

      test('a stop during the budget reset, before anything is sent: '
          'AssistantContextReset, then AssistantDone(stopped) with no text '
          'and no rates; nothing sent; the next turn runs on the fresh chat '
          'without another reset', () async {
        session().nativeInputTokens = limit - 100;
        final gate = model.createGate = Completer<void>();
        final turn = _Turn(repo, _tokens(5), image: cats);
        await settle();
        expect(repo.isOpen, isFalse, reason: 'the reset is under way');

        final stopping = repo.stop();
        await settle();
        gate.complete();
        await stopping;
        await turn.done;

        expect(turn.events, hasLength(2));
        expect(
          turn.events.first,
          isA<AssistantContextReset>().having(
            (e) => e.reason,
            'reason',
            ContextResetReason.budget,
          ),
        );
        expect(_fields(turn.metrics), {
          'chunks': 0,
          'stopped': true,
          'tokensPerSecondSource': TokenRateSource.chunks,
          'imageAttached': true,
          'imageSent': false,
          'imageResent': null,
          'contextReset': true,
          'contextResetReason': ContextResetReason.budget,
          'prefillTokens': null,
          'promptTokens': 5,
          'toolRounds': 0,
          'skillSteps': 0,
        });
        final m = turn.metrics;
        expect(m.timeToFirstToken, isNull);
        expect(m.tokensPerSecond, isNull);
        expect(m.contextTokens, isNull);
        expect(m.stopLatency, isNotNull);
        expect(session().queries, isEmpty);
        expect(session().responseRequests, 0);
        expect(repo.isOpen, isTrue);
        expect(repo.isGenerating.value, isFalse);
        expect(repo.imageInContext, isNull);

        final chats = model.chats.length;
        model.script.add(const TextTurn(['Two.']));
        final next = _Turn(repo, _tokens(3), image: cats);
        await next.done;
        expect(model.chats.length, chats);
        expect(next.events.whereType<AssistantContextReset>(), isEmpty);
        expect(next.metrics.imageSent, isTrue);
      });

      test('a native error mid-reply: the text so far, then AssistantFailed '
          'last; the next turn on the plain path keeps the chat, on the agent '
          'path rebuilds it and says so (interrupted skill)', () async {
        model.script.add(FailTurn(StateError('decode failed')));

        final failed = _Turn(repo, 'Hi');
        await failed.done;

        expect(
          failed.events.last,
          isA<AssistantFailed>().having(
            (e) => '${e.error}',
            'error',
            contains('decode failed'),
          ),
        );
        expect(failed.events.whereType<AssistantDone>(), isEmpty);
        expect(repo.isGenerating.value, isFalse);

        final chats = model.chats.length;
        model.script.add(const TextTurn(['Hello.']));
        final next = _Turn(repo, 'Again');
        await next.done;
        expect(next.deltas, ['Hello.']);
        switch (path) {
          case _Path.plain:
            expect(model.chats.length, chats);
            expect(next.events.whereType<AssistantContextReset>(), isEmpty);
            expect(next.metrics.contextReset, isFalse);
          case _Path.agent:
            expect(model.chats.length, chats + 1);
            expect(
              next.events.first,
              isA<AssistantContextReset>().having(
                (e) => e.reason,
                'reason',
                ContextResetReason.interruptedSkill,
              ),
            );
            expect(
              next.metrics.contextResetReason,
              ContextResetReason.interruptedSkill,
            );
        }
      });
    });
  }

  group('agent path only', () {
    setUp(() => open(repo, _Path.agent));

    test('a stop during the interrupted-skill rebuild: the reset is announced '
        'with its reason and the stopped Done carries it; the next turn needs '
        'no rebuild', () async {
      model.script.add(FailTurn(StateError('decode failed')));
      await _Turn(repo, 'Hi').done; // leaves the staged input behind
      final gate = model.createGate = Completer<void>();
      final turn = _Turn(repo, _tokens(4));
      await settle();

      final stopping = repo.stop();
      await settle();
      gate.complete();
      await stopping;
      await turn.done;

      expect(
        turn.events.first,
        isA<AssistantContextReset>().having(
          (e) => e.reason,
          'reason',
          ContextResetReason.interruptedSkill,
        ),
      );
      expect(turn.events, hasLength(2));
      final m = turn.metrics;
      expect(
        (m.stopped, m.chunks, m.contextReset, m.contextResetReason),
        (true, 0, true, ContextResetReason.interruptedSkill),
      );
      expect(m.promptTokens, 4);
      expect(session().queries, isEmpty);

      final chats = model.chats.length;
      model.script.add(const TextTurn(['Ok.']));
      final next = _Turn(repo, 'Again');
      await next.done;
      expect(model.chats.length, chats, reason: 'the stale tail is gone');
      expect(next.events.whereType<AssistantContextReset>(), isEmpty);
    });
  });
}
