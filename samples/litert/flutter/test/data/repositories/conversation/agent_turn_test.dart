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

import 'package:flutter_edge_ai/flutter_edge_ai.dart' show InferenceChat;
import 'package:flutter_edge_ai_agent/flutter_edge_ai_agent.dart'
    show
        AgentEvent,
        AgentSession,
        EmptyAnswerEvent,
        Skill,
        SkillLoadEvent,
        SkillRegistry,
        parseSkillMd;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/conversation/agent_turn.dart';
import 'package:litert_edge_demos/data/repositories/conversation/live_chat.dart';
import 'package:litert_edge_demos/data/repositories/conversation/native_chat.dart';
import 'package:litert_edge_demos/data/repositories/conversation_repository.dart';
import 'package:litert_edge_demos/data/services/llm/llm_service.dart';
import 'package:litert_edge_demos/domain/models/assistant_event.dart';
import 'package:litert_edge_demos/domain/models/skill_step.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../../fakes/fake_tool_model.dart';

Skill _bundled(String name) =>
    parseSkillMd(File('assets/skills/$name/SKILL.md').readAsStringSync());

/// An agent session whose turn is a scripted [AgentEvent] stream, given the
/// turn's `isCancelled` poll: the events the agent loop would emit, with a
/// test's own step between any two.
final class _ScriptedSession extends AgentSession {
  _ScriptedSession(InferenceChat chat, this._script)
    : super(chat: chat, registry: SkillRegistry(), executors: const []);

  final Stream<AgentEvent> Function(bool Function() isCancelled) _script;

  @override
  Stream<AgentEvent> ask(
    String userMessage, {
    bool Function()? isCancelled,
    Uint8List? imageBytes,
    Uint8List? audioBytes,
  }) => _script(isCancelled ?? () => false);
}

void main() {
  late FakeToolModel model;
  late LiveChat live;
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
    expect(await live.rebuild(kVoiceChatProfile, skills), isA<Ok<void>>());
  });

  tearDown(() => live.close());

  /// [live]'s agent chat with its session replaced by [script].
  AgentChat scripted(
    Stream<AgentEvent> Function(bool Function() isCancelled) script,
  ) {
    final real = live.agent!;
    return AgentChat(
      _ScriptedSession(real.session.chat, script),
      real.systemPrompt,
      real.skills,
    );
  }

  Future<List<AssistantEvent>> turn(AgentChat agent) =>
      askAgent(live, agent, 'What time is it?', null, (_) {}).toList();

  group('no answer after a loaded skill (EmptyAnswerEvent)', () {
    test('without a stop: the turn fails with an '
        'EmptyAnswerAfterSkillException naming the skill', () async {
      final events = await turn(
        scripted((_) async* {
          yield const SkillLoadEvent('current-time');
          yield const EmptyAnswerEvent('current-time');
        }),
      );

      expect(
        events.last,
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
    });

    test('a stop that lands after the loop\'s last isCancelled poll, before '
        'the turn reads how it ended: the turn ends stopped, not failed as '
        'an empty answer', () async {
      bool? polledBeforeStop;
      Future<void>? stopping;
      final events = await turn(
        scripted((isCancelled) async* {
          yield const SkillLoadEvent('current-time');
          // The loop's last poll, right before it gives up on an answer.
          polledBeforeStop = isCancelled();
          yield const EmptyAnswerEvent('current-time');
          // After that poll, before the stream ends (and so before the turn
          // reads how it ended).
          stopping = live.stop();
        }),
      );
      await stopping;

      expect(polledBeforeStop, isFalse, reason: 'no stop at the last poll');
      expect(stopping, isNotNull);
      expect(events.whereType<AssistantFailed>(), isEmpty);
      final done = events.last as AssistantDone;
      expect(done.metrics.stopped, isTrue);
      expect(done.metrics.skillSteps.single, isA<SkillLoaded>());
      expect(live.isGenerating.value, isFalse);
    });
  });
}
