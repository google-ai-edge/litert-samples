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

import 'package:flutter_edge_ai_agent/flutter_edge_ai_agent.dart'
    show Skill, parseSkillMd;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/conversation/native_chat.dart';
import 'package:litert_edge_demos/data/services/llm/llm_service.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../../fakes/fake_tool_model.dart';

Skill _bundled(String name) =>
    parseSkillMd(File('assets/skills/$name/SKILL.md').readAsStringSync());

/// [text]'s size for the fake tokenizer (length / 4).
int _tokens(String text) => (text.length / 4).ceil();

void main() {
  late FakeToolModel model;
  late LlmService llm;
  late ChatFactory chats;
  final skills = [_bundled('current-time'), _bundled('device-info')];

  setUp(() async {
    model = FakeToolModel();
    llm = LlmService(loadModel: (_) async => model);
    chats = ChatFactory(
      llm: llm,
      sampler: kSampler,
      executors: const [],
      maxIterations: kAgentMaxIterations,
      agentTools: kAgentTools,
    );
  });

  test('without a loaded chat model the capabilities say so', () {
    final none = chats.capabilities;

    expect(none.modelName, 'no chat model');
    expect((none.images, none.tools), (false, false));
  });

  test('a profile with a skills template and skills gets an agent chat; '
      'without skills, or without a template, a plain one', () async {
    expect(await llm.load(kDefineChatModel), isA<Ok<LlmInfo>>());

    final (_, agent) = await chats.build(kVoiceChatProfile, skills);
    expect(agent, isNotNull);
    expect(agent!.skills.map((s) => s.name), ['current-time', 'device-info']);
    expect(model.chats.last.tools, hasLength(kAgentTools.length));
    expect(model.chats.last.systemInstruction, agent.systemPrompt);

    final (_, noSkills) = await chats.build(kVoiceChatProfile, const []);
    expect(noSkills, isNull);
    final (_, camera) = await chats.build(kCameraProfile, skills);
    expect(camera, isNull);
    expect(model.chats.last.tools, isEmpty);
    expect(
      model.chats.last.systemInstruction,
      kCameraProfile.systemInstruction,
    );
  });

  test('the agent reserve: the system prompt with the tool declarations, and '
      'the tool rounds with the largest skill body', () async {
    await llm.load(kDefineChatModel);
    final (chat, agent) = await chats.build(kVoiceChatProfile, skills);

    final reserve = await agent!.reserve(chat);

    final largestSkill = skills
        .map((s) => _tokens('${s.name}\n${s.description}\n${s.instructions}'))
        .reduce((a, b) => a > b ? a : b);
    expect(
      reserve.system,
      _tokens(agent.systemPrompt) + kToolDeclarationTokens,
    );
    expect(
      reserve.rounds,
      kAgentToolRounds * (kToolCallTokens + kToolResultTokens) + largestSkill,
    );
  });

  test('session metrics come from the chat\'s live session', () async {
    await llm.load(kDefineChatModel);
    final chat = await chats.plain(kCameraProfile);
    model.lastSession.nativeInputTokens = 42;

    expect(sessionMetricsOf(chat)?.inputTokens, 42);
  });
}
