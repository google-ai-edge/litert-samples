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

import 'package:flutter/foundation.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart'
    show ModelType, PreferredBackend;
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
import 'package:litert_edge_demos/utils/wav.dart';

import '../../fakes/fake_tool_model.dart';

Skill _bundled(String name) =>
    parseSkillMd(File('assets/skills/$name/SKILL.md').readAsStringSync());

final _skills = [_bundled('current-time'), _bundled('device-info')];

/// A small-context build with its audio encoder, as a 1280-token NPU build
/// has (the window comes from the fake model; the backend is the GPU, which
/// the test machine has).
const _withAudio = ChatModelConfig(
  name: 'Gemma 4 E2B',
  modelType: ModelType.gemma4,
  llm: LlmConfig(
    maxTokens: 1280,
    backend: PreferredBackend.gpu,
    maxNumImages: 1,
    supportImage: true,
    supportAudio: true,
  ),
  tools: true,
);

/// [n] tokens for the fake tokenizer (length / 4).
String _tokens(int n) => 'abcd' * n;

Matcher _fits(int? index) =>
    isA<Ok<int?>>().having((r) => r.value, 'value', index);

/// `ConversationRepository.firstFittingPrompt`: which version of a turn's
/// prompt (fewer and fewer excerpts) a fresh chat holds, by the budget
/// guard's own figures, over the real repository and the fake engine.
void main() {
  late FakeToolModel model;
  late EdgeAiConversationRepository repo;
  final cats = Uint8List.fromList([0x89, 0x50, 0x4E, 0x47, 1]);
  // 2 s of 16 kHz mono PCM16: 50 tokens at 25 a second.
  final wav = wavFromPcm16(Uint8List(64000), sampleRate: 16000);

  Future<EdgeAiConversationRepository> repoOn(
    ChatModelConfig config, {
    required int window,
  }) async {
    model = FakeToolModel(maxTokens: window);
    final llm = LlmService(loadModel: (_) async => model);
    expect(await llm.load(config), isA<Ok<LlmInfo>>());
    return EdgeAiConversationRepository(
      llm: llm,
      stopTimeout: const Duration(seconds: 2),
    );
  }

  tearDown(() => repo.close());

  group('a 1280-token window (a small-context NPU build), plain chat', () {
    // What a prompt may take: the window less the headroom, the reply
    // allowance and the turn overhead (608 tokens).
    const room =
        1280 -
        kContextHeadroomTokens -
        384 -
        kTurnOverheadTokens; // kVoiceChatProfile.maxOutputTokens: 384

    setUp(() async {
      expect(kVoiceChatProfile.maxOutputTokens, 384);
      repo = await repoOn(_withAudio, window: 1280);
      expect(await repo.open(kVoiceChatProfile), isA<Ok<void>>());
      expect(repo.hasSkills, isFalse);
    });

    test('the first prompt a fresh chat holds beside the reply allowance and '
        'the headroom', () async {
      expect(
        await repo.firstFittingPrompt([
          _tokens(room + 1),
          _tokens(room),
          _tokens(1),
        ]),
        _fits(1),
      );
      expect(await repo.firstFittingPrompt([_tokens(1)]), _fits(0));
    });

    test('the budget guard agrees: the prompt that fits runs, one token more '
        'fails as too long', () async {
      model.script.add(const TextTurn(['Ok.']));
      final ran = await repo.ask(_tokens(room)).toList();
      expect(ran.last, isA<AssistantDone>());

      // A fresh chat for the second one: the first one's turn is in this one.
      expect(await repo.open(kVoiceChatProfile), isA<Ok<void>>());
      final refused = await repo.ask(_tokens(room + 1)).toList();
      expect(
        (refused.single as AssistantFailed).error,
        isA<ConversationTooLongException>(),
      );
    });

    test('the turn that sends the prompt that fit reuses its count: one '
        'tokenizer call for it, not two', () async {
      final fits = _tokens(room);
      final other = _tokens(room - 10);
      expect(
        await repo.firstFittingPrompt([_tokens(room + 1), fits]),
        _fits(1),
      );
      model.script.add(const TextTurn(['Ok.']));
      expect((await repo.ask(fits).toList()).last, isA<AssistantDone>());
      expect(model.lastSession.counted.where((t) => t == fits), hasLength(1));

      // Used once: another prompt, or the same one again, is counted.
      model.script.add(const TextTurn(['Ok.']));
      expect((await repo.ask(other).toList()).last, isA<AssistantDone>());
      expect(model.lastSession.counted.where((t) => t == other), hasLength(1));
      expect(await repo.open(kVoiceChatProfile), isA<Ok<void>>());
      model.script.add(const TextTurn(['Ok.']));
      expect((await repo.ask(fits).toList()).last, isA<AssistantDone>());
      expect(model.lastSession.counted.where((t) => t == fits), hasLength(1));
    });

    test('an attached image counts its allowance', () async {
      final prompts = [
        _tokens(room - kImageTokenAllowance + 1),
        _tokens(room - kImageTokenAllowance),
      ];
      expect(await repo.firstFittingPrompt(prompts), _fits(0));
      expect(await repo.firstFittingPrompt(prompts, image: cats), _fits(1));
    });

    test('a recording counts 25 tokens a second', () async {
      final prompts = [_tokens(room - 49), _tokens(room - 50)];
      expect(await repo.firstFittingPrompt(prompts), _fits(0));
      expect(await repo.firstFittingPrompt(prompts, audioWav: wav), _fits(1));
    });

    test('none fits: null (the ask then fails as too long)', () async {
      expect(
        await repo.firstFittingPrompt([_tokens(room + 200), _tokens(room + 1)]),
        _fits(null),
      );
    });

    test('a recording that is not a 16 kHz mono PCM16 WAV: an error', () async {
      final result = await repo.firstFittingPrompt([
        _tokens(1),
      ], audioWav: Uint8List.fromList([1, 2, 3, 4]));
      expect(
        result,
        isA<Error<int?>>().having(
          (e) => e.error,
          'error',
          isA<FormatException>(),
        ),
      );
    });
  });

  test('while a turn runs: an error, nothing counted beside its '
      'generation', () async {
    repo = await repoOn(_withAudio, window: 1280);
    expect(await repo.open(kVoiceChatProfile), isA<Ok<void>>());
    final prefill = Completer<void>();
    model.script.add(TextTurn(const ['Ok.'], gate: prefill));
    final running = repo.ask(_tokens(1)).toList();
    await pumpEventQueue();
    expect(repo.isGenerating.value, isTrue);

    expect(
      await repo.firstFittingPrompt([_tokens(1)]),
      isA<Error<int?>>().having(
        (e) => '${e.error}',
        'error',
        contains('already being generated'),
      ),
    );

    prefill.complete();
    expect((await running).last, isA<AssistantDone>());
    expect(await repo.firstFittingPrompt([_tokens(1)]), _fits(0));
  });

  test('no chat open: an error', () async {
    repo = await repoOn(_withAudio, window: 1280);
    final result = await repo.firstFittingPrompt([_tokens(1)]);
    expect(
      result,
      isA<Error<int?>>().having(
        (e) => e.error,
        'error',
        isA<ConversationNotReadyException>(),
      ),
    );
  });

  test('an agent chat counts its skills reserve (the system prompt with the '
      'tools, the tool rounds and the largest skill)', () async {
    repo = await repoOn(_withAudio, window: 2048);
    expect(
      await repo.open(kVoiceChatProfile, skills: _skills),
      isA<Ok<void>>(),
    );
    expect(repo.hasSkills, isTrue);
    final largestSkill = _skills
        .map(
          (s) => ('${s.name}\n${s.description}\n${s.instructions}'.length / 4)
              .ceil(),
        )
        .reduce((a, b) => a > b ? a : b);
    final system =
        (model.chats.last.systemInstruction!.length / 4).ceil() +
        kToolDeclarationTokens;
    final rounds =
        kAgentToolRounds * (kToolCallTokens + kToolResultTokens) + largestSkill;
    final plainRoom =
        2048 -
        kContextHeadroomTokens -
        kVoiceChatProfile.maxOutputTokens -
        kTurnOverheadTokens;
    final agentRoom = plainRoom - system - rounds;
    expect(agentRoom, greaterThan(0));

    expect(
      await repo.firstFittingPrompt([
        _tokens(plainRoom),
        _tokens(agentRoom + 1),
        _tokens(agentRoom),
      ]),
      _fits(2),
    );
  });
}
