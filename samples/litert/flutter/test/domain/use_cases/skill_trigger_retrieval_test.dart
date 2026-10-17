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

import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/knowledge_config.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/domain/use_cases/prompt_builder.dart';

import '../../fakes/fake_knowledge.dart';

/// Retrieval runs before every Demo 1 turn, so a skill question can arrive
/// with knowledge-base excerpts. Measured with the app's own embedding
/// pipeline (real EmbeddingGemma 2 numbers in
/// `test_assets/skill_trigger_similarity.json`, printed by
/// `integration_test/kb_retrieval_test.dart`):
///
/// - time triggers score 0.62–0.64: below the 0.70 gate, so their prompts
///   carry no excerpts;
/// - device questions score 0.68–0.77 against the GPU/backend docs: 5 of 6
///   DO get excerpts (with EmbeddingGemma-300M it was all 6, at 0.40–0.56
///   against a 0.40 gate). Routing by "skill description beats the best
///   excerpt" did not fix it with 300M either (4 of 6 device phrases scored
///   higher against the docs), so the policy is in the prompts: on an agent
///   chat the excerpts are reference only and live device facts come from
///   skills.
///
/// The policy alone was not enough (at temperature 0.6 Gemma still answered
/// some device questions from the docs), so on an agent chat whose skills
/// name `device_info` or `current_time`, `SkillQuestionRouter` skips
/// retrieval for live device and time questions
/// (`skill_question_router_test.dart` covers every phrase here). The prompt
/// policy stays as the backstop for phrasings the rules miss.
///
/// The golden also lists phrases for timer and camera-watch skills, which the
/// app does not have; the router test uses them as negatives.
///
/// Re-measure after changing the gate, the documents or the skills.
void main() {
  final golden = jsonDecode(
    File('test_assets/skill_trigger_similarity.json').readAsStringSync(),
  ) as Map<String, Object?>;
  final phrases = [
    for (final row in golden['phrases']! as List<Object?>)
      row! as Map<String, Object?>,
  ];
  Iterable<double> tops(String skill) => [
    for (final row in phrases)
      if (row['skill'] == skill) (row['top']! as num).toDouble(),
  ];

  test('measured at the gate in force', () {
    expect(golden['gate'], kKbMinSimilarityText);
  });

  test('every bundled skill has measured trigger phrases', () {
    final bundled = [
      for (final dir in Directory('assets/skills').listSync())
        if (dir is Directory) dir.uri.pathSegments.lastWhere((s) => s != ''),
    ];
    expect(bundled, isNotEmpty);
    for (final skill in bundled) {
      expect(tops(skill), isNotEmpty, reason: skill);
    }
  });

  test('time triggers stay below the gate: no excerpts reach their '
      'prompts', () {
    expect(tops('current-time'), isNotEmpty);
    for (final top in tops('current-time')) {
      expect(top, lessThan(kKbMinSimilarityText));
    }
  });

  test('device questions clear the gate, so the agent prompts must keep '
      'skills ahead of the excerpts', () {
    expect(
      tops('device-info').where((t) => t >= kKbMinSimilarityText),
      isNotEmpty,
      reason: 'if none did, the policy below could be relaxed',
    );
    const skills = ['current-time', 'device-info'];
    final prompt = PromptBuilder.build('Which accelerator is running?', [
      passage(1, similarity: 0.747),
    ], skills: skills);
    // An agent prompt that carries excerpts may still be a skill request
    // (the router acts only for skills that name its intents, and a skill
    // added at run time has no rule at all): the skills come first, and
    // anything else takes no tool call (a KB question was once pulled into
    // a tool call).
    expect(prompt, contains(PromptBuilder.excerptSkills(skills)));
    expect(kSkillsTemplate, contains('never from knowledge-base excerpts'));
  });
}
