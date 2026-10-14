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

import 'package:flutter_edge_ai_agent/flutter_edge_ai_agent.dart'
    show AgentToolNames;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/domain/skills/app_intents.dart';
import 'package:litert_edge_demos/domain/use_cases/prompt_builder.dart';

import '../../fakes/fake_knowledge.dart';

void main() {
  group('build', () {
    test('without passages the question goes out unchanged', () {
      expect(
        PromptBuilder.build('What is LiteRT?', const []),
        'What is LiteRT?',
      );
    });

    test('the spec template: excerpts numbered from 1, the instruction, '
        'then the question', () {
      final prompt = PromptBuilder.build('What is LiteRT?', [
        passage(1),
        passage(2),
        passage(3),
      ]);
      expect(
        prompt,
        'Excerpts from the knowledge base:\n\n'
        '[1] LiteRT overview › Section 1\n\nBody of excerpt 1.\n\n'
        '[2] LiteRT overview › Section 2\n\nBody of excerpt 2.\n\n'
        '[3] LiteRT overview › Section 3\n\nBody of excerpt 3.\n\n'
        'Answer the question. Use the excerpts only if they are relevant and '
        'cite each one you use as [1], [2] or [3].\n'
        'If they do not contain the answer, say so briefly and answer without '
        'citations.\n\n'
        'Question: What is LiteRT?',
      );
    });

    test('never invites a citation number that is not in the prompt', () {
      expect(
        PromptBuilder.build('q', [passage(1)]),
        contains('cite each one you use as [1].'),
      );
      final two = PromptBuilder.build('q', [passage(1), passage(2)]);
      expect(two, contains('cite each one you use as [1] or [2].'));
      expect(two, isNot(contains('[3]')));
    });

    test('on an agent chat: excerpts as reference and an explicit '
        '"no tool call" for this turn — live skill questions never get '
        'excerpts (SkillQuestionRouter), so one that does is a knowledge '
        'question; without excerpts the question is still unchanged', () {
      final prompt = PromptBuilder.build('q', [passage(1)], skills: true);
      expect(
        prompt,
        startsWith(
          'Reference excerpts from the knowledge base (general '
          'documentation):\n\n[1] ',
        ),
      );
      expect(prompt, contains(PromptBuilder.noToolCall));
      expect(prompt, isNot(contains('call the skill')));
      expect(prompt, contains('cite each one you use as [1]'));
      expect(prompt, endsWith('Question: q'));
      expect(PromptBuilder.build('q', const [], skills: true), 'q');
      expect(
        PromptBuilder.build('q', [passage(1)]),
        isNot(contains(PromptBuilder.noToolCall)),
        reason: 'a plain chat has no tools',
      );
    });
  });

  group('citedNumbers', () {
    test('single, listed and ranged markers, in range only', () {
      expect(PromptBuilder.citedNumbers('It runs on the GPU [1].', 3), {1});
      expect(PromptBuilder.citedNumbers('Both [1, 3] say so.', 3), {1, 3});
      expect(PromptBuilder.citedNumbers('See [1-3].', 3), {1, 2, 3});
      expect(PromptBuilder.citedNumbers('See [2–3] and [2].', 3), {2, 3});
      expect(PromptBuilder.citedNumbers('[1][2]', 3), {1, 2});
    });

    test('numbers outside 1..n and non-citations are ignored', () {
      expect(PromptBuilder.citedNumbers('As [4] and [0] say.', 3), isEmpty);
      expect(PromptBuilder.citedNumbers('See [1].', 0), isEmpty);
      expect(PromptBuilder.citedNumbers('A list [a] or [ 1 ]', 3), isEmpty);
      expect(PromptBuilder.citedNumbers('No citations here.', 3), isEmpty);
    });

    test('tensor shapes and code are not citations', () {
      expect(
        PromptBuilder.citedNumbers(
          'It takes a float32 tensor of shape [1, 3, 640, 640] [1].',
          3,
        ),
        {1},
      );
      expect(PromptBuilder.citedNumbers('Output [1, 300, 6] [2].', 3), {2});
      expect(PromptBuilder.citedNumbers('Input [1, 80, 3000].', 3), isEmpty);
      expect(PromptBuilder.citedNumbers('Use `[1]` in code [3].', 3), {3});
      expect(PromptBuilder.citedNumbers('Shape `[1, 3]` here.', 3), isEmpty);
      expect(PromptBuilder.citedNumbers('Both [1], [2].', 3), {1, 2});
      expect(PromptBuilder.citedNumbers('Both [1, 2].', 3), {1, 2});
    });
  });

  group('buildSpoken (audio mode: the question is the recording)', () {
    final passages = [passage(1), passage(2)];

    test('without excerpts: the minimal prompt, never empty', () {
      expect(PromptBuilder.buildSpoken(const []), PromptBuilder.spokenQuestion);
      expect(
        PromptBuilder.buildSpoken(const [], skills: true),
        PromptBuilder.spokenQuestionWithSkills,
      );
      expect(PromptBuilder.spokenQuestion, isNotEmpty);
      expect(
        PromptBuilder.buildSpoken(const [], skills: true, image: true),
        '${PromptBuilder.spokenQuestionWithSkills}'
        '${PromptBuilder.spokenImageHint}',
      );
      expect(
        PromptBuilder.buildSpoken(const [], image: true),
        PromptBuilder.spokenQuestion,
        reason: 'the hint is for agent chats only',
      );
    });

    test('on an agent chat without excerpts: a task as well as a question, '
        'and only the time and this device or app take a tool', () {
      const prompt = PromptBuilder.spokenQuestionWithSkills;
      expect(
        prompt,
        startsWith('Answer the question asked in this audio, or '),
      );
      expect(prompt, contains('do what it asks'));
      expect(prompt, contains('current-time only for the time'));
      expect(prompt, contains('device-info only for this device or app'));
      expect(prompt, endsWith('reply directly with no tool call.'));
      // The measured regression: a generic "use a skill if needed" made
      // Gemma call invented intents for stories, capitals and jokes.
      expect(prompt, isNot(contains('Use a skill only if')));
    });

    test('with excerpts: numbered from 1, the question is in the audio, '
        'cite only the numbers given; no "Question:" line', () {
      final prompt = PromptBuilder.buildSpoken(passages);
      expect(prompt, startsWith('Excerpts from the knowledge base:\n\n[1] '));
      expect(prompt, contains('[2] ${passages[1].content}'));
      expect(prompt, contains('The question is in this audio. Answer it.'));
      expect(prompt, contains('cite each one you use as [1] or [2]'));
      expect(prompt, isNot(contains('Question:')));
      expect(prompt, isNot(contains(PromptBuilder.noToolCall)));
    });

    test('on an agent chat the excerpts never forbid a skill: a live device '
        'or time question is told which intent answers it', () {
      final prompt = PromptBuilder.buildSpoken(passages, skills: true);
      expect(prompt, startsWith('Reference excerpts from the knowledge base'));
      expect(
        prompt,
        contains(
          'The question is in this audio. ${PromptBuilder.spokenExcerptSkills}'
          'Use the excerpts only if they are relevant',
        ),
      );
      expect(
        PromptBuilder.spokenExcerptSkills,
        contains(
          'call ${AgentToolNames.runIntent} with ${AppIntent.deviceInfo} '
          'instead of using the excerpts',
        ),
        reason: 'the tool and intent names the chat really has',
      );
      expect(
        PromptBuilder.spokenExcerptSkills,
        contains(
          'call ${AgentToolNames.runIntent} with ${AppIntent.currentTime}',
        ),
      );
      expect(
        prompt,
        contains('Otherwise answer it directly with no tool call'),
      );
      expect(
        PromptBuilder.buildSpoken(passages, skills: true, image: true),
        endsWith(PromptBuilder.spokenImageHint),
      );
    });
  });
}
