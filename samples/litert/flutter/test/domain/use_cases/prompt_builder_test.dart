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
    show onDemandAgentSystemPrompt;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/domain/skills/app_intents.dart';
import 'package:litert_edge_demos/domain/use_cases/prompt_builder.dart';

import '../../fakes/fake_knowledge.dart';

/// The loaded skills a hint names: one bundled, one dropped in at run time.
const _skills = ['current-time', 'kid-clock'];

/// What a skill-agnostic prompt must never contain: the app's intents, the
/// bundled skills' names and the topics they cover, and
/// a direct `runIntent` instruction (a skill is reached through `loadSkill`
/// and its own instructions).
final _named = RegExp(
  [
    for (final intent in AppIntent.all) RegExp.escape(intent),
    'current-time',
    'device-info',
    r'\btime\b',
    r'\bdate\b',
    r'\bclock\b',
    r'\bdevice\b',
    r'\bhardware\b',
    r'\baccelerators?\b',
    r'\bbackends?\b',
    r'\b[cgn]pu\b',
    r'\bmemory\b',
    r'\bmodels?\b',
    'runIntent',
    'run_intent',
  ].join('|'),
  caseSensitive: false,
);

void main() {
  group('agent prompts name no skill, intent or topic of their own', () {
    test('every hint, with no skill names given', () {
      for (final text in [
        PromptBuilder.excerptSkills(const []),
        PromptBuilder.directAnswerHint(const []),
        PromptBuilder.spokenQuestionWithSkills(const []),
        PromptBuilder.spokenImageHint,
      ]) {
        expect(_named.hasMatch(text), isFalse, reason: text);
      }
    });

    test('the skill names come from the loaded skills, in their order, and '
        'nothing else is added', () {
      const runtime = ['recipe-finder', 'weather-now'];
      for (final hint in [
        PromptBuilder.excerptSkills,
        PromptBuilder.directAnswerHint,
        PromptBuilder.spokenQuestionWithSkills,
      ]) {
        final text = hint(runtime);
        expect(text, contains('your skills (recipe-finder, weather-now)'));
        expect(_named.hasMatch(text), isFalse, reason: text);
        expect(
          hint(const []),
          isNot(contains('skills (')),
          reason: 'no empty list',
        );
      }
    });

    test('whole prompts on an agent chat: typed and spoken, with excerpts, '
        'a picture, or neither', () {
      const runtime = ['recipe-finder'];
      for (final prompt in [
        PromptBuilder.build('q', [passage(1)], skills: runtime),
        PromptBuilder.buildSpoken(const [], skills: runtime),
        PromptBuilder.buildSpoken(const [], skills: runtime, image: true),
        PromptBuilder.buildSpoken([passage(1)], skills: runtime),
        PromptBuilder.buildSpoken([passage(1)], skills: runtime, image: true),
      ]) {
        // The excerpts' own text is the knowledge base's, not the app's.
        final own = prompt.replaceAll(passage(1).content, '');
        expect(_named.hasMatch(own), isFalse, reason: own);
      }
    });

    test("the system template is the agent package's "
        "onDemandAgentSystemPrompt, then the app's read-aloud and retrieval "
        'lines', () {
      expect(kSkillsTemplate, startsWith(onDemandAgentSystemPrompt));
      expect('__SKILLS__'.allMatches(kSkillsTemplate), hasLength(1));
      expect(
        kSkillsTemplate.substring(onDemandAgentSystemPrompt.length),
        '\nReplies are read aloud: a few short sentences, plain text.\n'
        'Facts a skill provides come from that skill, never from '
        'knowledge-base excerpts.',
      );
      // The rules the app measured and sent before 0.3.0, word for word.
      expect(
        kSkillsTemplate,
        contains(
          'Skills:\n__SKILLS__\nOnly when a request needs one of these '
          'skills, call loadSkill with its name and follow its instructions. '
          'Answer everything else directly with no tool call: general '
          'questions, questions about a picture, chat.\n',
        ),
      );
    });

    test('the system template: the skill list is the only place skills '
        'appear', () {
      final [_, after] = kSkillsTemplate.split('__SKILLS__');
      expect(_named.hasMatch(after), isFalse, reason: after);
      expect(after, contains('call loadSkill with its name'));
      expect(
        after,
        contains(
          'Facts a skill provides come from that skill, never from '
          'knowledge-base excerpts.',
        ),
      );
    });
  });

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

    test('on an agent chat: excerpts as reference, and a request one of the '
        "skills serves goes to that skill (a run-time skill's request may "
        'clear the gate); without excerpts the question is unchanged', () {
      final prompt = PromptBuilder.build('q', [passage(1)], skills: _skills);
      expect(
        prompt,
        startsWith(
          'Reference excerpts from the knowledge base (general '
          'documentation):\n\n[1] ',
        ),
      );
      expect(
        prompt,
        contains(
          'If the request matches one of your skills (current-time, '
          'kid-clock), load it and follow it instead of the excerpts; '
          'otherwise answer from the excerpts with no tool call. Use the '
          'excerpts only if they are relevant',
        ),
      );
      expect(prompt, contains('cite each one you use as [1]'));
      expect(prompt, endsWith('Question: q'));
      expect(PromptBuilder.build('q', const [], skills: _skills), 'q');
      expect(
        PromptBuilder.build('q', [passage(1)]),
        isNot(contains('skills')),
        reason: 'a plain chat has no tools',
      );
    });
  });

  group('directAnswerHint (a typed picture turn on an agent chat)', () {
    test('leaves the skills open: the picture alone takes no tool', () {
      expect(
        PromptBuilder.directAnswerHint(_skills),
        '\n\n(A picture is attached. Use one of your skills (current-time, '
        'kid-clock) only when the request matches its description; otherwise '
        'answer about the picture directly with no tool call.)',
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
        PromptBuilder.buildSpoken(const [], skills: _skills),
        PromptBuilder.spokenQuestionWithSkills(_skills),
      );
      expect(PromptBuilder.spokenQuestion, isNotEmpty);
      expect(
        PromptBuilder.buildSpoken(const [], skills: _skills, image: true),
        '${PromptBuilder.spokenQuestionWithSkills(_skills)}'
        '${PromptBuilder.spokenImageHint}',
      );
      expect(
        PromptBuilder.buildSpoken(const [], image: true),
        PromptBuilder.spokenQuestion,
        reason: 'the hint is for agent chats only',
      );
    });

    test('on an agent chat without excerpts: a task as well as a question, a '
        'skill only when the request matches its description, and no tool '
        'for anything else', () {
      final prompt = PromptBuilder.spokenQuestionWithSkills(_skills);
      expect(
        prompt,
        'Answer the question asked in this audio, or do what it asks. Use one '
        'of your skills (current-time, kid-clock) only when the request '
        'matches its description; otherwise reply directly with no tool '
        'call.',
      );
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
      expect(prompt, isNot(contains('skills')));
    });

    test('on an agent chat the excerpts never forbid a skill: the same '
        'instruction as a typed turn', () {
      final prompt = PromptBuilder.buildSpoken(passages, skills: _skills);
      expect(prompt, startsWith('Reference excerpts from the knowledge base'));
      expect(
        prompt,
        contains(
          'The question is in this audio. '
          '${PromptBuilder.excerptSkills(_skills)}'
          'Use the excerpts only if they are relevant',
        ),
      );
      expect(
        PromptBuilder.buildSpoken(passages, skills: _skills, image: true),
        endsWith(PromptBuilder.spokenImageHint),
      );
    });
  });
}
