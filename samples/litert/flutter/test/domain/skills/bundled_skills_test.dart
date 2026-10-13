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
    show SkillMdParseException, SkillType, parseSkillMd;
import 'package:flutter_edge_ai_agent/flutter_edge_ai_agent.dart' as agent;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/domain/skills/app_intent_executor.dart';
import 'package:litert_edge_demos/domain/skills/app_intent_handlers.dart';
import 'package:litert_edge_demos/domain/skills/app_intents.dart';

import '../../../integration_test/support/skill_fixtures.dart';

/// The seed skills that ship in `assets/skills/`.
void main() {
  final folders =
      Directory('assets/skills').listSync().whereType<Directory>().toList()
        ..sort((a, b) => a.path.compareTo(b.path));
  String folderName(Directory d) =>
      d.uri.pathSegments.lastWhere((s) => s.isNotEmpty);

  test('the two seed skills are there', () {
    expect(folders.map(folderName), ['current-time', 'device-info']);
  });

  for (final folder in folders) {
    final name = folderName(folder);
    group(name, () {
      final text = File('${folder.path}/SKILL.md').readAsStringSync();

      test('parses; name is the folder; an intent skill', () {
        final skill = parseSkillMd(text);
        expect(skill.name, name);
        expect(skill.description, isNotEmpty);
        expect(skill.type, SkillType.intent);
      });

      test('names only intents the app has', () {
        final named = intentsNamedIn(parseSkillMd(text).instructions);
        expect(named, isNotEmpty);
        expect(AppIntent.all.containsAll(named), isTrue, reason: '$named');
      });

      test('is listed in pubspec.yaml (assets do not recurse)', () {
        expect(
          File('pubspec.yaml').readAsStringSync(),
          contains('- assets/skills/$name/'),
        );
      });
    });
  }

  test('every intent is used by some seed skill', () {
    final used = {
      for (final folder in folders)
        ...intentsNamedIn(
          parseSkillMd(File('${folder.path}/SKILL.md').readAsStringSync())
              .instructions,
        ),
    };
    expect(used, AppIntent.all);
  });

  group('the runtime-only kid-clock skill (never bundled)', () {
    test('parses as an intent skill naming current_time', () {
      final skill = parseSkillMd(kidClockSkillMd);
      expect(skill.name, 'kid-clock');
      expect(skill.type, SkillType.intent);
      expect(intentsNamedIn(skill.instructions), {AppIntent.currentTime});
      expect(
        Directory('assets/skills/kid-clock').existsSync(),
        isFalse,
        reason: 'it shows a skill added without a rebuild',
      );
    });

    test('called by its own name, it runs its one literal call', () async {
      final executor = AppIntentExecutor(
        buildAppIntents(
          deviceFacts: () => '',
          now: () => DateTime(2026, 10, 6, 9, 41),
        ),
        skillNamed: (name) =>
            name == 'kid-clock' ? parseSkillMd(kidClockSkillMd) : null,
      );
      // What the agent loop hands the executor for
      // runIntent(intent: "kid-clock") without a skillName.
      final result = await executor.execute(
        const agent.Skill(
          name: 'kid-clock',
          description: '',
          instructions: '',
          type: SkillType.intent,
        ),
        '{}',
      );

      expect(
        result,
        isA<agent.TextResult>().having(
          (r) => r.text,
          'text',
          'It is 9:41 AM on Tuesday, October 6, 2026.',
        ),
      );
    });

    test('the broken fixture is a parse error', () {
      expect(
        () => parseSkillMd(brokenSkillMd),
        throwsA(isA<SkillMdParseException>()),
      );
    });
  });
}
