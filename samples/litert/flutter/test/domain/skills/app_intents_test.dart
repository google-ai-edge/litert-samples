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
    show Skill, SkillType, parseSkillMd;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/domain/skills/app_intents.dart';

import '../../support/package_skill_fixtures.dart';

void main() {
  group('intentsNamedIn', () {
    test('intent `name` in backticks (the bundled skills\' form)', () {
      expect(
        intentsNamedIn(
          'Call the `run_intent` tool with intent `current_time` and '
          'parameters {}.',
        ),
        {AppIntent.currentTime},
      );
    });

    test('intent: name, with or without backticks or quotes, also as a list '
        'item', () {
      for (final body in const [
        'intent: current_time',
        '- intent: current_time',
        '  - intent: `current_time`',
        'Intent: "current_time"',
        "intent : 'current_time', then answer",
        'Call it with intent: current_time.',
      ]) {
        expect(intentsNamedIn(body), {AppIntent.currentTime}, reason: body);
      }
    });

    test('every name comes back as written, with no case or -/_ folding '
        '(the executor runs only exact names), so the scan can list it', () {
      expect(intentsNamedIn('intent `Device-Info`'), {'Device-Info'});
      expect(intentsNamedIn('intent ` current_time `'), {
        AppIntent.currentTime,
      });
      expect(intentsNamedIn('- intent: $packageGetCurrentTimeIntent'), {
        packageGetCurrentTimeIntent,
      });
      expect(intentsNamedIn('intent `current_tme`'), {'current_tme'});
      expect(intentsNamedIn('- intent: create_calendar_event'), {
        'create_calendar_event',
      });
    });

    test('every intent a body names, once each', () {
      expect(
        intentsNamedIn(
          'First intent `current_time`, then intent: device_info, then '
          'intent `current_time` again.',
        ),
        {AppIntent.currentTime, AppIntent.deviceInfo},
      );
    });

    test('prose that only says "intent" names nothing', () {
      for (final body in const [
        'Call the `run_intent` tool.',
        'This intent tells the time.',
        'Two intents are available.',
        'The intent:\nThen tell the user the time.',
        'Use the run_intent tool with the right intent.',
        '## Intent: tell the user the time',
        "Figure out the user's intent: they may want the time.",
        'Intent: Weather',
        'intent: answer briefly',
        '## Intent: real-time clock for kids',
        'Intent: kid-friendly time telling',
        "The user's intent: up-to-date facts.",
      ]) {
        expect(intentsNamedIn(body), isEmpty, reason: body);
      }
    });

    test('after a colon a bare word counts when it has `_` (a hyphen alone '
        'is prose); quoted, any value counts', () {
      expect(intentsNamedIn('intent: current_time'), {AppIntent.currentTime});
      expect(intentsNamedIn('intent: Current_Time'), {'Current_Time'});
      expect(intentsNamedIn('intent: Current-Time'), isEmpty);
      expect(intentsNamedIn('intent: make_coffee'), {'make_coffee'});
      expect(intentsNamedIn('intent: make-coffee'), isEmpty);
      expect(intentsNamedIn('intent: `make-coffee`'), {'make-coffee'});
      expect(intentsNamedIn('intent: `weather`'), {'weather'});
      expect(intentsNamedIn('intent: "weather"'), {'weather'});
      expect(
        intentsNamedIn('intent: "weather\''),
        isEmpty,
        reason: 'mismatched quotes are not a quoted value',
      );
    });

    test("the agent package's own get-current-time skill names its intent as "
        'written: a built-in intent of the package executor', () {
      expect(
        intentsNamedIn(parseSkillMd(packageGetCurrentTimeSkillMd).instructions),
        {packageGetCurrentTimeIntent},
      );
    });
  });

  group('intentsOfferedBy', () {
    test('the intents of intent-type skills only; an instruction-only skill '
        'that quotes one offers nothing', () {
      expect(
        intentsOfferedBy([
          parseSkillMd(packageGetCurrentTimeSkillMd),
          const Skill(
            name: 'clock',
            description: 'The time.',
            instructions:
                'Call the `run_intent` tool with intent `current_time`.',
            type: SkillType.intent,
          ),
          const Skill(
            name: 'device-notes',
            description: 'Notes.',
            instructions: 'Never call intent `device_info` for this.',
            type: SkillType.textOnly,
          ),
          const Skill(
            name: 'coffee',
            description: 'Coffee.',
            instructions:
                'Call the `run_intent` tool with intent `make_coffee`.',
            type: SkillType.intent,
          ),
        ]),
        {AppIntent.currentTime},
        reason:
            'only app intents are offered: the package built-in '
            'get_current_date_and_time and an unknown intent are not',
      );
      expect(intentsOfferedBy(const []), isEmpty);
    });
  });
}
