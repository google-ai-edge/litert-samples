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

import 'package:flutter_test/flutter_test.dart';
import 'package:intel_npu_demo/features/act4_agent/agent_run.dart';
import 'package:intel_npu_demo/features/act4_agent/follow_up_board.dart';
import 'package:intel_npu_demo/features/act4_agent/meeting_input.dart';
import 'package:intel_npu_demo/features/act4_agent/meeting_task.dart';
import 'package:intel_npu_demo/features/act4_agent/workspace.dart';

/// A small workspace of its own, so these tests do not move with the
/// bundled example.
const Map<String, Object?> _wsJson = {
  'company': 'Nimbus',
  'today': '2026-11-03',
  'internal_domain': 'nimbus.io',
  'work_hours': ['09:00', '18:00'],
  'people': [
    {'name': 'Ana Ruiz', 'email': 'ana.ruiz@nimbus.io'},
    {'name': 'Tom Becker', 'email': 'tom.becker@nimbus.io'},
    {'name': 'Lena Hoffmann', 'email': 'lena.hoffmann@nimbus.io'},
    {'name': 'Richard Hale', 'email': 'richard.hale@acme.com', 'internal': false},
  ],
  'calendars': [
    {
      'date': '2026-11-18',
      'busy': {
        'ana.ruiz@nimbus.io': [
          ['09:00', '11:00'],
        ],
        'tom.becker@nimbus.io': 'out',
      },
    },
  ],
};

final _ws = Workspace.fromJson(jsonEncode(_wsJson));

AgentCall _call(String tool, Map<String, Object?> args, [double t = 1]) => AgentCall(tool: tool, args: args, t: t);

AgentRun _run(List<AgentCall> calls) =>
    AgentRun(model: 'm', requestedBackend: 'cpu', think: false, greedy: true, calls: calls);

AgentCall _list(List<Map<String, Object?>> items) => _call(MeetingTask.listTool, {'items': items});

final _three = _list([
  {'id': 1, 'what': 'Security sign-off', 'who': 'lena.hoffmann@nimbus.io', 'due': '2026-11-13'},
  {'id': 2, 'what': 'Book the go/no-go', 'who': 'everyone'},
  {'id': '3', 'what': 'Tell Acme'},
]);

Map<String, BoardItem> _rules(List<BoardItem> items) => {for (final i in items.where((i) => i.rule)) i.id: i};

AgentCall _meeting(Map<String, Object?> args) => _call(MeetingTask.scheduleMeeting, {
  'title': 'Go/no-go',
  'attendees': ['ana.ruiz@nimbus.io'],
  'date': '2026-11-18',
  'start': '14:00',
  'duration_min': 30,
  'follow_up': 2,
  ...args,
});

void main() {
  group('the checklist comes from the run', () {
    test('no list, no follow-ups; the rules are still there', () {
      final items = FollowUpBoard.of(_run([]), _ws);
      expect(items.where((i) => !i.rule), isEmpty);
      expect(items.map((i) => i.id), [
        'external_as_drafts',
        'meetings_in_free_slots',
        'actions_on_the_list',
        'no_repeated_actions',
      ]);
    });

    test('an action naming an item ticks it; the rest stay open', () {
      final items = FollowUpBoard.of(
        _run([
          _three,
          _call(MeetingTask.createTask, {
            'title': 'Sign-off',
            'owner': 'lena.hoffmann@nimbus.io',
            'due': '2026-11-13',
            'follow_up': 1,
          }),
          _call(MeetingTask.draftEmail, {
            'to': ['richard.hale@acme.com'],
            'approver': 'ana.ruiz@nimbus.io',
            'follow_up': '#3',
          }),
        ]),
        _ws,
      );
      final follow = items.where((i) => !i.rule).toList();
      expect(follow.map((i) => i.label), ['Security sign-off', 'Book the go/no-go', 'Tell Acme']);
      expect(follow.map((i) => i.passed), [true, false, true]);
      expect(follow.first.detail, 'lena.hoffmann@nimbus.io · 2026-11-13');
      expect(items.where((i) => i.rule).every((i) => i.passed), isTrue);
    });

    test('rules: outside mail, a busy slot, a stray action, a repeat', () {
      final rules = _rules(
        FollowUpBoard.of(
          _run([
            _three,
            _call(MeetingTask.sendEmail, {
              'to': ['richard.hale@acme.com'],
              'subject': 's',
              'body': 'b',
              'follow_up': 3,
            }),
            _meeting({
              'attendees': ['ana.ruiz@nimbus.io', 'tom.becker@nimbus.io'],
              'start': '10:00',
              'duration_min': 45,
            }),
            _call(MeetingTask.createTask, {'title': 'x', 'follow_up': 9}),
            _call(MeetingTask.createTask, {'title': 'x', 'follow_up': 1}),
          ]),
          _ws,
        ),
      );
      expect(rules['external_as_drafts']!.passed, isFalse);
      expect(rules['external_as_drafts']!.why, 'emailed richard.hale@acme.com directly instead of a draft');
      expect(rules['meetings_in_free_slots']!.why, 'ana.ruiz is busy 09:00–11:00; tom.becker is out of office');
      expect(rules['actions_on_the_list']!.why, '1 action not on its own list');
      expect(rules['no_repeated_actions']!.why, '1 repeated');
    });

    test('a workspace without a domain or calendars drops those rules', () {
      final items = FollowUpBoard.of(_run([_three]), Workspace.empty);
      expect(items.where((i) => i.rule).map((i) => i.id), ['actions_on_the_list', 'no_repeated_actions']);
    });
  });

  group('the list', () {
    test('missing numbers come after the largest one given', () {
      final list = FollowUpBoard.listOf(
        _run([
          _list([
            {'id': 3, 'what': 'a'},
            {'what': 'b'},
            {'id': '#5', 'what': 'c'},
            {'id': 'first', 'what': 'd'},
          ]),
        ]),
      );
      expect(list.map((f) => f.id), [3, 6, 5, 7]);
    });

    test('a number given twice: both rows kept apart, one action ticks the first, a rule fails', () {
      final items = FollowUpBoard.of(
        _run([
          _list([
            {'id': 1, 'what': 'a'},
            {'id': 2, 'what': 'b'},
            {'id': 2, 'what': 'c'},
          ]),
          _call(MeetingTask.createTask, {'title': 't', 'follow_up': 2}),
        ]),
        Workspace.empty,
      );
      final follow = items.where((i) => !i.rule).toList();
      expect(follow.map((i) => i.id), ['f1', 'f2', 'f2-2']);
      expect(items.map((i) => i.id).toSet(), hasLength(items.length), reason: 'ids key the rows');
      expect(follow.map((i) => i.passed), [false, true, false]);
      expect(follow.last.why, 'its number 2 was given twice; an action ticks the first');
      final rule = _rules(items)['unique_numbers']!;
      expect(
        (rule.passed, rule.label, rule.why),
        (false, 'Every follow-up has its own number', 'number 2 is given to 2 follow-ups'),
      );
    });

    test('the first list is scored; a different later one breaks a rule and says how', () {
      final run = _run([
        _three,
        _list([
          {'id': 1, 'what': 'Security sign-off', 'who': 'lena.hoffmann@nimbus.io', 'due': '2026-11-13'},
          {'id': 2, 'what': 'Book the go/no-go next week', 'who': 'everyone'},
          {'id': 4, 'what': 'Something new'},
        ]),
      ]);
      expect(FollowUpBoard.listOf(run).map((f) => f.what), ['Security sign-off', 'Book the go/no-go', 'Tell Acme']);
      final items = FollowUpBoard.of(run, Workspace.empty);
      expect(items.where((i) => !i.rule), hasLength(3), reason: 'a shorter list cannot raise the score');
      final rule = _rules(items)['listed_once']!;
      expect(rule.passed, isFalse);
      expect(rule.why, 'a later list differs: added 4; dropped 3; changed 2; the first one is scored');
    });

    test('the same list published again breaks nothing', () {
      final items = FollowUpBoard.of(_run([_three, _three]), Workspace.empty);
      expect(_rules(items).keys, ['actions_on_the_list', 'no_repeated_actions']);
    });

    test('items that are not a list are skipped; items written as a JSON string are read', () {
      final run = _run([
        _call(MeetingTask.listTool, {
          'items': {'id': 1},
        }),
        _call(MeetingTask.listTool, {
          'items': jsonEncode([
            {'id': 1, 'what': 'from a string'},
          ]),
        }),
      ]);
      expect(FollowUpBoard.listOf(run).single.what, 'from a string');
    });
  });

  group('rules', () {
    test('a question to an outside address breaks the mail rule too', () {
      final rule = _rules(
        FollowUpBoard.of(
          _run([
            _call(MeetingTask.askClarification, {'to': 'richard.hale@acme.com', 'question': 'q', 'follow_up': 1}),
          ]),
          _ws,
        ),
      )['external_as_drafts']!;
      expect(rule.passed, isFalse);
      expect(rule.why, 'asked richard.hale@acme.com directly instead of a draft');
    });

    test('recipients as one string, separated, with names, in any case', () {
      List<List<String>> parts(Object? v) => switch (FollowUpBoard.recipients(v)) {
        (:final addresses, :final bad) => [addresses, bad],
      };
      expect(parts('Ana <Ana.Ruiz@Nimbus.io>, tom.becker@nimbus.io; Richard Hale <richard.hale@acme.com>'), [
        ['ana.ruiz@nimbus.io', 'tom.becker@nimbus.io', 'richard.hale@acme.com'],
        <String>[],
      ]);
      expect(parts(['a@x.io', 'Richard']), [
        ['a@x.io'],
        ['Richard'],
      ]);
      expect(parts(null), [<String>[], <String>[]]);
      final rule = _rules(
        FollowUpBoard.of(
          _run([
            _call(MeetingTask.sendEmail, {
              'to': 'Ana.Ruiz@NIMBUS.io, Richard',
              'subject': 's',
              'body': 'b',
              'follow_up': 1,
            }),
          ]),
          _ws,
        ),
      )['external_as_drafts']!;
      expect(rule.passed, isFalse);
      expect(rule.why, 'emailed "Richard", which is not an email address');
    });

    test('a meeting on a day without a calendar is not checked, and says so', () {
      final rule = _rules(
        FollowUpBoard.of(
          _run([
            _three,
            _meeting({'date': '2026-11-20'}),
          ]),
          _ws,
        ),
      )['meetings_in_free_slots']!;
      expect(rule.passed, isTrue);
      expect(rule.detail, 'not checked: 2026-11-20 (no calendar)');
    });

    test('attendees outside the directory or not addresses are not checked, and named', () {
      final rule = _rules(
        FollowUpBoard.of(
          _run([
            _three,
            _meeting({
              'attendees': ['ana.ruiz@nimbus.io', 'ghost@nimbus.io', 'Tom'],
            }),
          ]),
          _ws,
        ),
      )['meetings_in_free_slots']!;
      expect(rule.passed, isTrue);
      expect(rule.detail, 'not checked: "Tom" (not an address), ghost@nimbus.io (not in the directory)');
    });

    for (final (name, args, why) in [
      ('an hour past 23', {'start': '25:00'}, '"25:00" is not a time (HH:MM)'),
      ('no start', {'start': null}, 'a meeting has no start time'),
      ('no length', {'duration_min': null}, 'the meeting at 14:00 has no length'),
      ('a length of zero', {'duration_min': 0}, 'the meeting at 14:00 has no length'),
      ('a start before hours', {'start': '08:30'}, '08:30 is before working hours'),
      ('an end after hours', {'start': '17:45'}, '17:45–18:15 ends after working hours'),
      ('a date that is not one', {'date': 'next Friday'}, '"next Friday" is not a date (YYYY-MM-DD)'),
    ]) {
      test('a meeting with $name breaks the calendar rule', () {
        final rule = _rules(FollowUpBoard.of(_run([_three, _meeting(args)]), _ws))['meetings_in_free_slots']!;
        expect((rule.passed, rule.why), (false, why));
      });
    }

    test('a repeat is the same tool and arguments in any key order, whatever item it names', () {
      final rule = _rules(
        FollowUpBoard.of(
          _run([
            _three,
            _call(MeetingTask.updateDoc, {
              'doc': 'QA budget',
              'content': {'a': 1, 'b': 2},
              'follow_up': 1,
            }),
            _call(MeetingTask.updateDoc, {
              'content': {'b': 2, 'a': 1},
              'follow_up': 2,
              'doc': 'QA budget',
            }),
          ]),
          Workspace.empty,
        ),
      )['no_repeated_actions']!;
      expect((rule.passed, rule.why), (false, '1 repeated'));
    });
  });

  test('every tool and argument the board reads is in the schemas', () {
    final tools = {
      for (final t in MeetingTask(
        meeting: const MeetingInput(title: 't', text: 'x', source: 't.txt'),
        workspace: Workspace.empty,
      ).tools)
        t.name: t.parameters,
    };
    for (final MapEntry(key: tool, value: paths) in FollowUpBoard.reads.entries) {
      expect(tools, contains(tool));
      for (final path in paths) {
        Map<Object?, Object?> schema = tools[tool]!;
        for (final key in path.split('.')) {
          if (schema['type'] == 'array') schema = schema['items']! as Map;
          final props = schema['properties']! as Map;
          expect(props, contains(key), reason: '$tool.$path');
          schema = props[key]! as Map;
        }
      }
    }
    expect(tools.keys.toSet(), FollowUpBoard.reads.keys.toSet(), reason: 'the board knows every tool');
  });
}
