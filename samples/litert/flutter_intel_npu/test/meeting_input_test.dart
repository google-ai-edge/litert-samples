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
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:intel_npu_demo/features/act4_agent/agent_run.dart';
import 'package:intel_npu_demo/features/act4_agent/follow_up_board.dart';
import 'package:intel_npu_demo/features/act4_agent/meeting_input.dart';
import 'package:intel_npu_demo/features/act4_agent/meeting_task.dart';
import 'package:intel_npu_demo/features/act4_agent/workspace.dart';

Uint8List _b(String s) => Uint8List.fromList(utf8.encode(s));

Matcher _formatError(String part) =>
    throwsA(isA<FormatException>().having((e) => e.message, 'message', contains(part)));

void main() {
  group('a meeting from any transcript file', () {
    test('plain text keeps its lines; the first line names the meeting', () {
      final m = MeetingInput.parse('a/b/standup.txt', _b('﻿Weekly standup\r\n\r\n[00:01] Ana: hi'));
      expect(m.title, 'Weekly standup');
      expect(m.text, 'Weekly standup\n\n[00:01] Ana: hi');
    });

    test('a "Meeting transcript —" prefix and quotes are not the title; no first line, the file name', () {
      expect(MeetingInput.parse('m.txt', _b('MEETING TRANSCRIPT — "Weekly sync"\nAna: hi')).title, 'Weekly sync');
      expect(MeetingInput.parse('dir/m.txt', _b('[00:01] Ana: hi')).title, 'm.txt');
    });

    test('a UTF-8 byte order mark is dropped; old Mac line ends are lines', () {
      final m = MeetingInput.parse(
        'a.txt',
        Uint8List.fromList([0xEF, 0xBB, 0xBF, ...utf8.encode('Sync – Zoë\rAna: hi')]),
      );
      expect(m.text, 'Sync – Zoë\nAna: hi');
    });

    test('anything but .txt, an empty file or bytes that are not UTF-8 are refused, with the file name', () {
      expect(() => MeetingInput.parse('a.pdf', _b('%PDF')), _formatError('a.pdf: not a transcript'));
      expect(() => MeetingInput.parse('call.vtt', _b('WEBVTT')), _formatError('call.vtt: not a transcript'));
      expect(() => MeetingInput.parse('a.txt', _b('  \n ')), _formatError('a.txt: no text in it'));
      // "Zoë" in Windows-1252.
      final latin1 = Uint8List.fromList([0x5A, 0x6F, 0xEB, 0x0A]);
      expect(() => MeetingInput.parse('dir/notes.txt', latin1), _formatError('notes.txt: not UTF-8 text'));
    });

    test('the bundled example reads as a meeting', () async {
      final m = await MeetingInput.read('assets/agent/example/nimbus-android-3.0-beta.txt');
      expect(m.title, 'Nimbus Notes for Android 3.0: beta readiness');
    });
  });

  group('workspace', () {
    final ws = Workspace.fromJson(File('assets/agent/example/nimbus.workspace.json').readAsStringSync());

    test('the example reads, and only its own domain is internal', () {
      expect(ws.internalDomain, 'nimbus.io');
      expect(ws.isInternal('Ana.Ruiz@nimbus.io'), isTrue);
      expect(ws.isInternal('richard.hale@acme.com'), isFalse);
      final day = ws.calendarOn('2026-11-18')!;
      expect(day.clash('tom.becker@nimbus.io', 12 * 60, 13 * 60), 'out of office');
      expect(day.clash('ana.ruiz@nimbus.io', 10 * 60, 11 * 60), 'busy 09:00–11:00');
      expect(day.clash('ana.ruiz@nimbus.io', 12 * 60, 13 * 60), isNull);
      expect(day.clash('maria.costa@nimbus.io', 9 * 60, 18 * 60), isNull);
    });

    test('a bad field is named, not skipped', () {
      expect(
        () => Workspace.fromJson('{"calendars": [{"date": "2026-11-18", "busy": {"a@x": [["9", "10:00"]]}}]}'),
        throwsA(isA<FormatException>().having((e) => e.message, 'message', contains('calendars[0].busy'))),
      );
      expect(() => Workspace.fromJson('[1]'), throwsFormatException);
    });

    test('the prompt carries the workspace, and the rules follow it', () {
      final meeting = MeetingInput.parse('m.txt', _b('Standup\nAna: ship it'));
      final task = MeetingTask(meeting: meeting, workspace: ws);
      expect(task.system, contains('@nimbus.io addresses'));
      expect(task.system, contains('Calendar, Wednesday 2026-11-18'));
      expect(task.system, contains('Today is Tuesday 2026-11-03'));
      expect(task.user, endsWith('Standup\nAna: ship it'));
      expect(task.user, contains(MeetingTask.listTool));
      final update = task.tools.firstWhere((t) => t.name == 'update_doc');
      expect(update.description, contains('"QA budget"'));
      for (final t in task.tools.where((t) => t.name != MeetingTask.listTool)) {
        expect((t.parameters['required'] as List), contains(MeetingTask.itemArg), reason: t.name);
      }
      final bare = MeetingTask(meeting: meeting, workspace: Workspace.empty);
      expect(bare.system, contains('date of the meeting'));
      expect(bare.system, contains('no directory'));
      expect(bare.system, isNot(contains('@')));
      expect(bare.system, isNot(contains('Calendar')));
    });
  });

  group('the checklist comes from the run', () {
    final ws = Workspace.fromJson(File('assets/agent/example/nimbus.workspace.json').readAsStringSync());
    AgentCall call(String tool, Map<String, Object?> args, [double t = 1]) => AgentCall(tool: tool, args: args, t: t);
    AgentRun run(List<AgentCall> calls) =>
        AgentRun(model: 'm', requestedBackend: 'cpu', think: false, greedy: true, calls: calls);
    final list = call(MeetingTask.listTool, {
      'items': [
        {'id': 1, 'what': 'Security sign-off', 'who': 'lena.hoffmann@nimbus.io', 'due': '2026-11-13'},
        {'id': 2, 'what': 'Book the go/no-go', 'who': 'everyone'},
        {'id': '3', 'what': 'Tell Acme'},
      ],
    });

    test('no list, no follow-ups; the rules are still there', () {
      final items = FollowUpBoard.of(run([]), ws);
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
        run([
          list,
          call('create_task', {
            'title': 'Sign-off',
            'owner': 'lena.hoffmann@nimbus.io',
            'due': '2026-11-13',
            'follow_up': 1,
          }),
          call('draft_email', {
            'to': ['richard.hale@acme.com'],
            'approver': 'ana.ruiz@nimbus.io',
            'follow_up': '#3',
          }),
        ]),
        ws,
      );
      final follow = items.where((i) => !i.rule).toList();
      expect(follow.map((i) => i.label), ['Security sign-off', 'Book the go/no-go', 'Tell Acme']);
      expect(follow.map((i) => i.passed), [true, false, true]);
      expect(follow.first.detail, 'lena.hoffmann@nimbus.io · 2026-11-13');
      expect(items.where((i) => i.rule).every((i) => i.passed), isTrue);
    });

    test('rules: outside mail, a busy slot, a stray action, a repeat', () {
      final items = FollowUpBoard.of(
        run([
          list,
          call('send_email', {
            'to': ['richard.hale@acme.com'],
            'subject': 's',
            'body': 'b',
            'follow_up': 3,
          }),
          call('schedule_meeting', {
            'title': 'Go/no-go',
            'attendees': ['ana.ruiz@nimbus.io', 'tom.becker@nimbus.io'],
            'date': '2026-11-18',
            'start': '10:00',
            'duration_min': 45,
            'follow_up': 2,
          }),
          call('create_task', {'title': 'x', 'follow_up': 9}),
          call('create_task', {'title': 'x', 'follow_up': 1}),
        ]),
        ws,
      );
      final rules = {for (final i in items.where((i) => i.rule)) i.id: i};
      expect(rules['external_as_drafts']!.passed, isFalse);
      expect(rules['external_as_drafts']!.why, contains('richard.hale@acme.com'));
      expect(rules['meetings_in_free_slots']!.why, 'ana.ruiz is busy 09:00–11:00; tom.becker is out of office');
      expect(rules['actions_on_the_list']!.why, '1 action not on its own list');
      expect(rules['no_repeated_actions']!.why, '1 repeated');
    });

    test('a workspace without a domain or calendars drops those rules', () {
      final items = FollowUpBoard.of(run([list]), Workspace.empty);
      expect(items.where((i) => i.rule).map((i) => i.id), ['actions_on_the_list', 'no_repeated_actions']);
    });
  });
}
