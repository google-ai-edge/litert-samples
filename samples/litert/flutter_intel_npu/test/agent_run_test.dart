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

import 'package:flutter_edge_ai/flutter_edge_ai.dart' show Tool;
import 'package:flutter_test/flutter_test.dart';
import 'package:intel_npu_demo/features/act4_agent/agent_run.dart';
import 'package:intel_npu_demo/features/act4_agent/meeting_input.dart';
import 'package:intel_npu_demo/features/act4_agent/meeting_task.dart';
import 'package:intel_npu_demo/features/act4_agent/workspace.dart';

AgentRun _run({int maxOutputTokens = 6144}) => AgentRun(
  model: r'C:\models\gemma.litertlm',
  requestedBackend: 'npu',
  think: true,
  greedy: true,
  maxOutputTokens: maxOutputTokens,
);

/// Tools that take anything, for the tests of the turn clock.
final _any = [
  for (final n in ['a', 'b', 'c', 'create_task', 'update_doc', 'send_email'])
    Tool(name: n, description: '', parameters: const {'type': 'object'}),
];

/// The task's real tools.
final _tools = MeetingTask(
  meeting: const MeetingInput(title: 't', text: 'x', source: 't.txt'),
  workspace: Workspace.empty,
).tools;

void main() {
  group('RunRecorder', () {
    test('a turn that streams thinking and then calls', () {
      final run = _run();
      final rec = RunRecorder(run, maxCalls: 40, tools: _any)..begin(0);
      rec.thinking('Let me ', 1.0);
      rec.thinking('plan.', 1.1);
      rec.thinking(' Next.', 1.3);
      expect(rec.call('create_task', {'title': 'a'}, outTokens: 120, t: 5.0), {'status': 'ok', 'id': 'create_task-1'});
      rec.call('update_doc', {'doc': 'QA budget'}, outTokens: 120, t: 5.01);
      rec.text('Done.', 8.0);
      rec.finish(9.0, outTokens: 150);

      expect(run.turns, hasLength(2));
      final [t1, t2] = run.turns;
      expect((t1.start, t1.firstChunk, t1.end, t1.calls, t1.outTokens), (0, 1.0, 5.0, 2, 120));
      expect(t1.thought, [(1.0, 'Let me plan.'), (1.25, ' Next.')], reason: '250 ms buckets');
      expect((t2.start, t2.firstChunk, t2.end, t2.calls, t2.outTokens), (5.01, 8.0, 9.0, 0, 30));
      expect(run.finalText, 'Done.');
      expect(run.calls.map((c) => c.turn), [1, 1]);
      expect(run.runS, 9.0);
    });

    test('turns that stream nothing are told apart by the output-token count', () {
      final run = _run();
      final rec = RunRecorder(run, maxCalls: 40, tools: _any)..begin(0);
      rec.call('create_task', {}, outTokens: 300, t: 84);
      rec.call('create_task', {}, outTokens: 300, t: 84);
      rec.call('send_email', {}, outTokens: 340, t: 90); // a second turn, no chunks
      rec.finish(95, outTokens: 352); // a final turn, no chunks either

      expect(run.turns.map((t) => (t.n, t.start, t.end, t.calls, t.outTokens)), [
        (1, 0, 84, 2, 300),
        (2, 84, 90, 1, 40),
        (3, 90, 95, 0, 12),
      ]);
      expect(run.finalText, '');
    });

    test('a thinking turn that hits the output cap is flagged', () {
      final run = _run(maxOutputTokens: 512);
      final rec = RunRecorder(run, maxCalls: 40, tools: _any)..begin(0);
      rec.thinking('hmm', 1);
      rec.finish(30, outTokens: 512);
      expect(run.ranOutWhileThinking, isTrue);
      expect(run.turns.single.calls, 0);
    });

    test('calls past the limit are refused and kept as attempts', () {
      final run = _run();
      final rec = RunRecorder(run, maxCalls: 2, tools: _any)..begin(0);
      rec.call('a', {}, outTokens: 1, t: 1);
      rec.call('b', {}, outTokens: 1, t: 1);
      expect(rec.call('c', {}, outTokens: 1, t: 1), {'status': 'error', 'error': 'tool call limit reached'});
      expect(run.calls.map((c) => c.tool), ['a', 'b']);
      expect(run.attempts.map((c) => (c.tool, c.executed, c.rejected)), [
        ('a', true, null),
        ('b', true, null),
        ('c', false, null),
      ]);
    });

    test('a run that fails before generating keeps its time and no turns', () {
      final run = _run();
      RunRecorder(run, maxCalls: 40, tools: _any).finish(0.5, outTokens: 0);
      expect(run.turns, isEmpty);
      expect(run.runS, 0.5);
    });
  });

  group('calls are checked against the tool schemas', () {
    RunRecorder rec0(AgentRun run, {int maxCalls = 40}) =>
        RunRecorder(run, maxCalls: maxCalls, tools: _tools)..begin(0);
    const task = {'title': 'Sign-off', 'owner': 'a@x.io', 'due': '2026-11-13', 'follow_up': 1};

    test('a call that fits runs', () {
      final run = _run();
      expect(rec0(run).call('create_task', task, outTokens: 1, t: 1), {'status': 'ok', 'id': 'create_task-1'});
      expect(run.calls.single.rejected, isNull);
    });

    for (final (name, tool, args, why) in [
      ('an unknown tool', 'book_flight', <String, Object?>{}, startsWith('unknown tool "book_flight"; the tools are')),
      ('a missing required argument', 'create_task', {...task}..remove('owner'), 'missing required argument "owner"'),
      ('a null required argument', 'create_task', {...task, 'owner': null}, 'missing required argument "owner"'),
      (
        'a string for an array',
        'send_email',
        {'to': 'a@x.io', 'subject': 's', 'body': 'b', 'follow_up': 1},
        'argument "to" must be an array, got a string',
      ),
      (
        'a number in an array of strings',
        'send_email',
        {
          'to': ['a@x.io', 7],
          'subject': 's',
          'body': 'b',
          'follow_up': 1,
        },
        'argument "to[1]" must be a string, got an integer',
      ),
      (
        'a string for an integer',
        'create_task',
        {...task, 'follow_up': '1'},
        'argument "follow_up" must be an integer, got a string',
      ),
      (
        'a fraction for an integer',
        'create_task',
        {...task, 'follow_up': 1.5},
        'argument "follow_up" must be an integer, got a number',
      ),
      (
        'a list item of the wrong shape',
        MeetingTask.listTool,
        {
          'items': [
            {'id': 1, 'what': 'a', 'who': 'b'},
            {'id': '2', 'what': 'c', 'who': 'd'},
          ],
        },
        'argument "items[1].id" must be an integer, got a string',
      ),
      (
        'a list item without a required field',
        MeetingTask.listTool,
        {
          'items': [
            {'id': 1, 'what': 'a'},
          ],
        },
        'missing required argument "items[0].who"',
      ),
      ('items as a string', MeetingTask.listTool, {'items': '[]'}, 'argument "items" must be an array, got a string'),
    ]) {
      test('$name is answered with an error and kept as an attempt only', () {
        final run = _run();
        final reply = rec0(run).call(tool, args, outTokens: 1, t: 1);
        expect(reply['status'], 'error');
        expect(reply['error'], why);
        expect(run.calls, isEmpty);
        final a = run.attempts.single;
        expect((a.executed, a.rejected), (false, reply['error']));
      });
    }

    test('an integral number, an undeclared argument and a null optional one are accepted', () {
      final run = _run();
      final reply = rec0(run)
          .call('create_task', {...task, 'follow_up': 1.0, 'priority': 'high', 'notes': null}, outTokens: 1, t: 1);
      expect(reply['status'], 'ok');
    });

    test('rejected calls do not use up the call limit; refused ones say nothing more', () {
      final run = _run();
      final rec = rec0(run, maxCalls: 1);
      rec.call('create_task', {}, outTokens: 1, t: 1);
      rec.call('create_task', task, outTokens: 1, t: 1);
      expect(rec.call('create_task', task, outTokens: 1, t: 1), {
        'status': 'error',
        'error': 'tool call limit reached',
      });
      expect(run.attempts.map((c) => (c.executed, c.rejected != null)), [(false, true), (true, false), (false, false)]);
    });
  });
}
