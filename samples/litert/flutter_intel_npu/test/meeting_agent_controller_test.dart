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
import 'package:intel_npu_demo/features/act4_agent/meeting_agent_controller.dart';
import 'package:intel_npu_demo/features/act4_agent/agent_run.dart';
import 'package:intel_npu_demo/features/act4_agent/meeting_task.dart';
import 'package:intel_npu_demo/features/act4_agent/workspace.dart';

void main() {
  AgentRun run() => AgentRun(
    model: '/models/m.litertlm',
    requestedBackend: 'npu',
    backend: 'npu',
    think: true,
    greedy: true,
    runS: 12.5,
    meeting: 'Weekly sync',
    calls: [
      AgentCall(
        tool: MeetingTask.listTool,
        args: {
          'items': [
            {'id': 1, 'what': 'Book the review', 'who': 'ana@x.test'},
            {'id': 2, 'what': 'Send the notes', 'who': 'assistant'},
          ],
        },
        t: 3,
      ),
      AgentCall(
        tool: MeetingTask.createTask,
        args: {'title': 'Review', 'owner': 'ana@x.test', 'due': '2026-11-20', 'follow_up': 1},
        t: 9,
      ),
    ],
  );

  Map<Object?, Object?> line(Workspace w) =>
      jsonDecode(MeetingAgentController.resultsLine(run(), w, now: DateTime(2026))) as Map;

  test('the result line counts against the model\'s own list, rules against the workspace', () {
    final j = line(Workspace.fromJson('{"internal_domain": "x.test"}'));
    expect((j['meeting'], j['listed'], j['done'], j['model'], j['run_s']), ('Weekly sync', 2, 1, 'm.litertlm', 12.5));
    expect(j['rules'], '3/3');
    expect((j['failed'] as Map).keys, ['Send the notes']);
  });

  test('a workspace without a domain or calendars checks only the rules about the list', () {
    final j = line(Workspace.empty);
    expect(j['rules'], '2/2');
    expect((j['listed'], j['done']), (2, 1));
  });
}
