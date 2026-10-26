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

import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:intel_npu_demo/core/engines/engine_slot.dart';
import 'package:intel_npu_demo/features/act4_agent/agent_run.dart';
import 'package:intel_npu_demo/features/act4_agent/agent_runner.dart';
import 'package:intel_npu_demo/features/act4_agent/meeting_input.dart';
import 'package:intel_npu_demo/features/act4_agent/meeting_task.dart';
import 'package:intel_npu_demo/features/act4_agent/workspace.dart';

/// One step of a scripted generation.
sealed class _Step {}

class _Think extends _Step {
  _Think(this.s);
  final String s;
}

class _Text extends _Step {
  _Text(this.s);
  final String s;
}

/// A call at an output-token count.
class _Call extends _Step {
  _Call(this.name, this.args, this.outTokens);
  final String name;
  final Map<String, dynamic> args;
  final int outTokens;
}

class _Fail extends _Step {}

class _Session implements InferenceModelSession {
  int out = 0;
  bool metricsFail = false;

  @override
  SessionMetrics getSessionMetrics() {
    if (metricsFail) throw StateError('metrics gone');
    return SessionMetrics(inputTokens: 900, outputTokens: out, timeToFirstTokenMs: 120, tokensPerSecond: 20);
  }

  @override
  dynamic noSuchMethod(Invocation i) => super.noSuchMethod(i);
}

/// The members of [InferenceChat] the runner uses; the tool loop is played
/// from a script, answering calls through `onToolCall` as the plugin does.
class _Chat implements InferenceChat {
  _Chat(this.script);

  final List<_Step> script;
  final queries = <Message>[];
  final replies = <Map<String, dynamic>>[];
  final _session = _Session();
  bool stopped = false;
  bool closed = false;
  bool hitMaxTurns = false;

  @override
  InferenceModelSession get session => _session;

  @override
  Future<void> addQueryChunk(Message message, [bool noTool = false, bool prefix = false]) async => queries.add(message);

  @override
  Stream<ModelResponse> generateChatResponseWithTools({
    required FutureOr<Map<String, dynamic>> Function(FunctionCallResponse call) onToolCall,
    int maxToolTurns = 8,
    bool Function()? isCancelled,
    void Function()? onMaxToolTurns,
  }) async* {
    for (final s in script) {
      if (isCancelled?.call() ?? false) return;
      switch (s) {
        case _Think(:final s):
          _session.out += 5;
          yield ThinkingResponse(s);
        case _Text(:final s):
          _session.out += 5;
          yield TextResponse(s);
        case _Call(:final name, :final args, :final outTokens):
          _session.out = outTokens;
          replies.add(await onToolCall(FunctionCallResponse(name: name, args: args)));
        case _Fail():
          throw StateError('decode failed');
      }
    }
    if (hitMaxTurns) onMaxToolTurns?.call();
  }

  @override
  Future<void> stopGeneration() async => stopped = true;

  @override
  Future<void> close() async => closed = true;

  @override
  dynamic noSuchMethod(Invocation i) => super.noSuchMethod(i);
}

class _Slot implements EngineSlot {
  _Slot(this.chat, {this.openFails = false});

  final _Chat chat;
  final bool openFails;
  List<Tool>? tools;
  String? system;

  @override
  Future<InferenceChat> openChat({
    required List<Tool> tools,
    required String systemInstruction,
    required bool enableThinking,
    required double temperature,
    required int topK,
    required int randomSeed,
    int? maxOutputTokens,
  }) async {
    if (openFails) throw StateError('no NPU');
    this.tools = tools;
    system = systemInstruction;
    return chat;
  }

  @override
  dynamic noSuchMethod(Invocation i) => super.noSuchMethod(i);
}

/// Stderr, kept for the test to read instead of printed.
class _Err implements Stdout {
  final lines = <String>[];

  @override
  void writeln([Object? o = '']) => lines.add('$o');

  @override
  dynamic noSuchMethod(Invocation i) => super.noSuchMethod(i);
}

/// Runs [body] with stderr captured.
Future<(T, List<String>)> _quiet<T>(Future<T> Function() body) async {
  final err = _Err();
  final r = await IOOverrides.runZoned(body, stderr: () => err);
  return (r, err.lines);
}

final _task = MeetingTask(
  meeting: const MeetingInput(title: 'Standup', text: '[00:01] Ana: ship it', source: 'standup.txt'),
  workspace: Workspace.empty,
);

AgentRunner _runner(_Slot slot, {int maxCalls = 40}) => AgentRunner(
  slot: slot,
  task: _task,
  settings: AgentSettings(think: true, greedy: true, seed: 1, temp: 1, maxOutputTokens: 512, maxCalls: maxCalls),
  run: AgentRun(model: 'm', requestedBackend: 'npu', think: true, greedy: true),
);

const _list = {
  'items': [
    {'id': 1, 'what': 'Ship it', 'who': 'Ana'},
  ],
};
const _task1 = {'title': 'Ship', 'owner': 'ana@x.io', 'due': '2026-11-13', 'follow_up': 1};

void main() {
  test('one request: the prompt goes in once, and the tool loop does the rest', () async {
    final chat = _Chat([
      _Think('plan'),
      _Call(MeetingTask.listTool, _list, 40),
      _Call(MeetingTask.createTask, _task1, 80),
      _Call(MeetingTask.createTask, {'title': 'no owner'}, 80),
      _Text('Done.'),
    ]);
    final slot = _Slot(chat);
    var updates = 0;
    final run = await _runner(slot).start(onUpdate: () => updates++);

    expect(chat.queries, hasLength(1), reason: 'a second message after a tool loop fails on the NPU');
    expect((chat.queries.single.text, chat.queries.single.isUser), (_task.user, true));
    expect(slot.tools!.map((t) => t.name), _task.tools.map((t) => t.name));
    expect(slot.system, _task.system);
    expect(chat.replies.map((r) => r['status']), ['ok', 'ok', 'error']);
    expect(chat.replies.last['error'], 'missing required argument "owner"');
    expect(run.calls.map((c) => c.tool), [MeetingTask.listTool, MeetingTask.createTask]);
    expect(run.attempts, hasLength(3));
    expect(run.turns.map((t) => t.calls), [1, 2, 0]);
    expect(run.finalText, 'Done.');
    expect(run.error, isNull);
    expect(run.metrics, {'input_tokens': 900, 'output_tokens': 85, 'ttft_ms': 120.0, 'tok_s': 20.0});
    expect(run.runS, isNotNull);
    expect(updates, greaterThanOrEqualTo(5));
  });

  test('a failing generation lands in the run, not in a throw', () async {
    final (run, err) = await _quiet(() => _runner(_Slot(_Chat([_Think('a'), _Fail()]))).start(onUpdate: () {}));
    expect(run.error, 'Bad state: decode failed');
    expect(err.single, allOf(startsWith('ACT4 run: Bad state: decode failed\n'), contains('#0')), reason: 'the stack');
    expect(run.runS, isNotNull);
    expect(run.turns.single.thoughtText, 'a');
  });

  test('a chat that does not open lands in the run', () async {
    final (run, _) = await _quiet(() => _runner(_Slot(_Chat([]), openFails: true)).start(onUpdate: () {}));
    expect(run.error, 'Bad state: no NPU');
    expect(run.turns, isEmpty);
    expect(run.metrics, isNull);
  });

  test('metrics that cannot be read are an error, not a throw', () async {
    final chat = _Chat([_Text('hi')]).._session.metricsFail = true;
    final (run, _) = await _quiet(() => _runner(_Slot(chat)).start(onUpdate: () {}));
    expect(run.error, 'reading the session metrics failed: Bad state: metrics gone');
    expect(run.metrics, isNull);
    expect(run.finalText, 'hi');
  });

  test('an earlier error is kept first', () async {
    final chat = _Chat([_Call(MeetingTask.createTask, _task1, 10), _Fail()])..hitMaxTurns = false;
    final runner = _runner(_Slot(chat));
    runner.run.error = 'stopped after 50 turns with calls';
    final (run, _) = await _quiet(() => runner.start(onUpdate: () {}));
    expect(run.error, 'stopped after 50 turns with calls; then Bad state: decode failed');
  });

  test('the turn limit is reported', () async {
    final chat = _Chat([_Call(MeetingTask.createTask, _task1, 10)])..hitMaxTurns = true;
    final run = await _runner(_Slot(chat), maxCalls: 5).start(onUpdate: () {});
    expect(run.error, 'stopped after 15 turns with calls');
  });

  test('stop: the generation is stopped and the run says so', () async {
    final chat = _Chat([_Think('a'), _Think('b'), _Text('c')]);
    late AgentRunner runner;
    var first = true;
    runner = _runner(_Slot(chat));
    final run = await runner.start(
      onUpdate: () {
        if (first) {
          first = false;
          unawaited(runner.stop());
        }
      },
    );
    expect(chat.stopped, isTrue);
    expect(run.stopped, isTrue);
    expect(run.turns.single.thoughtText, 'a', reason: 'the loop checks isCancelled before the next step');
    await runner.close();
    expect(chat.closed, isTrue);
  });
}
