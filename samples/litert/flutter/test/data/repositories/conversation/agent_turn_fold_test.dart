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

import 'package:flutter/foundation.dart';
import 'package:flutter_edge_ai_agent/flutter_edge_ai_agent.dart'
    show
        AgentErrorEvent,
        DoneEvent,
        ErrorResult,
        ImageResult,
        MaxIterationsEvent,
        SkillLoadEvent,
        TextChunkEvent,
        TextResult,
        ToolCallEvent,
        ToolResultEvent;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/repositories/conversation/agent_turn_fold.dart';
import 'package:litert_edge_demos/domain/models/skill_step.dart';

const _rawToolCall =
    '{"role":"assistant","tool_calls":[{"type":"function","function":'
    '{"name":"loadSkill","arguments":{"skillName":"ma';

void main() {
  late Duration now;
  late List<SkillStep> heard;
  late AgentTurnFold fold;
  late List<String> log;
  late DebugPrintCallback originalDebugPrint;

  setUp(() {
    now = Duration.zero;
    heard = [];
    fold = AgentTurnFold(clock: () => now, onStep: heard.add);
    log = [];
    originalDebugPrint = debugPrint;
    debugPrint = (message, {wrapWidth}) => log.add(message ?? '');
  });

  tearDown(() => debugPrint = originalDebugPrint);

  test('reply text passes through; empty chunks do not', () {
    expect(
      fold.on(const TextChunkEvent('It is '), stopRequested: false),
      'It is ',
    );
    expect(fold.on(const TextChunkEvent(''), stopRequested: false), isNull);
    expect(fold.steps, isEmpty);
  });

  test('after a stop, text is dropped (partial tool-call JSON or a cut '
      'reply)', () {
    expect(fold.on(const TextChunkEvent('noon'), stopRequested: true), isNull);
    expect(
      fold.on(const TextChunkEvent(_rawToolCall), stopRequested: true),
      isNull,
    );
    expect(fold.steps, isEmpty, reason: 'nothing reported after a stop');
  });

  test('a tool call the SDK could not parse is never shown: one failed step '
      'without an intent, logged', () {
    now = const Duration(milliseconds: 40);

    expect(
      fold.on(const TextChunkEvent('  $_rawToolCall'), stopRequested: false),
      isNull,
    );

    final step = fold.steps.single as IntentFailed;
    expect(step.intent, isNull);
    expect(step.message, contains('could not be read'));
    expect(step.at, now);
    expect(heard, fold.steps);
    expect(
      log.where((l) => l.contains('unparsed tool call dropped')),
      hasLength(1),
    );
    expect(fold.toolRounds, 0);
  });

  test('a brace-leading answer without tool_calls is text', () {
    expect(
      fold.on(const TextChunkEvent('{"a": 1}'), stopRequested: false),
      '{"a": 1}',
    );
  });

  test('loadSkill → runIntent → result: three steps with their times, two '
      'tool rounds, the intent named from the call', () {
    now = const Duration(milliseconds: 100);
    fold.on(const SkillLoadEvent('current-time'), stopRequested: false);
    now = const Duration(milliseconds: 300);
    fold.on(
      const ToolCallEvent(
        toolName: 'runIntent',
        args: {'intent': 'current_time', 'parameters': '{}'},
      ),
      stopRequested: false,
    );
    now = const Duration(milliseconds: 350);
    fold.on(
      const ToolResultEvent(
        toolName: 'runIntent',
        result: TextResult('It is 2:03 PM.'),
      ),
      stopRequested: false,
    );

    expect(fold.toolRounds, 2);
    expect(fold.steps, hasLength(3));
    expect(heard, fold.steps);
    final [loaded, called, succeeded] = fold.steps;
    expect(
      loaded,
      isA<SkillLoaded>()
          .having((s) => s.name, 'name', 'current-time')
          .having((s) => s.found, 'found', isTrue)
          .having((s) => s.at, 'at', const Duration(milliseconds: 100)),
    );
    expect(
      called,
      isA<IntentCalled>()
          .having((s) => s.intent, 'intent', 'current_time')
          .having((s) => s.parameters, 'parameters', '{}')
          .having((s) => s.at, 'at', const Duration(milliseconds: 300)),
    );
    expect(
      succeeded,
      isA<IntentSucceeded>()
          .having((s) => s.intent, 'intent', 'current_time')
          .having((s) => s.result, 'result', 'It is 2:03 PM.')
          .having((s) => s.elapsed, 'elapsed', const Duration(milliseconds: 50))
          .having((s) => s.at, 'at', const Duration(milliseconds: 350)),
    );
  });

  test('a not-found skill load is a step and a tool round', () {
    fold.on(
      const SkillLoadEvent('weather', found: false),
      stopRequested: false,
    );

    expect(
      fold.steps.single,
      isA<SkillLoaded>().having((s) => s.found, 'found', isFalse),
    );
    expect(fold.toolRounds, 1);
  });

  test('tool arguments that are not strings are shown as JSON; without an '
      'intent the tool name stands in', () {
    fold.on(
      const ToolCallEvent(
        toolName: 'runIntent',
        args: {
          'parameters': {'city': 'Berlin'},
        },
      ),
      stopRequested: false,
    );

    expect(
      fold.steps.single,
      isA<IntentCalled>()
          .having((s) => s.intent, 'intent', 'runIntent')
          .having((s) => s.parameters, 'parameters', '{"city":"Berlin"}'),
    );
  });

  test('a result without a call before it: the tool name, no elapsed '
      'time', () {
    now = const Duration(milliseconds: 70);
    fold.on(
      const ToolResultEvent(toolName: 'runIntent', result: TextResult('ok')),
      stopRequested: false,
    );

    expect(
      fold.steps.single,
      isA<IntentSucceeded>()
          .having((s) => s.intent, 'intent', 'runIntent')
          .having((s) => s.elapsed, 'elapsed', Duration.zero),
    );
  });

  test('an image, widget or webview result is a success described by the '
      'result', () {
    fold.on(
      ToolResultEvent(toolName: 'runIntent', result: ImageResult(Uint8List(4))),
      stopRequested: false,
    );

    expect(
      fold.steps.single,
      isA<IntentSucceeded>().having(
        (s) => s.result,
        'result',
        'ImageResult(4 bytes)',
      ),
    );
  });

  test('an ErrorResult is one failed step: the loop\'s repeat of it as an '
      'error event is not a second one (agent_loop.dart:379)', () {
    fold
      ..on(
        const ToolCallEvent(
          toolName: 'runIntent',
          args: {'intent': 'make_coffee', 'parameters': '{}'},
        ),
        stopRequested: false,
      )
      ..on(
        const ToolResultEvent(
          toolName: 'runIntent',
          result: ErrorResult('Unknown intent "make_coffee"'),
        ),
        stopRequested: false,
      )
      ..on(
        const AgentErrorEvent(
          'Unknown intent "make_coffee"',
          toolName: 'runIntent',
        ),
        stopRequested: false,
      );

    final failed = fold.steps.whereType<IntentFailed>().single;
    expect(failed.intent, 'make_coffee');
    expect(failed.message, 'Unknown intent "make_coffee"');
  });

  test('an error event of its own is a failed step; the same message again '
      'after a repeat is one too', () {
    fold
      ..on(
        const ToolResultEvent(
          toolName: 'runIntent',
          result: ErrorResult('boom'),
        ),
        stopRequested: false,
      )
      ..on(const AgentErrorEvent('boom'), stopRequested: false)
      ..on(
        const AgentErrorEvent('no executor', toolName: 'runIntent'),
        stopRequested: false,
      )
      ..on(const AgentErrorEvent('boom'), stopRequested: false);

    expect(fold.steps.whereType<IntentFailed>().map((s) => s.message), [
      'boom',
      'no executor',
      'boom',
    ]);
    expect(
      fold.steps[1],
      isA<IntentFailed>().having((s) => s.intent, 'intent', 'runIntent'),
    );
  });

  test('a throwing step listener does not end the fold', () {
    final throwing = AgentTurnFold(
      clock: () => now,
      onStep: (_) => throw StateError('listener bug'),
    );

    throwing.on(const SkillLoadEvent('current-time'), stopRequested: false);

    expect(throwing.steps, hasLength(1));
    expect(log.where((l) => l.contains('onStep threw')), hasLength(1));
  });

  test('steps are logged with their times; no listener is fine', () {
    final quiet = AgentTurnFold(clock: () => const Duration(seconds: 1));

    quiet.on(const SkillLoadEvent('device-info'), stopRequested: false);

    expect(log.single, startsWith('[Conversation] skill step @1000ms: '));
  });

  test('the steps list given out is read-only', () {
    fold.on(const SkillLoadEvent('current-time'), stopRequested: false);

    expect(() => fold.steps.add(fold.steps.first), throwsUnsupportedError);
  });

  group('how the loop ended', () {
    test('DoneEvent: finished, not stopped unless a stop was requested', () {
      fold.on(const DoneEvent('It is noon.'), stopRequested: false);

      expect(fold.finished, isTrue);
      expect(fold.maxIterations, 0);
      expect(fold.stopped(stopRequested: false), isFalse);
      expect(fold.stopped(stopRequested: true), isTrue);
    });

    test('MaxIterationsEvent: not finished, not stopped, the cap kept', () {
      fold.on(const MaxIterationsEvent(5), stopRequested: false);

      expect(fold.finished, isFalse);
      expect(fold.maxIterations, 5);
      expect(fold.stopped(stopRequested: false), isFalse);
    });

    test('neither: the loop saw the cancel, so the turn was stopped', () {
      fold.on(const SkillLoadEvent('current-time'), stopRequested: false);

      expect(fold.finished, isFalse);
      expect(fold.stopped(stopRequested: false), isTrue);
    });
  });
}
