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

import 'package:fake_async/fake_async.dart';
import 'package:flutter_edge_ai_agent/flutter_edge_ai_agent.dart'
    show ErrorResult, Skill, SkillResult, SkillType, TextResult;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/domain/skills/app_intent_executor.dart';
import 'package:litert_edge_demos/domain/skills/app_intents.dart';

/// What the agent loop hands the executor for `runIntent(intent, …)`: a
/// synthetic skill named after the intent (`agent_loop.dart:400-423`).
Skill intent(String name) => Skill(
  name: name,
  description: '',
  instructions: '',
  type: SkillType.intent,
);

/// A registered skill, passed instead when the model adds a `skillName`
/// that resolves (`agent_loop.dart:332-333`).
const hardwareSkill = Skill(
  name: 'hardware',
  description: 'Hardware facts.',
  instructions: 'Call the `run_intent` tool with intent `device_info`.',
  type: SkillType.intent,
);

void main() {
  late List<(String, IntentParams)> calls;
  late AppIntentExecutor executor;

  AppIntentHandler recording(String name, [SkillResult? result]) => (params) {
    calls.add((name, params));
    return result ?? TextResult('$name ok');
  };

  setUp(() {
    calls = [];
    executor = AppIntentExecutor({
      for (final name in AppIntent.all)
        name: AppIntentSpec(handler: recording(name), usage: '{"minutes": 3}'),
    });
  });

  Future<String> errorOf(Skill skill, String data) async {
    final result = await executor.execute(skill, data);
    expect(result, isA<ErrorResult>(), reason: '$result');
    return (result as ErrorResult).message;
  }

  group('contract', () {
    test('priority 10, probed on the type alone (core probes a type-only '
        'skill whose other fields throw)', () {
      expect(executor.priority, 10);
      expect(executor.canExecute('intent'), isTrue);
      expect(executor.canExecute('text'), isFalse);
      expect(executor.canExecute('js'), isFalse);
    });

    test('a handler map that does not cover AppIntent.all is refused at '
        'construction', () {
      expect(
        () => AppIntentExecutor({
          AppIntent.deviceInfo: AppIntentSpec(handler: recording('x')),
        }),
        throwsArgumentError,
      );
    });
  });

  group('resolving the intent', () {
    test('from the probed name, lowercased, - → _', () async {
      expect(
        await executor.execute(intent('Current-Time'), '{}'),
        isA<TextResult>(),
      );
      expect(calls.single.$1, AppIntent.currentTime);
    });

    test('from {"intent", "parameters"} inside the data', () async {
      await executor.execute(
        intent('runIntent'),
        '{"intent": "current_time", "parameters": {"minutes": 3}}',
      );
      expect(calls.single.$1, AppIntent.currentTime);
      expect(calls.single.$2.wholeNumber('minutes'), 3);
    });

    test('a skill name is not an intent: retry with the intents that skill '
        'names', () async {
      final message = await errorOf(hardwareSkill, '{"verbose": true}');
      expect(message, contains('"hardware" is a skill, not an intent'));
      expect(message, contains('Call run_intent again'));
      expect(message, contains('Its intents: device_info with parameters'));
      expect(message, isNot(contains(AppIntent.currentTime)));
      expect(calls, isEmpty);
    });

    test('an unknown intent: retry with one of AppIntent.all', () async {
      final message = await errorOf(intent('make_coffee'), '{}');
      expect(message, contains('Unknown intent "make_coffee"'));
      // An invented intent is usually a plain question the
      // model tried to route through a tool; send it back to answering.
      expect(message, contains('answer them directly with no tool call'));
      expect(message, contains('call run_intent with intent set to one of'));
      expect(message, contains(AppIntent.listed));
    });
  });

  group('parameters', () {
    test('{"minutes": 3} arrives as 3', () async {
      await executor.execute(intent('device_info'), '{"minutes":3}');
      expect(calls.single.$2.wholeNumber('minutes'), 3);
    });

    test('"10" and 10.0 are coerced to 10', () async {
      await executor.execute(
        intent('device_info'),
        '{"seconds": "10", "minutes": 2.0}',
      );
      expect(calls.single.$2.wholeNumber('seconds'), 10);
      expect(calls.single.$2.wholeNumber('minutes'), 2);
    });

    test("'' and null are {}", () async {
      await executor.execute(intent('current_time'), '');
      await executor.execute(intent('current_time'), 'null');
      expect(calls.map((c) => c.$2.keys), [<String>{}, <String>{}]);
    });

    test("a Dart map's toString gets a JSON hint and the usage", () async {
      final message = await errorOf(intent('device_info'), '{minutes: 3}');
      expect(message, contains('parameters must be a JSON string, e.g. {}'));
      expect(message, contains('device_info'));
      expect(calls, isEmpty);
    });

    test('a JSON value that is not an object is refused', () async {
      await errorOf(intent('device_info'), '[3]');
      await errorOf(intent('device_info'), '"three minutes"');
    });

    test('IntentParams: ranges, text, flags', () {
      final p = IntentParams({
        'n': -1,
        'x': 'ten',
        'half': 1.5,
        'label': '  tea  ',
        'num': 7,
        'flag': 'true',
        'score': '0.95',
      });
      expect(
        () => p.wholeNumber('n'),
        throwsA(
          isA<IntentParamException>().having(
            (e) => e.message,
            'message',
            'n must be at least 0, got -1',
          ),
        ),
      );
      expect(() => p.wholeNumber('x'), throwsA(isA<IntentParamException>()));
      expect(() => p.wholeNumber('half'), throwsA(isA<IntentParamException>()));
      expect(p.wholeNumber('absent'), isNull);
      expect(p.text('label'), 'tea');
      expect(p.text('num'), '7');
      expect(
        () => p.text('label', maxLength: 2),
        throwsA(isA<IntentParamException>()),
      );
      expect(p.flag('flag'), isTrue);
      expect(
        () => p.number('score', min: 0.3, max: 0.9),
        throwsA(isA<IntentParamException>()),
      );
    });
  });

  group('handler failures never throw out of execute', () {
    test(
      'an IntentParamException becomes a retryable error with the usage',
      () async {
        executor = AppIntentExecutor({
          for (final name in AppIntent.all)
            name: AppIntentSpec(
              usage: '{"minutes": 3, "label": "tea"}',
              handler: (_) =>
                  throw const IntentParamException('the query is missing'),
            ),
        });
        final message = await errorOf(intent('device_info'), '{}');
        expect(message, contains('device_info: the query is missing'));
        expect(message, contains('{"minutes": 3, "label": "tea"}'));
      },
    );

    test('a handler that never completes times out after 3 s: the agent '
        'loop awaits execute with no timeout of its own', () {
      fakeAsync((async) {
        executor = AppIntentExecutor({
          for (final name in AppIntent.all)
            name: AppIntentSpec(
              handler: (_) => Completer<SkillResult>().future,
            ),
        });
        SkillResult? result;
        unawaited(
          executor.execute(intent('device_info'), '{}').then((r) => result = r),
        );

        async.elapse(const Duration(milliseconds: 2999));
        expect(result, isNull);
        async.elapse(const Duration(milliseconds: 1));

        expect(
          result,
          isA<ErrorResult>().having(
            (r) => r.message,
            'message',
            allOf(contains('device_info'), contains('timed out')),
          ),
        );
      });
    });

    test('any other throw becomes an error the model relays', () async {
      executor = AppIntentExecutor({
        for (final name in AppIntent.all)
          name: AppIntentSpec(handler: (_) => throw StateError('disk full')),
      });
      final message = await errorOf(intent('device_info'), '{}');
      expect(message, contains('device_info failed: Bad state: disk full'));
    });
  });

  group('a skill name passed as the intent', () {
    /// The bundled skill as the registry holds it.
    Skill bundled(String name) {
      final text = File('assets/skills/$name/SKILL.md').readAsStringSync();
      final body = text.substring(text.indexOf('---', 3) + 3).trim();
      return Skill(
        name: name,
        description: '',
        instructions: body,
        type: SkillType.intent,
      );
    }

    late AppIntentExecutor real;

    setUp(() {
      real = AppIntentExecutor({
        for (final name in AppIntent.all)
          name: AppIntentSpec(handler: recording(name)),
      });
    });

    test('a skill with exactly one parameterless intent runs it '
        '(device-info → device_info, current-time → current_time)', () async {
      for (final (skill, expected) in [
        ('device-info', AppIntent.deviceInfo),
        ('current-time', AppIntent.currentTime),
      ]) {
        calls.clear();
        final result = await real.execute(bundled(skill), '{}');
        expect(result, isA<TextResult>(), reason: skill);
        expect(calls.single.$1, expected);
      }
    });
  });

  test('run(intent, params) is a runIntent call made by the app', () async {
    final result = await executor.run('current_time', '{}');
    expect(result, isA<TextResult>());
    expect(calls.single.$1, AppIntent.currentTime);
    expect(calls.single.$2.keys, isEmpty);
  });

  test('a skill name passed as the intent with no skill attached (the '
      'agent loop only resolves a skillName argument) is looked up by name, '
      'so the listing names its intents and the model can retry', () async {
    final kid = const Skill(
      name: 'kid-clock',
      description: '',
      instructions:
          'Call the `run_intent` tool with intent `current_time` and '
          'parameters {}.',
      type: SkillType.intent,
    );
    final lookup = AppIntentExecutor({
      for (final name in AppIntent.all)
        name: AppIntentSpec(handler: recording(name)),
    }, skillNamed: (name) => name == 'kid-clock' ? kid : null);
    // Seen in a showcase run: the model claimed the action without
    // calling the intent. A skill whose instructions name exactly one
    // intent with literal parameters fully determines the call, so it runs.
    final result = await lookup.execute(intent('kid-clock'), '{}');
    expect(result, isA<TextResult>(), reason: '$result');
    expect(calls.single.$1, AppIntent.currentTime);
  });

  test('a skill with several intents or no literal parameters is '
      'listed, never run', () async {
    final both = const Skill(
      name: 'time-and-device',
      description: '',
      instructions:
          'Call the `run_intent` tool with intent `current_time` and '
          'parameters {}, then intent `device_info` and parameters {}.',
      type: SkillType.intent,
    );
    final vague = const Skill(
      name: 'device-vague',
      description: '',
      instructions: 'Call the `run_intent` tool with intent `device_info`.',
      type: SkillType.intent,
    );
    final lookup = AppIntentExecutor(
      {
        for (final name in AppIntent.all)
          name: AppIntentSpec(handler: recording(name)),
      },
      skillNamed: (name) => switch (name) {
        'time-and-device' => both,
        'device-vague' => vague,
        _ => null,
      },
    );
    for (final name in ['time-and-device', 'device-vague']) {
      final result = await lookup.execute(intent(name), '{}');
      expect(result, isA<ErrorResult>(), reason: name);
    }
    expect(calls, isEmpty);
  });
}
