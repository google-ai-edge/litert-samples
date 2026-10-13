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
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter_edge_ai_agent/flutter_edge_ai_agent.dart'
    show ErrorResult, Skill, SkillExecutor, SkillResult, SkillType;

import 'app_intents.dart';

/// What one intent does with its parsed parameters. A bad parameter throws
/// [IntentParamException]; the executor turns that, and any other throw, into
/// an [ErrorResult] the model can act on.
typedef AppIntentHandler = FutureOr<SkillResult> Function(IntentParams params);

/// One intent: its handler and the parameters it takes, shown in error
/// results so the model can call it again correctly.
final class const AppIntentSpec({
  required final AppIntentHandler handler,

  /// An example `parameters` value; `{}` for an intent that takes none (as
  /// both current ones do).
  final String usage = '{}',
});

/// A parameter the model got wrong; the message says how to fix it.
final class IntentParamException implements Exception {
  const IntentParamException(this.message);

  final String message;

  @override
  String toString() => message;
}

/// Runs every app intent. Registered with
/// priority 10 next to `TextSkillExecutor`; core never sees a
/// `NativeIntentExecutor`, whose validation rejects custom intents.
///
/// The intent comes from the probed skill's name, which the agent loop sets
/// to the `intent` argument of `runIntent` (`agent_loop.dart:400-423`), or
/// from `{"intent", "parameters"}` inside the data. A call that names a
/// skill (the model passed `skillName`), an unknown intent, unparsable
/// parameters or a handler that throws all come back as an [ErrorResult]
/// listing what to do instead: [execute] never throws.
final class AppIntentExecutor extends SkillExecutor {
  /// [intents] must cover [AppIntent.all] exactly: a missing handler would
  /// only show up when the model calls it.
  ///
  /// [timeout]: the longest a handler may take. The agent loop awaits
  /// `execute` with no timeout of its own (`AgentLoop._runExecutor`), so a
  /// handler that never completed would hang the turn.
  AppIntentExecutor(
    Map<String, AppIntentSpec> intents, {
    this.timeout = const Duration(seconds: 3),
    this._skillNamed,
  }) : _intents = Map.unmodifiable(intents) {
    final keys = intents.keys.toSet();
    if (!setEquals(keys, AppIntent.all)) {
      throw ArgumentError.value(
        keys,
        'intents',
        'must have exactly one handler per AppIntent (${AppIntent.listed})',
      );
    }
  }

  final Map<String, AppIntentSpec> _intents;
  final Duration timeout;

  /// Finds a loaded skill by name. The agent loop resolves a skill
  /// only from a `skillName` argument, so `runIntent(intent: "kid-clock")`
  /// arrives as a nameless skill; looked up here, its intents can be listed
  /// (or its one literal call run) and the turn does not end without the
  /// intent (seen in skills_test with a runtime skill).
  final Skill? Function(String name)? _skillNamed;

  @override
  String get name => 'AppIntentExecutor';

  /// Probed before the in-package executors (priority 0).
  @override
  int get priority => 10;

  /// On the type alone: core probes with a type-only skill whose other
  /// fields throw (`skill_executor.dart:42-45`).
  @override
  bool canExecuteSkill(Skill skill) => skill.type == SkillType.intent;

  /// Runs [intent] with [paramsJson] the way a `runIntent` call
  /// would, for the requests the app recognizes itself
  /// (`SkillQuestionRouter.action`).
  Future<SkillResult> run(String intent, String paramsJson) => execute(
    Skill(
      name: intent,
      description: '',
      instructions: '',
      type: SkillType.intent,
    ),
    paramsJson,
  );

  @override
  Future<SkillResult> execute(
    Skill skill,
    String dataJson, {
    String? secret,
  }) async {
    final String intent;
    final String data;
    switch (_resolve(skill, dataJson)) {
      case (final String resolved, final String resolvedData):
        intent = resolved;
        data = resolvedData;
      case null:
        // A skill's name passed as the intent (`device-info` and
        // `current-time` normalize to their only intent and run; a runtime
        // skill's name does not): name exactly that skill's intents with
        // their parameters, so a real request recovers in one round.
        final known = skill.instructions.isNotEmpty
            ? skill
            : _skillNamed?.call(skill.name);
        final offered = known == null
            ? const <String>[]
            : _intentsNamedIn(known.instructions);
        // A skill whose instructions name one intent with literal
        // parameters (kid-clock: current_time {}) fully determines the
        // call: run it. Otherwise the model was seen claiming the action
        // without calling the intent.
        if (offered.length == 1) {
          if (_literalParams(known!.instructions) case final params?) {
            debugPrint(
              '[Skills] "${skill.name}" called as an intent: running its '
              'only intent ${offered.single} $params',
            );
            return execute(
              Skill(
                name: offered.single,
                description: '',
                instructions: '',
                type: SkillType.intent,
              ),
              params,
            );
          }
        }
        if (offered.isNotEmpty) {
          final options = [
            for (final name in offered)
              '$name with parameters ${_intents[name]!.usage}',
          ].join('; ');
          // Its instructions too (what loadSkill would have returned): a
          // runtime skill names exact parameters there.
          return ErrorResult(
            '"${skill.name}" is a skill, not an intent. Its intents: '
            '$options. Its instructions: ${known!.instructions.trim()} '
            'Call run_intent again with one of its intents and the '
            "parameters from the user's request.",
          );
        }
        final what = skill.instructions.isNotEmpty
            ? '"${skill.name}" is a skill, not an intent.'
            : 'Unknown intent "${skill.name}".';
        // An invented intent is mostly a plain question the
        // model tried to route through a tool (photo questions in
        // particular); listing the intents alone invited it to pick one.
        return ErrorResult(
          '$what If the user did not ask for one of your skills, answer them '
          'directly with no tool call. Otherwise call run_intent with intent '
          'set to one of: ${AppIntent.listed}.',
        );
    }
    // Present: the constructor checked the keys and _resolve returns only
    // AppIntent.all members.
    final spec = _intents[intent]!;
    try {
      final result = await Future.sync(() => spec.handler(_parse(data)))
          .timeout(timeout);
      debugPrint('[Skills] $intent ${_short(data)} → $result');
      return result;
    } on TimeoutException {
      debugPrint('[Skills] $intent timed out after ${timeout.inSeconds}s');
      return ErrorResult(
        '$intent timed out after ${timeout.inSeconds} seconds. Tell the user '
        'it did not work.',
      );
    } on IntentParamException catch (e) {
      debugPrint('[Skills] $intent ${_short(data)} rejected: $e');
      return ErrorResult(
        '$intent: $e. Call run_intent again with parameters like '
        '${spec.usage}.',
      );
    } catch (e, st) {
      debugPrint('[Skills] $intent failed: $e\n$st');
      return ErrorResult('$intent failed: $e. Tell the user it did not work.');
    }
  }

  /// The intent and its parameters: the probed name, else
  /// `{"intent": …, "parameters": …}` in the data (a model that put both
  /// into `parameters`). Null when neither names an app intent.
  static (String, String)? _resolve(Skill skill, String dataJson) {
    if (AppIntent.normalize(skill.name) case final intent?) {
      return (intent, dataJson);
    }
    final Object? decoded;
    try {
      decoded = jsonDecode(dataJson);
    } on FormatException {
      return null;
    }
    if (decoded is! Map<String, Object?>) return null;
    final raw = decoded['intent'];
    final intent = raw is String ? AppIntent.normalize(raw) : null;
    if (intent == null) return null;
    return (
      intent,
      switch (decoded['parameters']) {
        null => '',
        final String text => text,
        final other => jsonEncode(other),
      },
    );
  }

  /// The one JSON object literal in [instructions] (the parameters a
  /// single-call skill spells out); null when there is none or several.
  static String? _literalParams(String instructions) {
    final found = <String>[];
    for (final m in RegExp(r'\{[^{}]*\}').allMatches(instructions)) {
      try {
        if (jsonDecode(m[0]!) is Map<String, Object?>) found.add(m[0]!);
      } on FormatException {
        // Not JSON: prose in braces.
      }
    }
    return found.length == 1 ? found.single : null;
  }

  /// The app intents [instructions] name, in order of first mention.
  static List<String> _intentsNamedIn(String instructions) => [
    ...{
      for (final m in RegExp(r'[a-z_]+').allMatches(instructions))
        if (AppIntent.all.contains(m[0])) m[0]!,
    },
  ];

  /// `''` and `null` are `{}`; anything but a JSON object is refused with a
  /// hint (a Dart map's `toString`, which the agent loop sends for map
  /// arguments, lands here).
  static IntentParams _parse(String data) {
    final trimmed = data.trim();
    if (trimmed.isEmpty) return IntentParams(const {});
    final Object? decoded;
    try {
      decoded = jsonDecode(trimmed);
    } on FormatException {
      throw const IntentParamException(_jsonHint);
    }
    return switch (decoded) {
      null => IntentParams(const {}),
      final Map<String, Object?> map => IntentParams(map),
      _ => throw const IntentParamException(_jsonHint),
    };
  }

  static const _jsonHint = 'parameters must be a JSON string, e.g. {}';

  static String _short(String data) =>
      data.length <= 80 ? data : '${data.substring(0, 80)}…';
}

/// One intent call's parameters, with the coercions small models need: a
/// number may come as `"10"` or `10.0`, a label as a number.
final class IntentParams {
  IntentParams(Map<String, Object?> values) : _values = Map.of(values);

  final Map<String, Object?> _values;

  /// Keys whose value is not null.
  Set<String> get keys => {
    for (final MapEntry(:key, :value) in _values.entries)
      if (value != null) key,
  };

  /// A whole number of at least [min] (and at most [max]); null when absent.
  /// Accepts `10`, `10.0` and `"10"`; anything else throws.
  int? wholeNumber(String key, {int min = 0, int? max}) {
    final raw = _values[key];
    if (raw == null) return null;
    final value = switch (raw) {
      final int n => n,
      final double d when d == d.roundToDouble() && d.isFinite => d.toInt(),
      final String s => int.tryParse(s.trim()) ?? _wholeDouble(s),
      _ => null,
    };
    if (value == null) {
      throw IntentParamException(
        '$key must be a whole number, got ${jsonEncode(raw)}',
      );
    }
    if (value < min || (max != null && value > max)) {
      throw IntentParamException(
        '$key must be ${max == null ? 'at least $min' : 'from $min to $max'}, '
        'got $value',
      );
    }
    return value;
  }

  /// A finite amount of at least 0 (`1.5`, `"0.5"`, `3`); null when absent.
  double? amount(String key) {
    final raw = _values[key];
    if (raw == null) return null;
    final value = switch (raw) {
      final num n => n.toDouble(),
      final String s => double.tryParse(s.trim()),
      _ => null,
    };
    if (value == null || !value.isFinite) {
      throw IntentParamException(
        '$key must be a number, got ${jsonEncode(raw)}',
      );
    }
    if (value < 0) {
      throw IntentParamException('$key must be at least 0, got $value');
    }
    return value;
  }

  /// A number from [min] to [max]; null when absent. Accepts numbers and
  /// numeric strings.
  double? number(String key, {required double min, required double max}) {
    final raw = _values[key];
    if (raw == null) return null;
    final value = switch (raw) {
      final num n => n.toDouble(),
      final String s => double.tryParse(s.trim()),
      _ => null,
    };
    if (value == null || !value.isFinite) {
      throw IntentParamException(
        '$key must be a number, got ${jsonEncode(raw)}',
      );
    }
    if (value < min || value > max) {
      throw IntentParamException('$key must be from $min to $max, got $value');
    }
    return value;
  }

  /// Trimmed text of at most [maxLength] characters; null when absent or
  /// blank. A number is taken as its text.
  String? text(String key, {int maxLength = 40}) {
    final raw = _values[key];
    final value = switch (raw) {
      null => null,
      final String s => s.trim(),
      final num n => '$n',
      _ => throw IntentParamException(
        '$key must be text, got ${jsonEncode(raw)}',
      ),
    };
    if (value == null || value.isEmpty) return null;
    if (value.length > maxLength) {
      throw IntentParamException('$key must be at most $maxLength characters');
    }
    return value;
  }

  /// `true`/`false`, also as `"true"`/`"false"`; null when absent.
  bool? flag(String key) => switch (_values[key]) {
    null => null,
    final bool b => b,
    'true' => true,
    'false' => false,
    final other => throw IntentParamException(
      '$key must be true or false, got ${jsonEncode(other)}',
    ),
  };

  static int? _wholeDouble(String s) {
    final d = double.tryParse(s.trim());
    return d != null && d.isFinite && d == d.roundToDouble() ? d.toInt() : null;
  }
}
