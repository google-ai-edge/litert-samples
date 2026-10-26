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

/// Text that arrived over time, merged into 250 ms buckets: `(t, text)` with
/// `t` the bucket's start in seconds from the start of the run.
typedef Timed = List<(double, String)>;

/// One tool call the model made. [t] is seconds from the start of the run;
/// [executed] is false for a call that was not run: refused at the call limit
/// ([rejected] null), or rejected because it does not fit the tool's schema
/// ([rejected] says why, in the words the model was answered with).
class AgentCall {
  const AgentCall({
    required this.tool,
    required this.args,
    required this.t,
    this.turn,
    this.executed = true,
    this.rejected,
  });

  final String tool;
  final Map<String, Object?> args;
  final double t;
  final int? turn;
  final bool executed;
  final String? rejected;
}

/// One generation of the model: what it thought and wrote, and the calls it
/// ended with. The plugin hands over a turn's calls when the turn ends, so
/// [end] is also when its calls arrived.
class AgentTurn {
  AgentTurn({
    required this.n,
    required this.start,
    this.firstChunk,
    this.end,
    this.outTokens,
    this.calls = 0,
    Timed? thought,
    Timed? text,
  }) : thought = thought ?? [],
       text = text ?? [];

  final int n;
  final double start;
  double? firstChunk;
  double? end;

  /// Tokens this turn generated, thinking included.
  int? outTokens;
  int calls;
  final Timed thought;
  final Timed text;

  String get thoughtText => thought.map((b) => b.$2).join();
  String get textText => text.map((b) => b.$2).join();
}

/// One run of the meeting task, filled in while it happens.
class AgentRun {
  AgentRun({
    required this.model,
    required this.requestedBackend,
    required this.think,
    required this.greedy,
    this.maxOutputTokens,
    this.backend,
    this.loadS,
    this.runS,
    List<AgentCall>? calls,
    List<AgentCall>? attempts,
    List<AgentTurn>? turns,
    this.finalText = '',
    this.error,
    this.metrics,
    this.stopped = false,
    this.meeting,
  }) : calls = calls ?? [],
       attempts = attempts ?? [],
       turns = turns ?? [];

  /// The model file it ran, as given.
  final String model;
  final String requestedBackend;

  /// The backend the plugin actually opened; null until it has.
  String? backend;
  final bool think;
  final bool greedy;
  final int? maxOutputTokens;
  double? loadS;
  double? runS;

  /// Executed calls, in order: what the checklist scores.
  final List<AgentCall> calls;

  /// Every call the model made, refused and rejected ones included.
  final List<AgentCall> attempts;

  final List<AgentTurn> turns;
  String finalText;
  String? error;
  Map<String, Object?>? metrics;

  /// Stopped by the presenter (Stop), not finished by the model.
  bool stopped;

  /// The meeting it worked on: the transcript's title.
  final String? meeting;

  String get modelName => model.split(RegExp(r'[\\/]')).last;

  /// The last turn wrote nothing and called nothing, and used the whole
  /// output cap: it ran out of tokens while thinking. The plugin gives no
  /// finish reason, so this is read from the token count.
  bool get ranOutWhileThinking {
    if (turns.isEmpty || maxOutputTokens == null) return false;
    final last = turns.last;
    return last.end != null &&
        last.calls == 0 &&
        last.textText.trim().isEmpty &&
        (last.outTokens ?? 0) >= maxOutputTokens!;
  }

  /// Seconds to the first call, or null if there was none.
  double? get firstCallS => attempts.isEmpty ? null : attempts.first.t;
}

/// Builds a live [AgentRun] from what the plugin's tool loop
/// (`InferenceChat.generateChatResponseWithTools`) shows: the text and
/// thinking chunks it streams and the calls it hands to `onToolCall`.
///
/// The loop does not say where one generation ends and the next begins, so
/// the recorder reads it off the session's output-token count, which only
/// grows while the model generates: the calls of one turn arrive one after
/// another at the same count, and a call at a new count opens a new turn
/// even when that turn streamed nothing.
///
/// Every call is checked against the declared [tools] before it counts: an
/// unknown tool, a missing required argument or a value of the wrong JSON
/// type is answered with an error the model can act on, and kept in
/// [AgentRun.attempts] only. Arguments the schema does not declare are
/// allowed, as JSON Schema allows them by default, and a null optional
/// argument counts as absent.
class RunRecorder {
  RunRecorder(this.run, {required this.maxCalls, required List<Tool> tools})
    : _schemas = {for (final t in tools) t.name: t.parameters};

  final AgentRun run;
  final int maxCalls;
  final Map<String, Map<String, Object?>> _schemas;

  /// Output tokens at the end of the last closed turn, and at the last call.
  int _outAtTurnEnd = 0;
  int? _outAtCall;
  double _lastCallT = 0;

  static const _bucket = 0.25;

  AgentTurn get _turn => run.turns.last;

  void begin(double t) => run.turns.add(AgentTurn(n: 1, start: t));

  void thinking(String s, double t) => _append(_open(t).thought, s, t);

  void text(String s, double t) => _append(_open(t).text, s, t);

  /// Records a call and returns the tool's answer. Calls past [maxCalls] are
  /// refused; calls that do not fit their tool's schema are rejected.
  Map<String, Object?> call(String tool, Map<String, Object?> args, {required int outTokens, required double t}) {
    if (_turn.end != null && outTokens != _outAtCall) _next(_lastCallT);
    if (_turn.end == null) _close(outTokens, t);
    _outAtCall = outTokens;
    _lastCallT = t;
    _turn.calls++;
    final overLimit = run.calls.length >= maxCalls;
    final rejected = overLimit ? null : misfit(tool, args);
    final c = AgentCall(
      tool: tool,
      args: args,
      t: _round(t),
      turn: _turn.n,
      executed: !overLimit && rejected == null,
      rejected: rejected,
    );
    run.attempts.add(c);
    if (overLimit) return const {'status': 'error', 'error': 'tool call limit reached'};
    if (rejected != null) return {'status': 'error', 'error': rejected};
    run.calls.add(c);
    return {'status': 'ok', 'id': '$tool-${run.calls.length}'};
  }

  /// Why a call of [tool] with [args] cannot run, or null if it fits.
  String? misfit(String tool, Map<String, Object?> args) {
    final schema = _schemas[tool];
    if (schema == null) return 'unknown tool "$tool"; the tools are ${_schemas.keys.join(', ')}';
    return _misfit(schema, args, '');
  }

  /// The subset of JSON Schema the tools use: object (properties,
  /// required), array (items), string, integer, number, boolean.
  static String? _misfit(Map<String, Object?> schema, Object? v, String at) {
    final type = schema['type'];
    String wrong(String want) => 'argument "$at" must be $want, got ${_kind(v)}';
    switch (type) {
      case 'object':
        if (v is! Map) return wrong('an object');
        for (final r in (schema['required'] as List?) ?? const []) {
          if (v[r] == null) return 'missing required argument "${_path(at, '$r')}"';
        }
        final props = (schema['properties'] as Map?) ?? const {};
        for (final MapEntry(:key, :value) in props.entries) {
          if (v[key] == null) continue;
          final why = _misfit((value as Map).cast(), v[key], _path(at, '$key'));
          if (why != null) return why;
        }
        return null;
      case 'array':
        if (v is! List) return wrong('an array');
        final items = schema['items'];
        if (items is! Map) return null;
        for (final (i, e) in v.indexed) {
          final why = _misfit(items.cast(), e, '$at[$i]');
          if (why != null) return why;
        }
        return null;
      case 'string':
        return v is String ? null : wrong('a string');
      case 'integer':
        return v is int || (v is double && v == v.roundToDouble()) ? null : wrong('an integer');
      case 'number':
        return v is num ? null : wrong('a number');
      case 'boolean':
        return v is bool ? null : wrong('true or false');
      case null:
        return null;
      default:
        throw StateError('tool schema at "$at": type "$type" is not supported by the recorder');
    }
  }

  static String _path(String at, String key) => at.isEmpty ? key : '$at.$key';

  /// The loop has returned (or failed). A turn still open is the final one;
  /// tokens generated after the last call with no chunk are a final turn
  /// that streamed nothing.
  void finish(double t, {required int outTokens}) {
    run.runS = _round(t);
    if (run.turns.isEmpty) return; // failed before the first generation
    if (_turn.end == null) {
      _close(outTokens, t);
    } else if (outTokens != (_outAtCall ?? _outAtTurnEnd)) {
      _next(_lastCallT);
      _close(outTokens, t);
    }
    run.finalText = _turn.calls == 0 ? _turn.textText : '';
  }

  AgentTurn _open(double t) {
    if (_turn.end != null) _next(_lastCallT);
    _turn.firstChunk ??= _round(t);
    return _turn;
  }

  void _next(double start) => run.turns.add(AgentTurn(n: _turn.n + 1, start: _round(start)));

  void _close(int outTokens, double t) {
    _turn
      ..end = _round(t)
      ..outTokens = outTokens - _outAtTurnEnd;
    _outAtTurnEnd = outTokens;
  }

  static void _append(Timed buckets, String s, double t) {
    final start = (t / _bucket).floor() * _bucket;
    if (buckets.isNotEmpty && buckets.last.$1 == start) {
      buckets.last = (start, buckets.last.$2 + s);
    } else {
      buckets.add((start, s));
    }
  }

  static double _round(double t) => (t * 100).round() / 100;
}

/// A JSON value's kind, in the words of an error message.
String _kind(Object? v) => switch (v) {
  null => 'null',
  String() => 'a string',
  int() => 'an integer',
  double() => 'a number',
  bool() => 'true/false',
  List() => 'an array',
  Map() => 'an object',
  _ => v.runtimeType.toString(),
};
