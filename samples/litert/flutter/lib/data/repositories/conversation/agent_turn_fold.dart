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

import 'package:flutter/foundation.dart';
import 'package:flutter_edge_ai_agent/flutter_edge_ai_agent.dart'
    show
        AgentErrorEvent,
        AgentEvent,
        DoneEvent,
        EmptyAnswerEvent,
        EmptyAnswerRetryEvent,
        ErrorResult,
        ImageResult,
        MaxIterationsEvent,
        SkillLoadEvent,
        TextChunkEvent,
        TextResult,
        ToolCallEvent,
        ToolResultEvent,
        WebviewResult,
        WidgetResult;

import '../../../domain/models/skill_step.dart';

/// Folds one agent turn's [AgentEvent]s into what the repository reports: the
/// reply text to show ([on]'s result), the skill steps — each logged and passed
/// to the step listener as it happens — and the counts the turn ends with.
final class AgentTurnFold {
  /// [clock]: the turn's time, for each step's `at` and an intent's
  /// elapsed time. [onStep]: the caller's step listener.
  AgentTurnFold({required this._clock, this._onStep});

  final Duration Function() _clock;
  final void Function(SkillStep step)? _onStep;
  final List<SkillStep> _steps = [];
  int _toolRounds = 0;
  String? _runningIntent;
  Duration? _intentStartedAt;
  String? _lastErrorResult;
  bool _finished = false;
  int _maxIterations = 0;
  String? _noAnswerAfter;

  /// Raw tool-call JSON that came as text ([_isRawToolCall]), with when it
  /// came. Since flutter_edge_ai 2.4.0 a call the SDK reads never reaches the
  /// text, also one that follows prose; this is only core's unparsed-call
  /// fallback (`InferenceChat.generateChatResponseAsync` surfaces the call
  /// text it held back when `lastRawResponse` has no call it can read). That
  /// generation has no call, so it is the final one, and the [DoneEvent] that
  /// ends it reports each as a failed step ([_reportUnreadCalls]).
  final List<({String text, Duration at})> _rawCalls = [];

  /// The turn's skill steps so far, in order.
  List<SkillStep> get steps => List.unmodifiable(_steps);

  /// Tool calls so far: loadSkill and runIntent each count (every one cost
  /// a generation before the answer).
  int get toolRounds => _toolRounds;

  /// The model produced something: text (spoken or not) or a tool call.
  bool get responded => _textSeen || _toolRounds > 0;
  bool _textSeen = false;

  /// The loop ended with a final, call-free generation ([DoneEvent]).
  bool get finished => _finished;

  /// The generations the loop used up without an answer
  /// ([MaxIterationsEvent]); 0 when it did not run out.
  int get maxIterations => _maxIterations;

  /// The skill whose `loadSkill` came right before a final generation that
  /// stayed empty ([EmptyAnswerEvent]: after the loop's retry, or with none
  /// when no generation was left for it); null when the turn ended otherwise.
  String? get noAnswerAfter => _noAnswerAfter;

  /// Whether the turn ended stopped: a stop was requested, or the loop ended
  /// without [DoneEvent] or [MaxIterationsEvent] (it saw the cancel).
  bool stopped({required bool stopRequested}) =>
      stopRequested || (!_finished && _maxIterations == 0);

  /// Folds [event] in; returns the reply text it adds, if any.
  /// [stopRequested]: a stop has been requested for this turn.
  String? on(AgentEvent event, {required bool stopRequested}) {
    switch (event) {
      case TextChunkEvent(:final text):
        // After a stop: partial tool-call JSON (flutter_edge_ai's
        // `InferenceChat.generateChatResponseAsync` surfaces it from its
        // swallowed-tool-call fallback, or for a text-stream format from
        // its end-of-stream buffer) or a cut reply; dropped either way.
        if (text.isNotEmpty) _textSeen = true;
        if (stopRequested || text.isEmpty) return null;
        // Tool-call JSON in the text channel is never spoken ([_rawCalls]).
        if (_isRawToolCall(text)) {
          _rawCalls.add((text: text, at: _clock()));
          return null;
        }
        return text;
      case SkillLoadEvent(:final skillName, :final found):
        _toolRounds++;
        _step(SkillLoaded(skillName, found: found, at: _clock()));
      case ToolCallEvent(:final toolName, :final args):
        _toolRounds++;
        final intent = _argText(args['intent']) ?? toolName;
        _runningIntent = intent;
        _intentStartedAt = _clock();
        _step(
          IntentCalled(
            intent,
            _argText(args['parameters']) ?? '',
            at: _clock(),
          ),
        );
      case ToolResultEvent(:final toolName, :final result):
        final elapsed = _clock() - (_intentStartedAt ?? _clock());
        final intent = _runningIntent ?? toolName;
        switch (result) {
          case ErrorResult(:final message):
            _lastErrorResult = message;
            _step(
              IntentFailed(intent, message, elapsed: elapsed, at: _clock()),
            );
          case TextResult(:final text):
            _step(
              IntentSucceeded(intent, text, elapsed: elapsed, at: _clock()),
            );
          case ImageResult() || WidgetResult() || WebviewResult():
            // No app executor returns these.
            _step(
              IntentSucceeded(
                intent,
                '$result',
                elapsed: elapsed,
                at: _clock(),
              ),
            );
        }
      case AgentErrorEvent(:final message, :final toolName):
        // Every error event comes from dispatching a call. The loop repeats
        // every ErrorResult as an error event (`AgentLoop._runExecutor`): one
        // failed step, not two.
        if (message == _lastErrorResult) {
          _lastErrorResult = null;
          return null;
        }
        _step(IntentFailed(toolName, message, at: _clock()));
      case EmptyAnswerRetryEvent(
        :final skillName,
        :final attempt,
        :final nudge,
      ):
        // The loop asks again after an empty answer to a loaded skill: a
        // generation of its own, with its nudge in the chat's history. The
        // package's README says to mirror the nudge in a copy of the history
        // an app keeps; the conversation's text mirror is no such copy: it
        // holds the user's words and the reply, no tool rounds, and a
        // rebuilt chat replays it as plain turns. The nudge answers a
        // loadSkill result the mirror never had, so a replayed nudge would
        // point at a skill the rebuilt chat never loaded: not added.
        _step(
          EmptyAnswerRetried(
            skillName,
            attempt: attempt,
            nudge: nudge,
            at: _clock(),
          ),
        );
      // A DoneEvent with empty text: matched before DoneEvent. Raw call
      // JSON would have made its generation's text non-empty (a DoneEvent).
      case EmptyAnswerEvent(:final skillName):
        _finished = true;
        _noAnswerAfter = skillName;
      case DoneEvent():
        _finished = true;
        _reportUnreadCalls();
      case MaxIterationsEvent(:final iterations):
        _maxIterations = iterations;
    }
    return null;
  }

  /// Reports each of [_rawCalls] as a failed step: a tool call the SDK could
  /// not read, which was not spoken.
  void _reportUnreadCalls() {
    for (final (:text, :at) in _rawCalls) {
      final short = text.length > 160 ? '${text.substring(0, 160)}…' : text;
      debugPrint(
        '[Conversation] unparsed tool call dropped from the reply: $short',
      );
      _step(
        IntentFailed(
          null,
          'The model wrote a tool call that could not be read; it '
          'was not spoken.',
          at: at,
        ),
      );
    }
    _rawCalls.clear();
  }

  void _step(SkillStep step) {
    _steps.add(step);
    debugPrint('[Conversation] skill step @${step.at.inMilliseconds}ms: $step');
    try {
      _onStep?.call(step);
    } catch (e, st) {
      // A listener's bug must not end the drain.
      debugPrint('[Conversation] onStep threw: $e\n$st');
    }
  }

  /// Raw `tool_calls` JSON in the text channel.
  static bool _isRawToolCall(String text) =>
      text.trimLeft().startsWith('{') && text.contains('"tool_calls"');

  /// A tool argument as text: strings as they are, anything else as JSON.
  static String? _argText(Object? value) => switch (value) {
    null => null,
    final String text => text,
    final other => jsonEncode(other),
  };
}
