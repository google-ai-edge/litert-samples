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

  /// The turn's skill steps so far, in order.
  List<SkillStep> get steps => List.unmodifiable(_steps);

  /// Tool calls so far: loadSkill and runIntent each count (every one cost
  /// a generation before the answer).
  int get toolRounds => _toolRounds;

  /// The loop ended with a final, call-free generation ([DoneEvent]).
  bool get finished => _finished;

  /// The generations the loop used up without an answer
  /// ([MaxIterationsEvent]); 0 when it did not run out.
  int get maxIterations => _maxIterations;

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
        if (stopRequested || text.isEmpty) return null;
        // A `{`-leading turn whose tool call the SDK could not parse is
        // surfaced as one text chunk (the swallowed-tool-call fallback at
        // the end of `InferenceChat.generateChatResponseAsync`): never
        // speak it.
        if (_isRawToolCall(text)) {
          debugPrint(
            '[Conversation] unparsed tool call dropped from the reply: '
            '${text.length > 160 ? '${text.substring(0, 160)}…' : text}',
          );
          _step(
            IntentFailed(
              null,
              'The model wrote a tool call that could not be read; it '
              'was not spoken.',
              at: _clock(),
            ),
          );
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
        // The loop repeats every ErrorResult as an error event
        // (agent_loop.dart:379-382): one failed step, not two.
        if (message == _lastErrorResult) {
          _lastErrorResult = null;
          return null;
        }
        _step(IntentFailed(toolName, message, at: _clock()));
      case DoneEvent():
        _finished = true;
      case MaxIterationsEvent(:final iterations):
        _maxIterations = iterations;
    }
    return null;
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
