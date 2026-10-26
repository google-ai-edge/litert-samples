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

import 'dart:io';

import 'package:flutter_edge_ai/flutter_edge_ai.dart';

import '../../core/engines/engine_slot.dart';
import 'agent_run.dart';
import 'meeting_task.dart';

/// How one run samples and where it stops.
class AgentSettings {
  const AgentSettings({
    required this.think,
    required this.greedy,
    required this.seed,
    required this.temp,
    required this.maxOutputTokens,
    required this.maxCalls,
  });

  final bool think;

  /// top-k 1. The NPU decodes greedily whatever is asked.
  final bool greedy;
  final int seed;

  /// Used when not [greedy], with top-k 64 and top-p 0.95.
  final double temp;

  /// Per turn, thinking included.
  final int maxOutputTokens;

  /// Executed calls; later ones are refused. The run stops 10 refusals later.
  final int maxCalls;
}

/// One run of the meeting task on a loaded engine, through the plugin's own
/// tool loop: the fake tools answer each call, and [run] is updated as the
/// model streams, for the screen to show while it happens.
class AgentRunner {
  AgentRunner({required this.slot, required this.task, required this.settings, required this.run});

  final EngineSlot slot;
  final MeetingTask task;
  final AgentSettings settings;
  final AgentRun run;

  InferenceChat? _chat;
  bool _stopped = false;
  final _sw = Stopwatch();

  /// Seconds since the prompt went in: the clock of [run]'s times.
  double get elapsed => _sw.elapsedMicroseconds / 1e6;

  /// Runs to the end (or to [stop]), calling [onUpdate] after every chunk and
  /// call. Never throws: a failure lands in [AgentRun.error] (its stack on
  /// stderr). The prompt goes in once: on the NPU a second message after a
  /// tool loop fails, so the loop's own tool answers are all that follows it.
  Future<AgentRun> start({required void Function() onUpdate}) async {
    final rec = RunRecorder(run, maxCalls: settings.maxCalls, tools: task.tools);
    double t() => elapsed;
    // Output tokens seen last: the turn clock keeps going on it if a read of
    // the session's metrics fails (that failure is the run's error).
    var lastOut = 0;
    int outTokens() => lastOut = _metrics()?.outputTokens ?? lastOut;
    try {
      final chat = _chat = await slot.openChat(
        tools: task.tools,
        systemInstruction: task.system,
        enableThinking: settings.think,
        temperature: settings.greedy ? 1.0 : settings.temp,
        topK: settings.greedy ? 1 : 64,
        randomSeed: settings.seed,
        maxOutputTokens: settings.maxOutputTokens,
      );
      _sw.start();
      rec.begin(0);
      await chat.addQueryChunk(Message.text(text: task.user, isUser: true));
      final stream = chat.generateChatResponseWithTools(
        onToolCall: (c) {
          final reply = rec.call(c.name, c.args, outTokens: outTokens(), t: t());
          onUpdate();
          return reply;
        },
        // Every turn has at least one call, so the call limit binds first.
        maxToolTurns: settings.maxCalls + 10,
        isCancelled: () => _stopped || run.attempts.length >= settings.maxCalls + 10,
        onMaxToolTurns: () => run.error ??= 'stopped after ${settings.maxCalls + 10} turns with calls',
      );
      await for (final r in stream) {
        switch (r) {
          case ThinkingResponse(:final content):
            rec.thinking(content, t());
          case TextResponse(:final token):
            rec.text(token, t());
          case FunctionCallResponse() || ParallelFunctionCallResponse():
            // The loop answers calls through onToolCall; it does not pass them on.
            break;
        }
        onUpdate();
      }
      if (run.attempts.length >= settings.maxCalls + 10) {
        run.error ??= 'stopped: ${run.attempts.length} calls, ${run.attempts.length - run.calls.length} not run';
      }
    } catch (e, st) {
      _fail('$e', st);
    }
    _sw.stop();
    try {
      rec.finish(t(), outTokens: outTokens());
      run.stopped = _stopped;
      if (_metrics() case final m?) {
        run.metrics = {
          'input_tokens': m.inputTokens,
          'output_tokens': m.outputTokens,
          'ttft_ms': m.timeToFirstTokenMs,
          'tok_s': m.tokensPerSecond,
        };
      }
      onUpdate();
    } catch (e, st) {
      _fail('finishing the run failed: $e', st);
    }
    return run;
  }

  /// The session's numbers, or null before the chat is open or when reading
  /// them fails (recorded as the run's error).
  SessionMetrics? _metrics() {
    final chat = _chat;
    if (chat == null) return null;
    try {
      return chat.session.getSessionMetrics();
    } catch (e, st) {
      _fail('reading the session metrics failed: $e', st);
      return null;
    }
  }

  /// Keeps the first failure as the run's error and adds later ones to it:
  /// the first is the cause, the rest may be its consequences.
  void _fail(String what, StackTrace st) {
    final before = run.error;
    run.error = before == null ? what : (before.contains(what) ? before : '$before; then $what');
    stderr.writeln('ACT4 run: $what\n$st');
  }

  /// Ends the generation in progress; the loop then answers the calls it has
  /// not run as cancelled and returns.
  Future<void> stop() async {
    _stopped = true;
    await _chat?.stopGeneration();
  }

  Future<void> close() async {
    final chat = _chat;
    _chat = null;
    await chat?.close();
  }
}
