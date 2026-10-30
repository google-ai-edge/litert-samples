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
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/scheduler.dart';

import '../../app/demo_config.dart';
import '../../core/engines/engine_pool.dart';
import 'agent_run.dart';
import 'agent_runner.dart';
import 'follow_up_board.dart';
import 'meeting_task.dart';
import 'workspace.dart';

enum Act4Phase { idle, loading, running, done, failed }

/// Act 4 — one model from the menu does the follow-up of a meeting, any
/// meeting the presenter opens: it lists the follow-ups it finds, then
/// carries them out with six fake tools, and the checklist ticks each item
/// as an action names it.
class MeetingAgentController extends ChangeNotifier {
  MeetingAgentController({required this.pool, required this.spec, required this.task})
    : selected = spec.modelIndex,
      greedy = spec.greedy,
      temp = spec.temp,
      seed = spec.seed {
    think = spec.think ?? (spec.models.isEmpty ? false : model.think);
  }

  final EnginePool pool;
  final Act4Spec spec;

  /// The meeting and its workspace; [openMeeting] replaces it.
  MeetingTask task;

  int selected;
  late bool think;
  bool greedy;
  double temp;
  int seed;

  Act4Model get model => spec.models[selected];

  Act4Phase phase = Act4Phase.idle;

  /// The run in progress, or the last one.
  AgentRun? live;

  AgentRunner? _runner;
  Act4Model? _held;
  bool _disposed = false;

  bool get busy => phase == Act4Phase.loading || phase == Act4Phase.running;

  /// The checklist of the run on screen, its rules checked against the open
  /// workspace.
  List<BoardItem> get board => FollowUpBoard.of(live, task.workspace);

  /// Seconds into the run on screen.
  double get now => _runner?.elapsed ?? live?.runS ?? 0;

  /// Another meeting: the screen starts over on it.
  void openMeeting(MeetingTask next) {
    if (busy) return;
    task = next;
    live = null;
    phase = Act4Phase.idle;
    notifyListeners();
  }

  void select(int i) {
    if (busy || i == selected) return;
    selected = i;
    think = model.think;
    notifyListeners();
  }

  void set({bool? think, bool? greedy, double? temp, int? seed}) {
    if (busy) return;
    this.think = think ?? this.think;
    this.greedy = greedy ?? this.greedy;
    this.temp = temp ?? this.temp;
    this.seed = seed ?? this.seed;
    notifyListeners();
  }

  /// Loads the selected model (releasing the one held before) and runs the
  /// task once.
  Future<AgentRun?> run() async {
    if (busy || spec.models.isEmpty) return null;
    final m = model;
    final maxOut = spec.maxOutputTokensFor(think: think);
    final run = live = AgentRun(
      model: m.path,
      requestedBackend: m.backend.name,
      think: think,
      greedy: greedy,
      maxOutputTokens: maxOut,
      meeting: task.meeting.title,
    );
    _set(Act4Phase.loading);
    try {
      final held = _held;
      if (held != null && !identical(held, m)) {
        _held = null;
        await pool.release(held.backend, maxTokens: held.maxTokens, modelPath: held.path);
      }
      _held = m;
      final slot = await pool.acquire(m.backend, maxTokens: m.maxTokens, modelPath: m.path);
      run
        ..backend = slot.activeBackend?.name
        ..loadS = slot.loadTime == null ? null : slot.loadTime!.inMilliseconds / 1000;
      if (_disposed) {
        await _release();
        return null;
      }
      _set(Act4Phase.running);
      final runner = _runner = AgentRunner(
        slot: slot,
        task: task,
        run: run,
        settings: AgentSettings(
          think: think,
          greedy: greedy,
          seed: seed,
          temp: temp,
          maxOutputTokens: maxOut,
          maxCalls: spec.maxCalls,
        ),
      );
      await runner.start(onUpdate: _notifySoon);
    } catch (e, st) {
      // Loading failed (the plugin's own words, e.g. wrongBackend); the runner
      // itself never throws. Keep an earlier error: it is the cause.
      run.error ??= '$e';
      _log('ACT4 run failed: $e\n$st');
    } finally {
      final runner = _runner;
      _runner = null;
      try {
        await runner?.close();
      } catch (e) {
        run.error ??= 'closing the chat failed: $e';
      }
      // Left the act mid-run: dispose stopped the run and left the model to us.
      if (_disposed) await _release();
    }
    _set(run.error == null ? Act4Phase.done : Act4Phase.failed);
    return run;
  }

  Future<void> stop() async {
    try {
      await _runner?.stop();
    } catch (e) {
      // The run winds down anyway; the failure to stop it early is worth a line.
      _log('ACT4 stop failed: $e');
    }
  }

  /// Streamed chunks arrive faster than frames: one rebuild per frame is enough.
  bool _notifyPending = false;
  void _notifySoon() {
    if (_notifyPending || _disposed) return;
    _notifyPending = true;
    SchedulerBinding.instance
      ..scheduleFrameCallback((_) {
        _notifyPending = false;
        notifyListeners();
      })
      ..scheduleFrame();
  }

  static void _log(String s) => stderr.writeln(s);

  /// One headless run (`ACT4_AUTORUN`): prints its result line and appends it
  /// to [resultsPath]. False if the run did not start or failed, or the line
  /// could not be written.
  Future<bool> autorun(String? resultsPath) async {
    final r = await run();
    if (r == null) {
      _log('ACT4_ERROR the run did not start');
      return false;
    }
    final line = resultsLine(r, task.workspace);
    // ignore: avoid_print
    print('ACT4_JSON $line');
    var written = true;
    if (resultsPath != null) {
      try {
        File(resultsPath).writeAsStringSync('$line\n', mode: FileMode.append, flush: true);
      } catch (e) {
        _log('RESULTS_PATH: cannot append to $resultsPath ($e)');
        written = false;
      }
    }
    return r.error == null && written;
  }

  /// The result of [run] as one JSON line, its rules checked against
  /// [workspace]: what a script compares across runs.
  static String resultsLine(AgentRun run, Workspace workspace, {DateTime? now}) {
    final items = FollowUpBoard.of(run, workspace);
    int count(bool rule, {bool passed = false}) => items.where((c) => c.rule == rule && (!passed || c.passed)).length;
    return jsonEncode({
      'act': 4,
      'captured_at': (now ?? DateTime.now()).toIso8601String(),
      'model': run.modelName,
      'requested_backend': run.requestedBackend,
      'backend': run.backend,
      'think': run.think,
      'greedy': run.greedy,
      'meeting': run.meeting,
      'listed': count(false),
      'done': count(false, passed: true),
      'rules': '${count(true, passed: true)}/${count(true)}',
      'calls': run.calls.length,
      'attempts': run.attempts.length,
      'turns': run.turns.length,
      'load_s': run.loadS,
      'run_s': run.runS,
      'first_call_s': run.firstCallS,
      'ran_out_while_thinking': run.ranOutWhileThinking,
      'metrics': run.metrics,
      'error': run.error,
      'failed': {
        for (final c in items)
          if (!c.passed) c.label: c.why,
      },
    });
  }

  Future<void> _release() async {
    final held = _held;
    _held = null;
    if (held != null) await pool.release(held.backend, maxTokens: held.maxTokens, modelPath: held.path);
  }

  void _set(Act4Phase p) {
    phase = p;
    notifyListeners();
  }

  @override
  void notifyListeners() {
    if (!_disposed) super.notifyListeners();
  }

  @override
  void dispose() {
    _disposed = true;
    // A model of this act holds up to 13 GB: it goes when the act closes. A
    // run (or load) in progress releases it when it has wound down.
    if (busy) {
      unawaited(stop());
    } else {
      unawaited(_release());
    }
    super.dispose();
  }
}
