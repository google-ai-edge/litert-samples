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
import 'dart:isolate';

import 'package:flutter/foundation.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart';

import '../scene/scene_events.dart';
import 'narrator_isolate.dart';

enum NarratorState { off, loading, idle, generating, failed }

/// One finished commentary line with the numbers behind it.
class NarrationTurn {
  const NarrationTurn({
    required this.seq,
    required this.at,
    required this.backend,
    required this.text,
    required this.eventsIn,
    required this.ttft,
    required this.tokensPerSecond,
    required this.outputTokens,
    required this.inputTokens,
  });

  /// 1, 2, 3… over the narrator's life (the history itself is capped).
  final int seq;
  final double at;
  final PreferredBackend backend;
  final String text;
  final int eventsIn;
  final Duration ttft;
  final double? tokensPerSecond;
  final int outputTokens;
  final int inputTokens;
}

/// Act 3's narrator, UI-side proxy. Batches the scene's event log and asks
/// the worker isolate ([narratorMain]) to comment on ONE multi-turn
/// conversation, so the log accumulates in context and the model can refer
/// back to something that happened minutes ago. All native work — engine
/// load, prefill, decode — happens on the worker; this side only formats
/// prompts and receives chunks.
class SceneNarrator extends ChangeNotifier {
  SceneNarrator({
    required this.events,
    required this.modelPath,
    this.modelPathOverrides = const {},
    this.maxTokens = 4096,
    this.interval = const Duration(seconds: 4),
    this.maxEventsPerTurn = 12,
    this.maxOutputTokens = 60,
    this.inline = false,
    this.preface = arenaPreface,
    this.situation,
  });

  /// Facts the scene adds beside the events each turn (what is coming, how
  /// the run is going, the hardware the narrator is on). Null or empty text
  /// leaves the prompt as it was.
  String Function()? situation;

  /// Run the engine host on the UI isolate instead of a worker (diagnostic /
  /// fallback; costs UI-thread time during prefill).
  final bool inline;

  final SceneEventLog events;
  final String modelPath;

  /// Per-backend model file, e.g. the NPU-compiled export on Windows.
  final Map<PreferredBackend, String> modelPathOverrides;

  String modelPathFor(PreferredBackend b) => modelPathOverrides[b] ?? modelPath;
  final int maxTokens;
  final Duration interval;
  final int maxEventsPerTurn;
  final int maxOutputTokens;

  NarratorState state = NarratorState.off;
  PreferredBackend? backend;
  String? error;

  SendPort? _worker;
  Isolate? _isolate;
  ReceivePort? _exitPort;
  final ReceivePort _inbox = ReceivePort();
  StreamSubscription<dynamic>? _inboxSub;
  Completer<void>? _workerReady;
  bool _disposed = false;

  /// start / stop / preload run one after another, in call order. Without
  /// this, two backend keys pressed during a turn both waited for it and
  /// then raced: both overwrote the pending `open`, and one never returned.
  Future<void> _lifecycle = Future.value();

  /// A load that takes longer is hung (on macOS an NPU load never returns):
  /// the worker is replaced and the load fails, instead of the narrator
  /// sitting in `loading` for the rest of the act.
  static const loadTimeout = Duration(minutes: 3);
  static const turnTimeout = Duration(minutes: 2);
  final Map<String, Completer<Map<String, Object?>>> _waiting = {};
  final Set<PreferredBackend> _loaded = {};
  int _turnSeq = 0;
  int? _liveTurnId;

  StreamSubscription<SceneEvent>? _sub;
  Timer? _timer;
  final List<SceneEvent> _pending = [];
  bool _conversationHasPreface = false;
  bool _busy = false;
  int _lastOutCumulative = 0;

  /// Text currently streaming in.
  String live = '';
  final List<NarrationTurn> history = [];
  int _turnsTotal = 0;
  int _phaseFrom = 0;

  double? lastTps;
  Duration? lastTtft;
  int contextTokens = 0;
  int compactions = 0;
  int droppedEvents = 0;
  final Map<PreferredBackend, int> loadMs = {};

  double get avgTps => _avgTps(history);
  Duration get avgTtft => _avgTtft(history);

  static double _avgTps(Iterable<NarrationTurn> turns) {
    final v = turns.map((t) => t.tokensPerSecond).whereType<double>().toList();
    return v.isEmpty ? 0 : v.reduce((a, b) => a + b) / v.length;
  }

  static Duration _avgTtft(Iterable<NarrationTurn> turns) {
    if (turns.isEmpty) return Duration.zero;
    final us = turns.map((t) => t.ttft.inMicroseconds).reduce((a, b) => a + b);
    return Duration(microseconds: us ~/ turns.length);
  }

  /// Starts a measurement phase: [toJson] with `phase: true` then covers only
  /// the turns from here on, on the backend the narrator is on. Without it a
  /// "cpu" phase averaged in every GPU turn before it.
  void markPhase() => _phaseFrom = _turnsTotal;

  @visibleForTesting
  void debugRecordTurn(NarrationTurn Function(int seq) make) => history.add(make(++_turnsTotal));

  List<NarrationTurn> get _phaseTurns => [
        for (final t in history)
          if (t.seq > _phaseFrom && t.backend == backend) t,
      ];

  /// What the model is told it is watching. The scene decides this: the same
  /// narrator drives the arena and the runner.
  /// Overridden per scene; defaults to the arena wording.
  final String preface;

  static const arenaPreface =
      'You are the live commentator of a physics arena where crates and balls '
      'keep dropping and colliding. I will send you batches of timestamped '
      'events. After each batch, reply with ONE or TWO short sentences of '
      'lively commentary about what just happened. Mention object names and '
      'numbers when they matter, and refer back to earlier events when '
      'something connects to them. Never list the events; talk about them.';

  static const runnerPreface =
      'You are the live commentator of an endless run. Dash, a small blue '
      'bird, is sprinting along a trail through open country: he jumps fallen '
      'logs and rivers, dodges boulders, sometimes runs into one, and passes '
      'through meadows, forest, canyon and snow while the day turns to night '
      'and back. Each turn you get the latest events and a short Situation: '
      'what is coming up, how Dash is doing, and the hardware you yourself '
      'are running on. Reply with ONE or TWO short sentences of lively '
      'commentary. Build suspense from what is ahead, but never announce a '
      'fall before it happens. Use at most one number per reply, and do not '
      'open two replies the same way. When you are moved to another chip, or '
      'your speed or the frame budget changes a lot, you may joke about it in '
      'a short aside, since you are part of the show. Never list the events; '
      'talk about them.';

  // ---------------------------------------------------------------- worker

  Future<void> _ensureWorker() async {
    if (_worker != null) return;
    if (_workerReady != null) return _workerReady!.future;
    _workerReady = Completer<void>();
    _inboxSub = _inbox.listen(_onWorkerMessage);
    if (inline) {
      // Same protocol, no isolate: commands go to a local host through a
      // ReceivePort so the reply path stays identical.
      final host = NarratorEngineHost();
      final cmdPort = ReceivePort();
      cmdPort.listen((raw) async {
        final alive = await host.handle(raw as Map<String, Object?>, _inbox.sendPort.send);
        if (!alive) cmdPort.close();
      });
      _inbox.sendPort.send({'type': NarratorMsg.ready, 'port': cmdPort.sendPort});
    } else {
      final exit = ReceivePort();
      _exitPort = exit;
      exit.listen((_) => _dropWorker('the narrator worker stopped'));
      _isolate = await Isolate.spawn(
        narratorMain,
        _inbox.sendPort,
        debugName: 'narrator',
        onExit: exit.sendPort,
        onError: exit.sendPort,
      );
      if (_disposed) _isolate?.kill();
    }
    return _workerReady!.future;
  }

  /// Forgets the worker and fails everything waiting on it. Kills it when it
  /// is an isolate (a call stuck in native code ends when it returns).
  void _dropWorker(String reason) {
    _exitPort?.close();
    _exitPort = null;
    _isolate?.kill(priority: Isolate.immediate);
    _isolate = null;
    _worker = null;
    final ready = _workerReady;
    _workerReady = null;
    if (ready != null && !ready.isCompleted) ready.completeError(StateError(reason));
    _loaded.clear();
    final e = StateError(reason);
    for (final c in _waiting.values) {
      if (!c.isCompleted) c.completeError(e);
    }
    _waiting.clear();
    if (state != NarratorState.off) {
      error = reason;
      _set(NarratorState.failed);
    }
  }

  void _onWorkerMessage(dynamic raw) {
    final msg = raw as Map<String, Object?>;
    switch (msg['type'] as String) {
      case NarratorMsg.ready:
        final port = msg['port'] as SendPort;
        if (_disposed) {
          port.send({'type': NarratorCmd.dispose});
          return;
        }
        _worker = port;
        _workerReady?.complete();
      case NarratorMsg.loaded:
        final b = PreferredBackend.values.byName(msg['backend'] as String);
        _loaded.add(b);
        loadMs[b] = (msg['loadMs'] as int?) ?? 0;
        _waiting.remove('load:${b.name}')?.complete(msg);
      case NarratorMsg.opened:
        _waiting.remove('open')?.complete(msg);
      case NarratorMsg.chunk:
        if (msg['turnId'] == _liveTurnId) {
          live += msg['text'] as String;
          notifyListeners();
        }
      case NarratorMsg.done:
        _waiting.remove('turn:${msg['turnId']}')?.complete(msg);
      case NarratorMsg.error:
        final op = msg['op'] as String;
        final key = switch (op) {
          NarratorCmd.generate => 'turn:${msg['turnId']}',
          NarratorCmd.open => 'open',
          NarratorCmd.load => null,
          _ => null,
        };
        final e = StateError(msg['message'] as String);
        if (key != null && _waiting.containsKey(key)) {
          _waiting.remove(key)!.completeError(e);
        } else if (op == NarratorCmd.load) {
          // load errors are keyed by backend, which the error message lacks;
          // fail every pending load.
          for (final k in _waiting.keys.where((k) => k.startsWith('load:')).toList()) {
            _waiting.remove(k)!.completeError(e);
          }
        } else {
          error = msg['message'] as String;
          _set(NarratorState.failed);
        }
    }
  }

  Future<Map<String, Object?>> _ask(String key, Map<String, Object?> cmd, Duration timeout) async {
    final worker = _worker;
    if (worker == null) throw StateError('the narrator worker is not running');
    final c = Completer<Map<String, Object?>>();
    _waiting[key] = c;
    worker.send(cmd);
    try {
      return await c.future.timeout(timeout);
    } on TimeoutException {
      _waiting.remove(key);
      _dropWorker('${cmd['type']} ${cmd['backend'] ?? ''} took over ${timeout.inSeconds} s; the narrator worker was restarted');
      rethrow;
    }
  }

  Future<T> _serial<T>(Future<T> Function() op) {
    final result = _lifecycle.then((_) => op());
    _lifecycle = result.then((_) {}, onError: (Object _) {});
    return result;
  }

  // ---------------------------------------------------------------- public

  /// Load an engine without narrating on it (so the switch later is instant).
  Future<void> preload(PreferredBackend b) => _serial(() => _preload(b));

  Future<void> _preload(PreferredBackend b) async {
    await _ensureWorker();
    if (_loaded.contains(b)) return;
    await _ask(
      'load:${b.name}',
      {'type': NarratorCmd.load, 'backend': b.name, 'modelPath': modelPathFor(b), 'maxTokens': maxTokens},
      loadTimeout,
    );
  }

  /// Loads (or reuses) the engine for [b] and starts narrating on a fresh
  /// conversation, recapping earlier commentary so the thread continues.
  /// Already narrating on [b]: nothing to do.
  Future<void> start(PreferredBackend b) => _serial(() => _start(b));

  Future<void> _start(PreferredBackend b) async {
    if (backend == b && (state == NarratorState.idle || state == NarratorState.generating)) return;
    await _waitIdle();
    if (backend != b) {
      // The last turn's speed belongs to the old backend: carried over, it
      // was sampled into the new backend's meter and baseline.
      lastTps = null;
      lastTtft = null;
    }
    backend = b;
    error = null;
    live = '';
    _set(NarratorState.loading);
    try {
      await _preload(b);
      await _openConversation();
      _sub ??= events.stream.listen(_onEvent);
      _timer?.cancel();
      _timer = Timer.periodic(interval, (_) => _tick());
      _set(NarratorState.idle);
    } catch (e, st) {
      error = '$e\n$st';
      _set(NarratorState.failed);
    }
  }

  /// Instant when the target engine is already resident; otherwise a load.
  Future<void> switchTo(PreferredBackend b) => start(b);

  /// Take a turn now instead of at the next interval, if there is anything
  /// pending: a backend switch is the moment the audience is watching.
  void nudge() => unawaited(_tick());

  Future<void> stop() => _serial(_stop);

  Future<void> _stop() async {
    _timer?.cancel();
    _timer = null;
    await _waitIdle();
    _worker?.send({'type': NarratorCmd.close});
    _pending.clear();
    live = '';
    _set(NarratorState.off);
  }

  Future<void> _openConversation() async {
    await _ask(
      'open',
      {
        'type': NarratorCmd.open,
        'backend': backend!.name,
        'modelPath': modelPathFor(backend!),
        'maxTokens': maxTokens,
        'maxOutputTokens': maxOutputTokens,
      },
      turnTimeout,
    );
    _conversationHasPreface = false;
    contextTokens = 0;
    _lastOutCumulative = 0;
  }

  void _onEvent(SceneEvent e) {
    if (state == NarratorState.off) return;
    if (e.kind == 'despawn') return;
    _pending.add(e);
    final cap = maxEventsPerTurn * 3;
    if (_pending.length > cap) {
      droppedEvents += _pending.length - cap;
      _pending.removeRange(0, _pending.length - cap);
    }
  }

  Future<void> _tick() async {
    if (_busy || state != NarratorState.idle || _pending.isEmpty || _worker == null) return;
    _busy = true;
    final b = backend!;
    final batch = _pending.take(maxEventsPerTurn).toList();
    _pending.removeRange(0, batch.length);
    try {
      if (contextTokens > maxTokens * 0.75) {
        compactions++;
        await _openConversation();
      }
      final prompt = StringBuffer();
      if (!_conversationHasPreface) {
        prompt.writeln(preface);
        if (history.isNotEmpty) {
          prompt.writeln('Recap of your earlier commentary:');
          for (final h in history.reversed.take(4).toList().reversed) {
            prompt.writeln('- ${h.text}');
          }
        }
        prompt.writeln();
        _conversationHasPreface = true;
      }
      prompt.writeln('Events:');
      for (final e in batch) {
        prompt.writeln(e.asText);
      }
      final facts = situation?.call() ?? '';
      if (facts.isNotEmpty) prompt.writeln(facts);
      prompt.writeln('Commentary:');

      final id = ++_turnSeq;
      _liveTurnId = id;
      live = '';
      _set(NarratorState.generating);
      final done = await _ask(
        'turn:$id',
        {'type': NarratorCmd.generate, 'prompt': prompt.toString(), 'turnId': id},
        turnTimeout,
      );
      final tps = done['tps'] as double?;
      final ttft = Duration(milliseconds: done['ttftMs'] as int);
      // LiteRT-LM benchmark info is CUMULATIVE per conversation (measured:
      // in 486 → 900 → 1324 over three turns), so the context size is the
      // latest reading, not a sum; this turn's output is the difference.
      final inTok = (done['inputTokens'] as int?) ?? 0;
      final outTok = (done['outputTokens'] as int?) ?? 0;
      final turnOut = outTok > 0 ? outTok - _lastOutCumulative : live.length ~/ 4;
      _lastOutCumulative = outTok;
      contextTokens = inTok + outTok > 0 ? inTok + outTok : contextTokens + prompt.length ~/ 4 + live.length ~/ 4;
      lastTps = tps;
      lastTtft = ttft;
      history.add(NarrationTurn(
        seq: ++_turnsTotal,
        at: batch.last.at,
        backend: b,
        text: live.trim(),
        eventsIn: batch.length,
        ttft: ttft,
        tokensPerSecond: tps,
        outputTokens: turnOut,
        inputTokens: inTok,
      ));
      if (history.length > 200) history.removeAt(0);
      _liveTurnId = null;
      if (state == NarratorState.generating) _set(NarratorState.idle);
    } catch (e, st) {
      error = '$e\n$st';
      _set(NarratorState.failed);
    } finally {
      _busy = false;
    }
  }

  Future<void> _waitIdle() async {
    while (_busy) {
      await Future<void>.delayed(const Duration(milliseconds: 50));
    }
  }

  /// With [phase], the turns since [markPhase] on the current backend only.
  Map<String, Object?> toJson({bool phase = false}) {
    final turns = phase ? _phaseTurns : history;
    return {
        'backend': backend?.name,
        'state': state.name,
        'turns': turns.length,
        'avg_tps': double.parse(_avgTps(turns).toStringAsFixed(1)),
        'avg_ttft_ms': _avgTtft(turns).inMilliseconds,
        'context_tokens': contextTokens,
        'compactions': compactions,
        'dropped_events': droppedEvents,
        'load_ms': {for (final e in loadMs.entries) e.key.name: e.value},
        'last_lines': turns.reversed.take(3).map((t) => t.text).toList(),
        // Every line, oldest first, with the backend it came from: what the
        // audience actually read.
        'transcript': [for (final t in turns) '[${t.backend.name.toUpperCase()}] ${t.text}'],
        'error': error?.split('\n').first,
      };
  }

  void _set(NarratorState s) {
    if (_disposed) return;
    state = s;
    notifyListeners();
  }

  @override
  void dispose() {
    _disposed = true;
    _timer?.cancel();
    _sub?.cancel();
    final worker = _worker;
    if (worker != null) {
      // The worker closes its engines and exits by itself.
      worker.send({'type': NarratorCmd.dispose});
      _exitPort?.close();
    } else {
      // Still spawning (or dead): nothing to close politely.
      _exitPort?.close();
      _isolate?.kill();
    }
    final e = StateError('narrator disposed');
    for (final c in _waiting.values) {
      if (!c.isCompleted) c.completeError(e);
    }
    _waiting.clear();
    _inboxSub?.cancel();
    _inbox.close();
    super.dispose();
  }
}
