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
import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/services.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:flutter_scene/scene.dart';
import 'package:vector_math/vector_math.dart' as vm;

import '../../../app/demo_config.dart';
import '../../../core/engines/backend_catalog.dart';
import '../../../core/narration/scene_narrator.dart';
import '../../../core/power/power_source.dart';
import '../../../core/scene/scene_events.dart';
import '../../../core/telemetry/frame_time_probe.dart';
import 'frame_recorder.dart';
import 'load_calibrator.dart';
import 'runner_game.dart';
import 'stage_hud.dart';

/// Act 3 — Dash runs a generated trail on the GPU while the model narrates
/// the run live. Frame time and tokens/sec share one HUD; the narrator's
/// backend is a chip, which is the whole point of the act.
class RunnerScreen extends StatefulWidget {
  const RunnerScreen({super.key, required this.config, required this.power});

  final DemoConfig config;
  final PowerSource power;

  @override
  State<RunnerScreen> createState() => _RunnerScreenState();
}

class _RunnerScreenState extends State<RunnerScreen> {
  final SceneEventLog _events = SceneEventLog();
  /// The display decides the budget when it cannot show more than the
  /// requested rate (see [FrameBudget]).
  late final FrameBudget _budget = FrameBudget.resolve(
    requested: widget.config.act3.targetFps,
    refreshRate: ui.PlatformDispatcher.instance.implicitView?.display.refreshRate ?? 0,
  );
  late final FrameTimeProbe _probe = FrameTimeProbe(targetFps: _budget.fps);
  final GlobalKey _captureKey = GlobalKey();
  final FocusNode _focus = FocusNode();

  late final RunnerGame _game = RunnerGame(
    events: _events,
    settings: RunnerSettings(
      viewChunks: widget.config.act3.viewChunks,
      propDensity: widget.config.act3.propDensity,
      shadows: widget.config.act3.shadows,
      post: widget.config.act3.post,
      renderScale: widget.config.act3.renderScale,
    ),
    missRate: widget.config.act3.pilotMissRate,
    dayCycleSeconds: widget.config.act3.dayCycleSeconds,
    dayStart: widget.config.act3.dayStart,
    startZ: widget.config.act3.startZ,
  );

  late final SceneNarrator _narrator = SceneNarrator(
    events: _events,
    modelPath: widget.config.modelPath,
    modelPathOverrides: {
      if (widget.config.act3.npuModelPath != null) PreferredBackend.npu: widget.config.act3.npuModelPath!,
    },
    maxTokens: widget.config.act3.narratorMaxTokens,
    interval: Duration(milliseconds: widget.config.act3.narratorIntervalMs),
    preface: SceneNarrator.runnerPreface,
  );

  bool _ready = false;
  String? _error;
  late bool _showDetails = widget.config.act3.details;
  final List<Map<String, Object?>> _phases = [];
  final StageSeries _series = StageSeries();
  Timer? _sampler;
  late final LoadCalibrator _calibrator = LoadCalibrator(
    probe: _probe,
    apply: (step) {
      if (mounted) applyStepTo(_game, step);
    },
    budget: _budget,
  );

  // Camera state, smoothed between frames so the chase does not snap.
  vm.Vector3? _camPos;
  vm.Vector3? _camTarget;

  @override
  void initState() {
    super.initState();
    unawaited(_init());
  }

  Future<void> _init() async {
    try {
      await Scene.initializeStaticResources();
      await _game.build();
      // Every await below can outlive the screen (Back during a load): stop
      // there rather than drive a disposed game and narrator.
      if (!mounted) return;
      _narrator.situation = _situationText;
      _probe.start();
      setState(() => _ready = true);
      final record = widget.config.act3.recordPath;
      if (record != null) {
        // With a plan, the file starts with the plan's first phase; the
        // calibration and baseline passes before it are grabbed (so their
        // load includes the read-back) but not written.
        _recorder = FrameRecorder(
          path: record,
          fps: widget.config.act3.recordFps,
          pixelRatio: widget.config.act3.recordScale,
        )
          ..writing = widget.config.act3.narratorPlan.isEmpty
          ..start(_captureKey);
        _log('ACT3_RECORD $record @ ${widget.config.act3.recordFps} fps');
      }
      _sampler = Timer.periodic(const Duration(milliseconds: 100), (_) => _sample());
      // The scene has to sit on the frame budget before anything is measured
      // or shown: with headroom, moving the model to the GPU costs nothing
      // and the act has nothing to demonstrate.
      _log('ACT3_BUDGET ${jsonEncode(_budget.toJson())}');
      if (widget.config.act3.calibrate) {
        await _calibrator.run();
        if (!mounted) return;
        _log('ACT3_CALIBRATED level ${_calibrator.level} ${_calibrator.step.label} :: ${_calibrator.log.join(' | ')}');
      }
      final plan = widget.config.act3.narratorPlan;
      if (plan.isNotEmpty) {
        if (widget.config.act3.baseline) {
          await _measureBaselines([
            for (final (b, _) in plan)
              if (b != 'off') b,
          ]);
        }
        if (!mounted) return;
        await _runPlan(plan);
      } else if (widget.config.act3.autorunSeconds > 0) {
        await Future<void>.delayed(Duration(seconds: widget.config.act3.autorunSeconds));
        if (!mounted) return;
        await _captureFrame();
        await _stopRecording();
        _emitResults();
      }
    } catch (e, st) {
      _log('ACT3_INIT_FAILED $e\n$st');
      if (mounted) setState(() => _error = '$e\n$st');
      if (widget.config.autorunExit) exit(3);
    }
  }

  /// Headless measurement: phases like `off:15,gpu:30,npu:30` — each resets
  /// frame stats, runs the narrator on that backend (or off) for N seconds,
  /// and records frame, narrator and scene stats. Engines are preloaded first
  /// so a switch is a pointer swap, not a load.
  Future<void> _runPlan(List<(String, int)> plan) async {
    final preloadErrors = <String, String>{};
    for (final (b, _) in plan) {
      if (b == 'off') continue;
      final status = _catalog.status(DemoConfig.parseBackend(b));
      if (!status.isAvailable) {
        preloadErrors[b] = 'not available here: ${status.reason}';
        _log('ACT3_PRELOAD $b SKIPPED: ${preloadErrors[b]}');
        continue;
      }
      try {
        await _narrator.preload(DemoConfig.parseBackend(b));
        _log('ACT3_PRELOAD $b ok ${_narrator.loadMs}');
      } catch (e) {
        preloadErrors[b] = e.toString().split('\n').first;
        _log('ACT3_PRELOAD $b FAILED: ${preloadErrors[b]}');
      }
    }
    _recorder?.writing = true;
    for (final (b, secs) in plan) {
      if (!mounted) return;
      String? phaseError = preloadErrors[b];
      try {
        if (b == 'off') {
          await _narrator.stop();
        } else if (phaseError == null) {
          await _switchNarrator(DemoConfig.parseBackend(b));
          if (_narrator.state == NarratorState.failed) phaseError = _narrator.error?.split('\n').first;
        }
      } catch (e) {
        phaseError = e.toString().split('\n').first;
      }
      // A phase whose backend failed runs without the narrator; left alone,
      // it kept talking on the previous phase's backend under this label.
      if (b != 'off' && phaseError != null) await _narrator.stop();
      _narrator.markPhase();
      _log('ACT3_PHASE $b start${phaseError != null ? ' (narrator failed: $phaseError)' : ''}');
      _probe.reset();
      final (_, powerSummary) = await widget.power.measure(() => Future<void>.delayed(Duration(seconds: secs)));
      if (!mounted) return;
      _phases.add({
        'phase': b,
        'seconds': secs,
        'frame': _probe.toJson(),
        'narrator': b == 'off' ? null : _narrator.toJson(phase: true),
        'narrator_error': phaseError,
        'scene': _game.toJson(),
        'power': powerSummary.toJson(),
      });
      await _captureFrame(suffix: '-$b');
    }
    await _stopRecording();
    _emitResults();
  }

  /// Measures each backend's solo speed with the scene at its lightest, so
  /// the MODEL SPEED meter has an honest denominator. Comparing the model
  /// against its own best-under-load reading always shows 100 % and says
  /// nothing.
  Future<void> _measureBaselines(List<String> requested) async {
    final backends = [
      for (final b in requested)
        if (_catalog.status(DemoConfig.parseBackend(b)).isAvailable) b,
    ];
    if (backends.isEmpty || _baselining) return;
    _baselining = true;
    final saved = _game.settings;
    applyStepTo(_game, LoadCalibrator.ladder.first);
    await Future<void>.delayed(const Duration(milliseconds: 900));
    for (final b in backends) {
      if (!mounted) break;
      try {
        final backend = DemoConfig.parseBackend(b);
        await _narrator.switchTo(backend);
        // Two turns: the first one on a fresh conversation is slower.
        await Future<void>.delayed(const Duration(seconds: 11));
        // lastTps is cleared on a switch, so a number here is a turn on THIS
        // backend; a failed switch leaves the previous backend in place.
        final tps = _narrator.lastTps;
        if (_narrator.state != NarratorState.failed && _narrator.backend == backend && tps != null && tps > 0) {
          _series.setBaseline(backendLabel(_narrator.backend), tps);
          _log('ACT3_BASELINE $b ${tps.toStringAsFixed(1)} tok/s solo');
        }
      } catch (e) {
        _log('ACT3_BASELINE $b failed: $e');
      }
    }
    _baselining = false;
    if (!mounted) return;
    await _narrator.stop();
    _game.applySettings(saved);
    await Future<void>.delayed(const Duration(milliseconds: 600));
    _probe.reset();
  }

  bool _baselining = false;

  bool get _streamingLine => _narrator.state == NarratorState.generating && _narrator.live.trim().isNotEmpty;

  /// Moves the narrator and tells it so: the switch is the moment the act
  /// is about, so it gets a line right away rather than at the next tick.
  Future<void> _switchNarrator(PreferredBackend to) async {
    final from = _narrator.state == NarratorState.off ? null : _narrator.backend;
    await _narrator.switchTo(to);
    if (from == to || _narrator.state == NarratorState.failed) return;
    final label = backendLabel(to);
    _events.add(SceneEvent(
      at: _game.elapsed,
      kind: 'hardware',
      subject: 'you',
      object: label,
      detail: from == null
          ? 'you just came on air, running on the $label'
          : 'you were just moved from the ${backendLabel(from)} to the $label',
    ));
    _narrator.nudge();
  }

  /// The narrator's Situation block: the trail from the game, the hardware
  /// from the meters the audience is looking at.
  String _situationText() {
    final active = _narrator.state != NarratorState.off && _narrator.backend != null;
    final label = active ? backendLabel(_narrator.backend) : null;
    final base = label == null ? null : _series.baselineFor(label);
    final tps = _narrator.lastTps;
    final frameMs = _probe.avgInterval.inMicroseconds / 1000;
    final budgetMs = 1000 / _budget.fps;
    final p = widget.power.latest;
    double? share(double? percent) => percent == null ? null : percent / 100; // the sidecar writes 0..100
    return _game
        .situation(
          backend: label,
          speedShare: base != null && base > 0 && tps != null && tps > 0 ? tps / base : null,
          frameBudget: frameMs > 0 ? frameMs / budgetMs : null,
          cpuLoad: share(p?.cpuLoad),
          gpuLoad: share(p?.gpu3dLoad),
          npuLoad: share(p?.npuLoad),
        )
        .asText;
  }

  void _sample() {
    if (!mounted) return;
    final interval = _probe.avgInterval.inMicroseconds / 1000;
    _series.add(StageSample(
      frameMs: interval > 0 ? interval : 0,
      tps: _narrator.lastTps ?? 0,
      backend: _narrator.state == NarratorState.off ? 'off' : backendLabel(_narrator.backend),
    ));
  }

  void _log(String line) {
    // ignore: avoid_print
    print(line);
    final p = widget.config.resultsPath;
    if (p != null) {
      try {
        File('${p.replaceAll(RegExp(r'\.[a-z]+$'), '')}.log').writeAsStringSync('$line\n', mode: FileMode.append);
      } catch (_) {}
    }
  }

  /// Writes the composited frame next to the results file: headless runs have
  /// no screen to look at, and `screencapture` needs a permission we lack.
  Future<void> _captureFrame({String suffix = ''}) async {
    final p = widget.config.resultsPath;
    if (p == null) return;
    try {
      final boundary = _captureKey.currentContext?.findRenderObject() as RenderRepaintBoundary?;
      if (boundary == null) return;
      final image = await boundary.toImage(pixelRatio: 1.0);
      final bytes = await image.toByteData(format: ui.ImageByteFormat.png);
      if (bytes != null) {
        final out = '${p.replaceAll(RegExp(r'\.[a-z]+$'), '')}$suffix.png';
        File(out).writeAsBytesSync(bytes.buffer.asUint8List());
        _log('ACT3_FRAME $out ${image.width}x${image.height}');
      }
    } catch (e) {
      _log('ACT3_FRAME capture failed: $e');
    }
  }

  FrameRecorder? _recorder;

  Future<void> _stopRecording() async {
    final r = _recorder;
    if (r == null) return;
    _recorder = null;
    await r.stop();
    final stats = '${r.width}x${r.height}: ${r.frames} frames grabbed, ${r.repeated} repeated, ${r.grabLatency}';
    _log(r.error == null ? 'ACT3_RECORDED ${r.path} $stats' : 'ACT3_RECORD_FAILED ${r.path}: ${r.error} ($stats)');
  }

  void _emitResults() {
    final json = jsonEncode({
      'act': 3,
      'scene': 'runner',
      'captured_at': DateTime.now().toIso8601String(),
      'os': Platform.operatingSystem,
      'events_total': _events.total,
      'warnings': _events.recent.where((e) => e.kind == 'warning').map((e) => '${e.subject}: ${e.detail}').toList(),
      'run': _game.toJson(),
      'calibration': widget.config.act3.calibrate ? _calibrator.toJson() : null,
      'baselines': _series.bestTps.map((k, v) => MapEntry(k, {'tok_s': double.parse(v.toStringAsFixed(1)), 'measured': _series.isMeasured(k)})),
      'frame': _probe.toJson(),
      'narrator': _narrator.toJson(),
      'phases': _phases,
    });
    // ignore: avoid_print
    print('ACT3_JSON $json');
    final p = widget.config.resultsPath;
    if (p != null) File(p).writeAsStringSync('$json\n', mode: FileMode.append);
    if (widget.config.autorunExit) exit(0);
  }

  PerspectiveCamera _camera(Duration elapsed) {
    final dash = _game.dashPosition;
    final f = _game.forward;
    // Behind and above Dash, looking down the trail so the audience sees what
    // is coming at him, not just where he has been.
    final wantPos = vm.Vector3(dash.x - f.x * 7.5, dash.y + 3.0, dash.z - f.y * 7.5);
    final wantTarget = vm.Vector3(dash.x + f.x * 11, dash.y + 1.6, dash.z + f.y * 11);
    // Exponential smoothing: the camera lags a little, which reads as weight.
    _camPos = _camPos == null ? wantPos : _camPos! + (wantPos - _camPos!) * 0.18;
    _camTarget = _camTarget == null ? wantTarget : _camTarget! + (wantTarget - _camTarget!) * 0.25;

    // Shake goes on after the smoothing; before it, the lag ate almost all of
    // it. Offsets are in the camera's own frame: right, up, forward.
    final shake = _game.shakeOffset;
    final toTarget = _camTarget! - _camPos!;
    final reach = toTarget.length;
    final fwd = toTarget / reach;
    final right = fwd.cross(vm.Vector3(0, 1, 0))..normalize();
    final up = right.cross(fwd);
    final t = shake.translation;
    final r = shake.rotationEuler; // pitch, yaw, roll
    final pos = _camPos! + right * t.x + up * t.y + fwd * t.z;
    final target = _camTarget! + right * (t.x + r.y * reach) + up * (t.y + r.x * reach) + fwd * t.z;
    final tilted = up * math.cos(r.z) + right * math.sin(r.z);
    return PerspectiveCamera(position: pos, target: target, up: tilted);
  }

  void _onKey(KeyEvent event) {
    if (event is KeyUpEvent) {
      if (event.logicalKey == LogicalKeyboardKey.keyA ||
          event.logicalKey == LogicalKeyboardKey.keyD ||
          event.logicalKey == LogicalKeyboardKey.arrowLeft ||
          event.logicalKey == LogicalKeyboardKey.arrowRight) {
        _game.releaseControl();
      }
      return;
    }
    // A held key repeats; only steering wants that. A held H flickered the
    // panels and a held N skipped through the whole day.
    if (event is KeyRepeatEvent && !_steeringKeys.contains(event.logicalKey)) return;
    switch (event.logicalKey) {
      case LogicalKeyboardKey.keyA:
      case LogicalKeyboardKey.arrowLeft:
        _game.steer(-1);
      case LogicalKeyboardKey.keyD:
      case LogicalKeyboardKey.arrowRight:
        _game.steer(1);
      case LogicalKeyboardKey.space:
      case LogicalKeyboardKey.keyW:
      case LogicalKeyboardKey.arrowUp:
        _game.jump();
      case LogicalKeyboardKey.keyP:
        _game.togglePause();
      case LogicalKeyboardKey.keyN:
        _game.skipDayPhase();
      case LogicalKeyboardKey.keyH:
        setState(() => _showDetails = !_showDetails);
      case LogicalKeyboardKey.keyB:
        // Re-take the "model on its own" baselines, e.g. after changing the
        // scene load by hand.
        if (_baselining) return;
        _series.clearBaselines();
        unawaited(_measureBaselines(const ['gpu', 'npu', 'cpu']));
      case LogicalKeyboardKey.digit0:
        unawaited(_narrator.stop());
      case LogicalKeyboardKey.digit1:
        _switchIfAvailable(PreferredBackend.cpu);
      case LogicalKeyboardKey.digit2:
        _switchIfAvailable(PreferredBackend.gpu);
      case LogicalKeyboardKey.digit3:
        _switchIfAvailable(PreferredBackend.npu);
    }
  }

  static final _steeringKeys = {
    LogicalKeyboardKey.keyA,
    LogicalKeyboardKey.keyD,
    LogicalKeyboardKey.arrowLeft,
    LogicalKeyboardKey.arrowRight,
  };

  static const _catalog = BackendCatalog();

  /// Only what this OS can run: on macOS an NPU load hangs instead of failing.
  void _switchIfAvailable(PreferredBackend b) {
    if (!_catalog.status(b).isAvailable) {
      _log('ACT3_KEY ${b.name} not available here: ${_catalog.status(b).reason}');
      return;
    }
    unawaited(_switchNarrator(b));
  }

  @override
  Widget build(BuildContext context) {
    if (_error != null) {
      return Scaffold(appBar: AppBar(title: const Text('Act 3')), body: Center(child: Text(_error!)));
    }
    if (!_ready) return const Scaffold(body: Center(child: CircularProgressIndicator()));
    return Scaffold(
      body: KeyboardListener(
        focusNode: _focus,
        autofocus: true,
        onKeyEvent: _onKey,
        child: RepaintBoundary(
          key: _captureKey,
          child: Stack(
            children: [
              Positioned.fill(
                child: SceneView(
                  _game.scene,
                  cameraBuilder: _camera,
                  onTick: (elapsed, dt) => _game.tick(dt),
                ),
              ),
              Positioned(
                top: 12,
                left: 12,
                right: 12,
                child: ListenableBuilder(
                  listenable: Listenable.merge([_probe, _narrator, _series, _calibrator]),
                  builder: (context, _) => StageMeters(
                    series: _series,
                    targetFps: _budget.fps,
                    sustainedFps: _probe.sustainedFps,
                    frameMs: _probe.avgInterval.inMicroseconds / 1000,
                    tps: _narrator.lastTps ?? 0,
                    backend: _narrator.state == NarratorState.off ? 'off' : backendLabel(_narrator.backend),
                    calibrating: _calibrator.running,
                    calibrationStatus: _calibrator.status,
                  ),
                ),
              ),
              if (_showDetails)
                Positioned(
                  top: 190,
                  left: 12,
                  child: _Hud(
                    probe: _probe,
                    game: _game,
                    events: _events,
                    narrator: _narrator,
                    power: widget.power,
                    powerMode: widget.config.powerMode,
                  ),
                ),
              if (_showDetails)
                Positioned(top: 190, right: 12, child: _Controls(game: _game, probe: _probe, narrator: _narrator)),
              Positioned(
                left: 12,
                right: 12,
                bottom: 12,
                child: ListenableBuilder(
                  listenable: Listenable.merge([_narrator, _events]),
                  builder: (context, _) => StageCommentary(
                    // A streaming turn takes the screen with its first words;
                    // until then, and between turns, the last finished line
                    // stays up at the same size.
                    text: _streamingLine ? _narrator.live : (_narrator.history.isEmpty ? '' : _narrator.history.last.text),
                    turn: _streamingLine ? _narrator.history.length : _narrator.history.length - 1,
                    tps: _narrator.lastTps,
                    backend: _narrator.state == NarratorState.off ? 'off' : backendLabel(_narrator.backend),
                  ),
                ),
              ),
              // Off in the stage view, where it sat across the meters (and
              // across every recorded video); H brings it back with the panels.
              if (_showDetails)
                Positioned(
                  top: 4,
                  left: 0,
                  right: 0,
                  child: Center(
                    child: TextButton.icon(
                      onPressed: () => Navigator.of(context).maybePop(),
                      icon: const Icon(Icons.arrow_back),
                      label: const Text("Act 3 — The NPU's own argument"),
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    // Leaving the act mid-recording still finishes the file.
    unawaited(_stopRecording());
    _sampler?.cancel();
    _narrator.dispose();
    _probe.dispose();
    _game.dispose();
    _events.dispose();
    _focus.dispose();
    super.dispose();
  }
}

class _Panel extends StatelessWidget {
  const _Panel({required this.child});
  final Widget child;

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.all(10),
        decoration: BoxDecoration(color: Colors.black.withValues(alpha: 0.55), borderRadius: BorderRadius.circular(8)),
        child: child,
      );
}

class _Hud extends StatelessWidget {
  const _Hud({
    required this.probe,
    required this.game,
    required this.events,
    required this.narrator,
    required this.power,
    required this.powerMode,
  });

  final FrameTimeProbe probe;
  final RunnerGame game;
  final SceneEventLog events;
  final SceneNarrator narrator;
  final PowerSource power;
  final PowerMode powerMode;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: Listenable.merge([probe, game, events, narrator, power]),
      builder: (context, _) {
        String ms(Duration d) => (d.inMicroseconds / 1000).toStringAsFixed(1);
        // Sustained fps is the honest headline: the recent-window number
        // describes only the frames that happened, so it stays high while the
        // app stalls between them.
        final over = probe.sustainedFps < probe.targetFps * 0.92;
        final n = narrator;
        final llm = switch (n.state) {
          NarratorState.off => 'off',
          NarratorState.loading => 'loading ${backendLabel(n.backend)}…',
          NarratorState.failed => 'failed: ${n.error?.split('\n').first}',
          _ => '${backendLabel(n.backend)} · '
              '${n.lastTps?.toStringAsFixed(1) ?? '—'} tok/s · '
              'TTFT ${n.lastTtft?.inMilliseconds ?? '—'} ms · '
              'ctx ${n.contextTokens}/${n.maxTokens} · ${n.history.length} turns',
        };
        return _Panel(
          child: DefaultTextStyle(
            style: const TextStyle(fontFamily: 'monospace', fontSize: 13, color: Colors.white),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('${probe.sustainedFps.toStringAsFixed(1)} fps  (target ${probe.targetFps})',
                    style: TextStyle(
                        fontSize: 22,
                        fontWeight: FontWeight.bold,
                        color: over ? Colors.orangeAccent : Colors.greenAccent)),
                Text('frame  ${ms(probe.avgTotal)} ms avg · p95 ${ms(probe.p95Total)} · worst ${ms(probe.worstFrame)}'),
                Text('ui     ${ms(probe.avgBuild)} ms   raster ${ms(probe.avgRaster)} ms'),
                Text('recent ${probe.fps.toStringAsFixed(1)} fps · gap p95 ${(probe.p95Interval.inMicroseconds / 1000).toStringAsFixed(1)} ms'),
                Text('janky  ${probe.jankyFrames}/${probe.frames} · missed ${probe.missedFrames}'),
                const SizedBox(height: 6),
                Text('dash   ${game.distance.round()} m · ${game.speed.toStringAsFixed(1)} m/s · ${game.biomeLabel}'),
                Text('cleared ${game.obstaclesCleared} · jumps ${game.jumpsCleared}/${game.jumps} · near ${game.nearMisses} · stumbles ${game.stumbles}'),
                Text('world  ${game.liveChunks} chunks · ${game.liveProps} props'),
                const SizedBox(height: 6),
                Text('LLM    $llm',
                    style: TextStyle(color: n.state == NarratorState.generating ? Colors.lightBlueAccent : Colors.white)),
                Text('power  ${power.latest == null ? power.label : formatPower(PowerSummary.of([power.latest!]), powerMode)}',
                    style: const TextStyle(color: Colors.white70)),
              ],
            ),
          ),
        );
      },
    );
  }
}

class _Controls extends StatelessWidget {
  const _Controls({required this.game, required this.probe, required this.narrator});
  final RunnerGame game;
  final FrameTimeProbe probe;
  final SceneNarrator narrator;

  @override
  Widget build(BuildContext context) {
    const catalog = BackendCatalog();
    return ListenableBuilder(
      listenable: Listenable.merge([game, narrator]),
      builder: (context, _) => _Panel(
        child: SizedBox(
          width: 290,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text('Narrator  (keys 0–3)', style: TextStyle(color: Colors.white70, fontSize: 12)),
              Wrap(spacing: 6, runSpacing: 6, children: [
                ChoiceChip(
                  label: const Text('Off'),
                  selected: narrator.state == NarratorState.off,
                  onSelected: (_) => narrator.stop(),
                ),
                for (final b in PreferredBackend.values)
                  ChoiceChip(
                    label: Text(backendLabel(b)),
                    selected: narrator.backend == b && narrator.state != NarratorState.off,
                    onSelected: catalog.status(b).isAvailable && narrator.state != NarratorState.loading
                        ? (_) => narrator.switchTo(b)
                        : null,
                  ),
              ]),
              const SizedBox(height: 8),
              Text('draw distance ${game.settings.viewChunks} chunks (${game.settings.viewChunks * 64} m)',
                  style: const TextStyle(color: Colors.white, fontSize: 12)),
              Slider(
                value: game.settings.viewChunks.clamp(2, 14).toDouble(),
                min: 2,
                max: 14,
                divisions: 12,
                onChanged: (v) => game.applySettings(game.settings.copyWith(viewChunks: v.round())),
              ),
              Text('vegetation ×${game.settings.propDensity.toStringAsFixed(2)}',
                  style: const TextStyle(color: Colors.white, fontSize: 12)),
              Slider(
                // Ranges reach the calibrator's top rung (LoadCalibrator.ladder).
                value: game.settings.propDensity.clamp(0, 6.5),
                min: 0,
                max: 6.5,
                divisions: 26,
                onChanged: (v) => game.applySettings(game.settings.copyWith(propDensity: v)),
              ),
              Text('render scale ${game.settings.renderScale.toStringAsFixed(2)}',
                  style: const TextStyle(color: Colors.white, fontSize: 12)),
              Slider(
                value: game.settings.renderScale.clamp(0.5, 4.0),
                min: 0.5,
                max: 4.0,
                divisions: 14,
                onChanged: (v) => game.applySettings(game.settings.copyWith(renderScale: v)),
              ),
              Wrap(spacing: 6, runSpacing: 6, children: [
                FilledButton.tonal(
                  onPressed: () => game.applySettings(game.settings.copyWith(shadows: !game.settings.shadows)),
                  child: Text(game.settings.shadows ? 'Shadows on' : 'Shadows off'),
                ),
                FilledButton.tonal(
                  onPressed: () => game.applySettings(game.settings.copyWith(post: !game.settings.post)),
                  child: Text(game.settings.post ? 'Post on' : 'Post off'),
                ),
                FilledButton.tonal(onPressed: game.togglePause, child: Text(game.paused ? 'Resume' : 'Pause')),
                FilledButton.tonal(onPressed: probe.reset, child: const Text('Reset stats')),
              ]),
            ],
          ),
        ),
      ),
    );
  }
}

