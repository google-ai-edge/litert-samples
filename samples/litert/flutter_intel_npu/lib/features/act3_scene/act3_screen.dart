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
import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:flutter_scene/scene.dart';
import 'package:vector_math/vector_math.dart' as vm;

import '../../app/demo_config.dart';
import '../../core/engines/backend_catalog.dart';
import '../../core/narration/scene_narrator.dart';
import '../../core/power/power_source.dart';
import '../../core/scene/scene_events.dart';
import '../../core/telemetry/frame_time_probe.dart';
import 'runner/load_calibrator.dart' show FrameBudget;
import 'arena_game.dart';

/// Act 3 — heavy 3D on the GPU while the model narrates the scene live.
/// Frame time and tokens/sec share one HUD; the backend switch is a chip.
class Act3Screen extends StatefulWidget {
  const Act3Screen({super.key, required this.config, required this.power});

  final DemoConfig config;
  final PowerSource power;

  @override
  State<Act3Screen> createState() => _Act3ScreenState();
}

class _Act3ScreenState extends State<Act3Screen> {
  final SceneEventLog _events = SceneEventLog();
  late final ArenaGame _game = ArenaGame(events: _events);
  late final FrameTimeProbe _probe = FrameTimeProbe(
    targetFps: FrameBudget.resolve(
      requested: widget.config.act3.targetFps,
      refreshRate: ui.PlatformDispatcher.instance.implicitView?.display.refreshRate ?? 0,
    ).fps,
  );
  late final SceneNarrator _narrator = SceneNarrator(
    events: _events,
    modelPath: widget.config.modelPath,
    modelPathOverrides: {
      if (widget.config.act3.npuModelPath != null) PreferredBackend.npu: widget.config.act3.npuModelPath!,
    },
    maxTokens: widget.config.act3.narratorMaxTokens,
    interval: Duration(milliseconds: widget.config.act3.narratorIntervalMs),
  );
  final GlobalKey _captureKey = GlobalKey();
  bool _ready = false;
  String? _error;
  double _orbit = 0;
  final List<Map<String, Object?>> _phases = [];

  @override
  void initState() {
    super.initState();
    _game.bodyCount = widget.config.act3.bodyCount;
    unawaited(_init());
  }

  Future<void> _init() async {
    try {
      await Scene.initializeStaticResources();
      if (!mounted) return;
      _game.build();
      _game.scene.renderScale = widget.config.act3.renderScale;
      _probe.start();
      setState(() => _ready = true);
      final plan = widget.config.act3.narratorPlan;
      if (plan.isNotEmpty) {
        await _runPlan(plan);
      } else if (widget.config.act3.autorunSeconds > 0) {
        await Future<void>.delayed(Duration(seconds: widget.config.act3.autorunSeconds));
        if (!mounted) return;
        await _captureFrame();
        _emitResults();
      }
    } catch (e, st) {
      _log('ACT3_INIT_FAILED $e\n$st');
      if (mounted) setState(() => _error = '$e\n$st');
      if (widget.config.autorunExit) exit(3);
    }
  }

  /// Headless measurement: phases like `off:10,gpu:20,cpu:20` — each phase
  /// resets frame stats, runs the narrator on that backend (or off) for N
  /// seconds, and records frame + narrator stats. Engines are preloaded
  /// first so a switch is a pointer swap, not a load.
  Future<void> _runPlan(List<(String, int)> plan) async {
    final preloadErrors = <String, String>{};
    for (final (b, _) in plan) {
      if (b == 'off') continue;
      try {
        await _narrator.preload(DemoConfig.parseBackend(b));
        _log('ACT3_PRELOAD $b ok ${_narrator.loadMs}');
      } catch (e) {
        preloadErrors[b] = e.toString().split('\n').first;
        _log('ACT3_PRELOAD $b FAILED: ${preloadErrors[b]}');
      }
    }
    for (final (b, secs) in plan) {
      if (!mounted) return;
      String? phaseError = preloadErrors[b];
      try {
        if (b == 'off') {
          await _narrator.stop();
        } else if (phaseError == null) {
          await _narrator.switchTo(DemoConfig.parseBackend(b));
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
      _phases.add({
        'phase': b,
        'seconds': secs,
        'frame': _probe.toJson(),
        'narrator': b == 'off' ? null : _narrator.toJson(phase: true),
        'narrator_error': phaseError,
        'live_bodies': _game.liveBodies,
        'power': powerSummary.toJson(),
      });
      await _captureFrame(suffix: '-$b');
    }
    _emitResults();
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

  /// Writes the composited frame next to the results file (headless runs
  /// have no screen to look at; screencapture needs a permission we lack).
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
        // ignore: avoid_print
        print('ACT3_FRAME $out ${image.width}x${image.height}');
      }
    } catch (e) {
      // ignore: avoid_print
      print('ACT3_FRAME capture failed: $e');
    }
  }

  void _emitResults() {
    final json = jsonEncode({
      'act': 3,
      'captured_at': DateTime.now().toIso8601String(),
      'os': Platform.operatingSystem,
      'body_count': _game.bodyCount,
      'live_bodies': _game.liveBodies,
      'render_scale': _game.scene.renderScale,
      'events_total': _events.total,
      'warnings': _events.recent.where((e) => e.kind == 'warning').map((e) => '${e.subject}: ${e.detail}').toList(),
      'sample_heights': _game.sampleHeights(),
      'physics_backend': _game.world.backendName,
      'frame': _probe.toJson(),
      'narrator': _narrator.toJson(),
      'phases': _phases,
    });
    // ignore: avoid_print
    print('ACT3_JSON $json');
    final p = widget.config.resultsPath;
    if (p != null) File(p).writeAsStringSync('$json\n', mode: FileMode.append);
    if (widget.config.autorunExit) {
      exit(0);
    }
  }

  PerspectiveCamera _camera(Duration elapsed) {
    _orbit = elapsed.inMilliseconds / 1000 * 0.15;
    const r = 13.0;
    return PerspectiveCamera(
      position: vm.Vector3(r * math.cos(_orbit), 7.5, r * math.sin(_orbit)),
      target: vm.Vector3(0, 1.5, 0),
    );
  }

  @override
  Widget build(BuildContext context) {
    if (_error != null) {
      return Scaffold(appBar: AppBar(title: const Text('Act 3')), body: Center(child: Text(_error!)));
    }
    if (!_ready) return const Scaffold(body: Center(child: CircularProgressIndicator()));
    return Scaffold(
      body: RepaintBoundary(
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
            Positioned(top: 12, left: 12, child: _Hud(probe: _probe, game: _game, events: _events, narrator: _narrator, power: widget.power, powerMode: widget.config.powerMode)),
            Positioned(top: 12, right: 12, child: _Controls(game: _game, probe: _probe, narrator: _narrator)),
            Positioned(left: 12, right: 12, bottom: 12, child: _Commentary(narrator: _narrator, events: _events)),
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
    );
  }

  @override
  void dispose() {
    _narrator.dispose();
    _probe.dispose();
    _game.dispose();
    _events.dispose();
    super.dispose();
  }
}

class _Hud extends StatelessWidget {
  const _Hud({required this.probe, required this.game, required this.events, required this.narrator, required this.power, required this.powerMode});
  final FrameTimeProbe probe;
  final ArenaGame game;
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
                    style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold, color: over ? Colors.orangeAccent : Colors.greenAccent)),
                Text('frame  ${ms(probe.avgTotal)} ms avg · p95 ${ms(probe.p95Total)} · worst ${ms(probe.worstFrame)}'),
                Text('ui     ${ms(probe.avgBuild)} ms   raster ${ms(probe.avgRaster)} ms'),
                Text('recent ${probe.fps.toStringAsFixed(1)} fps · gap p95 ${(probe.p95Interval.inMicroseconds / 1000).toStringAsFixed(1)} ms'),
                Text('janky  ${probe.jankyFrames}/${probe.frames} · missed ${probe.missedFrames}'),
                const SizedBox(height: 6),
                Text('bodies ${game.liveBodies}/${game.bodyCount} · scale ${game.scene.renderScale.toStringAsFixed(2)}'),
                Text('events ${events.total} · ${events.rate.toStringAsFixed(0)}/s'),
                const SizedBox(height: 6),
                Text('LLM    $llm', style: TextStyle(color: n.state == NarratorState.generating ? Colors.lightBlueAccent : Colors.white)),
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
  final ArenaGame game;
  final FrameTimeProbe probe;
  final SceneNarrator narrator;

  @override
  Widget build(BuildContext context) {
    const catalog = BackendCatalog();
    return ListenableBuilder(
      listenable: Listenable.merge([game, narrator]),
      builder: (context, _) => _Panel(
        child: SizedBox(
          width: 280,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text('Narrator', style: TextStyle(color: Colors.white70, fontSize: 12)),
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
              Text('bodies ${game.bodyCount}', style: const TextStyle(color: Colors.white)),
              Slider(
                value: game.bodyCount.toDouble(),
                min: 10,
                max: 400,
                divisions: 39,
                onChanged: (v) => game.setBodyCount(v.round()),
              ),
              Text('render scale ${game.scene.renderScale.toStringAsFixed(2)}', style: const TextStyle(color: Colors.white)),
              Slider(
                value: game.scene.renderScale,
                min: 0.5,
                max: 2.0,
                divisions: 15,
                onChanged: game.setRenderScale,
              ),
              Wrap(spacing: 6, runSpacing: 6, children: [
                FilledButton.tonal(onPressed: () => game.spawnBurst(25), child: const Text('Burst ×25')),
                FilledButton.tonal(onPressed: game.clearBodies, child: const Text('Clear')),
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

class _Commentary extends StatelessWidget {
  const _Commentary({required this.narrator, required this.events});
  final SceneNarrator narrator;
  final SceneEventLog events;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: Listenable.merge([narrator, events]),
      builder: (context, _) {
        final n = narrator;
        final lines = n.history.reversed.take(3).toList().reversed.toList();
        return _Panel(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              if (n.state == NarratorState.off)
                DefaultTextStyle(
                  style: const TextStyle(fontFamily: 'monospace', fontSize: 12, color: Colors.white70),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      for (final e in events.recent.reversed.take(3).toList().reversed)
                        Text(e.asText, maxLines: 1, overflow: TextOverflow.ellipsis),
                    ],
                  ),
                )
              else ...[
                for (final t in lines)
                  Padding(
                    padding: const EdgeInsets.only(bottom: 4),
                    child: Text(
                      '[${_stamp(t.at)}] ${t.text}',
                      style: const TextStyle(color: Colors.white70, fontSize: 14),
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                if (n.state == NarratorState.generating)
                  Text(n.live.isEmpty ? '…' : n.live, style: const TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.w600)),
              ],
            ],
          ),
        );
      },
    );
  }

  static String _stamp(double at) {
    final s = at.round(); // rounding the seconds alone once printed 00:60
    return '${(s ~/ 60).toString().padLeft(2, '0')}:${(s % 60).toString().padLeft(2, '0')}';
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
