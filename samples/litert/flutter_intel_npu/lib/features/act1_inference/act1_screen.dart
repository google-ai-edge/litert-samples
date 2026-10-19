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
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/scheduler.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart';

import '../../app/demo_config.dart';
import '../../core/engines/backend_catalog.dart';
import '../../core/engines/benchmark_run.dart';
import '../../core/engines/benchmark_runner.dart';
import '../../core/engines/engine_pool.dart';
import '../../core/power/power_source.dart';
import 'race_view.dart';

/// Act 1 — the same prompt on the CPU, the GPU and the NPU, each on its own
/// lane. Measured one backend at a time; **Replay as a race** then plays the
/// recorded runs side by side at their real speed.
class Act1Screen extends StatefulWidget {
  const Act1Screen({super.key, required this.config, required this.pool, required this.power});

  final DemoConfig config;
  final EnginePool pool;
  final PowerSource power;

  @override
  State<Act1Screen> createState() => _Act1ScreenState();
}

class _Act1ScreenState extends State<Act1Screen> with SingleTickerProviderStateMixin {
  late final BenchmarkRunner _runner = BenchmarkRunner(
    pool: widget.pool,
    catalog: const BackendCatalog(),
    power: widget.power,
  );
  late PowerMode _powerMode = widget.config.powerMode;
  late final TextEditingController _prompt = TextEditingController(text: widget.config.prompt);

  final Map<PreferredBackend, BenchmarkRun> _rows = {};
  final StringBuffer _live = StringBuffer();
  PreferredBackend? _liveBackend;
  bool _running = false;
  String? _installError;
  bool _installed = false;

  late final Ticker _replay = createTicker(_onReplayTick);

  /// Where the race replay is; null when the lanes show the measured runs.
  Duration? _replayAt;

  @override
  void initState() {
    super.initState();
    unawaited(
      _install().then((_) async {
        if (!widget.config.autorun) return;
        for (var i = 0; i < widget.config.autorunRepeat && mounted; i++) {
          await _run();
        }
        if (!mounted) return;
        await _captureRace();
        if (widget.config.autorunExit) {
          await widget.pool.releaseAll();
          exit(0);
        }
      }),
    );
  }

  Future<void> _install() async {
    if (widget.config.modelPath.isEmpty) {
      setState(() => _installError = 'MODEL_PATH is not set: point it at gemma-4-E2B-it.litertlm (see README).');
      _print('BENCH_ERROR MODEL_PATH is not set');
      if (widget.config.autorunExit) exit(2);
      return;
    }
    try {
      await FlutterEdgeAi.installModel(
        modelType: ModelType.gemma4,
        fileType: ModelFileType.litertlm,
      ).fromFile(widget.config.modelPath).install();
      if (!mounted) return;
      setState(() => _installed = true);
    } catch (e) {
      if (!mounted) return;
      setState(() => _installError = '$e');
      _print('BENCH_ERROR install failed: $e');
      if (widget.config.autorunExit) exit(2);
    }
  }

  Future<void> _run() async {
    if (_running || !_installed) return;
    _stopReplay();
    setState(() {
      _running = true;
      _rows.clear();
      _live.clear();
      _liveBackend = null;
    });
    final stream = _runner.runComparison(
      modelPathFor: widget.config.modelPathFor,
      maxTokens: widget.config.maxTokens,
      backends: widget.config.backends,
      prompt: _prompt.text,
      warmup: widget.config.warmup,
      maxOutputTokens: widget.config.maxOutputTokens,
      topK: widget.config.topK,
      onChunk: (backend, chunk) {
        if (_liveBackend != backend) {
          _liveBackend = backend;
          _live.clear();
        }
        _live.write(chunk);
        if (mounted) setState(() {});
      },
    );
    await for (final run in stream) {
      if (!mounted) return;
      setState(() => _rows[run.requestedBackend] = run);
      if (run.status.index >= BenchmarkRunStatus.ok.index) {
        _print('BENCH_JSON ${jsonEncode(run.toJson())}');
      }
    }
    if (!mounted) return;
    setState(() {
      _running = false;
      _liveBackend = null;
    });
    final summary = jsonEncode({
      'models': {for (final b in widget.config.backends) b.name: DemoConfig.fileName(widget.config.modelPathFor(b))},
      'max_tokens': widget.config.maxTokens,
      'policy': widget.pool.policy.name,
      'warmup': widget.config.warmup,
      'prompt': _prompt.text,
      'os': Platform.operatingSystem,
      'os_version': Platform.operatingSystemVersion,
      'captured_at': DateTime.now().toIso8601String(),
      'runs': _rows.values.map((r) => r.toJson()).toList(),
    });
    _print('BENCH_DONE $summary');
    final resultsPath = widget.config.resultsPath;
    if (resultsPath != null) {
      // Append one JSON object per sweep (NDJSON) so repeats are all kept.
      await File(resultsPath).writeAsString('$summary\n', mode: FileMode.append);
      _print('BENCH_WRITTEN $resultsPath');
    }
  }

  /// Headless runs have no screen to look at: replay the race and save its
  /// middle and its finish next to the results file.
  Future<void> _captureRace() async {
    if (widget.config.resultsPath == null || !_canReplay) return;
    final length = _raceLength;
    _startReplay();
    await Future<void>.delayed(length ~/ 2);
    await _captureFrame('-race-mid');
    await Future<void>.delayed(length ~/ 2 + const Duration(milliseconds: 400));
    await _captureFrame('-race-end');
  }

  final GlobalKey _captureKey = GlobalKey();

  Future<void> _captureFrame(String suffix) async {
    final p = widget.config.resultsPath;
    if (p == null || !mounted) return;
    try {
      final boundary = _captureKey.currentContext?.findRenderObject() as RenderRepaintBoundary?;
      if (boundary == null) return;
      final image = await boundary.toImage(pixelRatio: MediaQuery.devicePixelRatioOf(context));
      final bytes = await image.toByteData(format: ui.ImageByteFormat.png);
      if (bytes == null) return;
      final out = '${p.replaceAll(RegExp(r'\.[a-z]+$'), '')}$suffix.png';
      File(out).writeAsBytesSync(bytes.buffer.asUint8List());
      _print('BENCH_FRAME $out ${image.width}x${image.height}');
    } catch (e) {
      _print('BENCH_FRAME capture failed: $e');
    }
  }

  // ------------------------------------------------------------- replay

  Duration get _raceLength {
    var longest = Duration.zero;
    for (final r in _rows.values) {
      final f = r.finishedAt;
      if (f != null && f > longest) longest = f;
    }
    return longest;
  }

  bool get _canReplay => !_running && _rows.values.where((r) => r.finishedAt != null).length >= 2;

  void _startReplay() {
    _replay.stop();
    setState(() => _replayAt = Duration.zero);
    _replay.start();
  }

  void _onReplayTick(Duration elapsed) {
    final end = _raceLength;
    setState(() => _replayAt = elapsed < end ? elapsed : end);
    if (elapsed >= end) _replay.stop(); // the finished race stays on screen
  }

  void _stopReplay() {
    _replay.stop();
    if (_replayAt != null) setState(() => _replayAt = null);
  }

  void _print(String line) {
    // stdout, not debugPrint: headless runs grep these lines from `flutter run`.
    // ignore: avoid_print
    print(line);
  }

  // ------------------------------------------------------------- UI

  @override
  Widget build(BuildContext context) {
    final cfg = widget.config;
    final theme = Theme.of(context);
    final places = placesOf(_rows.values);
    final replaying = _replayAt != null;
    final allDone = !_running && _rows.isNotEmpty;

    return RepaintBoundary(
      key: _captureKey,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Act 1 — Inference: CPU · GPU · NPU'),
          actions: [
            SegmentedButton<PowerMode>(
              segments: const [
                ButtonSegment(value: PowerMode.auto, label: Text('auto')),
                ButtonSegment(value: PowerMode.watts, label: Text('W')),
                ButtonSegment(value: PowerMode.utilization, label: Text('%')),
              ],
              selected: {_powerMode},
              onSelectionChanged: (v) => setState(() => _powerMode = v.first),
              style: const ButtonStyle(visualDensity: VisualDensity.compact),
            ),
            const SizedBox(width: 12),
          ],
        ),
        body: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              if (_installError != null)
                Padding(
                  padding: const EdgeInsets.only(bottom: 8),
                  child: Text(_installError!, style: const TextStyle(color: Colors.red)),
                ),
              Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _prompt,
                      maxLines: 2,
                      minLines: 1,
                      decoration: const InputDecoration(labelText: 'Prompt', border: OutlineInputBorder()),
                    ),
                  ),
                  const SizedBox(width: 12),
                  FilledButton.icon(
                    onPressed: _running || !_installed ? null : () => _run(),
                    icon: const Icon(Icons.play_arrow),
                    label: Text(_running ? 'Measuring…' : 'Run comparison'),
                  ),
                  const SizedBox(width: 8),
                  OutlinedButton.icon(
                    onPressed: _canReplay ? _startReplay : null,
                    icon: const Icon(Icons.flag),
                    label: const Text('Replay as a race'),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              ListenableBuilder(
                listenable: widget.power,
                builder: (context, _) => Text(
                  [
                    if (replaying)
                      'race replay ${(_replayAt!.inMilliseconds / 1000).toStringAsFixed(1)} s — the runs were measured one '
                          'after another and are replayed together at their real speed'
                    else if (allDone)
                      'measured one backend at a time: the iGPU and the NPU share memory bandwidth'
                    else
                      'maxTokens ${cfg.maxTokens} · policy ${widget.pool.policy.name}',
                    'power: ${widget.power.label}',
                  ].join(' · '),
                  style: theme.textTheme.bodySmall,
                ),
              ),
              const SizedBox(height: 8),
              Expanded(
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    for (final b in cfg.backends)
                      Expanded(
                        child: ListenableBuilder(
                          listenable: widget.power,
                          builder: (context, _) {
                            final live = _running && _rows[b]?.status == BenchmarkRunStatus.generating;
                            final sample = live ? widget.power.latest : null;
                            return RaceColumn(
                              backend: b,
                              run: _rows[b],
                              liveText: _liveBackend == b ? _live.toString() : null,
                              liveLoads: sample == null ? null : Loads.ofSample(sample),
                              replayAt: _replayAt,
                              place: allDone ? places[b] : null,
                              powerMode: _powerMode,
                            );
                          },
                        ),
                      ),
                  ],
                ),
              ),
              ExpansionTile(
                title: Text('Details', style: theme.textTheme.titleSmall),
                tilePadding: EdgeInsets.zero,
                children: [
                  SingleChildScrollView(
                    scrollDirection: Axis.horizontal,
                    child: _ComparisonTable(rows: cfg.backends.map((b) => _rows[b]).toList(), powerMode: _powerMode),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    _replay.dispose();
    _prompt.dispose();
    super.dispose();
  }
}

/// Every number, for whoever asks how it was measured: runtime-reported and
/// wall-clock side by side.
class _ComparisonTable extends StatelessWidget {
  const _ComparisonTable({required this.rows, required this.powerMode});

  final List<BenchmarkRun?> rows;
  final PowerMode powerMode;

  @override
  Widget build(BuildContext context) {
    String ms(Duration? d) => d == null ? '—' : '${d.inMilliseconds} ms';
    String num1(double? v) => v == null ? '—' : v.toStringAsFixed(1);

    return DataTable(
      columns: const [
        DataColumn(label: Text('Requested')),
        DataColumn(label: Text('Actual')),
        DataColumn(label: Text('Model')),
        DataColumn(label: Text('Status')),
        DataColumn(label: Text('Load')),
        DataColumn(label: Text('TTFT (rt / wall)')),
        DataColumn(label: Text('Decode tok/s (rt / wall)')),
        DataColumn(label: Text('Tokens in / out')),
        DataColumn(label: Text('Power / load')),
      ],
      rows: [
        for (final r in rows)
          if (r != null)
            DataRow(
              cells: [
                DataCell(Text(backendLabel(r.requestedBackend))),
                DataCell(
                  Text(
                    backendLabel(r.actualBackend),
                    style: r.actualBackend != null && r.actualBackend != r.requestedBackend
                        ? const TextStyle(color: Colors.orange, fontWeight: FontWeight.bold)
                        : null,
                  ),
                ),
                DataCell(Text(r.modelFile)),
                DataCell(Tooltip(message: r.error ?? r.reason ?? '', child: Text(r.status.name))),
                DataCell(Text(ms(r.loadTime))),
                DataCell(Text('${num1(r.metrics?.timeToFirstTokenMs)} / ${ms(r.wallTimeToFirstToken)}')),
                DataCell(Text('${num1(r.metrics?.tokensPerSecond)} / ${num1(r.wallDecodeTokensPerSecond)}')),
                DataCell(Text(r.metrics == null ? '—' : '${r.metrics!.inputTokens} / ${r.metrics!.outputTokens}')),
                DataCell(Text(formatPower(r.power, powerMode))),
              ],
            ),
      ],
    );
  }
}
