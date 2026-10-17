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

import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter_gemma/flutter_gemma.dart';

import '../../core/engines/backend_catalog.dart';
import '../../core/engines/benchmark_run.dart';
import '../../core/power/power_source.dart';

/// CPU, GPU and NPU load in percent (0..100), whichever the platform reports.
/// The GPU is whichever of its engines is busier: the scene draws on 3D, a
/// model runs on compute.
class Loads {
  const Loads({this.cpu, this.gpu, this.npu, this.cpuW, this.gpuW});

  factory Loads.ofSummary(PowerSummary p) => Loads(
    cpu: p.cpuLoad,
    gpu: _max(p.gpu3dLoad, p.gpuComputeLoad),
    npu: p.npuLoad,
    cpuW: p.cpuPackageW,
    gpuW: p.gpuW,
  );

  factory Loads.ofSample(PowerSample s) => Loads(
    cpu: s.cpuLoad,
    gpu: _max(s.gpu3dLoad, s.gpuComputeLoad),
    npu: s.npuLoad,
    cpuW: s.cpuPackageW,
    gpuW: s.gpuW,
  );

  final double? cpu;
  final double? gpu;
  final double? npu;
  final double? cpuW;
  final double? gpuW;

  bool get isEmpty => cpu == null && gpu == null && npu == null;

  static double? _max(double? a, double? b) => a == null ? b : (b == null ? a : math.max(a, b));
}

/// One backend's lane in Act 1: the speed in numbers the back of the room can
/// read, what the chips were doing meanwhile, and the answer as it streamed.
///
/// The runs are measured one after another (the iGPU and the NPU share memory
/// bandwidth, so running them together would spoil both numbers); [replayAt]
/// replays every lane from its recorded timeline at once, as a race.
class RaceColumn extends StatelessWidget {
  const RaceColumn({
    super.key,
    required this.backend,
    required this.run,
    required this.liveText,
    required this.liveLoads,
    required this.replayAt,
    required this.place,
    required this.powerMode,
  });

  final PreferredBackend backend;
  final BenchmarkRun? run;

  /// Text streaming in right now (this lane is the one generating).
  final String? liveText;

  /// The sidecar's latest reading while this lane generates.
  final Loads? liveLoads;

  /// Position of the race replay; null when not replaying.
  final Duration? replayAt;

  /// 1, 2, 3 once this lane has finished in the race; null before.
  final int? place;

  final PowerMode powerMode;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final r = run;
    final done = r?.status == BenchmarkRunStatus.ok;
    final replaying = replayAt != null;
    final finished = r?.finishedAt;
    final crossed = done && finished != null && (!replaying || replayAt! >= finished);

    final tps = r?.metrics?.tokensPerSecond ?? r?.wallDecodeTokensPerSecond;
    final ttft = r?.metrics?.timeToFirstTokenMs ?? r?.wallTimeToFirstToken?.inMilliseconds.toDouble();

    final text = replaying ? (r == null ? '' : textAt(r.timeline, replayAt!)) : (liveText ?? r?.responsePreview ?? '');
    final loads = liveLoads ?? (r?.power != null && r!.power!.samples > 0 ? Loads.ofSummary(r.power!) : null);

    return Card(
      clipBehavior: Clip.antiAlias,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              children: [
                Text(
                  backendLabel(backend),
                  style: theme.textTheme.headlineMedium?.copyWith(fontWeight: FontWeight.w800),
                ),
                const SizedBox(width: 12),
                if (place != null && crossed) _PlaceBadge(place: place!),
                const Spacer(),
                _StatusChip(run: r, racing: replaying && done && !crossed),
              ],
            ),
            Text(
              [
                if (r != null) r.modelFile,
                if (r?.actualBackend != null && r!.actualBackend != backend) 'ran on ${backendLabel(r.actualBackend)}',
              ].join(' · '),
              style: theme.textTheme.bodySmall?.copyWith(
                color: r?.actualBackend != null && r!.actualBackend != backend ? Colors.orange : null,
              ),
              overflow: TextOverflow.ellipsis,
            ),
            const SizedBox(height: 12),
            Row(
              crossAxisAlignment: CrossAxisAlignment.baseline,
              textBaseline: TextBaseline.alphabetic,
              children: [
                Text(
                  tps == null ? '—' : tps.toStringAsFixed(tps >= 10 ? 0 : 1),
                  style: theme.textTheme.displayMedium?.copyWith(fontWeight: FontWeight.w800),
                ),
                const SizedBox(width: 6),
                Text('tok/s', style: theme.textTheme.titleMedium),
              ],
            ),
            Text(
              [
                'first token ${ttft == null ? '—' : '${ttft.round()} ms'}',
                // In the replay the finishing time appears when the lane finishes.
                if (finished != null && crossed) 'answer ${_seconds(finished)}',
                if (r?.loadTime != null) 'load ${_seconds(r!.loadTime!)}',
              ].join(' · '),
              style: theme.textTheme.bodyMedium,
            ),
            const SizedBox(height: 12),
            if (loads != null && !loads.isEmpty) _LoadBars(loads: loads, live: liveLoads != null, powerMode: powerMode),
            const SizedBox(height: 8),
            Expanded(
              child: DecoratedBox(
                decoration: BoxDecoration(
                  border: Border.all(color: theme.dividerColor),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: SingleChildScrollView(
                  reverse: true, // keep the newest words in view while streaming
                  padding: const EdgeInsets.all(10),
                  child: Text(
                    text.isEmpty ? (r?.error?.split('\n').first ?? r?.reason ?? '') : text,
                    style: theme.textTheme.bodyMedium,
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  static String _seconds(Duration d) => '${(d.inMilliseconds / 1000).toStringAsFixed(1)} s';
}

class _StatusChip extends StatelessWidget {
  const _StatusChip({required this.run, this.racing = false});
  final BenchmarkRun? run;

  /// Still running in the race replay, though measured and done.
  final bool racing;

  @override
  Widget build(BuildContext context) {
    final r = run;
    final (label, color) = racing
        ? ('racing', Colors.blue)
        : switch (r?.status) {
            null || BenchmarkRunStatus.pending => ('waiting', Colors.grey),
            BenchmarkRunStatus.loading => ('loading', Colors.blueGrey),
            BenchmarkRunStatus.warming => ('warming up', Colors.blueGrey),
            BenchmarkRunStatus.generating => ('generating', Colors.blue),
            BenchmarkRunStatus.ok => ('done', Colors.green),
            BenchmarkRunStatus.failed => ('failed', Colors.red),
            BenchmarkRunStatus.unavailable => ('not on this machine', Colors.grey),
          };
    return Tooltip(
      message: r?.error?.split('\n').first ?? r?.reason ?? '',
      child: Chip(
        label: Text(label),
        side: BorderSide(color: color),
        labelStyle: TextStyle(color: color, fontWeight: FontWeight.w600),
        visualDensity: VisualDensity.compact,
      ),
    );
  }
}

class _PlaceBadge extends StatelessWidget {
  const _PlaceBadge({required this.place});
  final int place;

  @override
  Widget build(BuildContext context) {
    final label = switch (place) {
      1 => '1st',
      2 => '2nd',
      3 => '3rd',
      _ => '${place}th',
    };
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: place == 1 ? Colors.amber : Theme.of(context).colorScheme.secondaryContainer,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Text(
        label,
        style: const TextStyle(fontWeight: FontWeight.w800, color: Colors.black87),
      ),
    );
  }
}

/// What the three chips were doing while this lane generated. The NPU lane's
/// argument is the GPU bar sitting near zero.
class _LoadBars extends StatelessWidget {
  const _LoadBars({required this.loads, required this.live, required this.powerMode});
  final Loads loads;
  final bool live;
  final PowerMode powerMode;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final showWatts = switch (powerMode) {
      PowerMode.watts => true,
      PowerMode.utilization => false,
      PowerMode.auto => loads.cpuW != null || loads.gpuW != null,
    };
    Widget bar(String name, double? pct) => Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        children: [
          SizedBox(width: 40, child: Text(name, style: theme.textTheme.labelMedium)),
          Expanded(
            child: ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: LinearProgressIndicator(
                value: pct == null ? 0 : (pct / 100).clamp(0, 1),
                minHeight: 10,
                backgroundColor: theme.colorScheme.surfaceContainerHighest,
              ),
            ),
          ),
          SizedBox(
            width: 48,
            child: Text(
              pct == null ? '—' : '${pct.round()} %',
              textAlign: TextAlign.right,
              style: theme.textTheme.labelMedium,
            ),
          ),
        ],
      ),
    );
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Text(live ? 'chip load, now' : 'chip load during this answer', style: theme.textTheme.labelSmall),
        bar('CPU', loads.cpu),
        bar('GPU', loads.gpu),
        bar('NPU', loads.npu),
        if (showWatts)
          Text(
            'CPU ${loads.cpuW == null ? '—' : '${loads.cpuW!.toStringAsFixed(1)} W'} · '
            'GPU ${loads.gpuW == null ? '—' : '${loads.gpuW!.toStringAsFixed(1)} W'}',
            style: theme.textTheme.labelMedium,
          ),
      ],
    );
  }
}

/// Finishing order of the runs that completed: backend → 1, 2, 3.
Map<PreferredBackend, int> placesOf(Iterable<BenchmarkRun> runs) {
  final done = [
    for (final r in runs)
      if (r.status == BenchmarkRunStatus.ok && r.finishedAt != null) r,
  ]..sort((a, b) => a.finishedAt!.compareTo(b.finishedAt!));
  return {for (var i = 0; i < done.length; i++) done[i].requestedBackend: i + 1};
}
