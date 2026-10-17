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

/// One sample of the two things the act is about, taken ten times a second.
class StageSample {
  const StageSample({required this.frameMs, required this.tps, required this.backend});
  final double frameMs;
  final double tps;

  /// `off`, `CPU`, `GPU`, `NPU` — what the narrator was running on.
  final String backend;
}

/// Rolling history behind the stage meters, plus the per-backend best
/// throughput each backend has shown in this session. The "best" is the
/// honest denominator for "how much of its own speed is the model still
/// getting": it is that same engine on that same machine with the scene
/// leaving it alone.
class StageSeries extends ChangeNotifier {
  StageSeries({this.capacity = 200});

  final int capacity;
  final List<StageSample> samples = [];
  final Map<String, double> bestTps = {};

  void add(StageSample s) {
    samples.add(s);
    if (samples.length > capacity) samples.removeAt(0);
    // A baseline measured on purpose stays: the best reading under load is
    // only the fallback denominator for a backend that was never measured.
    if (s.tps > 0 && s.backend != 'off' && !measured.contains(s.backend)) {
      final best = bestTps[s.backend] ?? 0;
      if (s.tps > best) bestTps[s.backend] = s.tps;
    }
    notifyListeners();
  }

  /// A baseline measured on purpose: that backend, that machine, with the
  /// scene at its lightest. Without it the meter compares the model against
  /// itself under load and always reads 100 %.
  final Set<String> measured = {};

  void setBaseline(String backend, double tps) {
    if (tps <= 0) return;
    bestTps[backend] = tps;
    measured.add(backend);
    notifyListeners();
  }

  bool isMeasured(String backend) => measured.contains(backend);

  void clearBaselines() {
    bestTps.clear();
    measured.clear();
    notifyListeners();
  }

  double? baselineFor(String backend) => bestTps[backend];
}

/// The two costs, side by side and readable from the back of the room: how
/// much of the frame budget the scene is using, and how much of its own solo
/// speed the model is still getting. Whichever one the platform makes the
/// model pay, the audience sees a bar drop at the moment the backend changes.
class StageMeters extends StatelessWidget {
  const StageMeters({
    super.key,
    required this.series,
    required this.targetFps,
    required this.sustainedFps,
    required this.frameMs,
    required this.tps,
    required this.backend,
    required this.calibrating,
    required this.calibrationStatus,
  });

  final StageSeries series;
  final int targetFps;
  final double sustainedFps;
  final double frameMs;
  final double tps;
  final String backend;
  final bool calibrating;
  final String calibrationStatus;

  @override
  Widget build(BuildContext context) {
    final budgetMs = 1000 / targetFps;
    final frameUse = budgetMs <= 0 ? 0.0 : frameMs / budgetMs;
    final baseline = series.baselineFor(backend);
    final modelShare = (baseline == null || baseline <= 0) ? null : tps / baseline;

    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Expanded(
          child: _Meter(
            title: 'FRAME BUDGET',
            value: frameUse,
            headline: '${(frameUse * 100).round()}%',
            detail: calibrating
                ? calibrationStatus
                : '${sustainedFps.toStringAsFixed(0)} fps · ${frameMs.toStringAsFixed(1)} of ${budgetMs.toStringAsFixed(1)} ms',
            good: frameUse < 0.85,
            warn: frameUse < 1.0,
            series: series,
            pick: (s) => budgetMs <= 0 ? 0 : s.frameMs / budgetMs,
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: _Meter(
            title: 'MODEL SPEED',
            value: modelShare ?? 0,
            headline: modelShare == null ? '—' : '${(modelShare * 100).round()}%',
            detail: backend == 'off'
                ? 'narrator off'
                : baseline == null
                    ? '$backend · warming up'
                    : '$backend · ${tps.toStringAsFixed(0)} of ${baseline.toStringAsFixed(0)} tok/s'
                        '${series.isMeasured(backend) ? ' solo' : ' (best seen)'}',
            good: (modelShare ?? 1) > 0.85,
            warn: (modelShare ?? 1) > 0.6,
            invert: true,
            series: series,
            pick: (s) {
              final b = series.baselineFor(s.backend);
              if (b == null || b <= 0) return 0;
              return s.tps / b;
            },
          ),
        ),
      ],
    );
  }
}

class _Meter extends StatelessWidget {
  const _Meter({
    required this.title,
    required this.value,
    required this.headline,
    required this.detail,
    required this.good,
    required this.warn,
    required this.series,
    required this.pick,
    this.invert = false,
  });

  final String title;
  final double value;
  final String headline;
  final String detail;
  final bool good;
  final bool warn;
  final StageSeries series;
  final double Function(StageSample) pick;

  /// For the model meter, where full is good and empty is bad.
  final bool invert;

  @override
  Widget build(BuildContext context) {
    final color = good
        ? Colors.greenAccent
        : warn
            ? Colors.amberAccent
            : Colors.redAccent;
    return Container(
      padding: const EdgeInsets.fromLTRB(14, 10, 14, 10),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.62),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.baseline,
            textBaseline: TextBaseline.alphabetic,
            children: [
              Text(title,
                  style: const TextStyle(
                      color: Colors.white70, fontSize: 13, letterSpacing: 2, fontWeight: FontWeight.w600)),
              const Spacer(),
              Text(headline,
                  style: TextStyle(color: color, fontSize: 34, fontWeight: FontWeight.bold, height: 1)),
            ],
          ),
          const SizedBox(height: 6),
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: value.clamp(0.0, 1.0),
              minHeight: 12,
              backgroundColor: Colors.white12,
              valueColor: AlwaysStoppedAnimation(color),
            ),
          ),
          const SizedBox(height: 6),
          SizedBox(
            height: 46,
            child: CustomPaint(
              painter: _SeriesPainter(series: series, pick: pick, color: color, invert: invert),
              size: Size.infinite,
            ),
          ),
          Text(detail, style: const TextStyle(color: Colors.white70, fontSize: 13)),
        ],
      ),
    );
  }
}

/// A 20-second strip of the meter's own value, with a line at 100 % and a
/// tick wherever the narrator's backend changed — so the step lines up with
/// the moment the presenter pressed the key.
class _SeriesPainter extends CustomPainter {
  _SeriesPainter({required this.series, required this.pick, required this.color, required this.invert})
      : super(repaint: series);

  final StageSeries series;
  final double Function(StageSample) pick;
  final Color color;
  final bool invert;

  @override
  void paint(Canvas canvas, Size size) {
    final samples = series.samples;
    final grid = Paint()
      ..color = Colors.white24
      ..strokeWidth = 1;
    // 100 % line: the frame budget, or the model's own solo speed.
    final y100 = size.height * (invert ? 0.12 : 0.5);
    canvas.drawLine(Offset(0, y100), Offset(size.width, y100), grid);
    if (samples.length < 2) return;

    final dx = size.width / (series.capacity - 1);
    // The frame meter shows 0..2 × budget, the model meter 0..1.2 × solo.
    final span = invert ? 1.2 : 2.0;
    double yFor(double v) => size.height - (v / span * size.height).clamp(0.0, size.height);

    final path = Path();
    for (var i = 0; i < samples.length; i++) {
      final x = i * dx;
      final y = yFor(pick(samples[i]));
      if (i == 0) {
        path.moveTo(x, y);
      } else {
        path.lineTo(x, y);
      }
    }
    canvas.drawPath(
      path,
      Paint()
        ..color = color
        ..strokeWidth = 2
        ..style = PaintingStyle.stroke,
    );

    final markPaint = Paint()
      ..color = Colors.white70
      ..strokeWidth = 1.5;
    for (var i = 1; i < samples.length; i++) {
      if (samples[i].backend == samples[i - 1].backend) continue;
      final x = i * dx;
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), markPaint);
      final tp = TextPainter(
        text: TextSpan(
          text: samples[i].backend,
          style: const TextStyle(color: Colors.white, fontSize: 11, fontWeight: FontWeight.w600),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      tp.paint(canvas, Offset(math.min(x + 3, size.width - tp.width), 2));
    }
  }

  @override
  bool shouldRepaint(covariant _SeriesPainter old) => true;
}

/// The commentary, large. This is the third way the contention shows: the
/// text is streamed at the model's real token rate, so when the scene takes
/// the GPU away the words visibly crawl — no HUD reading required.
/// The commentary as subtitles: one line, large, readable from the back of
/// the room. It streams in place and stays up, the same size, until the
/// next line starts and cross-fades over it. (A history of smaller lines
/// above it shrank each line the moment it finished, while the room was
/// still reading it.)
class StageCommentary extends StatelessWidget {
  const StageCommentary({
    super.key,
    required this.text,
    required this.turn,
    required this.tps,
    required this.backend,
  });

  final String text;

  /// Which turn [text] belongs to: a new turn cross-fades, streamed chunks
  /// of the same turn update in place.
  final int turn;
  final double? tps;
  final String backend;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 14),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.66),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(children: [
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
              decoration: BoxDecoration(
                color: Colors.white12,
                borderRadius: BorderRadius.circular(4),
              ),
              child: Text(backend.toUpperCase(),
                  style: const TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.bold, letterSpacing: 1)),
            ),
            const SizedBox(width: 8),
            Text(tps == null ? '' : '${tps!.toStringAsFixed(0)} tok/s',
                style: const TextStyle(color: Colors.white70, fontSize: 13)),
          ]),
          const SizedBox(height: 6),
          AnimatedSwitcher(
            duration: const Duration(milliseconds: 350),
            layoutBuilder: (current, previous) => Stack(
              alignment: Alignment.topLeft,
              children: [...previous, ?current],
            ),
            child: Text(
              text.isEmpty ? '…' : text,
              key: ValueKey(turn),
              maxLines: 3,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(color: Colors.white, fontSize: 22, fontWeight: FontWeight.w600, height: 1.3),
            ),
          ),
        ],
      ),
    );
  }
}
