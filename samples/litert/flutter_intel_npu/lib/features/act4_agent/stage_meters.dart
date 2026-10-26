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

import 'agent_run.dart';
import 'follow_up_board.dart';
import 'meeting_task.dart';
import 'stage.dart';

/// The three stage meters: follow-ups done, rules kept, and the run's time
/// on this machine split into thinking and acting.
class StageMeters extends StatelessWidget {
  const StageMeters({super.key, required this.items, required this.run, required this.now, required this.live});

  final List<BoardItem> items;
  final AgentRun? run;
  final double now;
  final bool live;

  @override
  Widget build(BuildContext context) {
    final tasks = items.where((c) => !c.rule).toList();
    final rules = items.where((c) => c.rule).toList();
    final anyCall = (run?.attempts ?? const []).any((c) => c.tool != MeetingTask.listTool);
    final finished = run != null && !live && (run!.runS != null || run!.error != null);
    return SizedBox(
      height: 112,
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Expanded(
            flex: 4,
            child: _ScoreMeter(
              title: 'FOLLOW-UPS DONE',
              marks: [for (final c in tasks) c.passed ? _Mark.pass : (finished ? _Mark.fail : _Mark.open)],
              caption: tasks.isEmpty ? 'the model lists them first' : 'of the ${tasks.length} it found in the meeting',
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            flex: 3,
            child: _ScoreMeter(
              title: 'RULES KEPT',
              // The rules come with the list, not before it.
              marks: tasks.isEmpty && !finished
                  ? const []
                  : [for (final c in rules) !anyCall ? _Mark.open : (c.passed ? _Mark.pass : _Mark.fail)],
              caption: anyCall
                  ? 'checked against the workspace'
                  : (tasks.isEmpty ? 'they come with the list' : 'tested from the first action'),
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            flex: 5,
            child: _TimeMeter(run: run, now: now, live: live),
          ),
        ],
      ),
    );
  }
}

enum _Mark { pass, fail, open }

class _ScoreMeter extends StatelessWidget {
  const _ScoreMeter({required this.title, required this.marks, required this.caption});
  final String title;
  final List<_Mark> marks;
  final String caption;

  @override
  Widget build(BuildContext context) {
    final n = marks.where((m) => m == _Mark.pass).length;
    final total = marks.isEmpty ? '–' : '${marks.length}';
    return Container(
      padding: const EdgeInsets.fromLTRB(14, 12, 14, 12),
      decoration: Stage.panelBox(color: Stage.card),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Expanded(child: Text(title, style: Stage.label)),
              TweenAnimationBuilder<double>(
                tween: Tween(end: n.toDouble()),
                duration: const Duration(milliseconds: 900),
                curve: Curves.easeOutCubic,
                builder: (context, v, _) => Text.rich(
                  TextSpan(
                    children: [
                      TextSpan(text: '${v.round()}'),
                      TextSpan(
                        text: '/$total',
                        style: const TextStyle(fontSize: 17, color: Color(0xFF7FD9B5)),
                      ),
                    ],
                  ),
                  style: const TextStyle(
                    fontSize: 34,
                    height: 1,
                    fontWeight: FontWeight.w700,
                    color: Stage.mint,
                    fontFeatures: Stage.tabular,
                  ),
                ),
              ),
            ],
          ),
          const Spacer(),
          Row(
            children: [
              if (marks.isEmpty)
                Expanded(
                  child: Container(
                    height: 9,
                    decoration: BoxDecoration(color: Stage.track, borderRadius: BorderRadius.circular(2)),
                  ),
                ),
              for (final (i, m) in marks.indexed) ...[
                if (i > 0) const SizedBox(width: 3),
                Expanded(
                  child: AnimatedContainer(
                    // Segments fill one after another when a batch of calls lands.
                    duration: Duration(milliseconds: 350 + 60 * i),
                    curve: Curves.easeOut,
                    height: 9,
                    decoration: BoxDecoration(
                      color: switch (m) {
                        _Mark.pass => Stage.mint,
                        _Mark.fail => Stage.red,
                        _Mark.open => Stage.track,
                      },
                      borderRadius: BorderRadius.circular(2),
                      boxShadow: m == _Mark.pass
                          ? [BoxShadow(color: Stage.mint.withValues(alpha: 0.35), blurRadius: 6)]
                          : null,
                    ),
                  ),
                ),
              ],
            ],
          ),
          const SizedBox(height: 8),
          Text(
            caption,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: const TextStyle(color: Stage.dim, fontSize: 12),
          ),
        ],
      ),
    );
  }
}

/// Time on this machine: a bar that grows while the run goes, striped and
/// moving while the model thinks, with a glowing tick for every action.
class _TimeMeter extends StatefulWidget {
  const _TimeMeter({required this.run, required this.now, required this.live});
  final AgentRun? run;
  final double now;
  final bool live;

  @override
  State<_TimeMeter> createState() => _TimeMeterState();
}

class _TimeMeterState extends State<_TimeMeter> with SingleTickerProviderStateMixin {
  late final _flow = AnimationController(vsync: this, duration: const Duration(milliseconds: 900));

  @override
  void initState() {
    super.initState();
    _sync();
  }

  @override
  void didUpdateWidget(_TimeMeter old) {
    super.didUpdateWidget(old);
    _sync();
  }

  /// The stripes move only while a thought is open: no 60 fps repaint for a
  /// bar that is not changing.
  void _sync() {
    final run = widget.run;
    final spans = run == null || !widget.live ? const <(double, double)>[] : _thoughtSpans(run, widget.now);
    final thinking = spans.isNotEmpty && spans.last.$2 >= widget.now - 0.5;
    if (thinking && !_flow.isAnimating) {
      _flow.repeat();
    } else if (!thinking && _flow.isAnimating) {
      _flow.stop();
    }
  }

  @override
  void dispose() {
    _flow.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final run = widget.run;
    final elapsed = run == null ? 0.0 : (widget.live ? widget.now : (run.runS ?? 0));
    final thinking = run == null ? 0.0 : _thinkingSeconds(run, elapsed);
    final actions = run?.attempts.length ?? 0;
    final caption = run == null
        ? 'press Run'
        : [
            if (run.firstCallS != null) 'first action at ${clock(run.firstCallS!)}',
            if (run.firstCallS == null && widget.live) 'no action yet',
          ].join(' · ');
    return Container(
      padding: const EdgeInsets.fromLTRB(14, 12, 14, 12),
      decoration: Stage.panelBox(color: Stage.card),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              const Expanded(child: Text('ON THIS LAPTOP, NO CLOUD', style: Stage.label)),
              Text(
                clock(elapsed),
                style: const TextStyle(
                  fontSize: 34,
                  height: 1,
                  fontWeight: FontWeight.w700,
                  color: Stage.mint,
                  fontFeatures: Stage.tabular,
                ),
              ),
            ],
          ),
          const Spacer(),
          SizedBox(
            height: 14,
            child: AnimatedBuilder(
              animation: _flow,
              builder: (context, _) => CustomPaint(
                painter: _TimelinePainter(run: run, elapsed: elapsed, live: widget.live, phase: _flow.value),
              ),
            ),
          ),
          const SizedBox(height: 6),
          Row(
            children: [
              _Key(color: Stage.think, text: 'thinking ${clock(thinking)}'),
              const SizedBox(width: 12),
              _Key(color: Stage.mint, text: '$actions action${actions == 1 ? '' : 's'}'),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  caption,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  textAlign: TextAlign.end,
                  style: const TextStyle(color: Stage.dim, fontSize: 12),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  static double _thinkingSeconds(AgentRun run, double elapsed) {
    var s = 0.0;
    for (final (a, b) in _thoughtSpans(run, elapsed)) {
      s += b - a;
    }
    return s;
  }
}

/// Each turn thinks from its start until its first visible output: a call,
/// its text, or its end.
List<(double, double)> _thoughtSpans(AgentRun run, double elapsed) {
  final spans = <(double, double)>[];
  for (final turn in run.turns) {
    if (turn.thought.isEmpty) continue;
    final calls = [
      for (final c in run.attempts)
        if (c.turn == turn.n) c.t,
    ];
    final ends = [
      if (calls.isNotEmpty) calls.reduce(math.min),
      if (turn.text.isNotEmpty) turn.text.first.$1,
      ?turn.end,
      elapsed,
    ];
    final end = ends.reduce(math.min);
    if (end > turn.start) spans.add((turn.start, end));
  }
  return spans;
}

class _TimelinePainter extends CustomPainter {
  _TimelinePainter({required this.run, required this.elapsed, required this.live, required this.phase});
  final AgentRun? run;
  final double elapsed;
  final bool live;
  final double phase;

  @override
  void paint(Canvas canvas, Size size) {
    final r = Radius.circular(size.height / 2);
    final track = RRect.fromRectAndRadius(Offset(0, 3) & Size(size.width, size.height - 6), r);
    canvas.drawRRect(track, Paint()..color = Stage.track);
    final run = this.run;
    if (run == null || elapsed <= 0) return;
    // A live run's bar keeps room ahead of it; a finished one fills the track.
    final span = live ? math.max(elapsed * 1.15, 30.0) : math.max(elapsed, 1.0);
    double x(double t) => (t / span).clamp(0.0, 1.0) * size.width;

    canvas.save();
    canvas.clipRRect(track);
    canvas.drawRect(
      Rect.fromLTRB(0, 3, x(elapsed), size.height - 3),
      Paint()..color = Stage.mint.withValues(alpha: 0.18),
    );
    for (final (a, b) in _thoughtSpans(run, elapsed)) {
      final rect = Rect.fromLTRB(x(a), 3, x(b), size.height - 3);
      canvas.drawRect(rect, Paint()..color = Stage.think);
      // Moving stripes while the model thinks.
      final stripe = Paint()..color = const Color(0xFF5A66E0);
      const w = 12.0;
      final shift = live ? phase * w * 2 : 0.0;
      for (var s = rect.left - w * 2 + shift; s < rect.right; s += w * 2) {
        final p = Path()
          ..moveTo(s, rect.bottom)
          ..lineTo(s + w, rect.top)
          ..lineTo(s + w * 2, rect.top)
          ..lineTo(s + w, rect.bottom)
          ..close();
        canvas.save();
        canvas.clipRect(rect);
        canvas.drawPath(p, stripe);
        canvas.restore();
      }
    }
    canvas.restore();

    final tick = Paint()..color = Stage.mint;
    final glow = Paint()
      ..color = Stage.mint.withValues(alpha: 0.6)
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 4);
    for (final c in run.attempts) {
      if (c.t > elapsed) continue;
      final cx = x(c.t);
      final rect = Rect.fromCenter(center: Offset(cx, size.height / 2), width: 3, height: size.height);
      canvas.drawRRect(RRect.fromRectAndRadius(rect.inflate(1.5), const Radius.circular(2)), glow);
      canvas.drawRRect(RRect.fromRectAndRadius(rect, const Radius.circular(2)), tick);
    }
  }

  @override
  bool shouldRepaint(_TimelinePainter old) =>
      old.phase != phase || old.elapsed != elapsed || old.run != run || old.live != live;
}

class _Key extends StatelessWidget {
  const _Key({required this.color, required this.text});
  final Color color;
  final String text;

  @override
  Widget build(BuildContext context) => Row(
    mainAxisSize: MainAxisSize.min,
    children: [
      Container(
        width: 8,
        height: 8,
        decoration: BoxDecoration(color: color, borderRadius: BorderRadius.circular(2)),
      ),
      const SizedBox(width: 5),
      Text(
        text,
        style: const TextStyle(color: Stage.dim, fontSize: 12, fontFeatures: Stage.tabular),
      ),
    ],
  );
}
