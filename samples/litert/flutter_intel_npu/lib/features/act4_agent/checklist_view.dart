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

import 'package:flutter/material.dart';

import 'follow_up_board.dart';
import 'stage.dart';

/// The checklist: the follow-ups the model listed, then the rules. Until the
/// list arrives there is nothing to tick; until the first action the rules
/// are not tested; while the run goes an unmet follow-up is open, not failed.
class ChecklistView extends StatelessWidget {
  const ChecklistView({
    super.key,
    required this.items,
    required this.anyAction,
    required this.finished,
    required this.live,
  });

  final List<BoardItem> items;
  final bool anyAction;
  final bool finished;
  final bool live;

  @override
  Widget build(BuildContext context) {
    final follow = items.where((c) => !c.rule).toList();
    final rules = items.where((c) => c.rule).toList();
    final done = follow.where((c) => c.passed).length;
    // Nothing to check before the model has read the meeting: no list, and
    // rules shown then would look like a checklist known in advance.
    if (follow.isEmpty && !finished) return _Waiting(live: live);
    return ListView(
      padding: const EdgeInsets.fromLTRB(14, 12, 14, 12),
      children: [
        Row(
          children: [
            const Expanded(child: Text('FOLLOW-UPS THE MODEL FOUND', style: Stage.label)),
            if (follow.isNotEmpty) Text('$done / ${follow.length}', style: Stage.label.copyWith(color: Stage.text)),
          ],
        ),
        const SizedBox(height: 6),
        if (follow.isEmpty)
          const Padding(
            padding: EdgeInsets.symmetric(vertical: 18, horizontal: 6),
            child: Text(
              'The model listed no follow-ups.',
              style: TextStyle(color: Stage.dim, fontStyle: FontStyle.italic, height: 1.4),
            ),
          ),
        for (final (i, c) in follow.indexed)
          CheckRow(
            key: ValueKey(c.id),
            order: i,
            item: c,
            state: c.passed ? CheckState.pass : (finished ? CheckState.fail : CheckState.open),
          ),
        const SizedBox(height: 12),
        const Text('RULES', style: Stage.label),
        const SizedBox(height: 6),
        for (final (i, c) in rules.indexed)
          CheckRow(
            key: ValueKey(c.id),
            order: follow.length + i,
            item: c,
            state: !anyAction ? CheckState.open : (c.passed ? CheckState.pass : CheckState.fail),
          ),
      ],
    );
  }
}

enum CheckState { pass, fail, open }

/// One item. When it turns green it pops and its row flashes; a list that
/// arrives slides its rows in one after another.
class CheckRow extends StatefulWidget {
  const CheckRow({super.key, required this.item, required this.state, this.order = 0});
  final BoardItem item;
  final CheckState state;

  /// Place in the list: a list that arrives at once comes in row by row.
  final int order;

  @override
  State<CheckRow> createState() => _CheckRowState();
}

class _CheckRowState extends State<CheckRow> with TickerProviderStateMixin {
  late final _pop = AnimationController(vsync: this, duration: const Duration(milliseconds: 1100), value: 1);
  late final _in = AnimationController(vsync: this, duration: const Duration(milliseconds: 380));
  // Made once: a CurvedAnimation adds a listener to its parent until disposed.
  late final _curve = CurvedAnimation(parent: _in, curve: Curves.easeOutCubic);
  late final _slide = Tween(begin: const Offset(0.06, 0), end: Offset.zero).animate(_curve);

  @override
  void initState() {
    super.initState();
    Future<void>.delayed(Duration(milliseconds: 90 * widget.order), () {
      if (mounted) _in.forward();
    });
  }

  @override
  void didUpdateWidget(CheckRow old) {
    super.didUpdateWidget(old);
    if (old.state != widget.state && widget.state != CheckState.open) _pop.forward(from: 0);
  }

  @override
  void dispose() {
    _pop.dispose();
    _curve.dispose();
    _in.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return FadeTransition(
      opacity: _curve,
      child: SlideTransition(position: _slide, child: _row()),
    );
  }

  Widget _row() {
    final state = widget.state;
    final item = widget.item;
    final (bg, fg, glyph) = switch (state) {
      CheckState.pass => (Stage.mintDeep, Stage.mint, Icons.check_rounded),
      CheckState.fail => (Stage.redDeep, Stage.red, Icons.close_rounded),
      CheckState.open => (Stage.track, Stage.dim, null),
    };
    final flashColor = state == CheckState.fail ? Stage.red : Stage.mint;
    return AnimatedBuilder(
      animation: _pop,
      builder: (context, _) {
        final t = _pop.value;
        // Overshoot then settle: 0.4 → 1.35 → 1.
        final scale = t < 0.3 ? 0.4 + t / 0.3 * 0.95 : 1.35 - 0.35 * Curves.easeOut.transform((t - 0.3) / 0.7);
        final flash = (1 - t).clamp(0.0, 1.0);
        return Container(
          margin: const EdgeInsets.symmetric(vertical: 1),
          padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 4),
          decoration: BoxDecoration(
            color: flashColor.withValues(alpha: 0.16 * flash),
            borderRadius: BorderRadius.circular(6),
          ),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Transform.scale(
                scale: scale,
                child: Container(
                  width: 18,
                  height: 18,
                  margin: const EdgeInsets.only(top: 1),
                  decoration: BoxDecoration(color: bg, shape: BoxShape.circle),
                  child: glyph == null ? null : Icon(glyph, size: 13, color: fg),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      item.label,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(fontSize: 13, color: state == CheckState.open ? Stage.dim : Stage.text),
                    ),
                    if (state == CheckState.fail && item.why.isNotEmpty)
                      Text(
                        item.why,
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(fontSize: 11.5, color: Stage.red),
                      )
                    else if (item.detail.isNotEmpty)
                      Text(
                        _detail(item.detail),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(fontSize: 11.5, color: Stage.dim),
                      ),
                  ],
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  /// "jane.doe@example.com · YYYY-MM-DD" → "Jane · Fri 13 Nov": names only here.
  static String _detail(String d) => d
      .split(' · ')
      .map(
        (p) => RegExp(r'^\d{4}-\d\d-\d\d$').hasMatch(p) ? day(p) : (p.contains('@') ? person(p).split(' (').first : p),
      )
      .join(' · ');
}

/// Before the list: the model is reading. Placeholder lines shimmer where
/// the checklist will be.
class _Waiting extends StatefulWidget {
  const _Waiting({required this.live});
  final bool live;

  @override
  State<_Waiting> createState() => _WaitingState();
}

class _WaitingState extends State<_Waiting> with SingleTickerProviderStateMixin {
  late final _shine = AnimationController(vsync: this, duration: const Duration(milliseconds: 1600));

  @override
  void initState() {
    super.initState();
    if (widget.live) _shine.repeat();
  }

  @override
  void didUpdateWidget(_Waiting old) {
    super.didUpdateWidget(old);
    if (widget.live && !_shine.isAnimating) _shine.repeat();
    if (!widget.live && _shine.isAnimating) _shine.stop();
  }

  @override
  void dispose() {
    _shine.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    const widths = [0.82, 0.64, 0.74, 0.55, 0.7, 0.6];
    return Padding(
      padding: const EdgeInsets.fromLTRB(14, 12, 14, 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('FOLLOW-UPS THE MODEL FOUND', style: Stage.label),
          const SizedBox(height: 14),
          AnimatedBuilder(
            animation: _shine,
            builder: (context, _) => ShaderMask(
              blendMode: BlendMode.srcATop,
              shaderCallback: (r) => LinearGradient(
                colors: const [Stage.track, Color(0xFF4A4D58), Stage.track],
                stops: const [0.35, 0.5, 0.65],
                begin: Alignment(-2.2 + 4.4 * _shine.value, 0),
                end: Alignment(-1.2 + 4.4 * _shine.value, 0),
              ).createShader(r),
              child: Column(
                children: [
                  for (final w in widths)
                    Padding(
                      padding: const EdgeInsets.symmetric(vertical: 7),
                      child: Row(
                        children: [
                          Container(
                            width: 18,
                            height: 18,
                            decoration: const BoxDecoration(color: Stage.track, shape: BoxShape.circle),
                          ),
                          const SizedBox(width: 10),
                          Expanded(
                            child: FractionallySizedBox(
                              alignment: Alignment.centerLeft,
                              widthFactor: w,
                              child: Container(
                                height: 12,
                                decoration: BoxDecoration(color: Stage.track, borderRadius: BorderRadius.circular(4)),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 14),
          Text(
            widget.live
                ? 'Reading the meeting. The checklist is what the model finds in it.'
                : 'The model reads the meeting first and lists what has to happen: that list is the checklist.',
            style: const TextStyle(color: Stage.dim, fontStyle: FontStyle.italic, height: 1.4),
          ),
        ],
      ),
    );
  }
}
