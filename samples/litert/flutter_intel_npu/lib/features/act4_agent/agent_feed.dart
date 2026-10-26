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
import 'workspace.dart';

/// The agent's side: per turn, what it thought, then the actions it ended the
/// turn with. The plugin hands a turn's calls over when the turn ends, so
/// they arrive together; they are let in one after another.
class AgentFeed extends StatefulWidget {
  const AgentFeed({super.key, required this.run, required this.now, required this.live, this.workspace});

  final AgentRun run;

  /// Names people and tells internal mail from outside mail.
  final Workspace? workspace;

  /// Seconds into the run.
  final double now;

  /// Still going: the last turn may be open.
  final bool live;

  @override
  State<AgentFeed> createState() => _AgentFeedState();
}

class _AgentFeedState extends State<AgentFeed> {
  final _scroll = ScrollController();

  /// The turn whose thought last filled the feed, and the height it had: its
  /// finished card starts at that height and shrinks.
  int? _filledTurn;
  double? _filledHeight;

  @override
  void didUpdateWidget(AgentFeed old) {
    super.didUpdateWidget(old);
    // Follow the run while it grows, unless the presenter scrolled up.
    if (_scroll.hasClients && _scroll.position.maxScrollExtent - _scroll.offset < 120) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (_scroll.hasClients) {
          _scroll.animateTo(
            _scroll.position.maxScrollExtent,
            duration: const Duration(milliseconds: 400),
            curve: Curves.easeOut,
          );
        }
      });
    }
  }

  @override
  void dispose() {
    _scroll.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final run = widget.run;
    final children = <Widget>[
      for (final turn in run.turns) ..._turn(turn),
      // Between a turn's end and the next turn's first chunk the model is
      // already at work: show it, not an empty feed.
      if (widget.live && run.runS == null && run.turns.isNotEmpty && run.turns.last.end != null)
        _LiveThought(
          key: ValueKey('live${run.turns.length + 1}'),
          turn: run.turns.length + 1,
          text: '',
          seconds: widget.now - run.turns.last.end!,
          writing: !run.think,
          step: FollowUpBoard.listOf(run).isEmpty ? 'READING THE MEETING' : 'CARRYING IT OUT',
        ),
      if (run.finalText.trim().isNotEmpty) _Report(text: run.finalText),
      if (run.ranOutWhileThinking)
        _Note(
          'Ran out of output tokens while thinking (${run.maxOutputTokens} per turn): no answer, no action.',
          error: true,
        ),
      if (run.stopped) const _Note('Stopped.'),
      if (run.error != null) _Note(run.error!.split('\n').first, error: true),
    ];
    // While the model thinks, its thought takes all the room the feed has
    // left. One scroll view either way, so the position survives the switch.
    final live = children.isNotEmpty && children.last is _LiveThought ? children.removeLast() as _LiveThought : null;
    return CustomScrollView(
      controller: _scroll,
      slivers: [
        SliverPadding(
          padding: EdgeInsets.fromLTRB(12, 4, 12, live == null ? 12 : 0),
          sliver: SliverList.list(children: children),
        ),
        if (live != null)
          SliverLayoutBuilder(
            builder: (context, c) {
              // The rest of the viewport, but never less than a readable card.
              final h = math.max(_minLive, c.viewportMainAxisExtent - c.precedingScrollExtent - 12);
              _filledTurn = live.turn;
              _filledHeight = h;
              return SliverPadding(
                padding: const EdgeInsets.fromLTRB(12, 0, 12, 12),
                sliver: SliverToBoxAdapter(
                  child: SizedBox(
                    height: h,
                    child: _LiveThought(
                      key: live.key,
                      text: live.text,
                      seconds: live.seconds,
                      writing: live.writing,
                      turn: live.turn,
                      step: live.step,
                      fill: true,
                    ),
                  ),
                ),
              );
            },
          ),
      ],
    );
  }

  static const _minLive = 180.0;

  /// Place of attempt [i] among the calls that arrived with it, for the
  /// stagger: a turn's calls land at one instant.
  int _order(int i) {
    final a = widget.run.attempts;
    var k = 0;
    for (var j = i - 1; j >= 0 && (a[i].t - a[j].t).abs() < 0.5; j--) {
      k++;
    }
    return k;
  }

  List<Widget> _turn(AgentTurn turn) {
    final run = widget.run;
    final open = widget.live && turn.end == null;
    final thought = turn.thoughtText;
    final text = turn.textText;
    final last = turn.n == run.turns.length;
    final calls = [
      for (final (i, c) in run.attempts.indexed)
        if (c.turn == turn.n) (i, c),
    ];
    return [
      if (open && calls.isEmpty && text.isEmpty)
        _LiveThought(
          key: ValueKey('live${turn.n}'),
          turn: turn.n,
          text: thought,
          seconds: widget.now - turn.start,
          // With thinking on, the first second before a thought chunk is thinking too.
          writing: thought.isEmpty && !run.think,
          step: FollowUpBoard.listOf(run).isEmpty ? 'READING THE MEETING' : 'CARRYING IT OUT',
        )
      else if (thought.isNotEmpty)
        _Thought(
          key: ValueKey('t${turn.n}'),
          text: thought,
          seconds: _thoughtEnd(turn, calls) - turn.start,
          shrinkFrom: turn.n == _filledTurn ? _filledHeight : null,
        ),
      // The last turn's text is the final report, shown below.
      if (text.trim().isNotEmpty && !(last && calls.isEmpty)) _Aside(text),
      for (final (i, c) in calls)
        ActionCard(key: ValueKey('a$i'), call: c, order: _order(i), workspace: widget.workspace),
    ];
  }

  double _thoughtEnd(AgentTurn turn, List<(int, AgentCall)> calls) {
    final ends = [
      if (calls.isNotEmpty) calls.map((e) => e.$2.t).reduce(math.min),
      if (turn.text.isNotEmpty) turn.text.first.$1,
      ?turn.end,
    ];
    return ends.isEmpty ? turn.start : ends.reduce(math.min);
  }
}

/// Thought text without its markdown, on one line.
String _plain(String s) => s.replaceAll(RegExp(r'[*#`_]+'), '').replaceAll(RegExp(r'\s+'), ' ').trim();

/// While the model thinks: a card whose border light circles, three dots
/// that breathe, the clock, and the tail of the thought drifting up.
class _LiveThought extends StatefulWidget {
  const _LiveThought({
    super.key,
    required this.text,
    required this.seconds,
    required this.writing,
    required this.step,
    required this.turn,
    this.fill = false,
  });
  final String text;
  final int turn;

  /// Takes the height it is given (the rest of the feed), at stage size.
  final bool fill;
  final double seconds;

  /// No thought at all: thinking is off and the model is writing its calls.
  final bool writing;

  /// Which step the model is on: reading the meeting (no list yet) or carrying it out.
  final String step;

  @override
  State<_LiveThought> createState() => _LiveThoughtState();
}

class _LiveThoughtState extends State<_LiveThought> with SingleTickerProviderStateMixin {
  late final _spin = AnimationController(vsync: this, duration: const Duration(milliseconds: 2400))..repeat();

  @override
  void dispose() {
    _spin.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final keep = widget.fill ? 2400 : 700;
    final tail = widget.text.length > keep ? widget.text.substring(widget.text.length - keep) : widget.text;
    final words = Text(
      widget.writing ? 'The calls come all at once when the turn ends.' : _plain(tail),
      style: TextStyle(
        color: const Color(0xFFAEB4F5),
        fontStyle: FontStyle.italic,
        height: 1.4,
        fontSize: widget.fill ? 16 : 13.5,
      ),
    );
    final color = widget.writing ? Stage.mint : Stage.think;
    return AnimatedBuilder(
      animation: _spin,
      builder: (context, child) => Container(
        margin: const EdgeInsets.only(top: 8, bottom: 8),
        padding: const EdgeInsets.all(1.5),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(12),
          gradient: SweepGradient(
            colors: [color.withValues(alpha: 0.1), color, color.withValues(alpha: 0.1), color.withValues(alpha: 0.1)],
            stops: const [0, 0.18, 0.4, 1],
            transform: GradientRotation(_spin.value * 2 * math.pi),
          ),
          boxShadow: [BoxShadow(color: color.withValues(alpha: 0.18), blurRadius: 18)],
        ),
        // The words and the mask repaint only when they change, not with the border.
        child: RepaintBoundary(child: child),
      ),
      child: Container(
        padding: const EdgeInsets.fromLTRB(14, 12, 14, 12),
        decoration: BoxDecoration(color: Stage.thinkDeep, borderRadius: BorderRadius.circular(10.5)),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              children: [
                _Dots(animation: _spin, color: color),
                const SizedBox(width: 10),
                Text(
                  '${widget.step} · ${widget.writing ? 'WRITING' : 'THINKING'}',
                  style: Stage.label.copyWith(color: color),
                ),
                const Spacer(),
                Text(
                  clock(widget.seconds),
                  style: TextStyle(color: color, fontWeight: FontWeight.w700, fontFeatures: Stage.tabular),
                ),
              ],
            ),
            const SizedBox(height: 8),
            _sized(
              ShaderMask(
                // The thought drifts up and fades out at the top.
                shaderCallback: (r) => const LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [Colors.transparent, Colors.white, Colors.white],
                  stops: [0, 0.45, 1],
                ).createShader(r),
                blendMode: BlendMode.dstIn,
                // The newest words sit at the bottom; older ones scroll off the top.
                child: ClipRect(
                  child: OverflowBox(
                    alignment: Alignment.bottomLeft,
                    minHeight: 0,
                    maxHeight: double.infinity,
                    child: words,
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

extension on _LiveThoughtState {
  /// A fixed strip in a list, or all the height left when filling.
  Widget _sized(Widget child) => widget.fill
      ? Expanded(
          child: ConstrainedBox(constraints: const BoxConstraints(minHeight: 120), child: child),
        )
      : SizedBox(height: 92, child: child);
}

/// Listens to [animation] itself: it sits in the static part of the live
/// card, which the card's own builder does not rebuild.
class _Dots extends AnimatedWidget {
  const _Dots({required Animation<double> animation, required this.color}) : super(listenable: animation);
  final Color color;

  Animation<double> get animation => listenable as Animation<double>;

  @override
  Widget build(BuildContext context) => Row(
    mainAxisSize: MainAxisSize.min,
    children: [
      for (var i = 0; i < 3; i++)
        Container(
          margin: const EdgeInsets.only(right: 4),
          width: 7,
          height: 7,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: color.withValues(
              alpha: 0.3 + 0.7 * (0.5 + 0.5 * math.sin((animation.value * 3 - i / 3) * 2 * math.pi)).clamp(0, 1),
            ),
          ),
        ),
    ],
  );
}

/// A finished thought: one quoted line, the whole of it on a tap.
class _Thought extends StatefulWidget {
  const _Thought({super.key, required this.text, required this.seconds, this.shrinkFrom});
  final String text;
  final double seconds;

  /// The height the live thought had: the card starts there and shrinks.
  final double? shrinkFrom;

  @override
  State<_Thought> createState() => _ThoughtState();
}

class _ThoughtState extends State<_Thought> {
  double? _height;

  @override
  void initState() {
    super.initState();
    _height = widget.shrinkFrom;
    if (_height != null) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (mounted) setState(() => _height = null);
      });
    }
  }

  @override
  Widget build(BuildContext context) => AnimatedSize(
    duration: const Duration(milliseconds: 700),
    curve: Curves.easeInOutCubic,
    alignment: Alignment.topCenter,
    child: _height == null ? _card(context) : SizedBox(height: _height, child: _card(context, fill: true)),
  );

  Widget _card(BuildContext context, {bool fill = false}) {
    final text = widget.text;
    final seconds = widget.seconds;
    final flat = _plain(text);
    final quote = flat.length > 220 ? '…${flat.substring(flat.length - 220)}' : flat;
    return Container(
      margin: const EdgeInsets.only(top: 8, bottom: 8),
      constraints: fill ? const BoxConstraints.expand() : null,
      alignment: fill ? Alignment.topCenter : null,
      decoration: BoxDecoration(
        color: Stage.thinkDeep,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: Stage.think.withValues(alpha: 0.35)),
      ),
      child: Theme(
        data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
        child: ExpansionTile(
          dense: true,
          tilePadding: const EdgeInsets.symmetric(horizontal: 14),
          childrenPadding: const EdgeInsets.fromLTRB(14, 0, 14, 12),
          iconColor: Stage.think,
          collapsedIconColor: Stage.think,
          title: Text('THOUGHT FOR ${clock(seconds).toUpperCase()}', style: Stage.label.copyWith(color: Stage.think)),
          subtitle: Padding(
            padding: const EdgeInsets.only(top: 4),
            child: Text(
              '“$quote”',
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(color: Color(0xFFAEB4F5), fontStyle: FontStyle.italic, fontSize: 12.5),
            ),
          ),
          children: [
            SizedBox(
              height: 220,
              child: SingleChildScrollView(
                child: SelectableText(
                  text,
                  style: const TextStyle(color: Color(0xFFAEB4F5), fontStyle: FontStyle.italic, fontSize: 12.5),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// One action, in words: what, for whom, by when. It slides in, after the
/// ones that arrived with it.
class ActionCard extends StatefulWidget {
  const ActionCard({super.key, required this.call, this.order = 0, this.workspace});
  final AgentCall call;

  /// Names people and tells internal mail from outside mail.
  final Workspace? workspace;

  /// Place among the calls that arrived at the same moment.
  final int order;

  @override
  State<ActionCard> createState() => _ActionCardState();
}

class _ActionCardState extends State<ActionCard> with SingleTickerProviderStateMixin {
  late final _in = AnimationController(vsync: this, duration: const Duration(milliseconds: 420));
  // Made once: a CurvedAnimation adds a listener to its parent until disposed.
  late final _curve = CurvedAnimation(parent: _in, curve: Curves.easeOutCubic);
  late final _slide = Tween(begin: const Offset(0.08, 0), end: Offset.zero).animate(_curve);

  @override
  void initState() {
    super.initState();
    Future<void>.delayed(Duration(milliseconds: 110 * widget.order), () {
      if (mounted) _in.forward();
    });
  }

  @override
  void dispose() {
    _curve.dispose();
    _in.dispose();
    super.dispose();
  }

  static const _look = {
    MeetingTask.createTask: (Icons.task_alt, Color(0xFF203B5C), Color(0xFF7DB7FF)),
    MeetingTask.sendEmail: (Icons.send_rounded, Color(0xFF1F4D3B), Stage.mint),
    MeetingTask.draftEmail: (Icons.drafts_outlined, Color(0xFF5C4A1F), Stage.amber),
    MeetingTask.scheduleMeeting: (Icons.event, Color(0xFF3E2D5C), Color(0xFFC6A2FF)),
    MeetingTask.updateDoc: (Icons.description_outlined, Color(0xFF1F4A4D), Color(0xFF6FE0E6)),
    MeetingTask.askClarification: (Icons.help_outline, Color(0xFF4D2B3A), Color(0xFFFF9CC4)),
    MeetingTask.listTool: (Icons.checklist_rounded, Color(0xFF2A2E5C), Stage.think),
  };

  @override
  Widget build(BuildContext context) {
    final call = widget.call;
    final a = call.args;
    final (icon, tile, ink) = _look[call.tool] ?? (Icons.build, Stage.track, Stage.dim);
    final w = widget.workspace;
    // Outside only if the workspace says so: without a mail domain nobody knows.
    final outside =
        call.tool == MeetingTask.sendEmail &&
        w?.internalDomain != null &&
        FollowUpBoard.recipients(a['to']).addresses.any((t) => !w!.isInternal(t));
    // A malformed list is the recorder's to refuse; here it must not break the feed.
    final items = a['items'] is List ? a['items'] as List : const [];
    final (title, detail, tag, tagColor) = switch (call.tool) {
      MeetingTask.createTask => (
        '${a['title'] ?? 'Task'}',
        'task for ${person(a['owner'], w)} · due ${day(a['due'])}',
        null,
        null,
      ),
      MeetingTask.sendEmail => (
        '${a['subject'] ?? 'Email'}',
        'email to ${people(a['to'], w)}',
        outside ? 'SENT OUTSIDE' : 'SENT',
        outside ? Stage.red : Stage.mint,
      ),
      MeetingTask.draftEmail => (
        '${a['subject'] ?? 'Draft'}',
        'draft to ${people(a['to'], w)} · waits for ${person(a['approver'], w)}',
        'FOR APPROVAL',
        Stage.amber,
      ),
      MeetingTask.scheduleMeeting => (
        '${a['title'] ?? 'Meeting'}',
        '${day(a['date'])}, ${a['start'] ?? ''}, ${a['duration_min'] ?? '?'} min · ${people(a['attendees'], w)}',
        null,
        null,
      ),
      MeetingTask.updateDoc => (
        '${a['doc'] ?? 'Document'}',
        '${a['content'] ?? ''}'.replaceAll('\n', ' '),
        'UPDATED',
        null,
      ),
      MeetingTask.askClarification => (
        '${a['question'] ?? 'Question'}',
        'asks ${person(a['to'], w)}',
        'QUESTION',
        null,
      ),
      MeetingTask.listTool => (
        'Found ${items.length} follow-ups in the meeting',
        [
          for (final it in items)
            if (it is Map) '${it['what'] ?? ''}',
        ].join(' · '),
        'THE LIST',
        Stage.think,
      ),
      _ => (call.tool, '$a', null, null),
    };
    return FadeTransition(
      opacity: _curve,
      child: SlideTransition(
        position: _slide,
        child: Opacity(
          opacity: call.executed ? 1 : 0.45,
          child: Container(
            margin: const EdgeInsets.symmetric(vertical: 3),
            padding: const EdgeInsets.fromLTRB(10, 9, 12, 9),
            decoration: Stage.panelBox(color: Stage.card),
            child: Row(
              children: [
                Container(
                  width: 32,
                  height: 32,
                  decoration: BoxDecoration(color: tile, borderRadius: BorderRadius.circular(8)),
                  child: Icon(icon, size: 18, color: ink),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        title,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 14),
                      ),
                      const SizedBox(height: 2),
                      Text(
                        // A refused call says why: that is what the model was told.
                        call.rejected ?? detail,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(color: Stage.dim, fontSize: 12.5),
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 8),
                if (FollowUpBoard.itemOf(call) case final n?) ...[
                  Text(
                    '#$n',
                    style: const TextStyle(color: Stage.dim, fontWeight: FontWeight.w700, fontSize: 12),
                  ),
                  const SizedBox(width: 8),
                ],
                if (call.rejected != null)
                  const StageTag('REFUSED', color: Stage.red, solid: false)
                else if (!call.executed)
                  const StageTag('OVER THE LIMIT', color: Stage.dim, solid: false)
                else if (tag != null)
                  StageTag(tag, color: tagColor ?? ink, solid: false),
                const SizedBox(width: 8),
                Text(
                  clock(call.t),
                  style: const TextStyle(color: Stage.dim, fontSize: 11, fontFeatures: Stage.tabular),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _Aside extends StatelessWidget {
  const _Aside(this.text);
  final String text;

  @override
  Widget build(BuildContext context) => Padding(
    padding: const EdgeInsets.symmetric(vertical: 6, horizontal: 4),
    child: Text(
      text.trim(),
      maxLines: 4,
      overflow: TextOverflow.ellipsis,
      style: const TextStyle(color: Stage.dim),
    ),
  );
}

/// The model's own report at the end: three lines, the rest on a tap.
class _Report extends StatelessWidget {
  const _Report({required this.text});
  final String text;

  @override
  Widget build(BuildContext context) {
    final plain = text.replaceAll(RegExp(r'[*#`]'), '').trim();
    return Container(
      margin: const EdgeInsets.only(top: 10),
      decoration: Stage.panelBox(color: Stage.panel).copyWith(border: Border.all(color: Stage.line)),
      child: Theme(
        data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
        child: ExpansionTile(
          dense: true,
          tilePadding: const EdgeInsets.symmetric(horizontal: 14),
          childrenPadding: const EdgeInsets.fromLTRB(14, 0, 14, 12),
          title: const Text("THE AGENT'S REPORT", style: Stage.label),
          subtitle: Text(plain, maxLines: 2, overflow: TextOverflow.ellipsis, style: const TextStyle(fontSize: 12.5)),
          children: [SelectableText(plain, style: const TextStyle(fontSize: 12.5, height: 1.4))],
        ),
      ),
    );
  }
}

class _Note extends StatelessWidget {
  const _Note(this.text, {this.error = false});
  final String text;
  final bool error;

  @override
  Widget build(BuildContext context) => Padding(
    padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 4),
    child: Text(
      text,
      style: TextStyle(color: error ? Stage.red : Stage.dim, fontStyle: error ? null : FontStyle.italic),
    ),
  );
}
