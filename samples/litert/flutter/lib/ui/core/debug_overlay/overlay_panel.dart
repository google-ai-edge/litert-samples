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

import 'package:flutter/foundation.dart';
import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';

import '../../../domain/models/diagnostics_snapshot.dart';
import 'overlay_keys.dart';
import 'overlay_lines.dart';

/// The scroll strip on the panel's right edge while the list overflows.
const double _kStripWidth = 24;

/// The diagnostics panel: the full list ([debugOverlayLines]) or the
/// summary ([debugOverlaySummaryLines]), switched by the handle at its
/// bottom right ([DebugOverlayKeys.expand]). The parent bounds its height;
/// the text scrolls inside it.
///
/// The panel lets every pointer through to the screen under it, except on
/// the handle and, while the text overflows, on the scroll strip along its
/// right edge ([DebugOverlayKeys.scrollbar]). Rebuilds only when the
/// coalesced (≤4 Hz) snapshot changes, on the handle, and when the text
/// starts or stops overflowing.
class DebugOverlay extends StatefulWidget {
  const DebugOverlay({
    super.key,
    required this.snapshot,
    required this.expandedChoice,
    this.compact = false,
  });

  final ValueListenable<DiagnosticsSnapshot> snapshot;

  /// The user's expand/collapse choice, owned by the caller so it
  /// outlives the panel; null follows the window (collapsed when
  /// [compact]), so a window resized across the breakpoint follows too.
  final ValueNotifier<bool?> expandedChoice;

  /// A phone-sized window: starts collapsed to the summary, with 10 px text
  /// instead of 11 px.
  final bool compact;

  @override
  State<DebugOverlay> createState() => _DebugOverlayState();
}

class _DebugOverlayState extends State<DebugOverlay> {
  final ScrollController _scroll = ScrollController();

  bool get _expanded => widget.expandedChoice.value ?? !widget.compact;

  /// The text is taller than the panel (from the scroll view's metrics).
  bool _overflows = false;

  @override
  void initState() {
    super.initState();
    widget.expandedChoice.addListener(_onChoice);
  }

  @override
  void didUpdateWidget(DebugOverlay oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.expandedChoice != widget.expandedChoice) {
      oldWidget.expandedChoice.removeListener(_onChoice);
      widget.expandedChoice.addListener(_onChoice);
    }
  }

  @override
  void dispose() {
    widget.expandedChoice.removeListener(_onChoice);
    _scroll.dispose();
    super.dispose();
  }

  void _onChoice() => setState(() {});

  void _toggle() {
    // Either view starts at its top.
    if (_scroll.hasClients) _scroll.jumpTo(0);
    widget.expandedChoice.value = !_expanded;
  }

  bool _onMetrics(ScrollMetricsNotification notification) {
    final overflows = notification.metrics.maxScrollExtent > 0;
    if (notification.depth == 0 && overflows != _overflows) {
      setState(() => _overflows = overflows);
    }
    return false;
  }

  @override
  Widget build(BuildContext context) {
    final expanded = _expanded;
    final style = TextStyle(
      color: Colors.white,
      fontFamily: 'Menlo',
      fontFamilyFallback: const ['monospace', 'Courier'],
      fontSize: widget.compact ? 10 : 11,
      height: 1.3,
    );
    // The full list or the summary, and the handle's label: the summary's
    // last line (`+23 more`) or `less`.
    final text = ValueListenableBuilder<DiagnosticsSnapshot>(
      valueListenable: widget.snapshot,
      builder: (context, value, _) {
        final List<String> facts;
        final String label;
        if (expanded) {
          facts = debugOverlayLines(value);
          label = _kLessLabel;
        } else {
          final summary = debugOverlaySummaryLines(value);
          facts = summary.sublist(0, summary.length - 1);
          label = summary.last;
        }
        return DefaultTextStyle(
          style: style,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Flexible(
                child: Stack(
                  fit: StackFit.passthrough,
                  children: [
                    // The text never takes pointers: the screen under the
                    // panel keeps them in every state.
                    IgnorePointer(
                      child: NotificationListener<ScrollMetricsNotification>(
                        onNotification: _onMetrics,
                        child: RawScrollbar(
                          controller: _scroll,
                          interactive: false,
                          thumbVisibility: true,
                          trackVisibility: true,
                          // Thumb and track fill the scroll strip.
                          thickness: 4,
                          crossAxisMargin: (_kStripWidth - 4) / 2,
                          radius: const Radius.circular(2),
                          thumbColor: Colors.white54,
                          trackColor: const Color(0x14FFFFFF),
                          trackBorderColor: const Color(0x00000000),
                          trackRadius: const Radius.circular(4),
                          child: ScrollConfiguration(
                            // The thumb above is the only scrollbar
                            // (desktop adds one).
                            behavior: ScrollConfiguration.of(context)
                                .copyWith(scrollbars: false),
                            child: SingleChildScrollView(
                              key: DebugOverlayKeys.scroll,
                              controller: _scroll,
                              // The strip's width is kept free of text.
                              padding: EdgeInsets.fromLTRB(
                                8,
                                8,
                                _overflows ? _kStripWidth : 8,
                                0,
                              ),
                              // One paragraph per fact, a gap between them:
                              // a fact that wraps reads as one.
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                spacing: 3,
                                children: [
                                  for (final fact in facts)
                                    Text(_keepUnits(fact)),
                                ],
                              ),
                            ),
                          ),
                        ),
                      ),
                    ),
                    if (_overflows)
                      Positioned(
                        top: 0,
                        right: 0,
                        bottom: 0,
                        width: _kStripWidth,
                        child: _ScrollStrip(controller: _scroll),
                      ),
                  ],
                ),
              ),
              Align(
                alignment: Alignment.centerRight,
                child: _ExpandHandle(
                  expanded: expanded,
                  label: label,
                  onTap: _toggle,
                ),
              ),
            ],
          ),
        );
      },
    );
    return Stack(
      key: DebugOverlayKeys.panel,
      fit: StackFit.passthrough,
      children: [
        // The background takes no pointers (a decoration would).
        const Positioned.fill(
          child: IgnorePointer(
            child: DecoratedBox(
              decoration: BoxDecoration(
                // Dark enough that the screen under it does not show
                // through the text (it covers up to 45% of a phone when
                // expanded).
                color: Color(0xE6000000),
                borderRadius: BorderRadius.all(Radius.circular(8)),
              ),
            ),
          ),
        ),
        text,
      ],
    );
  }
}

/// The expanded panel's handle label.
const _kLessLabel = 'less';

/// The panel's only switch: `+23 more ⇕` collapsed, `less ⇕` expanded, at
/// least 32 dp high.
class _ExpandHandle extends StatelessWidget {
  const _ExpandHandle({
    required this.expanded,
    required this.label,
    required this.onTap,
  });

  final bool expanded;
  final String label;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    const dim = Colors.white70;
    // Its own node with the tap action: without `container` and `onTap`
    // the label would merge into the screen's root node, actionless.
    return Semantics(
      container: true,
      button: true,
      label: expanded ? 'Show fewer diagnostics' : 'Show all diagnostics',
      onTap: onTap,
      excludeSemantics: true,
      child: MouseRegion(
        cursor: SystemMouseCursors.click,
        child: GestureDetector(
          key: DebugOverlayKeys.expand,
          behavior: HitTestBehavior.opaque,
          onTap: onTap,
          child: ConstrainedBox(
            constraints: const BoxConstraints(minHeight: 32, minWidth: 48),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 8),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                spacing: 4,
                children: [
                  Text(label, style: const TextStyle(color: dim)),
                  Icon(
                    expanded ? Icons.unfold_less : Icons.unfold_more,
                    size: 14,
                    color: dim,
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

/// The panel's scroll bar while its text overflows: a strip along the right
/// edge (the thumb's track) that takes vertical drags (moving the thumb:
/// the track stands for the whole list), the mouse wheel and screen
/// readers' scroll actions. The text beside it lets pointers through.
class _ScrollStrip extends StatelessWidget {
  const _ScrollStrip({required this.controller});

  final ScrollController controller;

  void _scrollBy(double delta) {
    if (!controller.hasClients) return;
    final p = controller.position;
    controller.jumpTo(
      math.min(
        p.maxScrollExtent,
        math.max(p.minScrollExtent, p.pixels + delta),
      ),
    );
  }

  void _drag(DragUpdateDetails details) {
    if (!controller.hasClients) return;
    final p = controller.position;
    final listPerTrack =
        (p.maxScrollExtent + p.viewportDimension) / p.viewportDimension;
    _scrollBy(details.delta.dy * listPerTrack);
  }

  void _wheel(PointerSignalEvent event) {
    if (event is! PointerScrollEvent) return;
    GestureBinding.instance.pointerSignalResolver.register(
      event,
      (PointerSignalEvent resolved) =>
          _scrollBy((resolved as PointerScrollEvent).scrollDelta.dy),
    );
  }

  /// Most of a viewport, for a screen reader's scroll.
  double get _page =>
      controller.hasClients ? controller.position.viewportDimension * 0.8 : 0;

  @override
  Widget build(BuildContext context) {
    return Semantics(
      container: true,
      label: 'Diagnostics scroll bar',
      // On a list that grows downwards, scrolling "up" reveals what is
      // below (ScrollPosition's semantic actions).
      onScrollUp: () => _scrollBy(_page),
      onScrollDown: () => _scrollBy(-_page),
      excludeSemantics: true,
      child: Listener(
        onPointerSignal: _wheel,
        child: GestureDetector(
          key: DebugOverlayKeys.scrollbar,
          behavior: HitTestBehavior.opaque,
          onVerticalDragUpdate: _drag,
          child: const SizedBox.expand(),
        ),
      ),
    );
  }
}

/// A number and its unit (`148 ms`, `9.5 fps`, `8192 tok`).
final _numberUnit = RegExp(r'(\d) (ms|s|fps|tok|KB|MB|GB|dBFS)\b');

/// [line] for display: a number keeps its unit when the line wraps (a
/// no-break space between them).
String _keepUnits(String line) =>
    line.replaceAllMapped(_numberUnit, (m) => '${m[1]}\u00a0${m[2]}');
