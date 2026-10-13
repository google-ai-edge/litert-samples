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

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import '../../../domain/models/diagnostics_snapshot.dart';
import 'overlay_keys.dart';
import 'overlay_layout.dart';
import 'overlay_panel.dart';

/// Puts a toggleable diagnostics panel over [child] (the whole navigator, so
/// it is visible on every screen). Visible by default in debug builds. Screens
/// put a [DebugOverlayToggle] in their app bar; it finds this host through the
/// widget tree.
class DebugOverlayHost extends StatefulWidget {
  const DebugOverlayHost({
    super.key,
    required this.snapshot,
    required this.child,
  });

  final ValueListenable<DiagnosticsSnapshot> snapshot;
  final Widget child;

  /// The host's visibility switch, or null outside a host (e.g. in widget
  /// tests that pump a screen alone).
  static ValueNotifier<bool>? visibilityOf(BuildContext context) => context
      .dependOnInheritedWidgetOfExactType<_DebugOverlayScope>()
      ?.notifier;

  @override
  State<DebugOverlayHost> createState() => _DebugOverlayHostState();
}

class _DebugOverlayHostState extends State<DebugOverlayHost> {
  final ValueNotifier<bool> _visible = ValueNotifier(kDebugMode);

  /// The user's expand/collapse choice, kept here so it survives hiding
  /// the panel; null follows the window (collapsed when compact).
  final ValueNotifier<bool?> _expandedChoice = ValueNotifier(null);

  @override
  void dispose() {
    _visible.dispose();
    _expandedChoice.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return _DebugOverlayScope(
      notifier: _visible,
      child: Stack(
        children: [
          widget.child,
          ValueListenableBuilder<bool>(
            valueListenable: _visible,
            builder: (context, visible, _) => visible
                ? Positioned.fill(
                    child: _DebugOverlayFrame(
                      snapshot: widget.snapshot,
                      expandedChoice: _expandedChoice,
                    ),
                  )
                : const SizedBox.shrink(),
          ),
        ],
      ),
    );
  }
}

/// Lays the panel out under the app bar on the right, clear of the leading
/// slot (back button), inside the safe area (status bar, notch, landscape
/// insets) and above the keyboard, bounded so the demos' controls at the
/// bottom stay free ([DebugOverlayLayout]). Covers the screen but hit-tests
/// only the panel's handle and scroll strip: everywhere else the screen gets
/// its pointers as if the panel were not there.
/// Above the Navigator there is no Material, so the panel brings its own
/// (text style) without painting anything.
class _DebugOverlayFrame extends StatelessWidget {
  const _DebugOverlayFrame({
    required this.snapshot,
    required this.expandedChoice,
  });

  final ValueListenable<DiagnosticsSnapshot> snapshot;
  final ValueNotifier<bool?> expandedChoice;

  @override
  Widget build(BuildContext context) {
    final window = MediaQuery.sizeOf(context);
    return Padding(
      // The keyboard: the composer moves up above it.
      padding: EdgeInsets.only(bottom: MediaQuery.viewInsetsOf(context).bottom),
      child: SafeArea(
        child: LayoutBuilder(
          builder: (context, constraints) {
            final layout = DebugOverlayLayout(
              window: window,
              safe: constraints.biggest,
            );
            // A window too short or narrow for a line and the handle (a
            // short macOS window, a split screen) gets no panel.
            if (!layout.showsPanel) return const SizedBox.shrink();
            return Align(
              alignment: Alignment.topRight,
              child: Padding(
                padding: DebugOverlayLayout.margin,
                child: ConstrainedBox(
                  constraints: layout.constraints,
                  child: Material(
                    type: MaterialType.transparency,
                    child: DebugOverlay(
                      snapshot: snapshot,
                      compact: layout.compact,
                      expandedChoice: expandedChoice,
                    ),
                  ),
                ),
              ),
            );
          },
        ),
      ),
    );
  }
}

class _DebugOverlayScope extends InheritedNotifier<ValueNotifier<bool>> {
  const _DebugOverlayScope({required super.notifier, required super.child});
}

/// App-bar action that shows or hides the [DebugOverlayHost] panel. Renders
/// nothing outside a host.
class DebugOverlayToggle extends StatelessWidget {
  const DebugOverlayToggle({super.key});

  @override
  Widget build(BuildContext context) {
    final visible = DebugOverlayHost.visibilityOf(context);
    if (visible == null) return const SizedBox.shrink();
    return IconButton(
      key: DebugOverlayKeys.toggle,
      tooltip: visible.value ? 'Hide diagnostics' : 'Show diagnostics',
      onPressed: () => visible.value = !visible.value,
      icon: Icon(visible.value ? Icons.bug_report : Icons.bug_report_outlined),
    );
  }
}
