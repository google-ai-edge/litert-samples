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
import 'package:flutter/material.dart';

import '../../domain/models/voice.dart';

/// What both demos show while the capture starts ([TurnPhase.openingMic]):
/// on the phase line and as the mic button's label.
const kOpeningMicLabel = 'Opening the mic…';

/// Push-to-talk button: press opens the mic, release sends. While the
/// capture starts it shows a spinner instead of the level ring (nothing is
/// recorded yet); while listening the ring shows the input level and
/// repaints from [level] without rebuilding.
class MicButton extends StatelessWidget {
  const MicButton({
    super.key,
    required this.level,
    required this.phase,
    required this.enabled,
    required this.onDown,
    required this.onUp,
  });

  /// 0–1, updated per mic chunk.
  final ValueListenable<double> level;

  /// The turn's phase: [TurnPhase.openingMic] and [TurnPhase.listening]
  /// have their own looks; every other phase shows the button ready.
  final TurnPhase phase;
  final bool enabled;
  final VoidCallback onDown;
  final VoidCallback onUp;

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    final listening = phase == TurnPhase.listening;
    final opening = phase == TurnPhase.openingMic;
    final (background, foreground) = switch ((enabled, phase)) {
      (false, _) => (
        colors.surfaceContainerHighest,
        colors.onSurface.withValues(alpha: 0.38),
      ),
      (true, TurnPhase.openingMic) => (
        colors.errorContainer,
        colors.onErrorContainer,
      ),
      (true, TurnPhase.listening) => (colors.error, colors.onError),
      (true, _) => (colors.primary, colors.onPrimary),
    };
    return Tooltip(
      message: 'Hold to talk',
      child: Semantics(
        button: true,
        enabled: enabled,
        label: switch (phase) {
          TurnPhase.openingMic => kOpeningMicLabel,
          TurnPhase.listening => 'Release to send',
          _ => 'Hold to talk',
        },
        child: Listener(
          onPointerDown: enabled ? (_) => onDown() : null,
          onPointerUp: enabled ? (_) => onUp() : null,
          onPointerCancel: enabled ? (_) => onUp() : null,
          child: SizedBox.square(
            dimension: 56,
            child: Stack(
              fit: StackFit.expand,
              children: [
                if (opening && enabled)
                  CircularProgressIndicator(
                    strokeWidth: 2,
                    color: colors.error,
                  ),
                CustomPaint(
                  painter: LevelRingPainter(
                    level: level,
                    active: listening,
                    color: colors.error,
                  ),
                  child: Center(
                    child: DecoratedBox(
                      decoration: BoxDecoration(
                        color: background,
                        shape: BoxShape.circle,
                      ),
                      child: SizedBox.square(
                        dimension: 44,
                        child: Icon(
                          listening ? Icons.mic : Icons.mic_none,
                          color: foreground,
                        ),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

/// A ring whose thickness and opacity follow the mic level. Repaints when
/// [level] changes (high rate, no layout); other inputs come from rebuilds.
class LevelRingPainter extends CustomPainter {
  LevelRingPainter({
    required this.level,
    required this.active,
    required this.color,
  }) : super(repaint: level);

  final ValueListenable<double> level;
  final bool active;
  final Color color;

  @override
  void paint(Canvas canvas, Size size) {
    if (!active) return;
    final value = level.value.clamp(0.0, 1.0);
    final radius = math.min(size.width, size.height) / 2;
    final width = 2 + 4 * value;
    canvas.drawCircle(
      size.center(Offset.zero),
      radius - width / 2,
      Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = width
        ..color = color.withValues(alpha: 0.25 + 0.75 * value),
    );
  }

  @override
  bool shouldRepaint(LevelRingPainter old) =>
      old.level != level || old.active != active || old.color != color;
}
