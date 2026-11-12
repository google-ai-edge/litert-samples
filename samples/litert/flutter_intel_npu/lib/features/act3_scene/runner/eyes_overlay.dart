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

import 'runner_game.dart';
import 'vision_pilot.dart';

/// What the vision pilot looks at and what it made of it, drawn over the
/// scene (above the part it reads, so it never sees this).
class EyesOverlay extends StatelessWidget {
  const EyesOverlay({super.key, required this.eyes, required this.game});

  final VisionPilot eyes;
  final RunnerGame game;

  static const _words = {
    'trail': 'path',
    'grass': 'grass',
    'rock': 'rock',
    'jump': 'log',
  };

  @override
  Widget build(BuildContext context) {
    return IgnorePointer(
      child: ListenableBuilder(
        listenable: eyes,
        builder: (context, _) => LayoutBuilder(
          builder: (context, box) {
            final seen = eyes.seen;
            Widget part(String id, String? label, Color color, {bool loud = false}) {
              final (l, t, r, b) = VisionPilot.parts[id]!;
              return Positioned(
                left: l * box.maxWidth,
                top: t * box.maxHeight,
                width: (r - l) * box.maxWidth,
                height: (b - t) * box.maxHeight,
                child: DecoratedBox(
                  decoration: BoxDecoration(border: Border.all(color: color, width: loud ? 3 : 1.5)),
                  child: Align(
                    alignment: Alignment.topLeft,
                    child: label == null
                        ? const SizedBox.shrink()
                        : Container(
                            color: color,
                            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                            child: Text(label, style: const TextStyle(color: Colors.white, fontSize: 14, fontWeight: FontWeight.w600)),
                          ),
                  ),
                ),
              );
            }

            Color tone(String? s) => switch (s) {
                  'rock' => Colors.redAccent,
                  'jump' => Colors.orange,
                  'trail' => Colors.lightGreen,
                  _ => Colors.white70,
                };
            final c = seen['C'];
            final n = seen['N'];
            final s = seen['S'];
            return Stack(
              children: [
                Positioned(
                  left: 0,
                  right: 0,
                  top: 212,
                  child: Center(
                    child: Text('stumbles ${game.stumbles}',
                        style: const TextStyle(color: Colors.white, fontSize: 24, fontWeight: FontWeight.w700, shadows: [Shadow(blurRadius: 6)])),
                  ),
                ),
                for (final id in ['L', 'R'])
                  if (seen[id] != null) part(id, _words[seen[id]] ?? seen[id], tone(seen[id])),
                part('C', _words[c] ?? c, tone(c), loud: c == 'rock' || c == 'jump'),
                part('N', n == 'jump' ? 'JUMP' : null, n == 'jump' ? Colors.orange : Colors.white38, loud: n == 'jump'),
                if (s == 'left' || s == 'right')
                  Positioned(
                    left: 0,
                    right: 0,
                    top: VisionPilot.parts['S']!.$2 * box.maxHeight - 34,
                    child: Center(
                      child: Text(s == 'left' ? '◀ steer left' : 'steer right ▶',
                          style: const TextStyle(color: Colors.white, fontSize: 22, fontWeight: FontWeight.w700, shadows: [Shadow(blurRadius: 6)])),
                    ),
                  ),
              ],
            );
          },
        ),
      ),
    );
  }
}
