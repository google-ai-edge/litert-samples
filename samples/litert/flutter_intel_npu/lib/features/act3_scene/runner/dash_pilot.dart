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

import '../../../core/world/chunk_spec.dart';

/// What the pilot wants Dash to do this frame.
class PilotIntent {
  const PilotIntent({required this.lateral, required this.jump});

  /// Desired offset from the trail centre, metres to the right.
  final double lateral;

  /// Ask for a jump this frame (edge-triggered by the game).
  final bool jump;
}

/// Drives Dash when the presenter is not. It is deliberately imperfect:
/// [missRate] decides how often it fumbles an obstacle, because a clean run
/// is a screensaver and a stumble is the moment the narrator has something to
/// say.
class DashPilot {
  DashPilot({int seed = 7, this.lookahead = 15.0, this.missRate = 0.08}) : _rng = math.Random(seed);

  final math.Random _rng;

  /// How far down the trail the pilot reacts to obstacles, metres.
  final double lookahead;

  /// Probability that an obstacle is handled badly on purpose.
  double missRate;

  /// True while the presenter is driving.
  bool manual = false;
  double _manualIdle = 0;

  /// Set by the game so a decision is made once per obstacle, not per frame.
  int? _decidedFor;
  double _decidedLateral = 0;
  bool _decidedJump = false;
  bool _decidedFumble = false;

  /// The obstacle the pilot is currently committed to, for the HUD.
  int? get committedTo => _decidedFor;
  bool get fumbling => _decidedFumble;

  /// The presenter pressed something; hold manual control until they stop.
  void takeOver({double holdSeconds = 5}) {
    manual = true;
    _manualIdle = holdSeconds;
  }

  /// Returns null while the presenter is in control.
  PilotIntent? update({
    required double dt,
    required double z,
    required double lateral,
    required double halfWidth,
    required double speed,
    required bool airborne,
    required Obstacle? next,
  }) {
    if (manual) {
      _manualIdle -= dt;
      if (_manualIdle > 0) return null;
      manual = false;
    }

    if (next == null) {
      // Nothing ahead: drift back to the middle of the trail, where the
      // camera frames the landscape best.
      _decidedFor = null;
      return PilotIntent(lateral: lateral * 0.9, jump: false);
    }

    if (_decidedFor != next.id) {
      _decidedFor = next.id;
      _decidedFumble = _rng.nextDouble() < missRate;
      // Logs and streams are jumps by design; boulders are dodges. Only fall
      // back to the geometry when a boulder somehow blocks the whole trail.
      final mustJump = next.kind == ObstacleKind.log || next.kind == ObstacleKind.stream;
      final free = mustJump ? const <double>[] : _freeOffsets(next, halfWidth);
      if (free.isEmpty) {
        _decidedJump = true;
        // Jump down the middle: the take-off and landing both stay on the
        // path, which is what makes the jump readable.
        _decidedLateral = mustJump ? next.lateral : lateral;
      } else {
        _decidedJump = false;
        // Pick the side that costs the least sideways running.
        free.sort((a, b) => (a - lateral).abs().compareTo((b - lateral).abs()));
        _decidedLateral = free.first;
      }
      if (_decidedFumble) {
        // A fumble is a late, half-hearted attempt: aim at the obstacle
        // instead of past it, or skip the jump entirely.
        _decidedJump = false;
        _decidedLateral = next.lateral;
      }
    }

    final distance = next.z - z;
    // Take off a fixed *time* before the obstacle, not a fixed distance: at
    // 15 m/s a 3 m lead is half the lead it is at 8 m/s.
    final lead = (speed * 0.38).clamp(2.6, 5.5);
    final jumpNow = _decidedJump && !airborne && distance < lead && distance > 0;
    return PilotIntent(lateral: _decidedLateral, jump: jumpNow);
  }

  /// Offsets inside the corridor where this obstacle can be passed on foot.
  List<double> _freeOffsets(Obstacle o, double halfWidth) {
    final candidates = <double>[
      for (var x = -halfWidth + 0.6; x <= halfWidth - 0.6; x += 0.4) x,
    ];
    return candidates.where(o.clearsAt).toList();
  }
}
