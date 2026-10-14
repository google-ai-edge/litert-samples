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

import '../../../config/live_camera_config.dart';

/// When the visual memory may take its next frame (pure; the caller passes
/// its monotonic clock in µs):
///
/// - [due]: at least [_minInterval], and at least [_spacing] × the last
///   frame embedding's time, after the previous attempt ended (the first
///   attempt is due at once);
/// - [forced]: [_maxInterval] since the last stored frame (or none stored
///   yet): store even without a scene change.
///
/// Whether the demo is idle, the frame black or the scene changed is the
/// caller's and the worker's to say; this only keeps the times.
final class MemorySampler {
  MemorySampler({
    this._minInterval = kMemoryMinInterval,
    this._spacing = kMemoryEmbedSpacing,
    this._maxInterval = kMemoryMaxInterval,
  });

  final Duration _minInterval;
  final int _spacing;
  final Duration _maxInterval;

  int? _lastAttemptEnd;
  int? _lastStored;
  Duration _lastEmbed = Duration.zero;

  /// The last frame embedding's time (kept across [reset]: it is the
  /// device's, not the session's).
  Duration get lastEmbed => _lastEmbed;

  /// The wait between attempts: the longer of the minimum interval and the
  /// spacing times the last frame embedding.
  Duration get gap => Duration(
    microseconds: math.max(
      _minInterval.inMicroseconds,
      _lastEmbed.inMicroseconds * _spacing,
    ),
  );

  /// An attempt may start at [now].
  bool due(int now) {
    final last = _lastAttemptEnd;
    return last == null || now - last >= gap.inMicroseconds;
  }

  /// A frame taken at [now] is stored even if the scene did not change.
  bool forced(int now) {
    final last = _lastStored;
    return last == null || now - last >= _maxInterval.inMicroseconds;
  }

  /// An attempt ended at [now] (stored, unchanged, dropped or failed);
  /// [embedTime] when a frame was embedded (or failed embedding).
  void attemptEnded(int now, {Duration? embedTime}) {
    _lastAttemptEnd = now;
    if (embedTime != null) _lastEmbed = embedTime;
  }

  /// A frame taken at [at] was stored.
  void stored(int at) => _lastStored = at;

  /// Forgets the attempt and store times (a cleared memory starts at once).
  void reset() {
    _lastAttemptEnd = null;
    _lastStored = null;
  }
}
