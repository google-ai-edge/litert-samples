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

import 'dart:collection';
import 'dart:ui' show FramePhase, FrameTiming;

import 'package:flutter/foundation.dart';
import 'package:flutter/scheduler.dart' hide FrameTiming;

/// Frame timing from Flutter's own scheduler: per-frame UI (build) and
/// raster durations, the same numbers DevTools shows. One time source for
/// the whole demo; the LLM tokens/sec is measured against the same clock.
class FrameTimeProbe extends ChangeNotifier {
  FrameTimeProbe({this.window = 120, this.targetFps = 60});

  final int window;
  final int targetFps;

  final Queue<FrameTiming> _recent = Queue();
  final Queue<int> _intervalsUs = Queue();
  int _frames = 0;
  int _janky = 0;
  int _missed = 0;
  int _longestGapUs = 0;
  int? _lastVsyncUs;
  final Stopwatch _wall = Stopwatch();
  int _lastFrameWallUs = 0;
  Duration _worstTotal = Duration.zero;
  bool _listening = false;

  void start() {
    if (_listening) return;
    _listening = true;
    _wall.start();
    SchedulerBinding.instance.addTimingsCallback(_onTimings);
  }

  void stop() {
    if (!_listening) return;
    _listening = false;
    _wall.stop();
    SchedulerBinding.instance.removeTimingsCallback(_onTimings);
  }

  void reset() {
    _recent.clear();
    _intervalsUs.clear();
    _frames = 0;
    _janky = 0;
    _missed = 0;
    _longestGapUs = 0;
    _lastVsyncUs = null;
    _wall
      ..reset()
      ..start();
    _worstTotal = Duration.zero;
    notifyListeners();
  }

  void _onTimings(List<FrameTiming> timings) {
    final budgetUs = 1000000 ~/ targetFps;
    final budget = Duration(microseconds: budgetUs);
    for (final t in timings) {
      _recent.addLast(t);
      if (_recent.length > window) _recent.removeFirst();
      _frames++;
      if (t.totalSpan > budget) _janky++;
      if (t.totalSpan > _worstTotal) _worstTotal = t.totalSpan;

      // The gap BETWEEN frames, which per-frame durations cannot show. A run
      // that produced 380 frames in 30 s once reported "120 fps, 0 janky",
      // because every frame it did produce was fast — it simply produced
      // almost none.
      final vsync = t.timestampInMicroseconds(FramePhase.vsyncStart);
      final last = _lastVsyncUs;
      if (last != null && vsync > last) {
        final gap = vsync - last;
        _intervalsUs.addLast(gap);
        if (_intervalsUs.length > window) _intervalsUs.removeFirst();
        // Every budget slot the gap spans beyond the first is a frame that
        // never came: a one-second stall at 60 fps is ~59 missed, not 1.
        if (gap > budgetUs * 1.5) _missed += (gap / budgetUs).round() - 1;
        if (gap > _longestGapUs) _longestGapUs = gap;
      }
      _lastVsyncUs = vsync;
      _lastFrameWallUs = _wall.elapsedMicroseconds;
    }
    notifyListeners();
  }

  @visibleForTesting
  void debugAddTimings(List<FrameTiming> timings) => _onTimings(timings);

  int get frames => _frames;
  int get jankyFrames => _janky;
  Duration get worstFrame => _worstTotal;

  /// Frames the scheduler should have produced and did not: the budget slots
  /// inside gaps longer than 1.5 × budget between consecutive frames.
  int get missedFrames => _missed;

  Duration get longestGap => Duration(microseconds: _longestGapUs);

  /// Frames per second over the whole measurement window (frames ÷ wall
  /// clock). This is the honest headline number: [fps] only describes the
  /// frames that happened, so it stays high while the app stalls.
  double get sustainedFps {
    final s = _wall.elapsedMicroseconds / 1e6;
    if (s <= 0) return 0;
    return _frames / s;
  }

  /// How long ago the last frame arrived. A gap only becomes a "missed
  /// frame" when a frame closes it, so a window that stops drawing entirely
  /// (occluded on macOS, for one) is invisible to every other counter here.
  Duration get stalledFor =>
      _frames == 0 ? Duration.zero : Duration(microseconds: _wall.elapsedMicroseconds - _lastFrameWallUs);

  /// Mean gap between frames over the recent window: the honest "how long is
  /// a frame right now", because it counts the frames that never came.
  Duration get avgInterval {
    if (_intervalsUs.isEmpty) return Duration.zero;
    var us = 0;
    for (final i in _intervalsUs) {
      us += i;
    }
    return Duration(microseconds: us ~/ _intervalsUs.length);
  }

  /// 95th percentile of the gap between frames.
  Duration get p95Interval {
    if (_intervalsUs.isEmpty) return Duration.zero;
    final sorted = _intervalsUs.toList()..sort();
    return Duration(microseconds: sorted[((sorted.length - 1) * 0.95).round()]);
  }

  Duration _avg(Duration Function(FrameTiming t) pick) {
    if (_recent.isEmpty) return Duration.zero;
    var us = 0;
    for (final t in _recent) {
      us += pick(t).inMicroseconds;
    }
    return Duration(microseconds: us ~/ _recent.length);
  }

  Duration get avgBuild => _avg((t) => t.buildDuration);
  Duration get avgRaster => _avg((t) => t.rasterDuration);
  Duration get avgTotal => _avg((t) => t.totalSpan);

  /// Frames per second over the recent window, from frame timestamps.
  double get fps {
    if (_recent.length < 2) return 0;
    final first = _recent.first.timestampInMicroseconds(FramePhase.vsyncStart);
    final last = _recent.last.timestampInMicroseconds(FramePhase.vsyncStart);
    final span = last - first;
    if (span <= 0) return 0;
    return (_recent.length - 1) * 1e6 / span;
  }

  /// 95th percentile of total frame time in the recent window.
  Duration get p95Total {
    if (_recent.isEmpty) return Duration.zero;
    final sorted = _recent.map((t) => t.totalSpan.inMicroseconds).toList()..sort();
    return Duration(microseconds: sorted[((sorted.length - 1) * 0.95).round()]);
  }

  Map<String, Object?> toJson() => {
        'target_fps': targetFps,
        'frames': _frames,
        'janky_frames': _janky,
        'missed_frames': _missed,
        'seconds': double.parse((_wall.elapsedMicroseconds / 1e6).toStringAsFixed(2)),
        'fps_sustained': double.parse(sustainedFps.toStringAsFixed(1)),
        'fps_recent': double.parse(fps.toStringAsFixed(1)),
        'interval_p95_ms': p95Interval.inMicroseconds / 1000,
        'longest_gap_ms': _longestGapUs / 1000,
        'stalled_ms': stalledFor.inMicroseconds / 1000,
        'avg_build_ms': avgBuild.inMicroseconds / 1000,
        'avg_raster_ms': avgRaster.inMicroseconds / 1000,
        'avg_total_ms': avgTotal.inMicroseconds / 1000,
        'p95_total_ms': p95Total.inMicroseconds / 1000,
        'worst_total_ms': _worstTotal.inMicroseconds / 1000,
      };

  @override
  void dispose() {
    stop();
    super.dispose();
  }
}
