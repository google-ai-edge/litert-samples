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

import '../../../domain/models/live_state.dart';

/// Why the live pipeline's gate dropped a source frame.
enum FrameDrop {
  /// A frame was already in flight (latest frame wins).
  busy,

  /// The detector's duty is paused (Gemma generating).
  paused,

  /// The frame came too early for the rate gate.
  rate,
}

/// The live pipeline's rolling figures, as [LiveStats]: processed fps and
/// p50 timings over the last `window` results, the source's rate over its
/// last `window` frames and its newest frame size, and counters since the
/// last [reset].
///
/// Pure: each event carries the caller's monotonic clock time in µs, and the
/// caller publishes what [publishDue] hands back, at most once per
/// `interval`.
final class LiveStatsTracker {
  LiveStatsTracker({required this._window, required Duration interval})
    : _intervalMicros = interval.inMicroseconds;

  final int _window;
  final int _intervalMicros;

  /// The last [_window] results.
  final ListQueue<_Sample> _samples = ListQueue();

  /// Times of the last [_window] source frames (source fps).
  final ListQueue<int> _sourceTimes = ListQueue();
  int _sourceWidth = 0;
  int _sourceHeight = 0;
  int _processed = 0;
  int _sourceFrames = 0;
  int _droppedBusy = 0;
  int _droppedPaused = 0;
  int _droppedRate = 0;
  int? _lastPublishedAt;

  /// A [width]×[height] frame from the source at [atMicros], before the
  /// gate.
  void sourceFrame(int atMicros, {required int width, required int height}) {
    _sourceFrames++;
    _sourceTimes.addLast(atMicros);
    while (_sourceTimes.length > _window) {
      _sourceTimes.removeFirst();
    }
    _sourceWidth = width;
    _sourceHeight = height;
  }

  /// The gate dropped a source frame for [reason].
  void dropped(FrameDrop reason) {
    switch (reason) {
      case FrameDrop.busy:
        _droppedBusy++;
      case FrameDrop.paused:
        _droppedPaused++;
      case FrameDrop.rate:
        _droppedRate++;
    }
  }

  /// A result published at [atMicros], with its main-isolate copy, the
  /// worker's gather, run and decode, and its send → result time, in µs.
  void processed({
    required int atMicros,
    required int copyMicros,
    required int preMicros,
    required int runMicros,
    required int postMicros,
    required int latencyMicros,
  }) {
    _processed++;
    _samples.addLast(
      _Sample(
        at: atMicros,
        copy: copyMicros,
        pre: preMicros,
        run: runMicros,
        post: postMicros,
        latency: latencyMicros,
      ),
    );
    while (_samples.length > _window) {
      _samples.removeFirst();
    }
  }

  /// Forgets the results window (a resume: results from before a pause say
  /// nothing about the rate now). The source window and the counters stay.
  void clearSamples() => _samples.clear();

  /// Everything back to zero (a new source); the next [publishDue] is due.
  void reset() {
    _samples.clear();
    _sourceTimes.clear();
    _sourceWidth = 0;
    _sourceHeight = 0;
    _processed = 0;
    _sourceFrames = 0;
    _droppedBusy = 0;
    _droppedPaused = 0;
    _droppedRate = 0;
    _lastPublishedAt = null;
  }

  /// The figures at [nowMicros] when they have never been taken or the
  /// interval has passed since they last were; otherwise null.
  LiveStats? publishDue(int nowMicros) {
    final last = _lastPublishedAt;
    if (last != null && nowMicros - last < _intervalMicros) return null;
    _lastPublishedAt = nowMicros;
    return current();
  }

  /// The figures now.
  LiveStats current() {
    final samples = _samples;
    double? p50(int Function(_Sample) of) {
      if (samples.isEmpty) return null;
      final values = [for (final s in samples) of(s)]..sort();
      return values[values.length ~/ 2] / 1000;
    }

    return LiveStats(
      fps: _ratePerSecond(
        samples.length,
        samples.firstOrNull?.at,
        samples.lastOrNull?.at,
      ),
      sourceFps: _ratePerSecond(
        _sourceTimes.length,
        _sourceTimes.firstOrNull,
        _sourceTimes.lastOrNull,
      ),
      sourceWidth: _sourceWidth,
      sourceHeight: _sourceHeight,
      processed: _processed,
      sourceFrames: _sourceFrames,
      droppedBusy: _droppedBusy,
      droppedPaused: _droppedPaused,
      droppedRate: _droppedRate,
      copyMs: p50((s) => s.copy),
      preMs: p50((s) => s.pre),
      runMs: p50((s) => s.run),
      postMs: p50((s) => s.post),
      latencyMs: p50((s) => s.latency),
    );
  }

  /// Events per second for [count] events from [first] to [last] (µs); 0
  /// for fewer than two, or none apart.
  static double _ratePerSecond(int count, int? first, int? last) {
    if (count < 2 || first == null || last == null) return 0;
    final span = last - first;
    return span > 0 ? (count - 1) * 1e6 / span : 0;
  }
}

final class _Sample {
  const _Sample({
    required this.at,
    required this.copy,
    required this.pre,
    required this.run,
    required this.post,
    required this.latency,
  });

  final int at;
  final int copy;
  final int pre;
  final int run;
  final int post;
  final int latency;
}
