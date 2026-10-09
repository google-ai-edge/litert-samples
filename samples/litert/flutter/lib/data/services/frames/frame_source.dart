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

import '../../../domain/models/frame_source_info.dart';
import '../../../domain/models/frame_source_spec.dart';
import '../../../domain/models/preview_source.dart';
import '../../../utils/result.dart';

/// One plane of a [FrameView].
final class const FramePlane({
  required final Uint8List bytes,
  required final int bytesPerRow,
  required final int bytesPerPixel,
});

/// A frame as a source delivers it.
///
/// Valid only during the `onFrame` callback: a camera may recycle the buffer
/// (camera_desktop's FFI path does), so anything kept must be copied there.
abstract interface class FrameView {
  int get width;
  int get height;
  FramePixelFormat get format;

  /// Clockwise degrees (0, 90, 180, 270) that turn the buffer upright.
  int get rotationDeg;
  List<FramePlane> get planes;
}

/// Mean brightness (0–255) of [frame] from a sparse sample (every 16th pixel
/// of every 16th row). Near 0 means a covered, disconnected or zeroed camera,
/// not a detector problem.
double meanLuma(FrameView frame) => lumaStats(frame).mean;

/// The mean and standard deviation of brightness (0–255) over the sparse
/// sample of [meanLuma].
({double mean, double spread}) lumaStats(FrameView frame) {
  final plane = frame.planes.first;
  final bytes = plane.bytes;
  var sum = 0.0;
  var squares = 0.0;
  var n = 0;
  for (var y = 0; y < frame.height; y += 16) {
    for (var x = 0; x < frame.width; x += 16) {
      final double value;
      switch (frame.format) {
        case FramePixelFormat.bgra8888 || FramePixelFormat.rgba8888:
          final i = y * plane.bytesPerRow + x * 4;
          if (i + 2 >= bytes.length) continue;
          final (r, b) = frame.format == FramePixelFormat.bgra8888
              ? (bytes[i + 2], bytes[i])
              : (bytes[i], bytes[i + 2]);
          value = 0.299 * r + 0.587 * bytes[i + 1] + 0.114 * b;
        case FramePixelFormat.nv21 || FramePixelFormat.yuv420:
          final i = y * plane.bytesPerRow + x;
          if (i >= bytes.length) continue;
          value = bytes[i].toDouble();
      }
      sum += value;
      squares += value * value;
      n++;
    }
  }
  if (n == 0) return (mean: 0, spread: 0);
  final mean = sum / n;
  final variance = squares / n - mean * mean;
  return (mean: mean, spread: variance <= 0 ? 0 : math.sqrt(variance));
}

/// A source of frames for live detection.
abstract interface class FrameSource {
  /// Starts delivering frames to [onFrame] on the main isolate. A failure
  /// after a successful start (camera unplugged, unusable frame format) goes
  /// to [onError], once. A source is single-use: start it once, stop it once.
  Future<Result<FrameSourceInfo>> start(
    void Function(FrameView) onFrame, {
    void Function(Exception error)? onError,
  });

  /// Stops delivering frames and releases the device, buffers and preview.
  Future<void> stop();

  /// What to show under the boxes; valid after a successful [start].
  PreviewSource get preview;
}

/// The lifecycle the [FrameSource] contract asks of every source, written
/// once: a source starts once (a second start is an error), stops once (a
/// second stop does nothing), learns when [stop] lands while it is still
/// starting, and reports at most one error, never after stop.
///
/// A source mixes this in and supplies:
/// - [sourceKind] and [stoppedWhileStarting], the two errors' wording;
/// - [acquire], the one start's work: open the device or stream, check
///   [stopped] after every await (when set, free what it opened itself and
///   fail with [stoppedWhileStarting]), and [attach] the callbacks once
///   frames may flow;
/// - [release], which [stop] calls once, after detaching the callbacks.
///
/// Frames go out through [frameCallback]; the one error is claimed with
/// [markFailed] and goes to [errorCallback] (or wherever the source routes
/// it while still starting). After stop both callbacks are null.
mixin SingleUseStart implements FrameSource {
  bool _started = false;
  bool _stopped = false;
  bool _failed = false;
  void Function(FrameView)? _onFrame;
  void Function(Exception error)? _onError;

  /// `camera` in "A camera source is single-use".
  @protected
  String get sourceKind;

  /// What [start] fails with when [stop] ran while it was acquiring.
  @protected
  FrameSourceUnavailableException get stoppedWhileStarting;

  /// Whether [stop] has run (or is running).
  @protected
  bool get stopped => _stopped;

  /// Whether the source's one error was claimed ([markFailed]).
  @protected
  bool get failed => _failed;

  /// Where frames go: null until [attach], and after [stop].
  @protected
  void Function(FrameView)? get frameCallback => _onFrame;

  /// Where the one error goes: null until [attach] (or when start had no
  /// `onError`), and after [stop].
  @protected
  void Function(Exception error)? get errorCallback => _onError;

  @override
  Future<Result<FrameSourceInfo>> start(
    void Function(FrameView) onFrame, {
    void Function(Exception error)? onError,
  }) async {
    if (_started) {
      return Result.error(
        FrameSourceUnavailableException('A $sourceKind source is single-use'),
      );
    }
    _started = true;
    return acquire(onFrame, onError: onError);
  }

  @override
  Future<void> stop() async {
    if (_stopped) return;
    _stopped = true;
    _onFrame = null;
    _onError = null;
    await release();
  }

  /// Opens the source for the one [start] (see [SingleUseStart]).
  @protected
  Future<Result<FrameSourceInfo>> acquire(
    void Function(FrameView) onFrame, {
    void Function(Exception error)? onError,
  });

  /// Frees what the source holds; [stop] calls it once.
  @protected
  Future<void> release();

  /// From now on frames go to [onFrame] and the one error to [onError],
  /// until [stop]. Does nothing once stopped.
  @protected
  void attach(
    void Function(FrameView) onFrame, {
    void Function(Exception error)? onError,
  }) {
    if (_stopped) return;
    _onFrame = onFrame;
    _onError = onError;
  }

  /// Claims the source's one error: true the first time; false once an
  /// error was claimed or the source stopped (then nothing is reported).
  @protected
  bool markFailed() {
    if (_failed || _stopped) return false;
    _failed = true;
    return true;
  }
}
