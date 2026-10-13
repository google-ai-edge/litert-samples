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

import 'dart:async';
import 'dart:typed_data';

import 'package:litert_edge_demos/data/services/detector/detector_codec.dart';
import 'package:litert_edge_demos/data/services/detector/detector_service.dart';
import 'package:litert_edge_demos/data/services/detector/frame_message.dart';
import 'package:litert_edge_demos/domain/models/detection.dart';
import 'package:litert_edge_demos/utils/result.dart';

const kFakeDetectorInfo = DetectorInfo(
  backend: DetectorBackend.gpu,
  fullyAccelerated: true,
  verifyAbsolute: 1.8e-3,
  verifyRelative: 2e-6,
  verifyReference: VerifyReference.interpreter,
  createTime: Duration(milliseconds: 50),
  verifyTime: Duration(milliseconds: 300),
  firstRunTime: Duration(milliseconds: 5),
);

/// A [Detector] the test drives: each [detect] waits until the test calls
/// [complete] or [fail], unless [autoComplete] is set.
class FakeDetector implements Detector {
  FakeDetector({
    this.info = kFakeDetectorInfo,
    this.autoComplete = false,
    this.failWith,
    this.results,
  });

  @override
  DetectorInfo? info;
  bool autoComplete;

  /// What each frame "contains"; one cat box ([resultFor]) when null.
  DetectionFrame Function(FrameMessage frame)? results;

  /// When set, every [detect] fails with it at once.
  Exception? failWith;

  final List<FrameMessage> calls = [];

  /// Which [calls] asked for a snapshot (`detectWithSnapshot`).
  final List<bool> snapshotCalls = [];
  final List<Completer<Result<Object>>> _pending = [];

  /// Frames handed to the detector whose results are still pending.
  int get inFlight => _pending.length;

  @override
  Future<Result<DetectionFrame>> detect(FrameMessage frame) async =>
      switch (await _detect(frame, snapshot: false)) {
        Ok(:final value) => Result.ok(value as DetectionFrame),
        Error(:final error) => Result.error(error),
      };

  /// The real worker conversion (`uprightRgba`) on the frame's bytes.
  @override
  Future<Result<SnapshotDetection>> detectWithSnapshot(
    FrameMessage frame,
  ) async => switch (await _detect(frame, snapshot: true)) {
    Ok(:final value) => Result.ok(value as SnapshotDetection),
    Error(:final error) => Result.error(error),
  };

  Future<Result<Object>> _detect(FrameMessage frame, {required bool snapshot}) {
    calls.add(frame);
    snapshotCalls.add(snapshot);
    if (failWith case final error?) return Future.value(Result.error(error));
    if (autoComplete) return Future.value(Result.ok(_value(frame, snapshot)));
    final completer = Completer<Result<Object>>();
    _pending.add(completer);
    return completer.future;
  }

  /// Completes the oldest pending frame with a result for it.
  void complete() {
    final i = calls.length - _pending.length;
    _pending
        .removeAt(0)
        .complete(Result.ok(_value(calls[i], snapshotCalls[i])));
  }

  Object _value(FrameMessage frame, bool snapshot) {
    final detection = results?.call(frame) ?? resultFor(frame);
    if (!snapshot) return detection;
    return SnapshotDetection(
      frame: detection,
      pixels: uprightRgba(frame, frame.materialize()),
      convertTime: const Duration(milliseconds: 2),
    );
  }

  void fail(Exception error) =>
      _pending.removeAt(0).complete(Result.error(error));

  /// One cat box; timings 1/4/1 ms.
  static DetectionFrame resultFor(FrameMessage frame) =>
      catsFor(frame, cats: 1);

  /// [cats] cat boxes (score 0.9); timings 1/4/1 ms.
  static DetectionFrame catsFor(FrameMessage frame, {required int cats}) =>
      DetectionFrame(
        frameId: frame.frameId,
        width: frame.width,
        height: frame.height,
        boxes: Float32List.fromList([
          for (var i = 0; i < cats; i++) ...[10.0 + i, 20, 110, 220, 0.9, 15],
        ]),
        preMicros: 1000,
        runMicros: 4000,
        postMicros: 1000,
        backend: DetectorBackend.gpu,
      );
}
