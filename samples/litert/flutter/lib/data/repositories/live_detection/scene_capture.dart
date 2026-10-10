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
import 'dart:collection';
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';

import '../../../config/live_camera_config.dart';
import '../../../config/model_catalog.dart';
import '../../../domain/models/detection.dart';
import '../../../domain/models/detection_summary.dart';
import '../../../domain/models/frame_source_info.dart';
import '../../../domain/models/scene_snapshot.dart';
import '../../../utils/result.dart';
import '../../services/images/snapshot_encoder.dart';

/// The frame a question is about, for the live pipeline
/// (`LiveDetectionRepository`), which owns the source, the detector slot and
/// the sessions:
///
/// - the summary window: the last [kSummaryWindow] published detections,
///   each with the time it was published ([record], [clearWindow]);
/// - one pending capture at a time ([capture]): the pipeline asks
///   [wantsFrame] for every source frame, sends the next one it can for it
///   ([claimFrame]), and hands back its detections and pixels ([complete])
///   or the reason it failed ([fail]);
/// - captures waiting for a starting source ([releaseStartWaiters]);
/// - the snapshot's PNG for Gemma and its image for the frozen view.
final class SceneCapture {
  /// [isStarting]: whether the source is still inside its start.
  /// [runningSession]: the session a running source delivers frames in;
  /// null when none runs (or the pipeline is closed).
  SceneCapture({
    required this._isStarting,
    required this._runningSession,
    required int Function() clockMicros,
    this._timeout = kCaptureTimeout,
    this._countScore = kCountScore,
    Duration summaryMaxAge = kSummaryMaxAge,
    this._encoder = const SnapshotEncoder(),
    this._llmImageMaxSide = kLlmImageMaxSide,
  }) : _clock = clockMicros,
       _summaryMaxAgeMicros = summaryMaxAge.inMicroseconds;

  final bool Function() _isStarting;
  final int? Function() _runningSession;
  final int Function() _clock;
  final Duration _timeout;
  final double _countScore;
  final int _summaryMaxAgeMicros;
  final SnapshotEncoder _encoder;
  final int _llmImageMaxSide;

  /// The summary window: the last [kSummaryWindow] detections with the
  /// [_clock] time each was published.
  final ListQueue<({int at, DetectionFrame frame})> _recent = ListQueue();

  /// The pending [capture], if any (one at a time; callers share it).
  _CaptureRequest? _capture;

  /// Captures waiting for a starting source; completed when the start
  /// settles (or on close).
  final List<Completer<void>> _startWaiters = [];

  /// The last [kSummaryWindow] detections, oldest first.
  List<DetectionFrame> get recent =>
      List.unmodifiable([for (final r in _recent) r.frame]);

  /// Adds [frame], published at [atMicros], to the summary window.
  void record(DetectionFrame frame, {required int atMicros}) {
    _recent.addLast((at: atMicros, frame: frame));
    while (_recent.length > kSummaryWindow) {
      _recent.removeFirst();
    }
  }

  /// Forgets the summary window: the scene before a pause or a stop must
  /// not outvote a capture after it.
  void clearWindow() => _recent.clear();

  /// The frame a question is about: the next source frame, detected on its own.
  /// It bypasses the pause and the rate gate (the detector is idle then, or
  /// about to be), waits for a frame already in flight (one at a time), and
  /// completes with that frame's detections and the summary of the recent
  /// window ending with it. A second call while one is pending shares it.
  ///
  /// While the source is still starting (just after entering the demo), it
  /// first waits for the start to finish, bounded by the capture timeout.
  ///
  /// Fails with [CaptureUnavailableException]: [CaptureFailure.notRunning] when no
  /// source runs or it stops first, [CaptureFailure.failed] when the
  /// pipeline fails, [CaptureFailure.timedOut] when no frame is detected
  /// within the capture timeout.
  Future<Result<SceneSnapshot>> capture() {
    if (_isStarting()) {
      return _whenStarted().then((_) => _captureNow());
    }
    return _captureNow();
  }

  /// Whether the pending capture still waits for a frame of [session]: the
  /// pipeline then sends the next frame the detector can take for it,
  /// paused or not and whatever the rate gate says.
  bool wantsFrame(int session) {
    final request = _capture;
    return request != null &&
        request.session == session &&
        request.frameId == null;
  }

  /// Frame [frameId] goes to the detector for the pending capture (the one
  /// [wantsFrame] just reported).
  void claimFrame(int frameId) => _capture?.frameId = frameId;

  /// Completes the pending capture if [frame] is the one it took: its
  /// detections, [pixels], the summary of the fresh part of the window, and
  /// [source]'s mirroring.
  void complete(
    DetectionFrame frame,
    RgbaPixels pixels, {
    required FrameSourceInfo? source,
  }) {
    final request = _capture;
    if (request == null || request.frameId != frame.frameId) return;
    _capture = null;
    request.timer?.cancel();
    // Only frames young enough to describe the scene now (this one always
    // is: it was just published).
    final now = _clock();
    final window = [
      for (final r in _recent)
        if (now - r.at <= _summaryMaxAgeMicros) r.frame,
    ];
    final snapshot = SceneSnapshot(
      frameId: frame.frameId,
      detections: frame,
      summary: DetectionSummary.fromWindow(window, minScore: _countScore),
      pixels: pixels,
      mirrored: source?.mirrored ?? false,
      previewMirrored: source?.previewMirrored ?? false,
      latency: Duration(microseconds: _clock() - request.requestedAt),
    );
    debugPrint(
      '[LiveDetection] capture frame=${frame.frameId} '
      'latency=${snapshot.latency.inMilliseconds}ms '
      'boxes=${frame.count} summary=${snapshot.summary.counts}',
    );
    request.done.complete(Result.ok(snapshot));
  }

  /// Fails the pending capture, if any, with [message].
  void fail(
    String message, {
    CaptureFailure kind = CaptureFailure.notRunning,
  }) => _fail(_capture, message, kind: kind);

  /// Lets the captures waiting for a starting source go on: the start
  /// settled (Running, Paused, Failed or Stopped), or the pipeline closed.
  void releaseStartWaiters() {
    final waiters = List.of(_startWaiters);
    _startWaiters.clear();
    for (final waiter in waiters) {
      if (!waiter.isCompleted) waiter.complete();
    }
  }

  /// [snapshot] as the PNG Gemma gets: at most `kLlmImageMaxSide` on the long
  /// side, mirrored back when the source mirrors its frames, so text reads the
  /// right way round and "left" means the scene's left. The PNG is a new object
  /// per call: the conversation sends it as a new image.
  Future<Result<EncodedSnapshot>> encodeForLlm(SceneSnapshot snapshot) async {
    try {
      final encoded = await _encoder.toPng(
        snapshot.pixels,
        frameId: snapshot.frameId,
        maxSide: _llmImageMaxSide,
        unmirror: snapshot.mirrored,
      );
      debugPrint(
        '[LiveDetection] encoded frame=${encoded.frameId} '
        '${encoded.width}x${encoded.height} PNG ${encoded.png.length} B '
        'unmirrored=${encoded.unmirrored} '
        'in ${encoded.encodeTime.inMilliseconds}ms',
      );
      return Result.ok(encoded);
    } catch (e, st) {
      debugPrint('[LiveDetection] encoding the snapshot failed: $e\n$st');
      return Result.error(asException(e));
    }
  }

  /// [snapshot]'s frame as an image for the frozen view, at full size. The
  /// caller owns it and must dispose it.
  Future<Result<ui.Image>> snapshotImage(SceneSnapshot snapshot) async {
    try {
      return Result.ok(await _encoder.toImage(snapshot.pixels));
    } catch (e, st) {
      debugPrint('[LiveDetection] the snapshot image failed: $e\n$st');
      return Result.error(asException(e));
    }
  }

  Future<void> _whenStarted() {
    final waiter = Completer<void>();
    _startWaiters.add(waiter);
    return waiter.future.timeout(
      _timeout,
      onTimeout: () {
        // Still starting: the capture then fails as "not running".
        _startWaiters.remove(waiter);
      },
    );
  }

  Future<Result<SceneSnapshot>> _captureNow() {
    final session = _runningSession();
    if (session == null) {
      return Future.value(
        const Result.error(
          CaptureUnavailableException("The camera isn't running"),
        ),
      );
    }
    final pending = _capture;
    if (pending != null && pending.session == session) {
      return pending.done.future;
    }
    final request = _capture = _CaptureRequest(
      session: session,
      requestedAt: _clock(),
    );
    request.timer = Timer(_timeout, () {
      _fail(
        request,
        'No camera frame was detected within '
        '${_timeout.inMilliseconds} ms',
        kind: CaptureFailure.timedOut,
      );
    });
    return request.done.future;
  }

  void _fail(
    _CaptureRequest? request,
    String message, {
    CaptureFailure kind = CaptureFailure.notRunning,
  }) {
    if (request == null || request.done.isCompleted) return;
    request.timer?.cancel();
    if (identical(_capture, request)) _capture = null;
    debugPrint('[LiveDetection] capture failed (${kind.name}): $message');
    request.done.complete(
      Result.error(CaptureUnavailableException(message, kind: kind)),
    );
  }
}

final class _CaptureRequest {
  _CaptureRequest({required this.session, required this.requestedAt});

  final int session;
  final int requestedAt;
  final Completer<Result<SceneSnapshot>> done = Completer();
  Timer? timer;

  /// The frame sent for this capture; null until one is.
  int? frameId;
}
