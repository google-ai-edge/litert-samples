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

import 'package:flutter/foundation.dart';

import '../../../config/live_camera_config.dart';
import '../../../domain/models/scene_snapshot.dart';
import '../../../domain/models/visual_memory.dart';
import '../../../utils/result.dart';
import '../../services/detector/memory_shot.dart';

/// The visual memory's capture, the second bounded slot next to the
/// question's ([SceneCapture]) in the live pipeline
/// (`LiveDetectionRepository`), which owns the source, the detector slot
/// and the sessions:
///
/// - at most one memory capture pending; a second request while one is
///   pending fails at once (nothing queues);
/// - unlike the question's, it never bypasses the rate gate or a pause: the
///   pipeline asks [wantsFrame] only for a frame that passed both, and only
///   when no question capture wants that frame (a question always wins);
/// - it fails after [_timeout], when the source stops, when the pipeline
///   fails, and alone ([failFrame]) when the worker could not make the
///   memory frame of a frame it detected.
final class MemoryCaptureSlot {
  /// [runningSession]: the session a running source delivers frames in;
  /// null when none runs (or the pipeline is closed).
  MemoryCaptureSlot({
    required this._runningSession,
    this._timeout = kMemoryCaptureTimeout,
  });

  final int? Function() _runningSession;
  final Duration _timeout;
  _MemoryRequest? _pending;

  /// A memory capture waits for a frame or its result.
  bool get pending => _pending != null;

  /// Asks for the memory frame of the next live frame the detector takes,
  /// made as [spec] says. Fails with [CaptureUnavailableException]:
  /// [CaptureFailure.notRunning] when no source runs, one is already pending
  /// or the source stops first; [CaptureFailure.timedOut] when no frame comes
  /// back within the timeout; [CaptureFailure.failed] when the pipeline
  /// fails.
  Future<Result<MemoryShot>> request(MemoryShotSpec spec) {
    final session = _runningSession();
    if (session == null) {
      return Future.value(
        const Result.error(
          CaptureUnavailableException("The camera isn't running"),
        ),
      );
    }
    if (_pending != null) {
      return Future.value(
        const Result.error(
          CaptureUnavailableException('A memory capture is already pending'),
        ),
      );
    }
    final request = _pending = _MemoryRequest(session: session, spec: spec);
    request.timer = Timer(_timeout, () {
      _fail(
        request,
        'No memory frame within ${_timeout.inMilliseconds} ms',
        kind: CaptureFailure.timedOut,
      );
    });
    return request.done.future;
  }

  /// The spec of the pending capture when it still waits for a frame of
  /// [session]; null otherwise.
  MemoryShotSpec? wantsFrame(int session) {
    final request = _pending;
    if (request == null ||
        request.session != session ||
        request.frameId != null) {
      return null;
    }
    return request.spec;
  }

  /// Frame [frameId] goes to the detector for the pending capture (the one
  /// [wantsFrame] just returned a spec for).
  void claimFrame(int frameId) => _pending?.frameId = frameId;

  /// Completes the pending capture if [shot] is of the frame it took.
  void complete(MemoryShot shot) {
    final request = _pending;
    if (request == null || request.frameId != shot.frameId) return;
    _pending = null;
    request.timer?.cancel();
    request.done.complete(Result.ok(shot));
  }

  /// Fails the pending capture with a [MemoryFrameException] if [frameId] is
  /// the frame it took: the worker detected that frame but could not make
  /// its memory frame (the live pipeline goes on).
  void failFrame(int frameId, String message) {
    final request = _pending;
    if (request == null || request.frameId != frameId) return;
    _pending = null;
    request.timer?.cancel();
    debugPrint('[LiveDetection] memory capture failed: $message');
    request.done.complete(Result.error(MemoryFrameException(message)));
  }

  /// Fails the pending capture, if any, with [message].
  void fail(
    String message, {
    CaptureFailure kind = CaptureFailure.notRunning,
  }) => _fail(_pending, message, kind: kind);

  void _fail(
    _MemoryRequest? request,
    String message, {
    required CaptureFailure kind,
  }) {
    if (request == null || request.done.isCompleted) return;
    request.timer?.cancel();
    if (identical(_pending, request)) _pending = null;
    debugPrint(
      '[LiveDetection] memory capture dropped (${kind.name}): $message',
    );
    request.done.complete(
      Result.error(CaptureUnavailableException(message, kind: kind)),
    );
  }
}

final class _MemoryRequest {
  _MemoryRequest({required this.session, required this.spec});

  final int session;
  final MemoryShotSpec spec;
  final Completer<Result<MemoryShot>> done = Completer();
  Timer? timer;

  /// The frame sent for this capture; null until one is.
  int? frameId;
}
