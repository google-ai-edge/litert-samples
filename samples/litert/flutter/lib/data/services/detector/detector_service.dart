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
import 'dart:isolate';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart' show AssetBundle, rootBundle;

import '../../../domain/models/detection.dart';
import '../../../domain/models/detector_spec.dart';
import '../../../domain/models/scene_snapshot.dart';
import '../../../utils/result.dart';
import '../../../utils/worker_channel.dart';
import '../model_store/local_files.dart';
import 'detector_engine.dart';
import 'detector_worker.dart';
import 'frame_message.dart';

/// Runs the detector on one frame at a time. The live repository holds the
/// latest-frame-wins slot; a second detect while one runs is an error.
abstract interface class Detector {
  /// What the loaded detector reported; null until a load succeeded.
  DetectorInfo? get info;

  Future<Result<DetectionFrame>> detect(FrameMessage frame);

  /// [detect] plus the frame itself as upright RGBA: a question's snapshot,
  /// converted in the worker from the same bytes.
  Future<Result<SnapshotDetection>> detectWithSnapshot(FrameMessage frame);
}

/// A detected frame and its pixels (`Detector.detectWithSnapshot`).
final class const SnapshotDetection({
  required final DetectionFrame frame,
  required final RgbaPixels pixels,

  /// The worker's frame → upright RGBA conversion.
  required final Duration convertTime,
});

/// The detector is not loaded, busy, closed, or its worker died.
final class DetectorUnavailableException implements Exception {
  const DetectorUnavailableException(this.message);

  final String message;

  @override
  String toString() => message;
}

/// Where the detector model comes from.
sealed class DetectorModelSource {
  const DetectorModelSource();

  /// For logs and reports: the path, or `bundled asset`.
  String get label;
}

/// A file (the self-test's `--detector`, a test's copy).
final class DetectorFile extends DetectorModelSource {
  const DetectorFile(this.path);

  final String path;

  @override
  String get label => path;
}

/// The model's bytes: the asset built into the app ([kDetModelAsset]).
final class DetectorBytes extends DetectorModelSource {
  const DetectorBytes(this.bytes, {this.label = 'bundled asset'});

  final Uint8List bytes;

  @override
  final String label;
}

/// Reads the built-in detector from the asset bundle on the main isolate
/// (the worker then gets the bytes as `TransferableTypedData`).
Future<Uint8List> loadBundledDetector([AssetBundle? bundle]) async {
  final data = await (bundle ?? rootBundle).load(kDetModelAsset);
  return data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
}

/// Owns the detector worker isolate: [load] spawns it and runs the load checks
/// there (`DetectorEngine.load`), [detect] sends one frame, [close] waits for
/// the frame in flight, closes the model and ends the isolate.
///
/// Called by `ModelRepository` (load, close) and `LiveDetectionRepository`
/// (detect).
class DetectorService implements Detector {
  DetectorService({
    this._runtime = const LiteRtDetectorRuntime(),
    this._expectedModelBytes = kDetModelBytes,
    this._closeTimeout = const Duration(seconds: 3),
  });

  final DetectorRuntime _runtime;
  final int _expectedModelBytes;
  final Duration _closeTimeout;

  /// The worker answers with its port before any work: a worker that has
  /// not answered by then never will, and [load] says so.
  static const _handshakeTimeout = Duration(seconds: 10);

  WorkerChannel? _worker;
  DetectorInfo? _info;
  Future<Result<DetectorInfo>>? _loading;
  Completer<void>? _detectDone;
  bool _closed = false;

  @override
  DetectorInfo? get info => _info;

  bool get isLoaded => _info != null && _worker != null;

  /// Spawns the worker and loads [source] on [backend]: a file (absolute,
  /// or relative to the documents directory), or the bundled asset's bytes,
  /// moved to the worker without a copy. A failure ends the worker; calling
  /// it again is the retry.
  Future<Result<DetectorInfo>> load({
    required DetectorModelSource source,
    required DetectorBackend backend,
  }) {
    if (_closed) return Future.value(_closedError());
    return _loading ??= _load(
      source,
      backend,
    ).whenComplete(() => _loading = null);
  }

  Future<Result<DetectorInfo>> _load(
    DetectorModelSource source,
    DetectorBackend backend,
  ) async {
    await _shutdownWorker(); // a retry replaces the previous worker
    try {
      final (String? path, TransferableTypedData? bytes) = switch (source) {
        DetectorFile(:final path) => (await resolveLocalPath(path), null),
        DetectorBytes(:final bytes) => (null, transferBytes(bytes)),
      };
      final worker = await WorkerChannel.spawn(
        protocol: workerProtocol,
        entryPoint: detectorWorkerMain,
        boot: (replyTo) => DetectorWorkerBoot(replyTo, _runtime),
        handshakeTimeout: _handshakeTimeout,
      );
      _worker = worker;
      if (_closed) {
        await _shutdownWorker();
        return _closedError();
      }
      final value = await worker.request(
        (id) => DetectorLoadRequest(
          id,
          modelPath: path,
          modelBytes: bytes,
          backend: backend,
          expectedBytes: _expectedModelBytes,
        ),
      );
      if (_closed) {
        // close() is waiting for this load; it shuts the worker down.
        return _closedError();
      }
      final info = value! as DetectorInfo;
      _info = info;
      // Logged here: the worker isolate's prints do not reach the device log.
      debugPrint('[DetectorService] loaded $info from ${source.label}');
      if (info.backend == DetectorBackend.gpu && info.verifyAbsolute == 0.0) {
        debugPrint(
          '[DetectorService] warning: GPU output is bit-identical to the CPU '
          'reference; the GPU may not have run',
        );
      }
      return Result.ok(info);
    } on DetectorUnavailableException catch (e) {
      debugPrint('[DetectorService] load failed: $e');
      await _shutdownWorker();
      return Result.error(e);
    } catch (e, st) {
      debugPrint('[DetectorService] load failed: $e\n$st');
      await _shutdownWorker();
      return Result.error(asException(e));
    }
  }

  @override
  Future<Result<DetectionFrame>> detect(FrameMessage frame) async {
    final result = await _request(frame, withSnapshot: false);
    return switch (result) {
      Ok(:final value) => Result.ok(value! as DetectionFrame),
      Error(:final error) => Result.error(error),
    };
  }

  @override
  Future<Result<SnapshotDetection>> detectWithSnapshot(
    FrameMessage frame,
  ) async {
    final result = await _request(frame, withSnapshot: true);
    switch (result) {
      case Error(:final error):
        return Result.error(error);
      case Ok(:final value):
        final reply = value! as SnapshotReply;
        final pixels = RgbaPixels(
          width: reply.width,
          height: reply.height,
          bytes: receiveBytes(reply.rgba),
        );
        debugPrint(
          '[DetectorService] snapshot frame=${reply.frame.frameId} '
          '${pixels.width}x${pixels.height} '
          'convert=${reply.convertTime.inMicroseconds}µs',
        );
        return Result.ok(
          SnapshotDetection(
            frame: reply.frame,
            pixels: pixels,
            convertTime: reply.convertTime,
          ),
        );
    }
  }

  /// One frame through the worker; the slot is this service's own guard.
  Future<Result<Object?>> _request(
    FrameMessage frame, {
    required bool withSnapshot,
  }) async {
    final worker = _worker;
    if (_closed || worker == null || _info == null) {
      return const Result.error(
        DetectorUnavailableException('The detector is not loaded'),
      );
    }
    if (_detectDone != null) {
      return const Result.error(
        DetectorUnavailableException('A frame is already being detected'),
      );
    }
    final done = _detectDone = Completer<void>();
    try {
      final value = await worker.request(
        (id) => DetectorDetectRequest(id, frame, withSnapshot: withSnapshot),
      );
      return Result.ok(value);
    } on DetectorUnavailableException catch (e) {
      return Result.error(e);
    } catch (e) {
      return Result.error(asException(e));
    } finally {
      _detectDone = null;
      done.complete();
    }
  }

  /// Waits (each wait bounded by the close timeout) for a load or frame in
  /// flight, closes the model and ends the worker. A native call that never
  /// returns cannot hold up the app's shutdown: the worker is killed, which
  /// fails whatever was pending. Safe to call more than once.
  ///
  /// When a wait timed out the worker is stuck in that call and could not
  /// answer a close request either, so it is killed at once: close takes at
  /// most about one close timeout per pending call, never one more.
  Future<void> close() async {
    if (_closed) return;
    _closed = true;
    final loadSettled = await _bounded(_loading, 'the model load');
    final frameSettled = await _bounded(
      _detectDone?.future,
      'the frame in flight',
    );
    await _shutdownWorker(graceful: loadSettled && frameSettled);
  }

  /// Whether [pending] finished within the close timeout (true when there
  /// was nothing to wait for).
  Future<bool> _bounded(Future<Object?>? pending, String what) async {
    if (pending == null) return true;
    try {
      await pending.timeout(_closeTimeout);
      return true;
    } on TimeoutException {
      debugPrint(
        '[DetectorService] $what did not finish within '
        '${_closeTimeout.inMilliseconds} ms of close; ending the worker',
      );
      return false;
    }
  }

  /// Ends the worker: [graceful] asks it to close the model first (bounded
  /// by the close timeout); otherwise it is killed straight away.
  Future<void> _shutdownWorker({bool graceful = true}) async {
    final worker = _worker;
    _worker = null;
    _info = null;
    if (worker == null) return;
    if (graceful) {
      await worker.close(timeout: _closeTimeout);
    } else {
      worker.kill();
    }
  }

  static Result<T> _closedError<T>() => Result.error(
    const DetectorUnavailableException('The detector is closed'),
  );

  /// The worker's wire format ([detectorWorkerMain]), the
  /// [DetectorUnavailableException] texts and the `[DetectorService]` log lines.
  @visibleForTesting
  static const workerProtocol = WorkerProtocol(
    name: 'detector',
    noun: 'the worker',
    parseReply: _parseReply,
    closeMessage: DetectorCloseRequest.new,
    failure: _failure,
    log: _log,
  );

  static WorkerReply? _parseReply(Object? message) => switch (message) {
    DetectorReply(:final id, :final value, :final error) => (
      id: id,
      value: value,
      error: error,
    ),
    _ => null,
  };

  static Exception _failure(WorkerFailure kind, String reason) =>
      DetectorUnavailableException(switch (kind) {
        WorkerFailure.startFailed =>
          'Could not start the detector worker: $reason',
        WorkerFailure.notRunning ||
        WorkerFailure.lost => 'The detector worker is gone: $reason',
        WorkerFailure.replyError => reason,
      });

  static void _log(WorkerEvent event, String detail) {
    final line = switch (event) {
      WorkerEvent.unexpectedMessage => 'unexpected reply $detail',
      WorkerEvent.crashed => 'worker error: $detail',
      WorkerEvent.died => detail,
      WorkerEvent.closeFailed => 'close: $detail',
      WorkerEvent.closeTimedOut => null,
    };
    if (line != null) debugPrint('[DetectorService] $line');
  }
}
