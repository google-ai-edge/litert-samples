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
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';

import '../../config/live_camera_config.dart';
import '../../config/model_catalog.dart';
import '../../domain/models/detection.dart';
import '../../domain/models/frame_source_info.dart';
import '../../domain/models/frame_source_spec.dart';
import '../../domain/models/live_state.dart';
import '../../domain/models/preview_source.dart';
import '../../domain/models/scene_snapshot.dart';
import '../../utils/frame_rate_gate.dart';
import '../../utils/result.dart';
import '../../utils/serial_queue.dart';
import '../../utils/stall_watchdog.dart';
import '../services/detector/detector_service.dart';
import '../services/detector/frame_message.dart';
import '../services/frames/frame_source.dart';
import '../services/images/snapshot_encoder.dart';
import 'live_detection/black_frame_detector.dart';
import 'live_detection/live_stats_tracker.dart';
import 'live_detection/scene_capture.dart';

/// Builds a new source for a [FrameSourceSpec] (sources are single-use).
/// The app's is `createFrameSource` in `config/dependencies.dart`.
typedef FrameSourceFactory = Result<FrameSource> Function(FrameSourceSpec spec);

/// The one live-detection pipeline: frame source → latest-frame-wins gate →
/// detector worker → [frames].
///
/// - **One slot**: a frame goes to the detector only when none is in
///   flight, the duty is [DetectorDuty.live], and the [FrameRateGate] allows
///   it (≤ [kLiveDetectFps]). Everything else is dropped, never queued; the
///   slot is released in `finally`.
/// - **Owner token**: [start] and [stop] are serialized and carry an owner;
///   [stop] from anyone but the current owner is a no-op, so a late dispose
///   of the previous screen cannot stop the next one's source.
/// - **Rate discipline**: [frames] changes per processed frame (≤15 Hz) and
///   only painters listen to it; [stats] at most every [kLiveStatsInterval];
///   [state] on transitions only.
/// - **Watchdogs** (only while the duty is live): a frame the detector does
///   not answer within [kDetectorStallTimeout], or no source frame for
///   [kSourceStallTimeout] while Running, is a [LiveFailed] with Retry, never
///   a silent 0 fps. A pause disarms them and a resume re-arms them with a
///   fresh window; a check that fires late because the app itself was
///   suspended (debugger, sleep, background) re-arms instead of failing.
/// - **Scene capture** ([SceneCapture]): the summary window, the question's
///   frame ([capture]) and its encodings.
class LiveDetectionRepository {
  LiveDetectionRepository({
    required this._detector,
    required this._createSource,
    int fps = kLiveDetectFps,
    Duration gateSlack = kLiveGateSlack,
    Duration statsInterval = kLiveStatsInterval,
    int statsWindow = kLiveStatsWindow,
    this._stopTimeout = kLiveStopTimeout,
    this._stallTimeout = kDetectorStallTimeout,
    this._sourceTimeout = kSourceStallTimeout,
    this._watchdogInterval = kLiveWatchdogInterval,
    Duration captureTimeout = kCaptureTimeout,
    double countScore = kCountScore,
    SnapshotEncoder encoder = const SnapshotEncoder(),
    int llmImageMaxSide = kLlmImageMaxSide,
    Duration summaryMaxAge = kSummaryMaxAge,
    int Function()? clockMicros,
  }) : _gate = FrameRateGate(fps: fps, slack: gateSlack),
       _statsTracker = LiveStatsTracker(
         window: statsWindow,
         interval: statsInterval,
       ),
       _clock = clockMicros ?? _monotonicMicros {
    _scene = SceneCapture(
      isStarting: () => !_closed && _state.value is LiveStarting,
      runningSession: () =>
          _closed || _source == null || _info == null ? null : _session,
      clockMicros: _clock,
      timeout: captureTimeout,
      countScore: countScore,
      summaryMaxAge: summaryMaxAge,
      encoder: encoder,
      llmImageMaxSide: llmImageMaxSide,
    );
  }

  final Detector _detector;
  final FrameSourceFactory _createSource;
  final FrameRateGate _gate;
  final LiveStatsTracker _statsTracker;
  final Duration _stopTimeout;
  final Duration _stallTimeout;
  final Duration _sourceTimeout;
  final Duration _watchdogInterval;
  final int Function() _clock;
  late final SceneCapture _scene;

  static final Stopwatch _monotonic = Stopwatch()..start();
  static int _monotonicMicros() => _monotonic.elapsedMicroseconds;

  final ValueNotifier<LiveState> _state = ValueNotifier(const LiveStopped());
  final ValueNotifier<DetectionFrame?> _frames = ValueNotifier(null);
  final ValueNotifier<PreviewSource?> _preview = ValueNotifier(null);
  final ValueNotifier<LiveStats> _stats = ValueNotifier(const LiveStats());
  final ValueNotifier<bool> _blackFrames = ValueNotifier(false);
  final BlackFrameDetector _blackFrameDetector = BlackFrameDetector();

  /// Starts, stops and the close, one at a time. None runs inside the call
  /// that queues it: start's take-over and stop's owner check come first.
  final SerialQueue _ops = SerialQueue(
    onError: (e, st) => debugPrint('[LiveDetection] operation failed: $e\n$st'),
  );
  Object? _owner;
  FrameSource? _source;
  FrameSourceInfo? _info;

  /// The source inside [_start]'s `source.start`, and its owner. A network
  /// camera may take up to ~10 s there (connect, first frame); stop, close
  /// and a take-over reach it at once instead of queuing behind it.
  FrameSource? _starting;
  Object? _startingOwner;
  bool _startAborted = false;

  /// Bumped whenever a source stops: callbacks and results of an older
  /// session are ignored.
  int _session = 0;
  int _nextFrameId = 0;

  /// The single slot: true from the send until `_process` finishes.
  bool _busy = false;
  Completer<void>? _inFlightDone;

  /// Fires when the frame in flight has not come back within the stall
  /// timeout; armed only while the duty is live.
  late final DeadlineWatchdog _stallWatchdog = DeadlineWatchdog(
    timeout: _stallTimeout,
    lateTolerance: _watchdogInterval,
    clockMicros: _clock,
  );

  /// Checks every [_watchdogInterval] that the source still delivers frames;
  /// armed only while Running with the duty live.
  late final StallWatchdog _sourceWatchdog = StallWatchdog(
    interval: _watchdogInterval,
    clockMicros: _clock,
  );

  DetectorDuty _duty = DetectorDuty.live;
  String _pauseReason = 'paused';
  bool _closed = false;
  Future<void>? _closing;

  ValueListenable<LiveState> get state => _state;

  /// The newest detection, ≤ [kLiveDetectFps] updates per second; null when
  /// stopped.
  ValueListenable<DetectionFrame?> get frames => _frames;

  /// What the live view shows under the boxes; null when stopped.
  ValueListenable<PreviewSource?> get preview => _preview;
  ValueListenable<LiveStats> get stats => _stats;

  /// True while the source has delivered black frames (mean luma below
  /// [kBlackFrameLuma]) for about [kBlackFramesAfter]: a covered lens, or
  /// macOS zeroing the frames because camera access was attributed to the
  /// terminal. A warning the live screen shows, not a failure; changes on
  /// transitions only.
  ValueListenable<bool> get blackFrames => _blackFrames;

  /// The newest sampled mean luma (0–255) of the source's frames; null when
  /// stopped or before the first sample.
  @visibleForTesting
  double? get luma => _blackFrameDetector.luma;

  /// The running source's report; null when stopped.
  FrameSourceInfo? get sourceInfo => _info;

  /// The last [kSummaryWindow] detections, oldest first.
  @visibleForTesting
  List<DetectionFrame> get recent => _scene.recent;

  /// The loaded detector's report (backend label for the UI).
  DetectorInfo? get detectorInfo => _detector.info;

  @visibleForTesting
  DetectorDuty get duty => _duty;

  /// Starts [spec]'s source for [owner], first stopping whatever runs (a
  /// take-over). Serialized with [stop].
  Future<Result<FrameSourceInfo>> start(
    FrameSourceSpec spec, {
    required Object owner,
  }) {
    // A take-over: a source still starting is stopped now.
    _abortStarting();
    return _ops.run(() => _start(spec, owner));
  }

  /// Stops the source if [owner] still holds it; otherwise does nothing.
  /// Completes after the frame in flight (if any) has come back.
  Future<void> stop({required Object owner}) {
    if (identical(owner, _startingOwner)) _abortStarting();
    return _ops.run(() async {
      if (_closed || !identical(owner, _owner)) return;
      await _stopSource();
      _owner = null;
      _setState(const LiveStopped());
    });
  }

  /// Stops the source that is still inside its `start` (it then completes
  /// with an error, which [_start] reports as Stopped, not Failed).
  void _abortStarting() {
    final starting = _starting;
    if (starting == null || _startAborted) return;
    _startAborted = true;
    debugPrint('[LiveDetection] stopping a source that is still starting');
    unawaited(
      starting.stop().catchError(
        (Object e, StackTrace st) =>
            debugPrint('[LiveDetection] stopping a starting source: $e\n$st'),
      ),
    );
  }

  /// Opens or closes the gate. Paused: no frame is sent (the one in flight
  /// still lands); the state says why. Applies to a later start too.
  void setDuty(DetectorDuty duty, {String? reason}) {
    if (_closed) return;
    final newReason = reason ?? 'paused';
    if (duty == _duty &&
        (duty == DetectorDuty.live || newReason == _pauseReason)) {
      return;
    }
    _duty = duty;
    _pauseReason = newReason;
    // Paused, nothing is expected of the detector or the source: a frame in
    // flight may wait behind Gemma on the GPU. The summary window describes
    // the scene before the pause; a capture during or after it must not be
    // outvoted by it.
    if (duty == DetectorDuty.paused) {
      _disarmWatchdogs();
      _scene.clearWindow();
    }
    switch ((_state.value, duty)) {
      case (LiveRunning(:final source), DetectorDuty.paused):
        _setState(
          LivePaused(source: source, reason: newReason, since: DateTime.now()),
        );
      case (LivePaused(:final source), DetectorDuty.live):
        _gate.reset();
        _statsTracker.clearSamples();
        _setState(LiveRunning(source));
        _armWatchdogs(_session);
      case (LivePaused(:final source), DetectorDuty.paused):
        _setState(
          LivePaused(source: source, reason: newReason, since: DateTime.now()),
        );
      default:
        break; // stopped, starting or failed: applied when frames flow
    }
  }

  /// The frame a question is about: the next source frame, detected on its own,
  /// with the summary of the recent window ending with it; waits for a source
  /// that is still starting. See [SceneCapture.capture].
  Future<Result<SceneSnapshot>> capture() => _scene.capture();

  /// [snapshot] as the PNG Gemma gets; see [SceneCapture.encodeForLlm].
  Future<Result<EncodedSnapshot>> encodeForLlm(SceneSnapshot snapshot) =>
      _scene.encodeForLlm(snapshot);

  /// [snapshot]'s frame as an image for the frozen view, at full size. The
  /// caller owns it and must dispose it.
  Future<Result<ui.Image>> snapshotImage(SceneSnapshot snapshot) =>
      _scene.snapshotImage(snapshot);

  /// Stops everything and releases the notifiers. Safe to call more than
  /// once.
  Future<void> close() {
    _abortStarting();
    return _closing ??= _close();
  }

  Future<void> _close() async {
    await _ops.run(() async {
      _closed = true;
      _disarmWatchdogs();
      _scene.releaseStartWaiters();
      await _stopSource();
      _owner = null;
    });
    _state.dispose();
    _frames.dispose();
    _preview.dispose();
    _stats.dispose();
    _blackFrames.dispose();
  }

  Future<Result<FrameSourceInfo>> _start(
    FrameSourceSpec spec,
    Object owner,
  ) async {
    if (_closed) {
      return const Result.error(
        FrameSourceUnavailableException('Live detection is closed'),
      );
    }
    await _stopSource();
    _owner = owner;
    if (_busy) {
      // A frame from an earlier session never came back (stuck native call):
      // starting would show Running at 0 fps.
      return _failStart(
        const FrameSourceUnavailableException(
          'The detector is still busy with an earlier frame and may be stuck. '
          'Press Retry in a moment; if it stays stuck, restart the app.',
        ),
      );
    }
    if (_detector.info == null) {
      return _failStart(
        const FrameSourceUnavailableException(
          'The detector is not loaded (see its row on the setup screen)',
        ),
      );
    }
    final FrameSource source;
    switch (_createSource(spec)) {
      case Ok(:final value):
        source = value;
      case Error(:final error):
        return _failStart(error);
    }
    _setState(const LiveStarting());
    final session = ++_session;
    _source = source;
    _resetStats();
    final Result<FrameSourceInfo> started;
    _starting = source;
    _startingOwner = owner;
    _startAborted = false;
    try {
      started = await source.start(
        (frame) => _onFrame(session, frame),
        onError: (error) => _failPipeline(session, error.toString()),
      );
    } catch (e, st) {
      // A source must return an error, but a plugin can still throw (a raw
      // PlatformException): never leave the state at Starting without Retry.
      debugPrint('[LiveDetection] source start threw: $e\n$st');
      _clearStarting(source);
      await _stopSource();
      return _failStart(asException(e));
    }
    final aborted = _startAborted && identical(_starting, source);
    _clearStarting(source);
    if (aborted) {
      // Stopped by its owner, a take-over or close while starting: not a
      // failure to show (a source that started anyway is released).
      await _stopSource();
      _setState(const LiveStopped());
      return switch (started) {
        Error() => started,
        Ok() => const Result.error(
          FrameSourceUnavailableException('Stopped while starting'),
        ),
      };
    }
    switch (started) {
      case Ok() when _closed || session != _session:
        // It failed while starting (a frame's detect, or the source's
        // onError): keep the Failed state and release the source now.
        await _stopSource();
        return Result.error(
          FrameSourceUnavailableException(switch (_state.value) {
            LiveFailed(:final message) => message,
            _ => 'The source failed while starting',
          }),
        );
      case Ok(:final value):
        _info = value;
        _preview.value = source.preview;
        debugPrint(
          '[LiveDetection] source ${value.label} ${value.width}x${value.height} '
          '${value.format.name} mirrored=${value.mirrored}',
        );
        if (_duty == DetectorDuty.paused) {
          _setState(
            LivePaused(
              source: value.label,
              reason: _pauseReason,
              since: DateTime.now(),
            ),
          );
        } else {
          _setState(LiveRunning(value.label));
          _armWatchdogs(session);
        }
        return started;
      case Error(:final error):
        await _stopSource();
        return _failStart(error);
    }
  }

  void _clearStarting(FrameSource source) {
    if (!identical(_starting, source)) return;
    _starting = null;
    _startingOwner = null;
    _startAborted = false;
  }

  Result<FrameSourceInfo> _failStart(Exception error) {
    debugPrint('[LiveDetection] start failed: $error');
    _setState(LiveFailed(error.toString()));
    return Result.error(error);
  }

  /// Stops the source and waits (bounded) for the frame in flight.
  Future<void> _stopSource() async {
    _session++;
    _disarmWatchdogs();
    _scene.fail("The camera isn't running");
    final source = _source;
    _source = null;
    _info = null;
    _preview.value = null;
    try {
      await source?.stop();
    } catch (e, st) {
      // The source is dropped either way; a throw must not abort stop()
      // before Stopped and before the owner is cleared.
      debugPrint('[LiveDetection] stopping the source failed: $e\n$st');
    }
    final inFlight = _inFlightDone;
    if (inFlight != null) {
      await inFlight.future.timeout(
        _stopTimeout,
        onTimeout: () => debugPrint(
          '[LiveDetection] the frame in flight did not return within '
          '${_stopTimeout.inMilliseconds} ms of stop',
        ),
      );
    }
    // Nothing may watch a stopped source, whatever ran during the waits.
    _disarmWatchdogs();
    _frames.value = null;
    _scene.clearWindow();
    _resetLuma();
  }

  void _onFrame(int session, FrameView view) {
    if (_closed || session != _session) return;
    final now = _clock();
    _sourceWatchdog.activity(now);
    _statsTracker.sourceFrame(now, width: view.width, height: view.height);
    // A pending capture takes the next frame the detector can take, paused
    // or not and whatever the rate gate says.
    final forCapture = _scene.wantsFrame(session);
    if (!forCapture && _duty == DetectorDuty.paused) {
      _statsTracker.dropped(FrameDrop.paused);
      _maybePublishStats(now);
      return;
    }
    if (_busy) {
      _statsTracker.dropped(FrameDrop.busy);
      _maybePublishStats(now);
      return;
    }
    if (!forCapture && !_gate.tryPass(now)) {
      _statsTracker.dropped(FrameDrop.rate);
      _maybePublishStats(now);
      return;
    }
    final FrameMessage message;
    final copyStart = _clock();
    try {
      // The view is only valid during this callback: copy it now.
      message = FrameMessage.copyOf(view, frameId: ++_nextFrameId);
    } catch (e, st) {
      debugPrint('[LiveDetection] frame copy failed: $e\n$st');
      _failPipeline(session, 'Could not copy a frame: $e');
      return;
    }
    if (_blackFrameDetector.frameSent(view, now) case final change?) {
      _showBlackFrames(change);
    }
    if (forCapture) _scene.claimFrame(message.frameId);
    _busy = true;
    final done = _inFlightDone = Completer<void>();
    // Paused, the frame may wait behind Gemma on the GPU: only the capture
    // timeout bounds it then.
    if (_duty == DetectorDuty.live) _armStallWatchdog(session, done);
    // Never throws: _process reports every failure through the state.
    unawaited(
      _process(
        session,
        message,
        done,
        sentAt: now,
        copyMicros: _clock() - copyStart,
        withSnapshot: forCapture,
      ),
    );
  }

  Future<void> _process(
    int session,
    FrameMessage message,
    Completer<void> done, {
    required int sentAt,
    required int copyMicros,
    required bool withSnapshot,
  }) async {
    try {
      // A capture's frame also comes back as upright RGBA (its snapshot).
      final Result<(DetectionFrame, RgbaPixels?)> result = withSnapshot
          ? switch (await _detector.detectWithSnapshot(message)) {
              Ok(:final value) => Result.ok((value.frame, value.pixels)),
              Error(:final error) => Result.error(error),
            }
          : switch (await _detector.detect(message)) {
              Ok(:final value) => Result.ok((value, null)),
              Error(:final error) => Result.error(error),
            };
      if (_closed || session != _session) return;
      switch (result) {
        case Ok(value: (final frame, final pixels)):
          _publish(
            frame,
            latencyMicros: _clock() - sentAt,
            copyMicros: copyMicros,
          );
          if (pixels != null) _scene.complete(frame, pixels, source: _info);
        case Error(:final error):
          _failPipeline(session, 'Detector failed: $error');
      }
    } catch (e, st) {
      debugPrint('[LiveDetection] frame failed: $e\n$st');
      _failPipeline(session, 'Detection failed: $e');
    } finally {
      // Release the slot, even after a synchronous throw.
      _busy = false;
      if (identical(_inFlightDone, done)) {
        _inFlightDone = null;
        _stallWatchdog.disarm();
      }
      done.complete();
    }
  }

  void _publish(
    DetectionFrame frame, {
    required int latencyMicros,
    required int copyMicros,
  }) {
    _frames.value = frame;
    final now = _clock();
    _scene.record(frame, atMicros: now);
    _statsTracker.processed(
      atMicros: now,
      copyMicros: copyMicros,
      preMicros: frame.preMicros,
      runMicros: frame.runMicros,
      postMicros: frame.postMicros,
      latencyMicros: latencyMicros,
    );
    _maybePublishStats(now);
  }

  /// Publishes [stats] when the interval has passed since the last time.
  void _maybePublishStats(int now) {
    if (_statsTracker.publishDue(now) case final stats?) _stats.value = stats;
  }

  /// Arms both watchdogs for [session] on entering Running with the duty
  /// live, each with a fresh window. A frame still in flight (sent before a
  /// pause, or while starting) gets its stall watchdog back.
  ///
  /// Only while a source runs: during a stop the state is still Running or
  /// Paused while the frame in flight lands, but the source is already gone
  /// and nothing would disarm checks armed now (a resume then failed the
  /// stopped pipeline a timeout later).
  void _armWatchdogs(int session) {
    if (_closed || _source == null || session != _session) return;
    _armSourceWatchdog(session);
    if (_inFlightDone case final done?) _armStallWatchdog(session, done);
  }

  /// Fails [session] when the frame in flight ([done]) has not come back
  /// within the stall timeout, counted from now.
  void _armStallWatchdog(int session, Completer<void> done) {
    _stallWatchdog.arm(
      cancelIf: () => !identical(_inFlightDone, done),
      onExpired: () => _failPipeline(
        session,
        'The detector did not answer within '
        '${_stallTimeout.inMilliseconds} ms (it may be stuck). Press Retry; '
        'if it stays stuck, restart the app.',
      ),
    );
  }

  /// Starts the source watchdog for [session] with a fresh window: frames
  /// that stopped while paused or starting get the full timeout to resume.
  ///
  /// camera_desktop 2.0.0 drops runtime camera errors (and a failed native
  /// stream start), so a camera that is unplugged, interrupted or taken by
  /// another app just stops sending frames: that silence is the failure.
  void _armSourceWatchdog(int session) {
    _sourceWatchdog.arm(
      // A network camera reports its own stall (with a better message) and
      // sets a longer backstop here.
      timeout: () => _info?.stallTimeout ?? _sourceTimeout,
      cancelIf: () => _closed || session != _session,
      onStall: (silentMicros) => _failPipeline(
        session,
        'The camera stopped delivering frames (none for '
        '${silentMicros ~/ 1000} ms): it may be unplugged, interrupted or in '
        'use by another app. Press Retry.',
      ),
    );
  }

  void _disarmWatchdogs() {
    _stallWatchdog.disarm();
    _sourceWatchdog.disarm();
  }

  /// A detector, copy or source failure: shown, and the source is stopped.
  /// The owner keeps the slot, so its stop() or a retry start() still
  /// applies.
  void _failPipeline(int session, String message) {
    if (_closed || session != _session) return;
    debugPrint('[LiveDetection] $message');
    _disarmWatchdogs();
    _scene.fail(message, kind: CaptureFailure.failed);
    _session++; // drop anything else from this session
    _setState(LiveFailed(message));
    unawaited(
      _ops.run(() async {
        if (_closed || _session != session + 1) return;
        await _stopSource();
      }),
    );
  }

  void _setState(LiveState next) {
    if (!_closed) _state.value = next;
    if (next is! LiveStarting) _scene.releaseStartWaiters();
  }

  /// Shows a [BlackFrameChange] on [blackFrames], logged once per turn.
  void _showBlackFrames(BlackFrameChange change) {
    switch (change) {
      case BlackFramesEnded(:final luma):
        debugPrint(
          '[LiveDetection] black frames over: luma=${luma.toStringAsFixed(1)}',
        );
        _blackFrames.value = false;
      case BlackFramesStarted(:final luma, :final spread, :final darkMicros):
        debugPrint(
          '[LiveDetection] black frames: luma=${luma.toStringAsFixed(1)} '
          'spread=${spread.toStringAsFixed(1)} for '
          '${darkMicros ~/ 1000} ms (lens covered, or camera access '
          'attributed to another app)',
        );
        _blackFrames.value = true;
    }
  }

  void _resetLuma() {
    _blackFrameDetector.reset();
    if (!_closed) _blackFrames.value = false;
  }

  void _resetStats() {
    _gate.reset();
    _statsTracker.reset();
    _stats.value = const LiveStats();
    _resetLuma();
  }
}
