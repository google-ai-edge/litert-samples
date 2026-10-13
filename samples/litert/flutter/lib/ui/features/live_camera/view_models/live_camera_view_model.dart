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

import '../../../../config/demos.dart';
import '../../../../config/model_catalog.dart';
import '../../../../data/repositories/conversation_repository.dart';
import '../../../../data/repositories/diagnostics_repository.dart';
import '../../../../data/repositories/live_camera_settings_repository.dart';
import '../../../../data/repositories/live_detection_repository.dart';
import '../../../../domain/models/camera_side_event.dart';
import '../../../../domain/models/camera_source.dart';
import '../../../../domain/models/detection.dart';
import '../../../../domain/models/frame_source_info.dart';
import '../../../../domain/models/frame_source_spec.dart';
import '../../../../domain/models/live_state.dart';
import '../../../../domain/models/model_id.dart';
import '../../../../domain/models/model_state.dart';
import '../../../../domain/models/preview_source.dart';
import '../../../../domain/models/scene_snapshot.dart';
import '../../../../domain/models/voice.dart';
import '../../../../domain/use_cases/voice_assistant.dart';
import '../../../../utils/command.dart';
import '../../../../utils/result.dart';
import '../../../core/app_foreground.dart';
import 'camera_exchange_reducer.dart';
import 'live_settings_applier.dart';

/// What to check when the camera delivers black frames. On macOS a camera
/// opened from an app launched by a terminal is attributed to the terminal;
/// without that permission the frames arrive, but zeroed.
String blackFramesHint(TargetPlatform platform) => switch (platform) {
  TargetPlatform.macOS =>
    'Camera delivers black frames — lens covered, or camera access '
        'attributed to the terminal (System Settings › Privacy & Security › '
        'Camera)',
  _ => 'Camera delivers black frames — is the lens covered?',
};

/// The detector failed or is unavailable: Demo 3 shows [message], and when
/// it failed while loading on [backend] (`gpu`, `cpu`) offers the other one.
final class const DetectorFailure({
  required final String message,
  final String? backend,
});

/// The frame a detailed answer is about, shown instead of the live view: the
/// snapshot's own pixels and boxes.
final class FrozenFrame {
  FrozenFrame._({
    required this.frameId,
    required this.image,
    required this.detections,
    required this.flip,
  });

  final int frameId;

  /// Owned by the view model: disposed when the view unfreezes (`RawImage`
  /// draws its own clone).
  final ui.Image image;

  /// The snapshot's own detections, in the image's pixels.
  final DetectionFrame detections;

  /// Flip image and boxes horizontally so the frozen frame looks like the
  /// live preview (preview and frames mirrored differently).
  final bool flip;
}

/// Demo 3: the live view with boxes plus push-to-talk questions about it.
/// Entering opens the shared chat with the camera profile, makes moonshine
/// the active recognizer (both in parallel) and starts live detection as
/// this screen's owner; leaving stops it (only if this screen still owns
/// it) and disposes its voice assistant.
///
/// A detailed question freezes the view on its snapshot — the frame
/// Gemma is asked about, with that frame's own boxes — from the snapshot's
/// arrival until the answer has been spoken (playback drained), the mic goes
/// down again (barge-in), Stop, or a tap.
///
/// Demo 3's settings (with [settings]): the camera source — the device
/// camera or a network camera's MJPEG URL — and where the detector runs,
/// applied by its [LiveSettingsApplier]: its [applySource] and
/// [setDetectorBackend] commands, and this screen's [startLive] for every
/// start (so the screen shows every step).
/// A detector that failed on its backend shows the reason with the other
/// backend as the explicit way out; nothing switches by itself.
///
/// App lifecycle (with `foreground`, Android and iOS): leaving the app ends
/// a held push-to-talk press without a turn and releases the camera; coming
/// back starts live detection again. Desktop keeps both.
///
/// Rate discipline: [notifyListeners] fires on commands, phase changes and
/// commits. Boxes, preview, live state, the frozen frame, the partial reply
/// and the mic level are listenables the view binds directly.
class LiveCameraViewModel extends ChangeNotifier {
  LiveCameraViewModel({
    required this._conversation,
    required this._live,
    required this._assistant,
    required this._diagnostics,
    required this._activateStt,
    this._source,
    this._settings,
    this._models,
    this._foreground,
    Future<Result<void>> Function()? reloadDetector,
    TargetPlatform? platform,
  }) : _platform = platform ?? defaultTargetPlatform {
    open = Command0<void>(_open)..addListener(notifyListeners);
    selectStt = Command0<void>(_selectStt)..addListener(notifyListeners);
    _startLive = Command0<FrameSourceInfo>(_startLiveAction)
      ..addListener(notifyListeners);
    stop = Command0<void>(_stop)..addListener(notifyListeners);
    micAccess = Command0<void>(_micAccess)..addListener(notifyListeners);
    _applier = LiveSettingsApplier(
      start: _start,
      runningStart: () => _startRun,
      stop: () => _live.stop(owner: this),
      needsReconnect: () =>
          _startLive.result is Error<FrameSourceInfo> ||
          _live.state.value is LiveFailed,
      onSwitch: unfreeze,
      onChanged: _notify,
      settings: _settings,
      reloadDetector: reloadDetector,
    );
    for (final command in _settingsCommands) {
      command.addListener(notifyListeners);
    }
    _detectorState = _models?.value[ModelId.yolo26n];
    _models?.addListener(_onModels);
    _foreground?.addListener(_onForeground);
    _events = _assistant.events.listen(_onEvent);
    _assistant.phase.addListener(_onPhase);
    unawaited(_applier.load());
    // The chat and the recognizer switch in parallel (~0.8 s for moonshine).
    unawaited(open.execute());
    unawaited(selectStt.execute());
    // Microphone access right after the camera's.
    unawaited(_startLiveThenMic());
    // The first question must not pay for the audio device start.
    unawaited(_assistant.prepareAudio());
  }

  final ConversationRepository _conversation;
  final LiveDetectionRepository _live;

  /// Detailed questions send the frame to the chat model; off when it was
  /// loaded without images: then only answers from the detections remain.
  bool get detailedEnabled => _conversation.capabilities.images;

  /// Why detailed answers are off, for the label on the screen; null when
  /// they are on.
  String? get detailedOffReason => detailedEnabled
      ? null
      : 'Detailed answers are off: ${_chatModelName()} was set up without '
            'images, so the frame is never sent. Questions are answered from '
            'the detections only.';

  String _chatModelName() => _conversation.capabilities.modelName;

  /// The chat model detailed answers go to (`Gemma 4 E2B`, or the user's).
  String get chatModelName => _chatModelName();

  /// Owned: disposed with this view model.
  final VoiceAssistant<CameraSideEvent> _assistant;
  final DiagnosticsRepository _diagnostics;
  final SttActivation _activateStt;

  /// The source without [_settings] (tests); with them the settings resolve
  /// it at every start (the build's define still wins there). Neither is a
  /// start error, never a default read from the environment here.
  final Result<FrameSourceSpec>? _source;

  /// Demo 3's camera-source and detector settings; null hides them.
  final LiveCameraSettingsRepository? _settings;

  /// The model rows: the detector's state, its failure and its reload.
  final ValueListenable<Map<ModelId, ModelState>>? _models;

  final TargetPlatform _platform;

  /// The app's lifecycle on Android and iOS ([AppForeground]): out of the
  /// foreground the camera is released and starts again on return. Null:
  /// always in the foreground (desktop keeps the camera).
  final AppForeground? _foreground;

  bool get _inForeground => _foreground?.value ?? true;

  /// Live detection was released (or a start was asked for) while the app
  /// was out of the foreground: it starts again when the app comes back.
  bool _restartOnReturn = false;

  /// Applies Demo 3's settings: serialized, newest choice wins. Closed with
  /// this view model.
  late final LiveSettingsApplier _applier;

  ModelState? _detectorState;

  /// The spec the last start opened.
  FrameSourceSpec? _activeSpec;
  late final StreamSubscription<VoiceAssistantEvent<CameraSideEvent>> _events;

  /// The voice events' projection: the exchange on screen and the current
  /// turn's facts ([CameraExchangeReducer]).
  static const _reducer = CameraExchangeReducer();
  CameraExchangeState _exchangeState = const CameraExchangeState();
  bool _disposed = false;

  /// A failed post-turn reset: the camera chat may be gone; Retry reopens.
  String? _resetError;

  final ValueNotifier<FrozenFrame?> _frozen = ValueNotifier(null);
  final ValueNotifier<DetectionFrame?> _frozenBoxes = ValueNotifier(null);

  /// Bumped by every freeze and unfreeze: a snapshot image that finishes
  /// decoding after either is dropped.
  int _freezeToken = 0;

  /// Opens the chat with [kCameraProfile]; executing it again is the Retry.
  late final Command0<void> open;

  /// Makes this demo's recognizer (moonshine) the active one; executing it
  /// again is the Retry.
  late final Command0<void> selectStt;

  /// Live detection's start, whoever runs it: its progress and failure
  /// ([retryStart] runs it again).
  Command<FrameSourceInfo> get startLive => _startLive;
  late final Command0<FrameSourceInfo> _startLive;

  /// The start in progress, whoever asked for it (entering, Retry, a
  /// switch); null when none runs. A switch waits for it to end.
  Future<void>? _startRun;

  /// Stops the answer (or closes the mic).
  late final Command0<void> stop;

  /// Asks for microphone access, after the camera started (its
  /// own OS dialog first); executing it again is the Retry.
  late final Command0<void> micAccess;

  /// Saves a camera-source choice and restarts live detection on it: run by
  /// the settings' queue ([applySettings]); here for its progress and
  /// failure.
  Command<void> get applySource => _applier.applySource;

  /// Saves where the detector runs, then stops the source, reloads the
  /// detector and restarts the source: run by the settings' queue
  /// ([applySettings], [runDetectorOn]); here for its progress and failure.
  Command<void> get setDetectorBackend => _applier.setDetectorBackend;

  /// Owned here: disposed with this view model.
  List<Command<Object?>> get _commands => [
    open,
    selectStt,
    _startLive,
    stop,
    micAccess,
  ];

  /// Owned by [_applier], which disposes them when it closes.
  List<Command<void>> get _settingsCommands => [
    _applier.applySource,
    _applier.setDetectorBackend,
  ];

  // Demo 3's settings.

  /// The settings sheet is offered (the app has the settings repository).
  bool get settingsAvailable => _applier.available;

  /// What fixed the camera source (`FRAME_SOURCE=fixture`); null when the
  /// user chooses it.
  String? get sourceLock => _applier.sourceLock;

  /// The saved camera-source choice (the device camera until loaded).
  CameraSourceChoice get sourceChoice => _applier.sourceChoice;

  /// What fixed the detector backend (`DETECTOR_BACKEND=cpu`); null when the
  /// user chooses it.
  String? get backendLock => _applier.backendLock;

  /// The backend the detector loads on (the GPU until loaded).
  DetectorBackend get backend => _applier.backend;

  /// The user can switch the detector's backend here.
  bool get canChooseBackend => _applier.canChooseBackend;

  /// Why the saved settings could not be read; null when they were.
  String? get settingsError => _applier.loadError;

  /// Linux boards often have no camera of their own: the network camera is
  /// offered first there.
  bool get preferNetworkCamera => _platform == TargetPlatform.linux;

  /// Live detection runs (or last ran) on a network camera: its Retry is
  /// "Reconnect".
  bool get sourceIsNetwork => _activeSpec is NetworkSourceSpec;

  /// The failure card's button.
  String get retryLabel => sourceIsNetwork ? 'Reconnect' : 'Retry';

  /// The device camera failed (none, denied, unplugged) and the user may
  /// choose the source: the failure card offers the network camera.
  bool get offerNetworkCamera =>
      _applier.sourceEditable && _activeSpec is CameraSourceSpec;

  /// `null` when [text] is a usable network camera URL, else what to fix.
  String? validateNetworkUrl(String text) => _applier.validateNetworkUrl(text);

  /// Applies the settings sheet ([LiveSettingsApplier.apply]). Returns what
  /// to fix in the URL (nothing is applied then), or null. Applies run one
  /// at a time, in order, the newest pending choice winning field by field;
  /// within one the camera source goes first, then the detector backend.
  String? applySettings({
    required CameraSourceKind kind,
    required String networkUrl,
    required DetectorBackend backend,
  }) => _applier.apply(kind: kind, networkUrl: networkUrl, backend: backend);

  /// "Run detector on CPU" (or back on the GPU): queued like an Apply.
  void runDetectorOn(DetectorBackend backend) =>
      _applier.runDetectorOn(backend);

  /// An Apply is queued or running.
  bool get applying => _applier.applying;

  /// The detector failed or is unavailable: its reason, and the backend the
  /// load failed on when another backend may work (Demo 3 then offers it).
  DetectorFailure? get detectorFailure => switch (_detectorState) {
    ModelFailed(:final message, :final backend) => DetectorFailure(
      message: message,
      backend: backend,
    ),
    ModelUnavailable(:final reason) => DetectorFailure(message: reason),
    _ => null,
  };

  /// Saving a setting failed (storage), or a switch failed without a model
  /// state that says why; null otherwise.
  String? get settingsActionError {
    if (applySource.result case Error(:final error)) {
      return 'Could not apply the camera source: $error';
    }
    if (detectorFailure == null) {
      if (setDetectorBackend.result case Error(:final error)) {
        return 'Could not switch the detector: $error';
      }
    }
    return null;
  }

  /// The detector is loading again after a backend switch.
  bool get detectorReloading => setDetectorBackend.running;

  /// The live statistics (the network camera's rate and size on its chip).
  ValueListenable<LiveStats> get stats => _live.stats;

  /// Push-to-talk press; during a spoken answer it is the barge-in, which
  /// also unfreezes the view at once. Not a [Command], for the same reason
  /// as Demo 1's: a guarded release that is still pending would drop the
  /// next one.
  Future<void> pressMic() async {
    unfreeze();
    _exchangeState = _reducer.turnStarted(_exchangeState);
    await _assistant.micDown();
  }

  /// Push-to-talk release: snapshot (at this moment), STT, route, spoken
  /// answer. Completes when the turn ended or was superseded.
  Future<void> releaseMic() async {
    await _assistant.micUp();
  }

  TurnPhase get phase => _assistant.phase.value;

  /// The answer as it streams; empty between turns.
  ValueListenable<String> get partialReply => _assistant.partialReply;

  /// Mic level 0–1 while listening.
  ValueListenable<double> get inputLevel => _assistant.inputLevel;

  /// The mic button is held but the capture is still starting: nothing is
  /// recorded yet.
  bool get isOpeningMic => phase == TurnPhase.openingMic;

  /// The capture runs: what the user says now is recorded.
  bool get isListening => phase == TurnPhase.listening;

  /// The last question, its route and its answer.
  CameraExchange get exchange => _exchangeState.exchange;

  /// The frame a detailed answer is about; null while the view is live.
  ValueListenable<FrozenFrame?> get frozen => _frozen;

  /// The frozen frame's boxes, for a static painter; null while live.
  ValueListenable<DetectionFrame?> get frozenBoxes => _frozenBoxes;

  /// The last snapshot sent to Gemma (its frame id is the frozen frame's).
  @visibleForTesting
  EncodedSnapshot? get lastSentImage => _exchangeState.sentImage;

  /// The mic works whatever the camera does: without a frame the answer
  /// says the camera isn't running.
  bool get canTalk => !_disposed;

  /// This demo's chat is open. Not while [open] runs: the shared chat is
  /// then still the previous demo's.
  bool get chatReady =>
      !open.running && _conversation.profile == kCameraProfile;

  String? get chatError => switch (open.result) {
    Error(:final error) => 'Could not open the camera chat: $error',
    _ when _resetError != null =>
      'Resetting the camera chat failed: $_resetError',
    _ => null,
  };

  /// Microphone access is off (with the platform's settings
  /// path). Retry: [micAccess].
  String? get micAccessError => switch (micAccess.result) {
    Error(:final error) => '$error',
    _ => null,
  };

  /// Why this demo's recognizer could not be made active (Retry:
  /// [selectStt]).
  String? get sttError => switch (selectStt.result) {
    Error(:final error) => 'Could not switch the speech recognizer: $error',
    _ => null,
  };

  /// Why live detection could not start (a bad `FRAME_SOURCE`, no images,
  /// no detector). A failure after the start shows as [LiveFailed].
  String? get startError => switch (startLive.result) {
    Error(:final error) => error.toString(),
    _ => null,
  };

  ValueListenable<LiveState> get liveState => _live.state;
  ValueListenable<DetectionFrame?> get frames => _live.frames;
  ValueListenable<PreviewSource?> get preview => _live.preview;

  /// True while the camera delivers black frames (a warning chip, not a
  /// failure); [blackFramesWarning] says what to check.
  ValueListenable<bool> get blackFrames => _live.blackFrames;
  String get blackFramesWarning => blackFramesHint(defaultTargetPlatform);

  /// Mirror the boxes when exactly one of preview and frames is mirrored
  /// (`FrameSourceInfo.overlayMirrored`): false for camera_desktop on macOS,
  /// which mirrors both, and for the fixture.
  bool get mirrorBoxes => _live.sourceInfo?.overlayMirrored ?? false;

  /// `GPU fp32 full` or `CPU (chosen)`; null without a detector.
  String? get detectorLabel => _live.detectorInfo?.label;
  bool get detectorOnCpu => _live.detectorInfo?.backend == DetectorBackend.cpu;

  /// Back to the live view (a tap on the frozen frame, the end of the
  /// answer, a barge-in or Stop). The answer itself goes on.
  void unfreeze() {
    _freezeToken++;
    final frozen = _frozen.value;
    if (frozen == null || _disposed) return;
    _frozen.value = null;
    _frozenBoxes.value = null;
    frozen.image.dispose();
    _diagnostics.recordFrozen(null);
  }

  Future<Result<void>> _open() async {
    final result = await _conversation.open(kCameraProfile);
    if (result is Ok<void>) _resetError = null;
    return result;
  }

  Future<Result<void>> _selectStt() => _activateStt(Demo.liveCamera.stt);

  Future<Result<void>> _micAccess() => _assistant.requestMicAccess();

  /// Starts live detection, then asks for the microphone: the camera's OS
  /// dialog (inside the start) comes first, never both at once.
  Future<void> _startLiveThenMic() async {
    await _start();
    if (_disposed) return;
    await micAccess.execute();
  }

  /// Starts live detection again — the failure card's Retry (Reconnect for
  /// a network camera) — or joins the start in progress. Completes when
  /// that start ended.
  Future<void> retryStart() => _start();

  /// Runs [startLive], or joins the start in progress; completes when that
  /// start ended. Every start goes through here, so [_startRun] is the one
  /// that runs and a switch can await it. Out of the foreground nothing
  /// opens: the start runs when the app comes back.
  Future<void> _start() {
    if (!_inForeground) {
      _restartOnReturn = true;
      return Future<void>.value();
    }
    return _startRun ??= _startLive.execute().whenComplete(
      () => _startRun = null,
    );
  }

  /// The app's lifecycle changed (Android and iOS only; see [_foreground]).
  /// Leaving ends a held push-to-talk press without a turn and releases a
  /// running source (its CameraController and image stream), except while
  /// this screen's own microphone dialog covers the app (inactive): then
  /// only once the app is hidden or paused. A source still starting is
  /// released once its start ends if the app is hidden by then, so a camera
  /// permission dialog does not abort the start. Coming back starts live
  /// detection again if it was released here. A failed or stopped source
  /// stays as it is.
  void _onForeground() {
    final foreground = _foreground;
    if (_disposed || foreground == null) return;
    if (foreground.value) {
      if (!_restartOnReturn) return;
      _restartOnReturn = false;
      debugPrint('[LiveCamera] back in the foreground: starting the camera');
      unawaited(_start());
      return;
    }
    unawaited(_assistant.cancelCapture());
    if (micAccess.running && !foreground.hidden) return;
    switch (_live.state.value) {
      case LiveRunning() || LivePaused():
        _releaseForBackground();
      case LiveStopped() || LiveStarting() || LiveFailed():
        break;
    }
  }

  /// Stops this screen's source while the app is out of the foreground; it
  /// starts again on return.
  void _releaseForBackground() {
    _restartOnReturn = true;
    debugPrint('[LiveCamera] out of the foreground: releasing the camera');
    unawaited(
      _live
          .stop(owner: this)
          .catchError(
            (Object e, StackTrace st) => debugPrint(
              '[LiveCamera] releasing the camera in the background: $e\n$st',
            ),
          ),
    );
  }

  Future<Result<void>> _stop() async {
    unfreeze();
    await _assistant.stop();
    return const Result.ok(null);
  }

  /// Feeds captured audio through the same entry point mic release uses
  /// (integration tests; the mic is bypassed).
  @visibleForTesting
  Future<TurnResult> submitUtterance(Utterance utterance) {
    unfreeze();
    _exchangeState = _reducer.turnStarted(_exchangeState);
    return _assistant.submitUtterance(utterance);
  }

  void _onPhase() {
    if (_reducer.unfreezesAt(phase)) unfreeze();
    _notify();
  }

  /// Applies [CameraExchangeReducer]'s step for [event], effects in the
  /// update's order.
  void _onEvent(VoiceAssistantEvent<CameraSideEvent> event) {
    final update = _reducer.reduce(
      _exchangeState,
      event,
      _conversation.capabilities,
    );
    if (update.unfreeze) unfreeze();
    _exchangeState = update.state;
    if (update.generation case final metrics?) {
      _diagnostics.recordGeneration(metrics);
    }
    if (update.turn case final turn?) _diagnostics.recordCameraTurn(turn);
    if (update.freeze case final snapshot?) unawaited(_freeze(snapshot));
    if (update.chatReset case final reset?) {
      _diagnostics.recordCameraReset(reset.elapsed, error: reset.error);
      _resetError = reset.error;
    }
    if (update.notify) _notify();
  }

  /// Shows [snapshot] instead of the live view once its image is decoded
  /// (a few ms), unless the view was unfrozen (or frozen again) meanwhile.
  Future<void> _freeze(SceneSnapshot snapshot) async {
    final token = ++_freezeToken;
    final result = await _live.snapshotImage(snapshot);
    switch (result) {
      case Ok(:final value):
        if (_disposed || token != _freezeToken) {
          value.dispose();
          return;
        }
        final previous = _frozen.value;
        _frozen.value = FrozenFrame._(
          frameId: snapshot.frameId,
          image: value,
          detections: snapshot.detections,
          flip: snapshot.previewMirrored != snapshot.mirrored,
        );
        _frozenBoxes.value = snapshot.detections;
        previous?.image.dispose();
        _diagnostics.recordFrozen(snapshot.frameId);
      case Error(:final error):
        // The answer goes on over the live view; the frame still went to
        // Gemma.
        debugPrint('[LiveCamera] the frozen frame could not be shown: $error');
    }
  }

  void _notify() {
    if (!_disposed) notifyListeners();
  }

  Future<Result<FrameSourceInfo>> _startLiveAction() async {
    if (_disposed) return Result.error(asException(StateError('disposed')));
    final settings = _settings;
    final spec = switch ((settings, _source)) {
      (final settings?, _) => await settings.resolveSource(),
      (null, final source?) => source,
      (null, null) => const Result<FrameSourceSpec>.error(
        FrameSourceUnavailableException(
          'No camera source: neither the settings nor a source was given',
        ),
      ),
    };
    if (_disposed) return Result.error(asException(StateError('disposed')));
    switch (spec) {
      case Ok(:final value):
        _activeSpec = value;
        final started = await _live.start(value, owner: this);
        // The app was put away while the source started (a network camera
        // connecting): release it until the app comes back. Merely covered
        // (the camera permission dialog), it keeps the camera.
        if (started is Ok<FrameSourceInfo> &&
            (_foreground?.hidden ?? false) &&
            !_disposed) {
          _releaseForBackground();
        }
        return started;
      case Error(:final error):
        return Result.error(error);
    }
  }

  void _onModels() {
    final state = _models?.value[ModelId.yolo26n];
    if (identical(state, _detectorState)) return;
    _detectorState = state;
    _notify();
  }

  @override
  void dispose() {
    unfreeze();
    _disposed = true;
    for (final command in _settingsCommands) {
      command.removeListener(notifyListeners);
    }
    // Ends its queue and its waits, and disposes its commands.
    _applier.close();
    _models?.removeListener(_onModels);
    _foreground?.removeListener(_onForeground);
    unawaited(_events.cancel());
    _assistant.phase.removeListener(_onPhase);
    // Silences an answer in flight and closes the mic.
    unawaited(_assistant.dispose());
    // Stops only if this screen still owns the source (the owner token).
    unawaited(
      _live
          .stop(owner: this)
          .catchError(
            (Object e, StackTrace st) =>
                debugPrint('[LiveCamera] stopping live detection: $e\n$st'),
          ),
    );
    for (final command in _commands) {
      command
        ..removeListener(notifyListeners)
        ..dispose();
    }
    _frozen.dispose();
    _frozenBoxes.dispose();
    super.dispose();
  }
}
