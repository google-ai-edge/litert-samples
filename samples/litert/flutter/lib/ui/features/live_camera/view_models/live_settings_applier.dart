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

import '../../../../data/repositories/live_camera_settings_repository.dart';
import '../../../../domain/models/camera_source.dart';
import '../../../../domain/models/detection.dart';
import '../../../../domain/models/detector_choice.dart';
import '../../../../domain/models/frame_source_spec.dart';
import '../../../../utils/command.dart';
import '../../../../utils/result.dart';
import '../../../../utils/waits.dart';

/// Demo 3's settings, applied for one screen: the camera source — the
/// device camera or a network camera's MJPEG URL — and where the detector
/// runs (GPU or CPU), both persisted. Applying a source restarts live
/// detection on it; switching the detector stops the source, reloads the
/// detector and restarts. A detector that failed on its backend is the
/// screen's to show, with the other backend as the explicit way out; nothing
/// switches by itself.
///
/// Applies run one at a time, in order; one that arrives while another runs
/// waits, and the newest pending choice wins (field by field). Within one:
/// the camera source first ([applySource]: saved, a start still connecting
/// is stopped, live detection restarted on it), then the detector backend
/// ([setDetectorBackend]: source stopped, detector reloaded, source
/// restarted) — never both at once. Applying the same source again after a
/// failure reconnects.
///
/// Live detection is the screen's: this applier starts and stops it through
/// the closures it is given, so the screen's own start command shows every
/// start, whoever runs it (Retry, an Apply, the "Run detector on CPU"
/// action).
///
/// Owned by the screen's view model, which [close]s it when it leaves; every
/// wait here ends then.
final class LiveSettingsApplier {
  LiveSettingsApplier({
    required this._start,
    required this._runningStart,
    required this._stop,
    required this._needsReconnect,
    required this._onSwitch,
    required this._onChanged,
    this._settings,
    this._reloadDetector,
  }) {
    applySource = Command1<void, CameraSourceChoice>(_applySource);
    setDetectorBackend = Command1<void, DetectorBackend>(_setDetectorBackend);
  }

  /// Starts live detection as the screen's start command, or joins the
  /// start in progress; completes when that start ended. The start reads
  /// the saved source then.
  final Future<void> Function() _start;

  /// The start in progress, whoever ran it (the screen's first start, its
  /// Retry, a switch); null when none runs.
  final Future<void>? Function() _runningStart;

  /// Stops live detection for this screen; a start still connecting is
  /// aborted (it then ends at once).
  final Future<void> Function() _stop;

  /// The last start failed, or live detection failed since: the same source
  /// applied again reconnects.
  final bool Function() _needsReconnect;

  /// A switch is about to interrupt the live view (its choice was saved).
  final void Function() _onSwitch;

  /// [applying] or the loaded choices changed.
  final void Function() _onChanged;

  /// Null: nothing can be chosen (every switch refuses).
  final LiveCameraSettingsRepository? _settings;

  /// `ModelRepository.reloadDetector`; null: the detector cannot switch.
  final Future<Result<void>> Function()? _reloadDetector;

  CameraSourceChoice? _sourceChoice;
  DetectorBackendChoice? _backendChoice;

  /// Bumped when a step sets [_sourceChoice] / [_backendChoice]: a [load]
  /// that began before keeps that choice (its read may predate the save).
  int _sourceVersion = 0;
  int _backendVersion = 0;
  String? _loadError;
  ({CameraSourceChoice? source, DetectorBackend? backend})? _pending;
  Future<void>? _applying;

  /// Completed by [close]: every wait here ends with it (a start may never
  /// end once its screen is gone, and its command no longer notifies).
  final Completer<void> _closedSignal = Completer<void>();
  bool get _closed => _closedSignal.isCompleted;

  /// Saves a camera-source choice and restarts live detection on it: a
  /// start in flight (a network camera still connecting to the old choice)
  /// is stopped and ends first; the new choice is the next start.
  ///
  /// Run by the queue ([apply]); the view model exposes it read-only. A
  /// second execute while one runs is ignored ([Command]), which would drop
  /// a queued choice.
  late final Command1<void, CameraSourceChoice> applySource;

  /// Saves where the detector runs, then stops the source, reloads the
  /// detector and restarts the source. A failed load is the detector's
  /// failure to show (with the other backend offered): nothing restarts.
  ///
  /// Run by the queue ([apply], [runDetectorOn]); the view model exposes it
  /// read-only, as [applySource].
  late final Command1<void, DetectorBackend> setDetectorBackend;

  /// The settings can be chosen here (the app has the settings repository).
  bool get available => _settings != null;

  /// What fixed the camera source (`FRAME_SOURCE=fixture`); null when the
  /// user chooses it.
  String? get sourceLock => _settings?.sourceLock;

  /// What fixed the detector backend (`DETECTOR_BACKEND=cpu`); null when the
  /// user chooses it.
  String? get backendLock => _settings?.backendLock;

  /// The user can choose the camera source.
  bool get sourceEditable => _settings != null && sourceLock == null;

  /// The user can switch the detector's backend.
  bool get canChooseBackend =>
      _settings != null && _reloadDetector != null && backendLock == null;

  /// The saved camera-source choice (the device camera until loaded).
  CameraSourceChoice get sourceChoice =>
      _sourceChoice ?? CameraSourceChoice.standard;

  /// The backend the detector loads on (the GPU until loaded).
  DetectorBackend get backend => _backendChoice?.backend ?? DetectorBackend.gpu;

  /// Why the saved settings could not be read; null when they were.
  String? get loadError => _loadError;

  /// An Apply is queued or running.
  bool get applying => _applying != null;

  /// `null` when [text] is a usable network camera URL, else what to fix.
  String? validateNetworkUrl(String text) =>
      switch (parseNetworkCameraUrl(text)) {
        Ok() => null,
        Error(:final error) => error.toString(),
      };

  /// Reads the saved choices; an unreadable one keeps its default and says
  /// why in [loadError]. A choice an Apply set while this load ran is newer
  /// than what the load read: it is kept, and so is nothing said about its
  /// read.
  Future<void> load() async {
    final settings = _settings;
    if (settings == null) return;
    final sourceVersion = _sourceVersion;
    final backendVersion = _backendVersion;
    final source = await settings.readSource();
    final backend = await settings.readBackend();
    if (_closed) return;
    final errors = <String>[];
    if (sourceVersion == _sourceVersion) {
      switch (source) {
        case Ok(:final value):
          _sourceChoice = value;
        case Error(:final error):
          errors.add('$error');
      }
    }
    if (backendVersion == _backendVersion) {
      switch (backend) {
        case Ok(:final value):
          _backendChoice = value;
        case Error(:final error):
          errors.add('$error');
      }
    }
    _loadError = errors.isEmpty ? null : errors.join(' · ');
    _onChanged();
  }

  /// Applies the settings sheet. Returns what to fix in the URL (nothing is
  /// applied then), or null. A setting the user cannot choose is left out.
  String? apply({
    required CameraSourceKind kind,
    required String networkUrl,
    required DetectorBackend backend,
  }) {
    final editable = sourceEditable;
    if (editable && kind == CameraSourceKind.network) {
      if (validateNetworkUrl(networkUrl) case final String error) return error;
    }
    _enqueue(
      source: editable
          ? CameraSourceChoice(kind: kind, networkUrl: networkUrl.trim())
          : null,
      backend: canChooseBackend ? backend : null,
    );
    return null;
  }

  /// "Run detector on CPU" (or back on the GPU): queued like an Apply.
  void runDetectorOn(DetectorBackend backend) {
    if (canChooseBackend) _enqueue(backend: backend);
  }

  void _enqueue({CameraSourceChoice? source, DetectorBackend? backend}) {
    final pending = _pending;
    _pending = (
      source: source ?? pending?.source,
      backend: backend ?? pending?.backend,
    );
    if (_applying != null && _runningStart() != null) {
      // The running Apply waits on a start (a network camera still
      // connecting to the choice now replaced): stop it, so the newest
      // choice runs next instead of after the old one's timeout.
      _stopStart();
    }
    _applying ??= _drain().whenComplete(() {
      _applying = null;
      _onChanged();
    });
    _onChanged();
  }

  Future<void> _drain() async {
    while (!_closed) {
      final request = _pending;
      if (request == null) return;
      _pending = null;
      // Compared with the loaded choices, not their defaults: before [load]
      // has read them, the saved choice is unknown and the request is
      // applied (it may differ from what is saved).
      if (request.source case final choice?) {
        final loaded = _sourceChoice;
        if (loaded == null || choice != loaded || _needsReconnect()) {
          await applySource.execute(choice);
        }
      }
      if (_closed) return;
      if (request.backend case final backend?) {
        final loaded = _backendChoice?.backend;
        if (loaded == null || backend != loaded) {
          await setDetectorBackend.execute(backend);
        }
      }
    }
  }

  Future<Result<void>> _applySource(CameraSourceChoice choice) async {
    final settings = _settings;
    if (settings == null) {
      return const Result.error(
        FrameSourceUnavailableException(
          'The camera source cannot be chosen here',
        ),
      );
    }
    final saved = await settings.saveSource(choice);
    if (saved is Error<void>) return saved;
    _sourceChoice = choice;
    _sourceVersion++;
    if (_closed) return saved;
    _onSwitch();
    await _abortStart();
    if (_closed) return saved;
    await _untilClosed(_start());
    return saved;
  }

  Future<Result<void>> _setDetectorBackend(DetectorBackend backend) async {
    final settings = _settings;
    final reload = _reloadDetector;
    if (settings == null || reload == null) {
      return const Result.error(
        FrameSourceUnavailableException(
          'The detector backend cannot be chosen',
        ),
      );
    }
    final saved = await settings.saveBackend(backend);
    if (saved is Error<void>) return saved;
    _backendChoice = DetectorBackendChoice(
      backend: backend,
      source: DetectorChoiceSource.setting,
    );
    _backendVersion++;
    if (_closed) return saved;
    _onSwitch();
    await _abortStart();
    // Stop the frame source (its frame in flight comes back first), reload
    // the detector (the old worker ends), then start the source again.
    await _untilClosed(_stop());
    if (_closed) return saved;
    debugPrint('[LiveCamera] reloading the detector on ${backend.name}');
    final reloaded = await reload();
    if (_closed) return reloaded;
    // A failed load shows as the detector's failure, with the other backend
    // offered; starting would only add "the detector is not loaded".
    if (reloaded is Ok<void>) await _untilClosed(_start());
    return reloaded;
  }

  /// Stops a start still in progress (the repository aborts a source that
  /// is still connecting) and waits for it to end.
  Future<void> _abortStart() async {
    final running = _runningStart();
    if (running == null) return;
    _stopStart();
    await _untilClosed(running);
  }

  void _stopStart() => unawaited(
    _stop().catchError(
      (Object e, StackTrace st) =>
          debugPrint('[LiveCamera] stopping a start in progress: $e\n$st'),
    ),
  );

  /// Waits for [future], or until [close].
  Future<void> _untilClosed(Future<void> future) =>
      waitFor(future, closed: _closedSignal.future);

  /// The screen left: no queued step runs any more, a running one ends at
  /// its next step, and every wait ends now.
  void close() {
    if (_closed) return;
    _closedSignal.complete();
    applySource.dispose();
    setDetectorBackend.dispose();
  }
}
