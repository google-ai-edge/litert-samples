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

import 'package:flutter/foundation.dart';

import '../../config/env.dart';
import '../../config/live_camera_config.dart' show standardDetectorBackend;
import '../../domain/models/camera_source.dart';
import '../../domain/models/detection.dart';
import '../../domain/models/detector_choice.dart';
import '../../domain/models/frame_source_spec.dart';
import '../../utils/redact_url.dart';
import '../../utils/result.dart';
import '../services/settings/typed_settings.dart';

/// The saved camera source is not one the app writes.
final class InvalidCameraSettingException implements Exception {
  const InvalidCameraSettingException(this.message);

  final String message;

  @override
  String toString() => message;
}

/// Demo 3's two settings, persisted: the camera source (device camera or a
/// network camera's URL) and where the detector runs (GPU or CPU). The
/// build's defines win and lock the matching setting: `FRAME_SOURCE=fixture`
/// or `network` (and an injected test source), `DETECTOR_BACKEND`.
///
/// Read-backs never guess: an unreadable or foreign value is an error the
/// screen shows, not a silent default.
class LiveCameraSettingsRepository {
  LiveCameraSettingsRepository({
    required this._settings,
    required this._environment,
    this._backendDefine = kDetectorBackend,
    DetectorBackend? standardBackend,
  }) : _standardBackend = standardBackend ?? standardDetectorBackend();

  final TypedSettings _settings;

  /// `FRAME_SOURCE` (or the source a test injected). [CameraSourceSpec]
  /// leaves the choice to the user.
  final Result<FrameSourceSpec> _environment;
  final String _backendDefine;

  /// The backend with nothing chosen ([standardDetectorBackend]).
  final DetectorBackend _standardBackend;

  /// What fixed the camera source, for the screen (`FRAME_SOURCE=fixture`);
  /// null when the user chooses it.
  String? get sourceLock => switch (_environment) {
    Ok(value: CameraSourceSpec()) => null,
    Ok(value: FixtureSourceSpec()) => 'FRAME_SOURCE=fixture',
    Ok(value: NetworkSourceSpec(:final url)) =>
      'FRAME_SOURCE=network (${networkCameraLabel(url)})',
    Error(:final error) => '$error',
  };

  /// What fixed the detector backend (`DETECTOR_BACKEND=cpu`); null when the
  /// user chooses it.
  String? get backendLock => _backendDefine.trim().isEmpty
      ? null
      : 'DETECTOR_BACKEND=${_backendDefine.trim()}';

  /// The saved camera source; the device camera when none is saved.
  Future<Result<CameraSourceChoice>> readSource() async {
    final kind = await _settings.read(Settings.cameraSource);
    final url = await _settings.read(Settings.networkCameraUrl);
    switch ((kind, url)) {
      case (Error(:final error), _) || (_, Error(:final error)):
        return Result.error(error);
      case (Ok(value: final k), Ok(value: final u)):
        final parsed = k == null
            ? CameraSourceKind.device
            : CameraSourceKind.tryParse(k);
        if (parsed == null) {
          return Result.error(
            InvalidCameraSettingException(
              'The saved camera source "$k" is not device or network: choose '
              "one in Live camera's settings",
            ),
          );
        }
        return Result.ok(
          CameraSourceChoice(
            kind: parsed,
            networkUrl: u ?? kNetworkCameraUrlExample,
          ),
        );
    }
  }

  /// Saves [choice]. A network choice must have a usable URL
  /// ([parseNetworkCameraUrl]); the URL is kept with the device camera too.
  /// It is saved as typed, a user and password in it included (each
  /// reconnect needs them; the settings sheet says so), but logged only
  /// as [redactUrl]'s form.
  ///
  /// The two keys are written so that every step is a source a launch can
  /// open, and a failed second write puts the first back: for the network
  /// camera the checked URL goes first and the choice names it last; for
  /// the device camera the choice goes first, so a URL that was never
  /// checked is not briefly the live one.
  Future<Result<void>> saveSource(CameraSourceChoice choice) async {
    final url = (Settings.networkCameraUrl, choice.networkUrl.trim());
    final kind = (Settings.cameraSource, choice.kind.name);
    final Result<void> saved;
    if (choice.kind == CameraSourceKind.network) {
      if (parseNetworkCameraUrl(choice.networkUrl) case Error(:final error)) {
        return Result.error(error);
      }
      saved = await _settings.writePair(url, kind);
    } else {
      saved = await _settings.writePair(kind, url);
    }
    if (saved is Ok<void>) {
      debugPrint(
        '[LiveCameraSettings] camera source ${choice.kind.name}'
        '${choice.kind == CameraSourceKind.network ? ' ${_safeUrl(choice.networkUrl)}' : ''}',
      );
    }
    return saved;
  }

  /// The source Demo 3 opens now: the build's (or test's) source when it
  /// fixes one, otherwise the saved choice.
  Future<Result<FrameSourceSpec>> resolveSource() async {
    if (_environment case Ok(value: CameraSourceSpec())) {
      switch (await readSource()) {
        case Error(:final error):
          return Result.error(error);
        case Ok(value: CameraSourceChoice(kind: CameraSourceKind.device)):
          return const Result.ok(CameraSourceSpec());
        case Ok(value: CameraSourceChoice(:final networkUrl)):
          return switch (parseNetworkCameraUrl(networkUrl)) {
            Ok(:final value) => Result.ok(NetworkSourceSpec(value)),
            Error(:final error) => Result.error(error),
          };
      }
    }
    return _environment;
  }

  /// The backend the next detector load uses: the define, else the saved
  /// setting, else the platform's standard ([resolveDetectorBackend]).
  Future<Result<DetectorBackendChoice>> readBackend() async {
    if (_backendDefine.trim().isNotEmpty) {
      return resolveDetectorBackend(define: _backendDefine);
    }
    return switch (await _settings.read(Settings.detectorBackend)) {
      Ok(:final value) => resolveDetectorBackend(
        define: '',
        saved: value,
        standard: _standardBackend,
      ),
      Error(:final error) => Result.error(
        InvalidDetectorSettingException(
          'Could not read the detector setting: $error',
        ),
      ),
    };
  }

  /// Saves where the detector runs from the next load on. Refused while
  /// `DETECTOR_BACKEND` fixes it.
  Future<Result<void>> saveBackend(DetectorBackend backend) async {
    if (backendLock case final lock?) {
      return Result.error(
        InvalidDetectorSettingException(
          'The detector backend is fixed by $lock',
        ),
      );
    }
    final saved = await _settings.write(Settings.detectorBackend, backend.name);
    if (saved is Ok<void>) {
      debugPrint('[LiveCameraSettings] detector backend ${backend.name}');
    }
    return saved;
  }

  /// [url] for the log ([redactUrl]: no user, password, query or fragment),
  /// normalized the way it is opened (a typed `user:pw@host` gets its
  /// `http://` first).
  static String _safeUrl(String url) => switch (parseNetworkCameraUrl(url)) {
    Ok(:final value) => redactUrl(value),
    Error() => '(not a URL)',
  };
}
