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

import '../../../utils/result.dart';
import 'settings_store.dart';

/// A persisted setting: its key and its value type.
sealed class Setting<T extends Object> {
  const Setting(this.key);

  final String key;
}

final class StringSetting extends Setting<String> {
  const StringSetting(super.key);
}

final class BoolSetting extends Setting<bool> {
  const BoolSetting(super.key);
}

/// Every setting the app persists. Installs upgraded from an earlier build
/// may still hold `models.manifestUrl` (a remote model manifest that earlier
/// builds read): nothing reads it.
abstract final class Settings {
  /// Which chat model the demos use: `bundled` or `custom`
  /// (`ChatModelKind.name`). Unset is `bundled`.
  static const chatModel = StringSetting('chat.model');

  /// The user's own chat model (`CustomChatModelCodec.encode`), kept while
  /// the bundled one is chosen so switching back loses nothing.
  static const customChatModel = StringSetting('chat.custom');

  /// Demo 3's camera: `device` or `network` (`CameraSourceKind.name`). Unset
  /// is the device camera. `FRAME_SOURCE=fixture|network` wins over it.
  static const cameraSource = StringSetting('camera.source');

  /// Demo 3's network camera URL (an MJPEG stream), kept while the device
  /// camera is chosen.
  static const networkCameraUrl = StringSetting('camera.networkUrl');

  /// Where the detector runs: `gpu` or `cpu` (`DetectorBackend.name`). Unset
  /// is the GPU. `DETECTOR_BACKEND` wins over it.
  static const detectorBackend = StringSetting('detector.backend');
}

/// Reads and writes [Setting]s: a typed store over [SettingsStore], shared by
/// the repositories that own settings (`ChatModelRepository`,
/// `LiveCameraSettingsRepository`). A storage failure is a [Result.error],
/// never a silent default.
class TypedSettings {
  TypedSettings({required this._store});

  final SettingsStore _store;

  Future<Result<T?>> read<T extends Object>(Setting<T> setting) =>
      _guard('read ${setting.key}', () async {
        final Object? value = switch (setting) {
          StringSetting(:final key) => await _store.getString(key),
          BoolSetting(:final key) => await _store.getBool(key),
        };
        return value as T?;
      });

  Future<Result<void>> write<T extends Object>(Setting<T> setting, T value) =>
      _guard('write ${setting.key}', () async {
        switch (setting) {
          case StringSetting(:final key):
            await _store.setString(key, value as String);
          case BoolSetting(:final key):
            await _store.setBool(key, value: value as bool);
        }
      });

  Future<Result<void>> clear(Setting<Object> setting) =>
      _guard('clear ${setting.key}', () => _store.remove(setting.key));

  /// Writes [first], then [second]. When [second] fails, [first] is put
  /// back as it was (cleared when it was unset), so a failed save leaves
  /// the old pair rather than half of the new one. The caller orders the
  /// pair so that the settings after the first write alone are a complete
  /// configuration as well (each key is stored as before: no new format).
  Future<Result<void>> writePair<A extends Object, B extends Object>(
    (Setting<A>, A) first,
    (Setting<B>, B) second,
  ) async {
    final (firstSetting, firstValue) = first;
    final (secondSetting, secondValue) = second;
    // Unreadable: nothing valid to put back; the write goes ahead.
    final previous = await read(firstSetting);
    final wroteFirst = await write(firstSetting, firstValue);
    if (wroteFirst is Error<void>) return wroteFirst;
    final wroteSecond = await write(secondSetting, secondValue);
    if (wroteSecond is Ok<void>) return wroteSecond;
    final restored = switch (previous) {
      Ok(value: final value?) => await write(firstSetting, value),
      Ok() => await clear(firstSetting),
      Error(:final error) => Result<void>.error(error),
    };
    if (restored case Error(:final error)) {
      debugPrint(
        '[Settings] ${firstSetting.key} could not be put back after '
        '${secondSetting.key} failed ($error): the saved pair stays half '
        'new until the next save',
      );
    }
    return wroteSecond;
  }

  static Future<Result<T>> _guard<T>(
    String what,
    Future<T> Function() op,
  ) async {
    try {
      return Result.ok(await op());
    } catch (e, st) {
      debugPrint('[Settings] $what failed: $e\n$st');
      return Result.error(asException(e));
    }
  }
}
