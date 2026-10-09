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

import 'package:shared_preferences/shared_preferences.dart';

/// Persistent key-value storage for app settings. The app uses
/// [SharedPreferencesSettingsStore]; tests use an in-memory one.
abstract interface class SettingsStore {
  Future<String?> getString(String key);
  Future<bool?> getBool(String key);
  Future<void> setString(String key, String value);
  Future<void> setBool(String key, {required bool value});
  Future<void> remove(String key);
}

/// [SettingsStore] over `SharedPreferencesAsync`. Every key gets the `app.`
/// prefix: flutter_edge_ai keeps its install records in the same preferences
/// (legacy API, `flutter.` prefix on Apple platforms), and the two must
/// never collide.
final class SharedPreferencesSettingsStore implements SettingsStore {
  SharedPreferencesSettingsStore([SharedPreferencesAsync? preferences])
    : _preferences = preferences ?? SharedPreferencesAsync();

  final SharedPreferencesAsync _preferences;

  static String _key(String key) => 'app.$key';

  @override
  Future<String?> getString(String key) => _preferences.getString(_key(key));

  @override
  Future<bool?> getBool(String key) => _preferences.getBool(_key(key));

  @override
  Future<void> setString(String key, String value) =>
      _preferences.setString(_key(key), value);

  @override
  Future<void> setBool(String key, {required bool value}) =>
      _preferences.setBool(_key(key), value);

  @override
  Future<void> remove(String key) => _preferences.remove(_key(key));
}
