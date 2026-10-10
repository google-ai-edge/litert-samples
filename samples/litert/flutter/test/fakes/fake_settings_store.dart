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

import 'package:litert_edge_demos/data/services/settings/settings_store.dart';

/// [SettingsStore] in memory; [failWith] makes every call throw.
final class InMemorySettingsStore implements SettingsStore {
  final Map<String, Object> values = {};
  Exception? failWith;

  /// Keys whose writes (set and remove) throw; reads still work.
  final Set<String> failWritesOf = {};

  void _check() {
    if (failWith case final error?) throw error;
  }

  void _checkWrite(String key) {
    _check();
    if (failWritesOf.contains(key)) throw Exception('cannot write $key');
  }

  @override
  Future<String?> getString(String key) async {
    _check();
    return values[key] as String?;
  }

  @override
  Future<bool?> getBool(String key) async {
    _check();
    return values[key] as bool?;
  }

  @override
  Future<void> setString(String key, String value) async {
    _checkWrite(key);
    values[key] = value;
  }

  @override
  Future<void> setBool(String key, {required bool value}) async {
    _checkWrite(key);
    values[key] = value;
  }

  @override
  Future<void> remove(String key) async {
    _checkWrite(key);
    values.remove(key);
  }
}
