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

// Demo 3's saved settings for the integration tests.

import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/services/settings/settings_store.dart';
import 'package:litert_edge_demos/data/services/settings/typed_settings.dart';
import 'package:litert_edge_demos/utils/result.dart';

/// Clears Demo 3's saved camera source, network camera URL and detector
/// backend. They live in the app's shared preferences, which a manual run on
/// the same machine (same bundle id) or device shares: a test that expects
/// the device camera or the GPU must not inherit "network camera" or "CPU"
/// from one. The build's defines still win over the defaults.
Future<void> resetDemo3Settings() async {
  final settings = TypedSettings(store: SharedPreferencesSettingsStore());
  for (final setting in [
    Settings.cameraSource,
    Settings.networkCameraUrl,
    Settings.detectorBackend,
  ]) {
    if (await settings.clear(setting) case Error(:final error)) {
      fail('Could not reset ${setting.key}: $error');
    }
  }
  debugPrint('[test] Demo 3 settings reset (device camera, GPU)');
}
