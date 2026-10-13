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
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/voice_config.dart';

void main() {
  // Every "access is off" hint names the settings path of the platform it
  // runs on (Android has no "Privacy & Security" page).
  test('microphone: macOS, iOS and Android each get their own path', () {
    expect(
      micAccessMessage(TargetPlatform.macOS),
      contains('System Settings › Privacy & Security › Microphone'),
    );
    expect(
      micAccessMessage(TargetPlatform.iOS),
      contains('Settings › Privacy & Security › Microphone'),
    );
    expect(micAccessMessage(TargetPlatform.iOS), isNot(contains('System')));
    final android = micAccessMessage(TargetPlatform.android);
    expect(android, isNot(contains('Privacy & Security')));
    expect(android, contains('Settings › Apps › LiteRT Demos'));
    expect(android, contains('Permissions › Microphone'));
  });
}
