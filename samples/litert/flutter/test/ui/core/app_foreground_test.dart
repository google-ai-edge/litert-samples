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

import 'dart:ui' show AppLifecycleState;

import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/ui/core/app_foreground.dart';

void main() {
  for (final platform in [TargetPlatform.android, TargetPlatform.iOS]) {
    group('on ${platform.name}', () {
      late AppForeground foreground;
      late List<bool> changes;

      setUp(() {
        foreground = AppForeground(platform: platform);
        changes = [];
        foreground.addListener(() => changes.add(foreground.value));
      });

      tearDown(() => foreground.dispose());

      test('in the foreground until the app is left', () {
        expect(foreground.value, isTrue);
      });

      test('inactive is already out of the foreground (the app switcher, a '
          'call), but not hidden; every state change is heard', () {
        final hidden = <bool>[];
        foreground.addListener(() => hidden.add(foreground.hidden));
        foreground
          ..onStateChange(AppLifecycleState.inactive)
          ..onStateChange(AppLifecycleState.hidden)
          ..onStateChange(AppLifecycleState.paused);
        expect(foreground.value, isFalse);
        expect(foreground.state, AppLifecycleState.paused);
        foreground
          ..onStateChange(AppLifecycleState.hidden)
          ..onStateChange(AppLifecycleState.inactive)
          ..onStateChange(AppLifecycleState.resumed);
        expect(foreground.value, isTrue);
        expect(changes, [false, false, false, false, false, true]);
        expect(hidden, [false, true, true, true, false, false]);
      });

      test('the same state twice is heard once; detached is hidden', () {
        foreground
          ..onStateChange(AppLifecycleState.detached)
          ..onStateChange(AppLifecycleState.detached);
        expect(changes, [false]);
        expect(foreground.hidden, isTrue);
      });
    });
  }

  for (final platform in [TargetPlatform.macOS, TargetPlatform.linux]) {
    test('on ${platform.name} the devices stay: losing focus, hiding or '
        'pausing never leaves the foreground', () {
      final foreground = AppForeground(platform: platform);
      addTearDown(foreground.dispose);
      var notified = 0;
      foreground.addListener(() => notified++);
      for (final state in AppLifecycleState.values) {
        foreground.onStateChange(state);
        expect(foreground.value, isTrue, reason: state.name);
        expect(foreground.hidden, isFalse, reason: state.name);
      }
      expect(notified, 0);
    });
  }
}
