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

/// Whether the demos may hold the camera and the microphone, from the app's
/// lifecycle.
///
/// On Android and iOS only while the app is resumed: leaving it (inactive:
/// the app switcher, a call, Control Centre, a system dialog; then hidden and
/// paused) makes [value] false, and the demos release the camera and end a
/// held push-to-talk press; coming back makes it true and the live camera
/// starts again. On desktop it stays true: a window that loses focus, or is
/// hidden, keeps the camera and the microphone.
///
/// Listeners hear every lifecycle change on Android and iOS, not only the
/// changes of [value]: leaving goes inactive → hidden → paused, all out of
/// the foreground, and [state] tells an overlay (inactive: the app's own
/// permission dialog among them) from an app that is no longer seen.
///
/// Owned by the root widget, which feeds it [onStateChange] from its
/// `AppLifecycleListener` and disposes it.
final class AppForeground extends ChangeNotifier
    implements ValueListenable<bool> {
  AppForeground({TargetPlatform? platform})
    : _releasesDevices = switch (platform ?? defaultTargetPlatform) {
        TargetPlatform.android || TargetPlatform.iOS => true,
        _ => false,
      };

  /// Android and iOS: leaving the app releases the devices.
  final bool _releasesDevices;
  AppLifecycleState _state = AppLifecycleState.resumed;

  @override
  bool get value => _state == AppLifecycleState.resumed;

  /// The latest lifecycle state on Android and iOS; always resumed on
  /// desktop.
  AppLifecycleState get state => _state;

  /// Out of sight, not only covered: hidden, paused or detached (inactive is
  /// an overlay over the app, which may be the app's own dialog).
  bool get hidden => switch (_state) {
    AppLifecycleState.resumed || AppLifecycleState.inactive => false,
    AppLifecycleState.hidden ||
    AppLifecycleState.paused ||
    AppLifecycleState.detached => true,
  };

  /// The app's lifecycle moved to [state].
  void onStateChange(AppLifecycleState state) {
    if (!_releasesDevices || state == _state) return;
    _state = state;
    notifyListeners();
  }
}
