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

// Keeps the test app's window on screen on macOS.
//
// A hidden or fully covered macOS app gets AppLifecycleState.hidden: frames
// stop, and `tester.pump` waits until the window shows again. The app keeps
// working, but the test is frozen inside one pump, and pumpUntil's wall-clock
// timeout (checked between pumps) is not enforced meanwhile, so a covered test
// hangs for as long as it stays covered. A new test app can also open behind
// the frontmost window (Chrome, another test app at the same frame).
//
// `initIntegrationTest` brings the window to the front before every test and
// logs when it is hidden later. It calls AppKit in-process over FFI (the Dart
// UI isolate runs on the main thread on macOS: merged UI/platform thread), so
// it needs no Runner change and no Apple Events entitlement in the sandbox.

import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:flutter/widgets.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

/// [IntegrationTestWidgetsFlutterBinding.ensureInitialized], plus, on macOS,
/// a [setUp] that brings the app to the front ([bringAppToFront]) and logs
/// every lifecycle change during the test (`APP_WINDOW …`), so a frozen
/// stretch is visible in the log. Elsewhere the binding only.
IntegrationTestWidgetsFlutterBinding initIntegrationTest() {
  final binding = IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  if (Platform.isMacOS) {
    setUp(() async {
      await bringAppToFront();
      final logger = _LifecycleLogger();
      WidgetsBinding.instance.addObserver(logger);
      addTearDown(() => WidgetsBinding.instance.removeObserver(logger));
    });
  }
  return binding;
}

/// Unhides the app, asks macOS to activate it, orders its windows in front
/// of every other app's, then waits until a window shows: macOS reports part
/// of one visible, or the app is active and the engine has resumed (it
/// resumes from NSWindow.isVisible on activation, because occlusionState can
/// latch stale: flutter#155977). Fails after [timeout] if it stays hidden (a
/// locked screen hides every window): every pump would wait. A no-op on
/// other platforms.
Future<void> bringAppToFront({
  Duration timeout = const Duration(seconds: 15),
}) async {
  if (!Platform.isMacOS) return;
  final app = _AppKit();
  if (!app.onMainThread) {
    fail(
      'bringAppToFront: AppKit needs the main thread, and the UI isolate '
      'is not on it (FLTEnableMergedPlatformUIThread off?)',
    );
  }
  bool resumed() =>
      app.active &&
      WidgetsBinding.instance.lifecycleState == AppLifecycleState.resumed;
  final wasVisible = app.visible;
  final windows = app.bringToFront();
  final watch = Stopwatch()..start();
  while (!app.visible && !resumed()) {
    if (watch.elapsed > timeout) {
      final state = WidgetsBinding.instance.lifecycleState?.name;
      fail(
        app.screenLocked
            ? 'The screen is locked: no window is visible, so frames stop '
                  'and tester.pump would wait until it is unlocked. Unlock '
                  'the Mac and run again.'
            : 'The app window is still hidden or covered after $timeout '
                  '(windows ordered front: $windows, active: ${app.active}, '
                  'lifecycle: $state). Frames stop while it is, and '
                  'tester.pump waits for them: uncover the window.',
      );
    }
    await Future<void>.delayed(const Duration(milliseconds: 50));
  }
  debugPrint(
    'APP_WINDOW front: was_visible=$wasVisible windows=$windows '
    'visible_after=${watch.elapsedMilliseconds}ms '
    'by=${app.visible ? 'occlusion' : 'resumed'} active=${app.active}',
  );
}

final class _LifecycleLogger with WidgetsBindingObserver {
  final _since = Stopwatch()..start();

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final previous = _since.elapsedMilliseconds;
    _since.reset();
    debugPrint(
      'APP_WINDOW lifecycle=${state.name} after ${previous}ms'
      '${state == AppLifecycleState.hidden ? ' (frames stop; pumps wait)' : ''}',
    );
  }
}

typedef _Id = Pointer<Void>;

/// The few NSApplication / NSWindow messages [bringAppToFront] sends, as
/// typed `objc_msgSend` casts (arm64 needs the exact signature per call).
final class _AppKit {
  _AppKit() {
    _app = _sendId(_class('NSApplication'), _sel('sharedApplication'));
  }

  static final _lib = DynamicLibrary.process();

  static final _getClass = _lib
      .lookupFunction<_Id Function(Pointer<Utf8>), _Id Function(Pointer<Utf8>)>(
        'objc_getClass',
      );
  static final _registerName = _lib
      .lookupFunction<_Id Function(Pointer<Utf8>), _Id Function(Pointer<Utf8>)>(
        'sel_registerName',
      );
  static final _pthreadMainNp = _lib
      .lookupFunction<Int Function(), int Function()>('pthread_main_np');
  // CoreGraphics: the login session's facts (+1 CFDictionary, or NULL).
  static final _sessionDictionary = _lib
      .lookupFunction<_Id Function(), _Id Function()>(
        'CGSessionCopyCurrentDictionary',
      );
  static final _cfRelease = _lib
      .lookupFunction<Void Function(_Id), void Function(_Id)>('CFRelease');

  static final _msgSend = _lib.lookup<Void>('objc_msgSend');
  static final _sendId = _msgSend
      .cast<NativeFunction<_Id Function(_Id, _Id)>>()
      .asFunction<_Id Function(_Id, _Id)>();
  static final _sendBool = _msgSend
      .cast<NativeFunction<Bool Function(_Id, _Id)>>()
      .asFunction<bool Function(_Id, _Id)>();
  static final _sendUlong = _msgSend
      .cast<NativeFunction<UnsignedLong Function(_Id, _Id)>>()
      .asFunction<int Function(_Id, _Id)>();
  static final _sendVoid = _msgSend
      .cast<NativeFunction<Void Function(_Id, _Id)>>()
      .asFunction<void Function(_Id, _Id)>();
  // A BOOL argument as a byte (YES = 1).
  static final _sendVoidByte = _msgSend
      .cast<NativeFunction<Void Function(_Id, _Id, Uint8)>>()
      .asFunction<void Function(_Id, _Id, int)>();
  static final _sendVoidId = _msgSend
      .cast<NativeFunction<Void Function(_Id, _Id, _Id)>>()
      .asFunction<void Function(_Id, _Id, _Id)>();
  static final _sendBoolSel = _msgSend
      .cast<NativeFunction<Bool Function(_Id, _Id, _Id)>>()
      .asFunction<bool Function(_Id, _Id, _Id)>();
  static final _sendIdAt = _msgSend
      .cast<NativeFunction<_Id Function(_Id, _Id, UnsignedLong)>>()
      .asFunction<_Id Function(_Id, _Id, int)>();
  static final _sendIdId = _msgSend
      .cast<NativeFunction<_Id Function(_Id, _Id, _Id)>>()
      .asFunction<_Id Function(_Id, _Id, _Id)>();
  static final _sendIdUtf8 = _msgSend
      .cast<NativeFunction<_Id Function(_Id, _Id, Pointer<Utf8>)>>()
      .asFunction<_Id Function(_Id, _Id, Pointer<Utf8>)>();
  static final _sendInt = _msgSend
      .cast<NativeFunction<Int Function(_Id, _Id)>>()
      .asFunction<int Function(_Id, _Id)>();

  /// NSApplicationOcclusionStateVisible.
  static const _occlusionVisible = 1 << 1;

  late final _Id _app;

  static _Id _class(String name) =>
      using((arena) => _getClass(name.toNativeUtf8(allocator: arena)));

  static _Id _sel(String name) =>
      using((arena) => _registerName(name.toNativeUtf8(allocator: arena)));

  bool get onMainThread => _pthreadMainNp() != 0;

  /// Some part of a window is on screen (what the engine maps to
  /// resumed/inactive rather than hidden).
  bool get visible =>
      (_sendUlong(_app, _sel('occlusionState')) & _occlusionVisible) != 0;

  bool get active => _sendBool(_app, _sel('isActive'));

  /// The session's CGSSessionScreenIsLocked (the dictionary is toll-free
  /// bridged to NSDictionary; the value is an NSNumber).
  bool get screenLocked {
    final session = _sessionDictionary();
    if (session == nullptr) return false;
    try {
      final key = using(
        (arena) => _sendIdUtf8(
          _class('NSString'),
          _sel('stringWithUTF8String:'),
          'CGSSessionScreenIsLocked'.toNativeUtf8(allocator: arena),
        ),
      );
      final value = _sendIdId(session, _sel('objectForKey:'), key);
      return value != nullptr && _sendInt(value, _sel('intValue')) != 0;
    } finally {
      _cfRelease(session);
    }
  }

  /// Returns how many windows were ordered front.
  int bringToFront() {
    if (_sendBool(_app, _sel('isHidden'))) {
      _sendVoid(_app, _sel('unhideWithoutActivation'));
    }
    // macOS 14+: cooperative `activate`; it may be declined while the user
    // works in another app, which is why the windows are ordered front too.
    final activate = _sel('activate');
    if (_sendBoolSel(_app, _sel('respondsToSelector:'), activate)) {
      _sendVoid(_app, activate);
    } else {
      // macOS 12-13.
      _sendVoidByte(_app, _sel('activateIgnoringOtherApps:'), 1);
    }
    final windows = _sendId(_app, _sel('windows'));
    final count = _sendUlong(windows, _sel('count'));
    var ordered = 0;
    for (var i = 0; i < count; i++) {
      final window = _sendIdAt(windows, _sel('objectAtIndex:'), i);
      if (_sendBool(window, _sel('isMiniaturized'))) {
        _sendVoidId(window, _sel('deminiaturize:'), nullptr);
      }
      if (_sendBool(window, _sel('isVisible'))) {
        // In front of its level even when the app is not active.
        _sendVoid(window, _sel('orderFrontRegardless'));
        ordered++;
      }
    }
    return ordered;
  }
}
