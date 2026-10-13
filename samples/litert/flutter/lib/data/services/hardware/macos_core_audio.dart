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

import 'dart:ffi';

import 'package:ffi/ffi.dart';

/// `AudioObjectPropertyAddress` (CoreAudio/AudioHardwareBase.h).
final class _Address extends Struct {
  @Uint32()
  external int selector;
  @Uint32()
  external int scope;
  @Uint32()
  external int element;
}

typedef _GetPropertyC = Int32 Function(
  Uint32,
  Pointer<_Address>,
  Uint32,
  Pointer<Void>,
  Pointer<Uint32>,
  Pointer<Void>,
);
typedef _GetPropertyDart = int Function(
  int,
  Pointer<_Address>,
  int,
  Pointer<Void>,
  Pointer<Uint32>,
  Pointer<Void>,
);
typedef _CFStringGetCStringC = Uint8 Function(
  Pointer<Void>,
  Pointer<Utf8>,
  Long,
  Uint32,
);
typedef _CFStringGetCStringDart = int Function(
  Pointer<Void>,
  Pointer<Utf8>,
  int,
  int,
);
typedef _CFReleaseC = Void Function(Pointer<Void>);
typedef _CFReleaseDart = void Function(Pointer<Void>);

const _kAudioObjectSystemObject = 1;
const _kAudioObjectUnknown = 0;
const _kDefaultInputDevice = 0x64496E20; // 'dIn '
const _kDefaultOutputDevice = 0x644F7574; // 'dOut'
const _kObjectName = 0x6C6E616D; // 'lnam' (kAudioObjectPropertyName)
const _kScopeGlobal = 0x676C6F62; // 'glob'
const _kElementMain = 0;
const _kCFStringEncodingUTF8 = 0x08000100;

/// The system default input or output device's name, from Core Audio over
/// FFI (`AudioObjectGetPropertyData`). macOS only; works in the App Sandbox
/// (it reads properties, it opens nothing).
final class CoreAudioDefaults {
  const CoreAudioDefaults();

  static final _coreAudio = DynamicLibrary.open(
    '/System/Library/Frameworks/CoreAudio.framework/CoreAudio',
  );
  static final _cf = DynamicLibrary.open(
    '/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation',
  );
  static final _get = _coreAudio
      .lookupFunction<_GetPropertyC, _GetPropertyDart>(
        'AudioObjectGetPropertyData',
      );
  static final _cfString = _cf
      .lookupFunction<_CFStringGetCStringC, _CFStringGetCStringDart>(
        'CFStringGetCString',
      );
  static final _cfRelease = _cf.lookupFunction<_CFReleaseC, _CFReleaseDart>(
    'CFRelease',
  );

  /// Null when the system has no such default device. Throws [StateError]
  /// when Core Audio answers with an error status.
  String? defaultDeviceName({required bool input}) {
    final address = calloc<_Address>();
    final size = calloc<Uint32>();
    final id = calloc<Uint32>();
    final name = calloc<Pointer<Void>>();
    try {
      address.ref
        ..selector = input ? _kDefaultInputDevice : _kDefaultOutputDevice
        ..scope = _kScopeGlobal
        ..element = _kElementMain;
      size.value = sizeOf<Uint32>();
      final status = _get(
        _kAudioObjectSystemObject,
        address,
        0,
        nullptr,
        size,
        id.cast(),
      );
      if (status != 0) {
        throw StateError('Core Audio default device query: OSStatus $status');
      }
      if (id.value == _kAudioObjectUnknown) return null;
      address.ref.selector = _kObjectName;
      size.value = sizeOf<Pointer<Void>>();
      final nameStatus = _get(id.value, address, 0, nullptr, size, name.cast());
      if (nameStatus != 0 || name.value == nullptr) {
        return 'Core Audio device ${id.value}';
      }
      try {
        const capacity = 512;
        final buffer = calloc<Uint8>(capacity);
        try {
          final ok = _cfString(
            name.value,
            buffer.cast(),
            capacity,
            _kCFStringEncodingUTF8,
          );
          return ok == 0
              ? 'Core Audio device ${id.value}'
              : buffer.cast<Utf8>().toDartString();
        } finally {
          calloc.free(buffer);
        }
      } finally {
        _cfRelease(name.value);
      }
    } finally {
      calloc
        ..free(address)
        ..free(size)
        ..free(id)
        ..free(name);
    }
  }
}
