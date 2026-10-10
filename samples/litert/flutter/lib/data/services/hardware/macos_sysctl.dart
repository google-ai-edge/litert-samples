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

/// `sysctlbyname` reads (macOS). Tests pass a map.
abstract interface class Sysctl {
  /// A string value (`machdep.cpu.brand_string`); null when absent.
  String? string(String name);

  /// An integer value of 4 or 8 bytes (`hw.memsize`); null when absent.
  int? integer(String name);
}

typedef _SysctlByNameC = Int32 Function(
  Pointer<Utf8>,
  Pointer<Void>,
  Pointer<Size>,
  Pointer<Void>,
  Size,
);
typedef _SysctlByNameDart = int Function(
  Pointer<Utf8>,
  Pointer<Void>,
  Pointer<Size>,
  Pointer<Void>,
  int,
);

/// The real `sysctlbyname` over FFI. Works inside the App Sandbox (the
/// design measured `hw.*` and `machdep.*` in a sandboxed bundle).
final class FfiSysctl implements Sysctl {
  const FfiSysctl();

  static final _sysctlbyname = DynamicLibrary.process()
      .lookupFunction<_SysctlByNameC, _SysctlByNameDart>('sysctlbyname');

  @override
  String? string(String name) {
    final cName = name.toNativeUtf8(allocator: calloc);
    final size = calloc<Size>();
    try {
      if (_sysctlbyname(cName, nullptr, size, nullptr, 0) != 0) return null;
      final length = size.value;
      if (length == 0) return '';
      final buffer = calloc<Uint8>(length);
      try {
        if (_sysctlbyname(cName, buffer.cast(), size, nullptr, 0) != 0) {
          return null;
        }
        // The value ends in a NUL that `size` counts.
        var end = size.value;
        while (end > 0 && buffer[end - 1] == 0) {
          end--;
        }
        return buffer.cast<Utf8>().toDartString(length: end);
      } finally {
        calloc.free(buffer);
      }
    } finally {
      calloc
        ..free(cName)
        ..free(size);
    }
  }

  @override
  int? integer(String name) {
    final cName = name.toNativeUtf8(allocator: calloc);
    final size = calloc<Size>()..value = 8;
    final buffer = calloc<Int64>();
    try {
      if (_sysctlbyname(cName, buffer.cast(), size, nullptr, 0) != 0) {
        return null;
      }
      return switch (size.value) {
        4 => buffer.cast<Int32>().value,
        8 => buffer.value,
        _ => null,
      };
    } finally {
      calloc
        ..free(cName)
        ..free(size)
        ..free(buffer);
    }
  }
}
