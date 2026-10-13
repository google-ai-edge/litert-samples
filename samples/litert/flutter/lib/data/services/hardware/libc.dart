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

/// `gnu_get_libc_version()` (Linux): `2.35`. Null when the C library is not
/// glibc (musl has no such symbol).
String? glibcVersion() {
  try {
    final fn = DynamicLibrary.process()
        .lookupFunction<Pointer<Utf8> Function(), Pointer<Utf8> Function()>(
          'gnu_get_libc_version',
        );
    return fn().toDartString();
    // dart:ffi reports a symbol the process lacks (musl: no
    // gnu_get_libc_version) as an ArgumentError; that is the probe's answer.
    // ignore: avoid_catching_errors
  } on ArgumentError {
    return null; // symbol not found
  }
}

typedef _FopenC = Pointer<Void> Function(Pointer<Utf8>, Pointer<Utf8>);
typedef _FilenoC = Int32 Function(Pointer<Void>);
typedef _FilenoDart = int Function(Pointer<Void>);
typedef _Dup2C = Int32 Function(Int32, Int32);
typedef _Dup2Dart = int Function(int, int);

/// Points the process's fd 2 (native stderr: LiteRT's absl/glog lines) at
/// [path], truncating it: `fopen(path, "w")`, `dup2(fileno, 2)`,
/// `fclose`. Dart's stdout (fd 1) is untouched. Returns null on success,
/// else what failed.
///
/// What flutter_edge_ai_litertlm's `stream_proxy_redirect_stderr` does in
/// debug builds;
/// here for release builds, where nothing captures those lines.
String? redirectStderrTo(String path) {
  final lib = DynamicLibrary.process();
  final fopen = lib.lookupFunction<_FopenC, _FopenC>('fopen');
  final fileno = lib.lookupFunction<_FilenoC, _FilenoDart>('fileno');
  final dup2 = lib.lookupFunction<_Dup2C, _Dup2Dart>('dup2');
  final fclose = lib.lookupFunction<_FilenoC, _FilenoDart>('fclose');
  final cPath = path.toNativeUtf8();
  final mode = 'w'.toNativeUtf8();
  try {
    final file = fopen(cPath, mode);
    if (file == nullptr) return 'fopen("$path") failed';
    final fd = fileno(file);
    // fd 2 was closed at launch (`2>&-`): fopen took it, so the file already
    // is fd 2; closing the FILE would close fd 2 again.
    if (fd == 2) return null;
    final rc = dup2(fd, 2);
    // fd 2 keeps the file open on its own.
    fclose(file);
    return rc == 2 ? null : 'dup2 failed (rc=$rc)';
  } finally {
    malloc
      ..free(cPath)
      ..free(mode);
  }
}
