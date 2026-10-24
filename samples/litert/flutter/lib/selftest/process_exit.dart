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
import 'dart:io';

typedef _GetCurrentProcessC = IntPtr Function();
typedef _GetCurrentProcessDart = int Function();
typedef _TerminateProcessC = Int32 Function(IntPtr, Uint32);
typedef _TerminateProcessDart = int Function(int, int);

/// Ends the process with [code] once stdout and stderr are flushed.
///
/// On Windows `exit()` runs every DLL's detach while the window's WinRT
/// message infrastructure is still up, and `coremessaging.dll` then fails
/// fast (0xE0464645): the process exit code is that, not [code].
/// `TerminateProcess` ends the process with [code] without the detach. Every
/// other platform uses `exit()`.
Future<Never> exitProcess(int code) async {
  await stdout.flush();
  await stderr.flush();
  if (Platform.isWindows) {
    final kernel32 = DynamicLibrary.open('kernel32.dll');
    final current = kernel32
        .lookupFunction<_GetCurrentProcessC, _GetCurrentProcessDart>(
          'GetCurrentProcess',
        );
    final terminate = kernel32
        .lookupFunction<_TerminateProcessC, _TerminateProcessDart>(
          'TerminateProcess',
        );
    terminate(current(), code);
  }
  exit(code);
}
