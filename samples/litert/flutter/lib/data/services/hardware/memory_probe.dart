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
import 'dart:io' show ProcessInfo;

import 'package:ffi/ffi.dart';
import 'package:flutter/foundation.dart';

import '../../../domain/models/hardware_profile.dart';
import 'macos_sysctl.dart';
import 'system_access.dart';

/// System and process memory now. On a Jetson the iGPU allocates from system
/// RAM, so the drop in [availableBytes] across a load is what a model costs;
/// the RSS alone misses GPU allocations.
abstract interface class MemoryProbe {
  /// System memory the OS can hand out now; null when unreadable.
  int? availableBytes();

  /// [availableBytes] plus this process's current and peak RSS.
  MemorySnapshot snapshot();
}

/// `MemTotal` and `MemAvailable` from `/proc/meminfo` text, in bytes.
({int? total, int? available}) parseMeminfo(String text) {
  int? field(String key) {
    final m = RegExp(
      '^$key:\\s+(\\d+)\\s*kB',
      multiLine: true,
    ).firstMatch(text);
    return m == null ? null : int.parse(m.group(1)!) * 1024;
  }

  return (total: field('MemTotal'), available: field('MemAvailable'));
}

MemorySnapshot _processSnapshot(int? available) => MemorySnapshot(
  availableBytes: available,
  rssBytes: ProcessInfo.currentRss,
  peakRssBytes: ProcessInfo.maxRss,
);

/// Linux and Android: `MemAvailable` (the kernel's estimate, page cache
/// included; `/proc/meminfo` is readable from an Android app).
final class LinuxMemoryProbe implements MemoryProbe {
  const LinuxMemoryProbe([this._files = const LocalSystemFiles()]);

  final SystemFiles _files;

  @override
  int? availableBytes() {
    final text = _files.read('/proc/meminfo');
    return text == null ? null : parseMeminfo(text).available;
  }

  @override
  MemorySnapshot snapshot() => _processSnapshot(availableBytes());
}

typedef _HostStatistics64C = Int32 Function(
  Uint32,
  Int32,
  Pointer<Int32>,
  Pointer<Uint32>,
);
typedef _HostStatistics64Dart = int Function(
  int,
  int,
  Pointer<Int32>,
  Pointer<Uint32>,
);

/// `HOST_VM_INFO64` and its count (`sizeof(vm_statistics64) / sizeof(int)`).
const _hostVmInfo64 = 4;
const _hostVmInfo64Count = 38;

/// Word offsets in `vm_statistics64`: free, inactive. `free_count` already
/// includes the speculative pages (checked against `vm_stat`).
const _freeIndex = 0;
const _inactiveIndex = 2;

/// macOS: free + inactive pages (`host_statistics64`), roughly what Activity
/// Monitor calls available.
final class MacMemoryProbe implements MemoryProbe {
  MacMemoryProbe({this._sysctl = const FfiSysctl()});

  final Sysctl _sysctl;

  static final DynamicLibrary _lib = DynamicLibrary.process();

  /// One send right for the process's lifetime (each `mach_host_self` call
  /// adds a reference).
  static final int _host = _lib
      .lookupFunction<Uint32 Function(), int Function()>('mach_host_self')();
  static final _hostStatistics64 = _lib
      .lookupFunction<_HostStatistics64C, _HostStatistics64Dart>(
        'host_statistics64',
      );

  bool _loggedFailure = false;

  @override
  int? availableBytes() {
    final pageSize = _sysctl.integer('vm.pagesize');
    if (pageSize == null) return null;
    final info = calloc<Int32>(_hostVmInfo64Count);
    final count = calloc<Uint32>()..value = _hostVmInfo64Count;
    try {
      final rc = _hostStatistics64(_host, _hostVmInfo64, info, count);
      if (rc != 0) {
        if (!_loggedFailure) {
          _loggedFailure = true;
          debugPrint('[Hardware] host_statistics64 failed (kern_return $rc)');
        }
        return null;
      }
      final pages =
          info[_freeIndex].toUnsigned(32) + info[_inactiveIndex].toUnsigned(32);
      return pages * pageSize;
    } finally {
      calloc
        ..free(info)
        ..free(count);
    }
  }

  @override
  MemorySnapshot snapshot() => _processSnapshot(availableBytes());
}

/// Elsewhere: process RSS only.
final class ProcessMemoryProbe implements MemoryProbe {
  const ProcessMemoryProbe();

  @override
  int? availableBytes() => null;

  @override
  MemorySnapshot snapshot() => _processSnapshot(null);
}

/// The probe for [platform].
MemoryProbe memoryProbeFor(HostPlatform platform) => switch (platform) {
  HostPlatform.linux || HostPlatform.android => const LinuxMemoryProbe(),
  HostPlatform.macos => MacMemoryProbe(),
  _ => const ProcessMemoryProbe(),
};
