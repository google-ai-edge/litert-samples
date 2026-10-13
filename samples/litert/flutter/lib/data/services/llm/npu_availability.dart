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

import 'package:flutter/foundation.dart';

import '../../../domain/models/npu_availability.dart';
import '../hardware/android_props.dart';

/// The app's copy of flutter_edge_ai_litertlm 1.9.0's NPU gate
/// (`lib/src/ffi/backend_preference.dart`): `npuDispatchShipsFor` — Windows
/// yes, Android only when `libcdsprpc.so` opens (`_androidHasFastRpc`),
/// every other OS no — with the reason text of `npuUnavailableReason`.
/// Mirrored, not imported: the package keeps it in `src/` and marks the
/// reason `@visibleForTesting`.
///
/// The Models screen offers `npu` only when this says [NpuAvailable];
/// `LlmService` checks it again before a load so an NPU request on a host
/// without the stack fails with the reason instead of loading on the GPU
/// (where flutter_edge_ai would put it).
NpuAvailability npuAvailabilityFor(
  String operatingSystem, {
  required ({bool opened, String? error}) Function() openFastRpc,
  String? Function()? probeLinuxNpu,
  String? soc,
}) {
  switch (operatingSystem) {
    case 'windows':
      return NpuAvailable(soc: soc);
    case 'android':
      final probe = openFastRpc();
      if (probe.opened) return NpuAvailable(soc: soc);
      return NpuUnavailable(
        'this device has no Qualcomm FastRPC (libcdsprpc.so did not open'
        '${probe.error == null ? '' : ': ${probe.error}'}), so the bundled '
        'NPU stack cannot run here',
        soc: soc,
      );
    case 'linux':
      // flutter_edge_ai_litertlm 1.11.0 ships the Qualcomm stack for
      // linux_arm64 (`qualcomm_npu: true`); the board must expose it.
      final probe = probeLinuxNpu?.call() ?? _probeLinuxQualcommNpu();
      if (probe == null) return NpuAvailable(soc: soc);
      return NpuUnavailable(probe, soc: soc);
    default:
      return NpuUnavailable(
        'no NPU dispatch stack ships for $operatingSystem',
        soc: soc,
      );
  }
}

/// The Linux arm64 Qualcomm NPU probe of flutter_edge_ai_litertlm 1.11.0
/// (`_linuxHasQualcommNpu`): null when the stack can run, else why not.
String? _probeLinuxQualcommNpu() {
  if (Abi.current() != Abi.linuxArm64) {
    return 'the Qualcomm NPU stack is built for linux_arm64 only, and this is '
        '${Abi.current()}';
  }
  for (final node in const ['/dev/fastrpc-cdsp', '/dev/dma_heap/system']) {
    try {
      File(node).openSync().closeSync();
    } on FileSystemException catch (e) {
      final errno = e.osError?.errorCode;
      if (errno == 2 && node == '/dev/fastrpc-cdsp') {
        return 'this machine has no Qualcomm compute DSP ($node does not '
            'exist)';
      }
      if (errno == 2) continue;
      return errno == 13
          ? '$node is not accessible to this user. Add the user to group '
                'fastrpc (sudo usermod -aG fastrpc \$USER) and log in again'
          : '$node could not be opened: $e';
    }
  }
  try {
    DynamicLibrary.open('libcdsprpc.so.1');
  } on Object catch (e) {
    return "Qualcomm's FastRPC library libcdsprpc.so.1 did not open ($e); on "
        'Ubuntu it comes with the qcom-fastrpc1 package';
  }
  return null;
}

NpuAvailability? _probed;

/// [npuAvailabilityFor] on this device, probed once per process (the answer
/// cannot change): one `dlopen('libcdsprpc.so')` on Android, nothing
/// elsewhere.
NpuAvailability probeNpu() => _probed ??= () {
  final os = Platform.operatingSystem;
  final result = npuAvailabilityFor(
    os,
    openFastRpc: () {
      try {
        DynamicLibrary.open('libcdsprpc.so');
        return (opened: true, error: null);
      } on Object catch (e) {
        return (opened: false, error: '$e');
      }
    },
    soc: os == 'android' ? _androidSoc() : null,
  );
  debugPrint('[NPU] $os: ${describeNpu(result)}');
  return result;
}();

/// The SoC Android names (`QTI SM8650`), from the properties the hardware
/// probe reads too (`androidProperties`: one `getprop` run per process).
/// Null when `getprop` cannot be run or names no SoC.
String? _androidSoc() {
  final props = androidProperties();
  if (props.error case final error?) {
    debugPrint('[NPU] getprop failed: $error');
  }
  return switch (socFromProperties(props)) {
    final soc? => [?soc.manufacturer, soc.model].join(' '),
    null => null,
  };
}
