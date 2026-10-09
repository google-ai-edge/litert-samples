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
    default:
      return NpuUnavailable(
        'no NPU dispatch stack ships for $operatingSystem',
        soc: soc,
      );
  }
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
