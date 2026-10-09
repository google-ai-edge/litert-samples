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

import 'dart:ffi' show Abi;
import 'dart:io' show Platform;

import '../../../domain/hardware/soc_table.dart';
import '../../../domain/models/hardware_profile.dart';
import '../../../domain/models/npu_availability.dart';
import '../llm/npu_availability.dart';
import 'android_props.dart';
import 'hardware_info_service.dart';
import 'memory_probe.dart' show parseMeminfo;
import 'system_access.dart';

/// Android: the SoC, phone model and Android version from `getprop` (shared
/// with the NPU gate), RAM from `/proc/meminfo`, the GPU and NPU from the app's
/// SoC table (inferred: no OpenCL query), and the Hexagon hint from the NPU
/// gate's `libcdsprpc.so` check. No Kotlin.
final class AndroidHardwareInfoService implements HardwareInfoService {
  AndroidHardwareInfoService({
    this._properties,
    this._files = const LocalSystemFiles(),
    NpuAvailability Function()? npu,
  }) : _npu = npu ?? probeNpu;

  final AndroidProperties? _properties;
  final SystemFiles _files;
  final NpuAvailability Function() _npu;

  @override
  Future<HardwareProfile> probe() async {
    // getprop off the main isolate first; the NPU gate then reads the cache.
    final props = _properties ?? await loadAndroidProperties();
    return androidProfileFrom(
      props,
      files: _files,
      npu: _npu(),
      cores: Platform.numberOfProcessors,
      architecture: Abi.current() == Abi.androidX64 ? 'x86_64' : 'arm64',
      osFallback: Platform.operatingSystemVersion,
    );
  }
}

/// The profile from the properties, `/proc` and the NPU gate (pure; the
/// tests' seam). Unknown facts stay unknown, with the reason in the notes.
HardwareProfile androidProfileFrom(
  AndroidProperties props, {
  required SystemFiles files,
  required NpuAvailability npu,
  required int cores,
  String architecture = 'arm64',
  String? osFallback,
}) {
  final notes = <String>[];
  if (props.error case final error?) {
    notes.add(
      'getprop did not run ($error): the SoC, phone model and Android '
      'version are unknown.',
    );
  }

  final soc = socFromProperties(props);
  final spec = soc == null ? null : lookupSoc(soc.model);
  if (props.error == null) {
    if (soc == null) {
      notes.add(
        'Android names no SoC (ro.soc.model, ro.board.platform and '
        'ro.hardware are empty).',
      );
    } else {
      if (soc.source != 'ro.soc.model') {
        notes.add(
          'ro.soc.model is empty (Android 11 or a vendor build without it): '
          'the SoC is taken from ${soc.source}.',
        );
      }
      if (spec == null) {
        notes.add(
          'SoC ${soc.label} is not in the app\'s table: the GPU is not named '
          '(no OpenCL query).',
        );
      }
    }
  }

  final memText = files.read('/proc/meminfo');
  final mem = memText == null ? null : parseMeminfo(memText);
  if (mem?.total == null) notes.add('/proc/meminfo is not readable.');

  final gpus = [
    if ((soc, spec?.gpu) case (final soc?, final gpu?))
      GpuInfo(
        name: gpu,
        source: 'the SoC table (${soc.model}); not queried',
        inferred: true,
        kind: GpuKind.integrated,
      ),
  ];
  if (gpus.isNotEmpty) {
    notes.add(
      'The GPU is named from the SoC, not queried (no OpenCL device '
      'query).',
    );
  }

  final npuHints = <NpuHint>[];
  switch (npu) {
    case NpuAvailable():
      final hexagon = switch (spec) {
        SocSpec(vendor: SocVendor.qualcomm, :final npu?) => npu,
        _ => 'Hexagon',
      };
      npuHints.add(
        NpuHint(
          'Qualcomm $hexagon · libcdsprpc.so opens · a chat model compiled '
          'for this SoC can use it',
        ),
      );
    case NpuUnavailable(:final reason):
      if (spec case SocSpec(:final npu?, :final vendor)
          when vendor != SocVendor.qualcomm) {
        npuHints.add(
          NpuHint(
            "$npu (not used: flutter_edge_ai's NPU path needs Qualcomm "
            'FastRPC)',
          ),
        );
      }
      if (spec?.vendor == SocVendor.qualcomm ||
          soc?.manufacturer?.toUpperCase() == 'QTI') {
        notes.add('Qualcomm SoC, but the NPU is unavailable: $reason.');
      }
  }

  return HardwareProfile(
    platform: HostPlatform.android,
    os: _androidVersion(props) ?? 'Android ${osFallback ?? '?'}',
    kernel: files.read('/proc/sys/kernel/osrelease')?.trim(),
    machine: _phoneModel(props),
    soc: soc,
    cpu: CpuInfo(
      model: soc?.label ?? 'unknown',
      cores: cores,
      architecture: architecture,
    ),
    memory: mem?.total == null
        ? null
        : MemoryInfo(totalBytes: mem!.total!, availableBytes: mem.available),
    gpus: List.unmodifiable(gpus),
    npuHints: List.unmodifiable(npuHints),
    notes: List.unmodifiable(notes),
  );
}

/// `Android 14 (API 34)`; null without `ro.build.version.release`.
String? _androidVersion(AndroidProperties props) {
  final release = props['ro.build.version.release'];
  if (release == null) return null;
  final sdk = props['ro.build.version.sdk'];
  return 'Android $release${sdk == null ? '' : ' (API $sdk)'}';
}

/// `Samsung SM-S921B`, `Google Pixel 8` (the manufacturer once, even when
/// the model repeats it); null when neither is set.
String? _phoneModel(AndroidProperties props) {
  final maker = switch (props['ro.product.manufacturer']) {
    final m? when m == m.toLowerCase() =>
      '${m[0].toUpperCase()}'
          '${m.substring(1)}',
    final m => m,
  };
  final model = props['ro.product.model'];
  if (model == null) return maker;
  if (maker == null || model.toLowerCase().startsWith(maker.toLowerCase())) {
    return model;
  }
  return '$maker $model';
}
