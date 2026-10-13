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

import '../../../domain/models/hardware_profile.dart';
import 'hardware_info_service.dart';
import 'macos_sysctl.dart';
import 'memory_probe.dart';

/// macOS: `sysctlbyname` over FFI (verified inside an App-Sandbox bundle). The
/// GPU is named after the chip (inferred): the Metal device name and the GPU
/// core count would need a Metal query from native code.
final class MacHardwareInfoService implements HardwareInfoService {
  MacHardwareInfoService({
    this._sysctl = const FfiSysctl(),
    MemoryProbe? memory,
  }) : _memory = memory ?? MacMemoryProbe();

  final Sysctl _sysctl;
  final MemoryProbe _memory;

  @override
  Future<HardwareProfile> probe() async =>
      macProfileFrom(_sysctl, availableBytes: _memory.availableBytes());
}

/// The profile from sysctl values (pure; the tests' seam).
HardwareProfile macProfileFrom(Sysctl sysctl, {int? availableBytes}) {
  final notes = <String>[];
  final brand = sysctl.string('machdep.cpu.brand_string');
  final appleSilicon = sysctl.integer('hw.optional.arm64') == 1;
  final version = sysctl.string('kern.osproductversion');
  final build = sysctl.string('kern.osversion');
  final levels = sysctl.integer('hw.nperflevels') ?? 1;
  final pCores = levels >= 2
      ? sysctl.integer('hw.perflevel0.physicalcpu')
      : null;
  final eCores = levels >= 2
      ? sysctl.integer('hw.perflevel1.physicalcpu')
      : null;
  final cores =
      sysctl.integer('hw.logicalcpu') ?? sysctl.integer('hw.ncpu') ?? 0;
  final memsize = sysctl.integer('hw.memsize');
  if (brand == null) notes.add('machdep.cpu.brand_string is not readable.');
  if (memsize == null) notes.add('hw.memsize is not readable.');
  if (availableBytes == null) {
    notes.add('Available memory (host_statistics64) is not readable.');
  }

  return HardwareProfile(
    platform: HostPlatform.macos,
    os: ['macOS ${version ?? '?'}', if (build != null) '($build)'].join(' '),
    machine: sysctl.string('hw.model'),
    cpu: CpuInfo(
      model: brand ?? 'unknown',
      cores: cores,
      performanceCores: pCores,
      efficiencyCores: eCores,
      architecture: appleSilicon ? 'arm64' : 'x86_64',
    ),
    memory: memsize == null
        ? null
        : MemoryInfo(totalBytes: memsize, availableBytes: availableBytes),
    gpus: [
      if (appleSilicon && brand != null)
        GpuInfo(
          name: brand,
          source: 'chip name (sysctl); Metal device not queried',
          inferred: true,
          kind: GpuKind.integrated,
          api: 'Metal',
        ),
    ],
    npuHints: [
      if (appleSilicon) const NpuHint('Apple Neural Engine (not used)'),
    ],
    notes: [
      ...notes,
      if (!appleSilicon)
        'Intel Mac: the GPU is not identified (its name would need a Metal '
            'device query in native code).',
    ],
  );
}
