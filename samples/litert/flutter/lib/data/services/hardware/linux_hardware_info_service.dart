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

import '../../../domain/models/hardware_profile.dart';
import 'hardware_info_service.dart';
import 'libc.dart';
import 'linux_audio_probe.dart';
import 'memory_probe.dart' show parseMeminfo;
import 'system_access.dart';

/// Linux: `/proc`, `/sys`, `/etc`, `vulkaninfo`, `nvpmodel`, `pactl` and glibc.
/// Every read goes through [SystemFiles] / [ProcessRunner], so tests run on
/// fake trees (a T4 VM, a Jetson, an arm64 VM, llvmpipe only, a Qualcomm board
/// whose GPU only Vulkan names).
final class LinuxHardwareInfoService implements HardwareInfoService {
  LinuxHardwareInfoService({
    this._files = const LocalSystemFiles(),
    this._processes = const LocalProcessRunner(),
    this._libcVersion = glibcVersion,
    String? architecture,
  }) : _architecture = architecture ?? _currentArchitecture();

  final SystemFiles _files;
  final ProcessRunner _processes;
  final String? Function() _libcVersion;
  final String _architecture;

  static String _currentArchitecture() => switch (Abi.current()) {
    Abi.linuxX64 => 'x86_64',
    Abi.linuxArm64 => 'arm64',
    final other => other.toString(),
  };

  @override
  Future<HardwareProfile> probe() async {
    final notes = <String>[];

    final cpuText = _files.read('/proc/cpuinfo');
    final present = switch (_files.read('/sys/devices/system/cpu/present')) {
      final text? => countCpuList(text),
      null => null,
    };
    final cpu = cpuText == null
        ? CpuInfo(model: 'unknown', cores: 0, architecture: _architecture)
        : parseCpuInfo(cpuText, architecture: _architecture, present: present);
    if (cpuText == null) notes.add('/proc/cpuinfo is not readable.');

    final memText = _files.read('/proc/meminfo');
    final mem = memText == null ? null : parseMeminfo(memText);
    final memory = mem?.total == null
        ? null
        : MemoryInfo(totalBytes: mem!.total!, availableBytes: mem.available);

    final os = switch (_files.read('/etc/os-release')) {
      final text? => parseOsRelease(text) ?? 'Linux',
      null => 'Linux',
    };
    final kernel = _files.read('/proc/sys/kernel/osrelease')?.trim();

    final glibc = _libcVersion();
    final libc = glibc == null ? 'not glibc' : 'glibc $glibc';
    notes.addAll(libcNotes(glibc, _architecture));

    final jetson = await _jetson();
    final machine = jetson?.model ?? _dmiMachine();

    final gpus = <GpuInfo>[];
    final nvidiaBuses = <String>{};
    final driver = switch (_files.read('/proc/driver/nvidia/version')) {
      final text? => parseNvidiaDriverVersion(text),
      null => null,
    };
    for (final entry in _files.list('/proc/driver/nvidia/gpus')) {
      final text = _files.read('/proc/driver/nvidia/gpus/$entry/information');
      if (text == null) continue;
      final info = parseNvidiaInformation(text);
      if (info == null) continue;
      nvidiaBuses.add(_normalizeBus(info.bus ?? entry));
      gpus.add(
        GpuInfo(
          name: info.name,
          source: '/proc/driver/nvidia',
          kind: GpuKind.discrete,
          driver: driver,
        ),
      );
    }
    gpus.addAll(_pciGpus(skipBuses: nvidiaBuses, driver: driver));
    if (jetson != null) {
      if (tegraGpuName(jetson.soc) case final name?) {
        gpus.add(
          GpuInfo(
            name: name,
            source: 'device tree (${jetson.soc})',
            inferred: true,
            kind: GpuKind.integrated,
          ),
        );
      }
    }

    final icds = _vulkanIcds();
    final (vulkan, vulkanNote) = await _vulkan(icds);
    gpus.addAll(gpusFromVulkan(gpus, vulkan));
    notes.addAll(gpuNotes(gpus: gpus, vulkan: vulkan, icds: icds));

    final audio = await LinuxAudioProbe(
      processes: _processes,
      files: _files,
    ).probe();
    notes.addAll(audioNotes(audio));

    return HardwareProfile(
      platform: HostPlatform.linux,
      os: os,
      kernel: kernel,
      libc: libc,
      machine: machine,
      cpu: cpu,
      memory: memory,
      gpus: List.unmodifiable(gpus),
      vulkan: vulkan,
      vulkanNote: vulkanNote,
      vulkanIcds: List.unmodifiable(icds),
      jetson: jetson,
      npuHints: _npuHints(),
      audio: audio,
      notes: List.unmodifiable(notes),
    );
  }

  /// A Jetson by its device tree or L4T release file; null elsewhere.
  Future<JetsonInfo?> _jetson() async {
    final model = _cString(_files.read('/proc/device-tree/model'));
    final compatible = _files.read('/proc/device-tree/compatible') ?? '';
    final release = _files.read('/etc/nv_tegra_release');
    final isTegra = compatible.contains('nvidia,tegra');
    if (release == null && !isTegra) return null;
    final soc = RegExp(r'nvidia,(tegra\d+)').firstMatch(compatible)?.group(1);
    final l4t = release == null ? null : parseTegraRelease(release);
    return JetsonInfo(
      model: model ?? 'NVIDIA Jetson (no device-tree model)',
      soc: soc,
      l4tRelease: l4t?.release,
      jetpack: l4t?.jetpack,
      powerMode: await _powerMode(),
    );
  }

  /// `nvpmodel -q`, else the mode number from its status file.
  Future<String?> _powerMode() async {
    final out = await _processes.run('nvpmodel', const ['-q']);
    if (out != null && out.exitCode == 0) {
      if (parseNvpmodel(out.stdout) case final mode?) return mode;
    }
    final status = _files.read('/var/lib/nvpmodel/status');
    final m = status == null ? null : RegExp(r'pmode:(\d+)').firstMatch(status);
    return m == null ? null : 'mode ${int.parse(m.group(1)!)}';
  }

  String? _dmiMachine() {
    String? clean(String? value) {
      final v = value?.trim();
      if (v == null || v.isEmpty) return null;
      if (RegExp(
        r'O\.E\.M\.|System Product Name',
        caseSensitive: false,
      ).hasMatch(v)) {
        return null;
      }
      return v;
    }

    final product = clean(_files.read('/sys/class/dmi/id/product_name'));
    final vendor = clean(_files.read('/sys/class/dmi/id/sys_vendor'));
    if (product == null) return vendor;
    if (vendor == null || product.contains(vendor)) return product;
    return '$vendor $product';
  }

  /// Display-class PCI devices (`/sys/bus/pci/devices/*/class` 0x03xxxx):
  /// works without nvidia-drm. Then DRM cards that are not PCI devices.
  List<GpuInfo> _pciGpus({
    required Set<String> skipBuses,
    required String? driver,
  }) {
    final found = <GpuInfo>[];
    final seen = <String>{};
    for (final bus in _files.list('/sys/bus/pci/devices')) {
      final base = '/sys/bus/pci/devices/$bus';
      final cls = _files.read('$base/class')?.trim();
      if (cls == null || !cls.toLowerCase().startsWith('0x03')) continue;
      final vendor = _hex(_files.read('$base/vendor'));
      final device = _hex(_files.read('$base/device'));
      if (vendor == null || device == null) continue;
      seen.add('$vendor:$device');
      if (skipBuses.contains(_normalizeBus(bus))) continue;
      found.add(
        pciGpu(vendor, device, driver: vendor == 0x10de ? driver : null),
      );
    }
    for (final card in _files.list('/sys/class/drm')) {
      if (!RegExp(r'^card\d+$').hasMatch(card)) continue;
      final base = '/sys/class/drm/$card/device';
      final vendor = _hex(_files.read('$base/vendor'));
      final device = _hex(_files.read('$base/device'));
      if (vendor == null || device == null) continue;
      if (!seen.add('$vendor:$device')) continue;
      found.add(pciGpu(vendor, device, source: 'DRM $card'));
    }
    return found;
  }

  List<String> _vulkanIcds() => [
    for (final dir in const ['/usr/share/vulkan/icd.d', '/etc/vulkan/icd.d'])
      ..._files.list(dir),
  ];

  Future<(List<VulkanDevice>?, String?)> _vulkan(List<String> icds) async {
    final icdText = icds.isEmpty ? 'no ICD files' : 'ICDs: ${icds.join(', ')}';
    final out = await _processes.run('vulkaninfo', const ['--summary']);
    if (out == null) {
      return (null, 'vulkaninfo not installed or timed out ($icdText)');
    }
    final devices = parseVulkanSummary(out.stdout);
    if (devices.isEmpty) {
      return (
        null,
        'vulkaninfo found no devices (exit ${out.exitCode}; $icdText)',
      );
    }
    return (
      List<VulkanDevice>.unmodifiable(devices),
      out.exitCode == 0 ? null : 'vulkaninfo exited with ${out.exitCode}',
    );
  }

  /// `/dev/accel/*` (the kernel's compute-accelerator class), with the driver
  /// when sysfs names it. Information only.
  List<NpuHint> _npuHints() => [
    for (final node in _files.list('/dev/accel'))
      NpuHint(switch (RegExp(
        r'DRIVER=(\S+)',
      ).firstMatch(_files.read('/sys/class/accel/$node/device/uevent') ?? '')) {
        final m? => '/dev/accel/$node (${m.group(1)})',
        null => '/dev/accel/$node',
      }),
  ];
}

/// `0000:00:04.0` → `00:04.0` (nvidia's `Bus Location` and sysfs agree
/// after this).
String _normalizeBus(String bus) {
  final parts = bus.trim().toLowerCase().split(':');
  return parts.length > 2 ? parts.sublist(parts.length - 2).join(':') : bus;
}

int? _hex(String? text) {
  final t = text?.trim().toLowerCase();
  if (t == null || t.isEmpty) return null;
  return int.tryParse(t.startsWith('0x') ? t.substring(2) : t, radix: 16);
}

/// A device-tree string: NUL-terminated.
String? _cString(String? text) {
  final t = text?.replaceAll('\u0000', '').trim();
  return t == null || t.isEmpty ? null : t;
}

/// The CPU from `/proc/cpuinfo` text. x86 names its model; arm64 lists
/// implementer and part per core, grouped here: `6× Cortex-A78AE`,
/// `4× Cortex-A76 + 4× Cortex-A55`.
CpuInfo parseCpuInfo(String text, {String? architecture, int? present}) {
  final blocks = text
      .split(RegExp(r'\n\s*\n'))
      .map(_keyValues)
      .where((b) => b.containsKey('processor'))
      .toList();
  final cores = blocks.length;
  String? model;
  final parts = <String, int>{};
  for (final b in blocks) {
    model ??= b['model name'];
    final implementer = _hex(b['CPU implementer']);
    final part = _hex(b['CPU part']);
    if (implementer != null && part != null) {
      final name = armCpuName(implementer, part);
      parts[name] = (parts[name] ?? 0) + 1;
    }
  }
  // arm64 kernels may print a useless `model name : ARMv8 Processor`; the
  // part table is better there.
  if (parts.isNotEmpty &&
      (model == null || model.startsWith('ARMv') || model.isEmpty)) {
    model = [for (final MapEntry(:key, :value) in parts.entries) '$value× $key']
        .join(' + ');
  }
  return CpuInfo(
    model: model ?? 'unknown',
    cores: cores,
    presentCores: present != null && present > cores ? present : null,
    architecture: architecture,
  );
}

Map<String, String> _keyValues(String block) => {
  for (final line in block.split('\n'))
    if (line.indexOf(':') case final i when i > 0)
      line.substring(0, i).trim(): line.substring(i + 1).trim(),
};

/// Arm cores by `CPU implementer` / `CPU part` (Arm TRM part numbers);
/// unknown pairs are spelled out rather than guessed.
String armCpuName(int implementer, int part) {
  const arm = {
    0xd03: 'Cortex-A53',
    0xd04: 'Cortex-A35',
    0xd05: 'Cortex-A55',
    0xd07: 'Cortex-A57',
    0xd08: 'Cortex-A72',
    0xd09: 'Cortex-A73',
    0xd0a: 'Cortex-A75',
    0xd0b: 'Cortex-A76',
    0xd0c: 'Neoverse-N1',
    0xd0d: 'Cortex-A77',
    0xd40: 'Neoverse-V1',
    0xd41: 'Cortex-A78',
    0xd42: 'Cortex-A78AE',
    0xd44: 'Cortex-X1',
    0xd46: 'Cortex-A510',
    0xd47: 'Cortex-A710',
    0xd48: 'Cortex-X2',
    0xd49: 'Neoverse-N2',
    0xd4b: 'Cortex-A78C',
    0xd4f: 'Neoverse-V2',
    0xd80: 'Cortex-A520',
    0xd81: 'Cortex-A720',
    0xd82: 'Cortex-X4',
  };
  const nvidia = {0x003: 'Denver 2', 0x004: 'Carmel'};
  const ampere = {0xac3: 'Ampere-1', 0xac4: 'Ampere-1a'};
  final hex = '0x${part.toRadixString(16).padLeft(3, '0')}';
  final name = switch (implementer) {
    0x41 => arm[part],
    0x4e => nvidia[part],
    0xc0 => ampere[part],
    _ => null,
  };
  return name ??
      'CPU part $hex (implementer '
          '0x${implementer.toRadixString(16).padLeft(2, '0')})';
}

/// Cores in a sysfs CPU list (`0-5`, `0-3,6-7`).
int? countCpuList(String text) {
  var n = 0;
  for (final range in text.trim().split(',')) {
    if (range.isEmpty) continue;
    final ends = range.split('-').map(int.tryParse).toList();
    if (ends.any((e) => e == null)) return null;
    n += ends.length == 1 ? 1 : ends[1]! - ends[0]! + 1;
  }
  return n == 0 ? null : n;
}

/// `PRETTY_NAME` (or `NAME VERSION_ID`) from `/etc/os-release`.
String? parseOsRelease(String text) {
  final values = <String, String>{};
  for (final line in text.split('\n')) {
    final i = line.indexOf('=');
    if (i <= 0) continue;
    var v = line.substring(i + 1).trim();
    if (v.length >= 2 && (v.startsWith('"') || v.startsWith("'"))) {
      v = v.substring(1, v.length - 1);
    }
    values[line.substring(0, i).trim()] = v;
  }
  return values['PRETTY_NAME'] ??
      [values['NAME'], values['VERSION_ID']].nonNulls.join(' ').nullIfEmpty;
}

/// `Model:` and `Bus Location:` from `/proc/driver/nvidia/gpus/*/information`.
({String name, String? bus})? parseNvidiaInformation(String text) {
  final kv = _keyValues(text);
  final name = kv['Model'];
  if (name == null || name.isEmpty) return null;
  return (name: name, bus: kv['Bus Location']);
}

/// The kernel module version from `/proc/driver/nvidia/version`
/// (`NVRM version: NVIDIA UNIX x86_64 Kernel Module  580.82.07  …`).
String? parseNvidiaDriverVersion(String text) =>
    RegExp(r'Kernel Module(?: for \S+)?\s+(\d+\.\d+(?:\.\d+)?)')
        .firstMatch(text)
        ?.group(1);

/// The `Devices:` section of `vulkaninfo --summary`.
List<VulkanDevice> parseVulkanSummary(String text) {
  final devices = <VulkanDevice>[];
  final start = text.indexOf('Devices:');
  if (start < 0) return devices;
  for (final block in text.substring(start).split(RegExp(r'\n\s*GPU\d+:'))) {
    final kv = <String, String>{};
    for (final line in block.split('\n')) {
      final i = line.indexOf('=');
      if (i <= 0) continue;
      kv[line.substring(0, i).trim()] = line.substring(i + 1).trim();
    }
    final name = kv['deviceName'];
    if (name == null) continue;
    devices.add(
      VulkanDevice(
        name: name,
        type: (kv['deviceType'] ?? 'UNKNOWN').replaceFirst(
          'PHYSICAL_DEVICE_TYPE_',
          '',
        ),
        apiVersion: _vulkanVersion(kv['apiVersion']),
        driver: kv['driverInfo'] ?? kv['driverName'],
      ),
    );
  }
  return devices;
}

/// `1.3.255`; older vulkaninfo prints `4206847 (1.3.255)`.
String? _vulkanVersion(String? text) {
  if (text == null) return null;
  return RegExp(r'\((\d+\.\d+\.\d+)\)').firstMatch(text)?.group(1) ?? text;
}

/// `/etc/nv_tegra_release` → L4T `36.4.0` and its JetPack. Exact JetPack
/// versions only where NVIDIA's release table pairs them; otherwise the
/// major line (`6.x`).
({String release, String? jetpack})? parseTegraRelease(String text) {
  final m = RegExp(r'R(\d+)\s*\(release\),\s*REVISION:\s*(\d+)\.(\d+)')
      .firstMatch(text);
  if (m == null) return null;
  final release = '${m.group(1)}.${m.group(2)}.${m.group(3)}';
  const exact = {
    '36.4.4': '6.2.1',
    '36.4.3': '6.2',
    '36.4.0': '6.1',
    '36.3.0': '6.0',
    '35.6.1': '5.1.5',
    '35.6.0': '5.1.4',
    '35.5.0': '5.1.3',
    '35.4.1': '5.1.2',
  };
  const major = {'38': '7.x', '36': '6.x', '35': '5.x', '32': '4.x'};
  return (release: release, jetpack: exact[release] ?? major[m.group(1)]);
}

/// `NV Power Mode: 15W` from `nvpmodel -q`.
String? parseNvpmodel(String text) =>
    RegExp(r'NV Power Mode:\s*(.+)').firstMatch(text)?.group(1)?.trim();

/// The Jetson iGPU for a Tegra SoC (from the device tree; inferred).
String? tegraGpuName(String? soc) => switch (soc) {
  'tegra234' => 'NVIDIA Orin iGPU (Ampere)',
  'tegra194' => 'NVIDIA Xavier iGPU (Volta)',
  'tegra186' => 'NVIDIA TX2 iGPU (Pascal)',
  'tegra210' => 'NVIDIA Tegra X1 iGPU (Maxwell)',
  _ => null,
};

/// A PCI display device by vendor and device id: a few known data-centre
/// GPUs by name, VM display adapters marked as such, the rest by id.
GpuInfo pciGpu(int vendor, int device, {String? source, String? driver}) {
  String hex4(int v) => v.toRadixString(16).padLeft(4, '0');
  final id = '${hex4(vendor)}:${hex4(device)}';
  const named = {
    '10de:1eb8': 'NVIDIA Tesla T4',
    '10de:27b8': 'NVIDIA L4',
    '10de:2237': 'NVIDIA A10G',
    '10de:20b0': 'NVIDIA A100 SXM4 40GB',
    '10de:20b5': 'NVIDIA A100 PCIe 80GB',
    '10de:20f1': 'NVIDIA A100 PCIe 40GB',
    '10de:1db4': 'NVIDIA Tesla V100 PCIe 16GB',
    '10de:1db1': 'NVIDIA Tesla V100 SXM2 16GB',
    '10de:1bb3': 'NVIDIA Tesla P4',
    '10de:2331': 'NVIDIA H100 PCIe',
  };
  const virtualDisplays = {
    '1234:1111': 'QEMU standard VGA',
    '1af4:1050': 'virtio-gpu',
    '15ad:0405': 'VMware SVGA II',
    '1d0f:1111': 'Amazon EC2 VGA',
    '1013:00b8': 'Cirrus VGA',
    '1414:5353': 'Hyper-V video',
    '1b36:0100': 'QXL',
  };
  const vendors = {
    0x10de: 'NVIDIA',
    0x1002: 'AMD',
    0x8086: 'Intel',
    0x13b5: 'Arm',
    0x5143: 'Qualcomm',
  };
  final where = source ?? 'PCI $id';
  if (virtualDisplays[id] case final name?) {
    return GpuInfo(
      name: '$name (VM display, no compute)',
      source: where,
      kind: GpuKind.virtualDisplay,
    );
  }
  if (named[id] case final name?) {
    return GpuInfo(
      name: name,
      source: where,
      kind: GpuKind.discrete,
      driver: driver,
    );
  }
  return GpuInfo(
    name:
        '${vendors[vendor] ?? 'vendor 0x${hex4(vendor)}'} GPU 0x${hex4(device)}',
    source: where,
    driver: driver,
  );
}

/// The GPUs only Vulkan names. When the probe found no GPU of its own in
/// [found] — no NVIDIA driver, no PCI or DRM GPU with vendor and device ids,
/// no Jetson iGPU — each hardware device in [vulkan], inferred from
/// `vulkaninfo`: an SoC's GPU is often a platform device with no ids in
/// sysfs (the Adreno 623 of a Qualcomm QCS8275 board under Mesa turnip).
/// Software devices (llvmpipe, lavapipe, SwiftShader) never count, a device
/// listed twice under the same name counts once, and a VM display adapter in
/// [found] is no GPU. The API is plain `Vulkan`: the version is on the
/// device's own Vulkan line, and `device_info` speaks this one. Empty when a
/// GPU was found or `vulkaninfo` did not run.
List<GpuInfo> gpusFromVulkan(List<GpuInfo> found, List<VulkanDevice>? vulkan) {
  final hasGpu = found.any(
    (g) => g.kind != GpuKind.virtualDisplay && g.kind != GpuKind.software,
  );
  if (hasGpu || vulkan == null) return const [];
  final names = <String>{};
  return [
    for (final d in vulkan)
      if (!d.isSoftware && names.add(d.name))
        GpuInfo(
          name: d.name,
          source: 'vulkaninfo',
          inferred: true,
          kind: switch (d.type) {
            'INTEGRATED_GPU' => GpuKind.integrated,
            'DISCRETE_GPU' => GpuKind.discrete,
            _ => GpuKind.unknown,
          },
          driver: d.driver,
          api: 'Vulkan',
        ),
  ];
}

/// What the C library means for this app: the LiteRT-LM native bundle's Linux
/// libraries need glibc 2.35; the detector's TFLite interpreter check needs
/// 2.38 and x86_64, else it is verified against LiteRT CPU.
List<String> libcNotes(String? glibc, String architecture) {
  if (glibc == null) {
    return const [
      'The C library is not glibc (musl?): the LiteRT-LM native libraries '
          'need glibc 2.35 or newer.',
    ];
  }
  final v = _version(glibc);
  return [
    if (v != null && _below(v, const [2, 35]))
      'glibc $glibc is older than 2.35: the LiteRT-LM native libraries will '
          'not load.',
    if (architecture != 'x86_64' || (v != null && _below(v, const [2, 38])))
      'The TFLite interpreter needs x86_64 and glibc 2.38: the detector is '
          'verified against LiteRT CPU instead.',
  ];
}

/// Warnings about what a GPU backend would open.
List<String> gpuNotes({
  required List<GpuInfo> gpus,
  required List<VulkanDevice>? vulkan,
  List<String> icds = const [],
}) {
  final realGpus = gpus.where(
    (g) => g.kind != GpuKind.virtualDisplay && g.kind != GpuKind.software,
  );
  final softwareIcdsOnly = icds.isNotEmpty && icds.every(isSoftwareVulkanIcd);
  return [
    if (vulkan != null &&
        vulkan.isNotEmpty &&
        vulkan.every((d) => d.isSoftware))
      'Vulkan lists only a software device '
          '(${vulkan.map((d) => d.name).join(', ')}): a GPU backend would run '
          'on the CPU.'
    else if (vulkan == null && realGpus.isEmpty && softwareIcdsOnly)
      'Only software Vulkan drivers are installed (${icds.join(', ')}): a GPU '
          'backend would run on the CPU (inferred; vulkaninfo did not run).',
    if (realGpus.isEmpty &&
        (vulkan == null || vulkan.every((d) => d.isSoftware)))
      vulkan == null
          ? 'No GPU found (no NVIDIA driver, no PCI or DRM GPU); Vulkan devices '
                'are unknown without vulkaninfo (apt install vulkan-tools).'
          : 'No GPU found (no NVIDIA driver, no PCI or DRM GPU, no hardware '
                'Vulkan device).',
  ];
}

List<int>? _version(String text) {
  final parts = text.split('.').map(int.tryParse).toList();
  return parts.any((p) => p == null) ? null : parts.cast<int>();
}

bool _below(List<int> v, List<int> min) {
  for (var i = 0; i < min.length; i++) {
    final x = i < v.length ? v[i] : 0;
    if (x != min[i]) return x < min[i];
  }
  return false;
}

extension on String {
  String? get nullIfEmpty => isEmpty ? null : this;
}
