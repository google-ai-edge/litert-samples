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

/// What the device is: chip, RAM, OS, GPUs and their API, NPU hints. Filled
/// once by a `HardwareInfoService`; immutable.
library;

import 'audio_devices.dart';

/// The OS the app runs on; decides which GPU API a model is inferred to use.
enum HostPlatform {
  linux,
  macos,
  ios,
  android,
  windows,
  other;

  /// [name] as `Platform.operatingSystem` spells it; [other] for anything
  /// else.
  static HostPlatform fromOperatingSystem(String os) => switch (os) {
    'linux' => linux,
    'macos' => macos,
    'ios' => ios,
    'android' => android,
    'windows' => windows,
    _ => other,
  };
}

/// The CPU as the OS reports it.
final class const CpuInfo({
  /// `Apple M4 Pro`, `Intel(R) Xeon(R) CPU @ 2.00GHz`, `6× Cortex-A78AE`.
  required final String model,

  /// Online logical cores.
  required final int cores,

  /// Apple performance/efficiency cores (`hw.perflevel0/1.physicalcpu`).
  final int? performanceCores,
  final int? efficiencyCores,

  /// Linux: cores the kernel knows (`/sys/devices/system/cpu/present`), when
  /// more than [cores] (a Jetson power mode can take cores offline).
  final int? presentCores,

  /// `arm64` or `x86_64`.
  final String? architecture,
});

/// System RAM. [availableBytes]: what the OS can hand out now (Linux
/// `MemAvailable`; macOS free + inactive pages).
final class const MemoryInfo({
  required final int totalBytes,
  final int? availableBytes,
});

/// What kind of device a [GpuInfo] is.
enum GpuKind {
  discrete,
  integrated,

  /// A VM's display adapter (QEMU, virtio, VMware, EC2): no compute.
  virtualDisplay,

  /// llvmpipe, lavapipe, SwiftShader, softpipe: runs on the CPU.
  software,
  unknown,
}

/// One GPU, from whichever source found it.
final class const GpuInfo({
  /// `Tesla T4`, `Apple M4 Pro`, `NVIDIA Tegra Orin (nvgpu)`.
  required final String name,

  /// Where [name] came from: `/proc/driver/nvidia`, `PCI 10de:1eb8`,
  /// `vulkaninfo`, `device tree`, `chip name (sysctl)`.
  required final String source,

  /// True when [name] is deduced (an Apple GPU named after its chip, a Jetson
  /// iGPU from the SoC, a Linux GPU only Vulkan names because sysfs has no
  /// ids for it), not read from the GPU itself.
  final bool inferred = false,
  final GpuKind kind = GpuKind.unknown,

  /// Driver version (`580.82.07`), when known.
  final String? driver,

  /// `Metal`, `Vulkan 1.4.312`.
  final String? api,
});

/// One device from `vulkaninfo --summary`.
final class const VulkanDevice({
  required final String name,

  /// `PHYSICAL_DEVICE_TYPE_DISCRETE_GPU` without the prefix: `DISCRETE_GPU`,
  /// `INTEGRATED_GPU`, `CPU`, …
  required final String type,
  final String? apiVersion,
  final String? driver,
}) {
  /// Runs on the CPU: a CPU device type or a known software rasterizer.
  bool get isSoftware => type == 'CPU' || isSoftwareGpuName(name);
}

/// Mesa's lavapipe (`lvp_icd.*`) or SwiftShader: a Vulkan driver that runs
/// on the CPU.
bool isSoftwareVulkanIcd(String fileName) =>
    fileName.startsWith('lvp_icd') ||
    fileName.toLowerCase().contains('swiftshader');

/// llvmpipe, lavapipe, SwiftShader or softpipe: a GPU API on the CPU.
bool isSoftwareGpuName(String name) => RegExp(
  r'llvmpipe|lavapipe|swiftshader|softpipe',
  caseSensitive: false,
).hasMatch(name);

/// An NVIDIA Jetson (L4T).
final class const JetsonInfo({
  /// `/proc/device-tree/model`: `NVIDIA Jetson Orin Nano Developer Kit`.
  required final String model,

  /// The SoC from the device tree's `compatible`: `tegra234`.
  final String? soc,

  /// `/etc/nv_tegra_release`: `36.4.0`.
  final String? l4tRelease,

  /// The JetPack that ships [l4tRelease] (`6.1`), or `6.x` when only the
  /// major release is known.
  final String? jetpack,

  /// `nvpmodel -q`: `15W`, or `mode 0` from its status file.
  final String? powerMode,
});

/// An NPU the device has. Information only: no model here runs on it.
final class const NpuHint(final String text);

/// The system-on-chip the OS names (Android `ro.soc.manufacturer` and
/// `ro.soc.model`, or the board on Android 11).
final class const SocInfo({
  /// `QTI`, `Samsung`, `Google`; null when the OS does not say.
  final String? manufacturer,

  /// The part as the OS names it: `SM8650`, `s5e9945`, `Tensor G3`, or the
  /// board's part (`SM8650` from `pineapple`).
  required final String model,

  /// Where [model] came from: `ro.soc.model`, `ro.board.platform pineapple`.
  required final String source,

  /// The marketing name from the app's table (`Snapdragon 8 Gen 3`); null
  /// when the table does not know [model].
  final String? name,
}) {
  /// Android did not name the part ([source] is a board property, not
  /// `ro.soc.model`): the chip is the app's reading of the board.
  bool get inferred => source != 'ro.soc.model';

  /// `Snapdragon 8 Gen 3 (SM8650)`, `Google Tensor G3`, or `QTI SM7435`
  /// when the table does not know the part.
  String get label => switch (name) {
    null => [?manufacturer, model].join(' '),
    final n when n.toLowerCase().contains(model.toLowerCase()) => n,
    final n => '$n ($model)',
  };
}

/// Everything the probe found. Missing facts are null; why they are missing
/// (and anything that limits the app) is in [notes].
final class const HardwareProfile({
  required final HostPlatform platform,

  /// `macOS 26.5.1 (25F71)`, `Ubuntu 22.04.5 LTS`.
  required final String os,
  required final CpuInfo cpu,

  /// `6.8.0-1069-gcp`.
  final String? kernel,

  /// `glibc 2.35`; `not glibc` when the symbol is missing (musl).
  final String? libc,

  /// `Mac16,8`, `NVIDIA Jetson Orin Nano Developer Kit`,
  /// `Google Compute Engine`, `Samsung SM-S921B`.
  final String? machine,

  /// The SoC the OS names (Android); null elsewhere.
  final SocInfo? soc,
  final MemoryInfo? memory,
  final List<GpuInfo> gpus = const [],

  /// Null when `vulkaninfo` did not run (see [vulkanNote]).
  final List<VulkanDevice>? vulkan,

  /// Why [vulkan] is missing, and the ICD files found instead.
  final String? vulkanNote,

  /// Vulkan ICD manifests (`nvidia_icd.json`, `lvp_icd.x86_64.json`) in
  /// `/usr/share/vulkan/icd.d` and `/etc/vulkan/icd.d`.
  final List<String> vulkanIcds = const [],
  final JetsonInfo? jetson,
  final List<NpuHint> npuHints = const [],

  /// Linux: the sound server and its sources and sinks (`pactl`); null on
  /// the other platforms (their devices come from the audio engine).
  final AudioSystem? audio,

  /// Limits and probe problems, one sentence each.
  final List<String> notes = const [],
});

/// Process and system memory at one moment.
final class const MemorySnapshot({
  /// System memory the OS can hand out ([MemoryInfo.availableBytes]); null
  /// when unreadable.
  final int? availableBytes,
  required final int rssBytes,
  required final int peakRssBytes,
});

/// What this binary is.
final class const BuildInfo({
  required final String appVersion,

  /// `debug`, `profile` or `release`.
  required final String buildMode,

  /// Null when the flutter tool did not pass `FLUTTER_VERSION`.
  required final String? flutterVersion,
  required final String dartVersion,

  /// Package name → resolved version.
  required final Map<String, String> packages,
});
