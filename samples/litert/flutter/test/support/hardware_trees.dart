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

// Fake /proc, /sys and /etc trees plus canned tool output for the Linux
// hardware probe: a GCP T4 VM (pactl installed, no sound server), a Jetson
// Orin Nano (L4T R36.4, PulseAudio, USB headset), an arm64 cloud VM without a
// GPU or pactl, an x86 VM where Vulkan lists only llvmpipe (PipeWire), and an
// Arduino VENTUNO Q (Qualcomm QCS8275), whose Adreno only Vulkan names.
// The texts follow the real files' layout (tabs, NULs in the device tree).

import 'package:litert_edge_demos/data/services/hardware/system_access.dart';

/// A file tree from a path → text map; directories are implied by paths.
final class FakeSystemFiles implements SystemFiles {
  const FakeSystemFiles(this.files);

  final Map<String, String> files;

  @override
  String? read(String path) => files[path];

  @override
  List<String> list(String dir) {
    final prefix = dir.endsWith('/') ? dir : '$dir/';
    return {
      for (final path in files.keys)
        if (path.startsWith(prefix))
          path.substring(prefix.length).split('/').first,
    }.toList()..sort();
  }
}

/// Canned output per `executable arg…`; anything else is "not installed".
final class FakeProcessRunner implements ProcessRunner {
  const FakeProcessRunner(this.outputs);

  final Map<String, ProcessOutput> outputs;

  @override
  Future<ProcessOutput?> run(
    String executable,
    List<String> arguments, {
    Duration timeout = const Duration(seconds: 10),
    Map<String, String>? environment,
  }) async => outputs[[executable, ...arguments].join(' ')];
}

/// One test machine.
final class const HardwareTree({
  required final Map<String, String> files,
  final Map<String, ProcessOutput> tools = const {},
  required final String? glibc,
  required final String architecture,
}) {
  FakeSystemFiles get systemFiles => FakeSystemFiles(files);
  FakeProcessRunner get processes => FakeProcessRunner(tools);
}

String _cpuinfoX86(String model, int cores) => [
  for (var i = 0; i < cores; i++)
    'processor\t: $i\n'
        'vendor_id\t: GenuineIntel\n'
        'cpu family\t: 6\n'
        'model\t\t: 85\n'
        'model name\t: $model\n'
        'cpu MHz\t\t: 2000.170\n'
        'cache size\t: 39424 KB\n'
        'flags\t\t: fpu vme de pse tsc msr pae avx2 avx512f\n',
].join('\n');

String _cpuinfoArm(int part, int cores) =>
    _cpuinfoArmParts([for (var i = 0; i < cores; i++) part]);

/// One `processor` block per entry of [parts] (Arm part numbers), in order.
String _cpuinfoArmParts(List<int> parts) => [
  for (final (i, part) in parts.indexed)
    'processor\t: $i\n'
        'BogoMIPS\t: 62.50\n'
        'Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics\n'
        'CPU implementer\t: 0x41\n'
        'CPU architecture: 8\n'
        'CPU variant\t: 0x0\n'
        'CPU part\t: 0x${part.toRadixString(16)}\n'
        'CPU revision\t: 1\n',
].join('\n');

String _meminfo(int totalKb, int availableKb) =>
    'MemTotal:       $totalKb kB\n'
    'MemFree:          812344 kB\n'
    'MemAvailable:   $availableKb kB\n'
    'Buffers:           98232 kB\n'
    'Cached:          2147200 kB\n';

String _osRelease(String pretty, String version) =>
    'PRETTY_NAME="$pretty"\n'
    'NAME="Ubuntu"\n'
    'VERSION_ID="$version"\n'
    'ID=ubuntu\n';

String _vulkanDevice(
  int index, {
  required String name,
  required String type,
  required String api,
  required String driverInfo,
  String vendorId = '0x10de',
  String deviceId = '0x1eb8',
  String driverId = 'DRIVER_ID_UNKNOWN',
  String driverName = 'driver',
}) =>
    'GPU$index:\n'
    '\tapiVersion         = $api\n'
    '\tdriverVersion      = 0.0.1\n'
    '\tvendorID           = $vendorId\n'
    '\tdeviceID           = $deviceId\n'
    '\tdeviceType         = PHYSICAL_DEVICE_TYPE_$type\n'
    '\tdeviceName         = $name\n'
    '\tdriverID           = $driverId\n'
    '\tdriverName         = $driverName\n'
    '\tdriverInfo         = $driverInfo\n'
    '\tconformanceVersion = 1.3.8.0\n';

String _vulkanSummary(List<String> devices) =>
    '==========\n'
    'VULKANINFO\n'
    '==========\n\n'
    'Vulkan Instance Version: 1.3.204\n\n\n'
    'Instance Extensions: count = 20\n'
    '-------------------------------\n'
    'VK_KHR_surface : extension revision 25\n\n'
    'Devices:\n'
    '========\n'
    '${devices.join()}';

const _llvmpipe = 'llvmpipe (LLVM 15.0.7, 256 bits)';

/// GCP n1-standard-4 with a Tesla T4, driver 580, Ubuntu 22.04.5. No
/// nvidia-drm: the PCI scan still sees the T4.
final t4Vm = HardwareTree(
  glibc: '2.35',
  architecture: 'x86_64',
  files: {
    '/proc/cpuinfo': _cpuinfoX86('Intel(R) Xeon(R) CPU @ 2.00GHz', 4),
    '/sys/devices/system/cpu/present': '0-3\n',
    '/proc/meminfo': _meminfo(15358316, 14309088),
    '/etc/os-release': _osRelease('Ubuntu 22.04.5 LTS', '22.04'),
    '/proc/sys/kernel/osrelease': '6.8.0-1069-gcp\n',
    '/sys/class/dmi/id/product_name': 'Google Compute Engine\n',
    '/sys/class/dmi/id/sys_vendor': 'Google\n',
    '/proc/driver/nvidia/version':
        'NVRM version: NVIDIA UNIX x86_64 Kernel Module  580.82.07  '
        'Fri Aug 29 17:58:12 UTC 2025\n'
        'GCC version:  gcc version 12.3.0 (Ubuntu 12.3.0-1ubuntu1~22.04)\n',
    '/proc/driver/nvidia/gpus/0000:00:04.0/information':
        'Model: \t\t Tesla T4\n'
        'IRQ:   \t\t 35\n'
        'GPU UUID: \t GPU-6b1c1a8e-0000-0000-0000-000000000000\n'
        'Video BIOS: \t 90.04.96.00.02\n'
        'Bus Type: \t PCIe\n'
        'Bus Location: \t 0000:00:04.0\n'
        'Device Minor: \t 0\n',
    '/sys/bus/pci/devices/0000:00:03.0/class': '0x010000\n',
    '/sys/bus/pci/devices/0000:00:03.0/vendor': '0x1af4\n',
    '/sys/bus/pci/devices/0000:00:03.0/device': '0x1004\n',
    '/sys/bus/pci/devices/0000:00:04.0/class': '0x030200\n',
    '/sys/bus/pci/devices/0000:00:04.0/vendor': '0x10de\n',
    '/sys/bus/pci/devices/0000:00:04.0/device': '0x1eb8\n',
    '/usr/share/vulkan/icd.d/lvp_icd.x86_64.json': '{}',
    '/usr/share/vulkan/icd.d/nvidia_icd.json': '{}',
    '/proc/asound/cards': '--- no soundcards ---\n',
  },
  tools: {
    'vulkaninfo --summary': ProcessOutput(
      exitCode: 0,
      stdout: _vulkanSummary([
        _vulkanDevice(
          0,
          name: 'Tesla T4',
          type: 'DISCRETE_GPU',
          api: '1.4.312',
          driverInfo: '580.82.07',
        ),
        _vulkanDevice(
          1,
          name: _llvmpipe,
          type: 'CPU',
          api: '1.3.255',
          driverInfo: 'Mesa 23.2.1-1ubuntu3.1~22.04.3 (LLVM 15.0.7)',
        ),
      ]),
    ),
    'pactl info': const ProcessOutput(
      exitCode: 1,
      stdout: '',
      stderr: 'Connection failure: Connection refused\n',
    ),
  },
);

/// NVIDIA Jetson Orin Nano Developer Kit, JetPack 6.1 (L4T R36.4.0), 15 W.
final jetsonOrinNano = HardwareTree(
  glibc: '2.35',
  architecture: 'arm64',
  files: {
    '/proc/cpuinfo': _cpuinfoArm(0xd42, 6),
    '/sys/devices/system/cpu/present': '0-5\n',
    '/proc/meminfo': _meminfo(7802384, 6214048),
    '/etc/os-release': _osRelease('Ubuntu 22.04.5 LTS', '22.04'),
    '/proc/sys/kernel/osrelease': '5.15.148-tegra\n',
    '/proc/device-tree/model': 'NVIDIA Jetson Orin Nano Developer Kit\u0000',
    '/proc/device-tree/compatible':
        'nvidia,p3768-0000+p3767-0005\u0000nvidia,p3767-0005\u0000'
        'nvidia,tegra234\u0000',
    '/etc/nv_tegra_release':
        '# R36 (release), REVISION: 4.0, GCID: 37537400, BOARD: generic, '
        'EABI: aarch64, DATE: Fri Sep 13 04:36:44 UTC 2024\n'
        '# KERNEL_VARIANT: oot\n'
        'TARGET_USERSPACE_LIB_DIR=nvidia\n',
    '/usr/share/vulkan/icd.d/nvidia_icd.json': '{}',
    '/proc/asound/cards': _jetsonCards,
  },
  tools: {
    'nvpmodel -q': const ProcessOutput(
      exitCode: 0,
      stdout: 'NV Power Mode: 15W\n0\n',
    ),
    'vulkaninfo --summary': ProcessOutput(
      exitCode: 0,
      stdout: _vulkanSummary([
        _vulkanDevice(
          0,
          name: 'NVIDIA Tegra Orin (nvgpu)',
          type: 'INTEGRATED_GPU',
          api: '1.3.280',
          driverInfo: '540.4.0',
        ),
      ]),
    ),
    ...jetsonPulseAudio,
  },
);

/// GCP t2a-standard-4 (Ampere Altra, Neoverse-N1): no GPU, no Vulkan tools.
final arm64CloudVm = HardwareTree(
  glibc: '2.35',
  architecture: 'arm64',
  files: {
    '/proc/cpuinfo': _cpuinfoArm(0xd0c, 4),
    '/sys/devices/system/cpu/present': '0-3\n',
    '/proc/meminfo': _meminfo(16365544, 15214380),
    '/etc/os-release': _osRelease('Ubuntu 22.04.5 LTS', '22.04'),
    '/proc/sys/kernel/osrelease': '6.8.0-1015-gcp\n',
    '/sys/class/dmi/id/product_name': 'Google Compute Engine\n',
    '/sys/class/dmi/id/sys_vendor': 'Google\n',
    '/sys/bus/pci/devices/0000:00:01.0/class': '0x010802\n',
    '/sys/bus/pci/devices/0000:00:01.0/vendor': '0x1ae0\n',
    '/sys/bus/pci/devices/0000:00:01.0/device': '0x001f\n',
    '/proc/asound/cards': '--- no soundcards ---\n',
  },
);

/// An x86 VM without a GPU but with Mesa's Vulkan: llvmpipe only.
final llvmpipeOnlyVm = HardwareTree(
  glibc: '2.39',
  architecture: 'x86_64',
  files: {
    '/proc/cpuinfo': _cpuinfoX86('AMD EPYC 7B12', 2),
    '/proc/meminfo': _meminfo(8131208, 7340032),
    '/etc/os-release': _osRelease('Ubuntu 24.04.3 LTS', '24.04'),
    '/proc/sys/kernel/osrelease': '6.14.0-1017-gcp\n',
    '/usr/share/vulkan/icd.d/lvp_icd.x86_64.json': '{}',
    '/proc/asound/cards': _intelHdaCard,
  },
  tools: {
    'vulkaninfo --summary': ProcessOutput(
      exitCode: 0,
      stdout: _vulkanSummary([
        _vulkanDevice(
          0,
          name: 'llvmpipe (LLVM 19.1.1, 256 bits)',
          type: 'CPU',
          api: '1.4.305',
          driverInfo: 'Mesa 25.0.7-0ubuntu0.24.04.2 (LLVM 19.1.1)',
        ),
      ]),
    ),
    ...pipewireDesktopAudio,
  },
);

/// Arduino VENTUNO Q (Qualcomm Dragonwing IQ8, QCS8275): Ubuntu 24.04.4,
/// kernel 6.8.0-1080-qcom, 4× Cortex-A78C + 4× Cortex-A55, 16 GB (14.93 GB
/// visible), as its self-test report shows it. The Adreno 623
/// is a platform device: its DRM card has no PCI vendor or device id, so
/// only Vulkan (Mesa turnip) names it; llvmpipe is listed too. No pactl
/// (pulseaudio-utils missing), one ALSA card. The card's text is a stand-in:
/// the probe only counts the cards.
final ventunoQ = HardwareTree(
  glibc: '2.39',
  architecture: 'arm64',
  files: {
    '/proc/cpuinfo': _cpuinfoArmParts([
      for (var i = 0; i < 4; i++) 0xd4b,
      for (var i = 0; i < 4; i++) 0xd05,
    ]),
    '/sys/devices/system/cpu/present': '0-7\n',
    '/proc/meminfo': _meminfo(15654840, 14512292),
    '/etc/os-release': _osRelease('Ubuntu 24.04.4 LTS', '24.04'),
    '/proc/sys/kernel/osrelease': '6.8.0-1080-qcom\n',
    '/proc/device-tree/model':
        'Qualcomm Technologies, Inc. Monaco Monza addons\u0000',
    '/sys/bus/pci/devices/0000:00:00.0/class': '0x060400\n',
    '/sys/bus/pci/devices/0000:00:00.0/vendor': '0x17cb\n',
    '/sys/bus/pci/devices/0000:00:00.0/device': '0x0115\n',
    '/sys/class/drm/card0/dev': '226:0\n',
    '/sys/class/drm/renderD128/dev': '226:128\n',
    '/usr/share/vulkan/icd.d/freedreno_icd.aarch64.json': '{}',
    '/usr/share/vulkan/icd.d/lvp_icd.aarch64.json': '{}',
    '/proc/asound/cards':
        ' 0 [Board          ]: qcs8275 - VENTUNO Q\n'
        '                      VENTUNO Q\n',
  },
  tools: {
    'vulkaninfo --summary': ProcessOutput(
      exitCode: 0,
      stdout: _vulkanSummary([
        _vulkanDevice(
          0,
          name: 'Adreno623',
          type: 'INTEGRATED_GPU',
          api: '1.3.318',
          driverInfo: 'Mesa 25.2.8-0ubuntu0.24.04.2',
          vendorId: '0x5143',
          deviceId: '0x6020300',
          driverId: 'DRIVER_ID_MESA_TURNIP',
          driverName: 'turnip Mesa driver',
        ),
        _vulkanDevice(
          1,
          name: 'llvmpipe (LLVM 20.1.2, 128 bits)',
          type: 'CPU',
          api: '1.4.318',
          driverInfo: 'Mesa 25.2.8-0ubuntu0.24.04.2 (LLVM 20.1.2)',
          vendorId: '0x10005',
          deviceId: '0x0000',
          driverId: 'DRIVER_ID_MESA_LLVMPIPE',
          driverName: 'llvmpipe',
        ),
      ]),
    ),
  },
);

const _intelHdaCard =
    ' 0 [Intel          ]: HDA-Intel - HDA Intel\n'
    '                      HDA Intel at 0xfebf0000 irq 34\n';

const _jetsonCards =
    ' 0 [HDA            ]: tegra-hda - NVIDIA Jetson Orin Nano HDA\n'
    '                      NVIDIA Jetson Orin Nano HDA at 0x3518000 irq 105\n'
    ' 1 [Device         ]: USB-Audio - USB Audio Device\n'
    '                      C-Media Electronics Inc. USB Audio Device at '
    'usb-3610000.usb-2.3, full speed\n';

/// `pactl info` in the C locale.
String pactlInfo({
  required String server,
  required String version,
  String? sink,
  String? source,
}) =>
    'Server String: /run/user/1000/pulse/native\n'
    'Library Protocol Version: 35\n'
    'Server Protocol Version: 35\n'
    'Is Local: yes\n'
    'Client Index: 98\n'
    'Tile Size: 65472\n'
    'User Name: tester\n'
    'Host Name: vm\n'
    'Server Name: $server\n'
    'Server Version: $version\n'
    'Default Sample Specification: float32le 2ch 48000Hz\n'
    'Default Channel Map: front-left,front-right\n'
    'Default Sink: ${sink ?? 'n/a'}\n'
    'Default Source: ${source ?? 'n/a'}\n'
    'Cookie: 3b8a:9f21\n';

/// One `Source #n` block of `pactl list sources` (PipeWire adds `node.name`
/// and `device.class` properties; PulseAudio does not).
String pactlSource(
  int index, {
  required String name,
  required String description,
  String? monitorOf,
  bool pipewire = true,
}) =>
    'Source #$index\n'
    '\tState: SUSPENDED\n'
    '\tName: $name\n'
    '\tDescription: $description\n'
    '\tDriver: ${pipewire ? 'PipeWire' : 'module-alsa-card.c'}\n'
    '\tSample Specification: s32le 2ch 48000Hz\n'
    '\tChannel Map: front-left,front-right\n'
    '\tOwner Module: 4294967295\n'
    '\tMute: no\n'
    '\tVolume: front-left: 65536 / 100% / 0.00 dB,   front-right: 65536 / '
    '100% / 0.00 dB\n'
    '\t        balance 0.00\n'
    '\tBase Volume: 65536 / 100% / 0.00 dB\n'
    '\tMonitor of Sink: ${monitorOf ?? 'n/a'}\n'
    '\tLatency: 0 usec, configured 0 usec\n'
    '\tFlags: HARDWARE DECIBEL_VOLUME LATENCY\n'
    '\tProperties:\n'
    '\t\tdevice.description = "$description"\n'
    '${pipewire ? '\t\tdevice.class = "${monitorOf == null ? 'sound' : 'monitor'}"\n'
              '\t\tnode.name = "$name"\n' : ''}'
    '${monitorOf == null ? '\tPorts:\n\t\tanalog-input-mic: Microphone (type: '
              'Mic, priority: 8700, availability unknown)\n'
              '\tActive Port: analog-input-mic\n' : ''}'
    '\tFormats:\n'
    '\t\tpcm\n';

/// One `Sink #n` block of `pactl list sinks`.
String pactlSink(
  int index, {
  required String name,
  required String description,
}) =>
    'Sink #$index\n'
    '\tState: SUSPENDED\n'
    '\tName: $name\n'
    '\tDescription: $description\n'
    '\tDriver: PipeWire\n'
    '\tSample Specification: s32le 2ch 48000Hz\n'
    '\tMonitor Source: $name.monitor\n'
    '\tProperties:\n'
    '\t\tdevice.description = "$description"\n';

const _hdaOut = 'alsa_output.pci-0000_00_1b.0.analog-stereo';
const _hdaIn = 'alsa_input.pci-0000_00_1b.0.analog-stereo';

/// Ubuntu 24.04 desktop VM: PipeWire, the emulated Intel HDA's output, its
/// monitor and its microphone.
final pipewireDesktopAudio = {
  'pactl info': ProcessOutput(
    exitCode: 0,
    stdout: pactlInfo(
      server: 'PulseAudio (on PipeWire 1.0.5)',
      version: '15.0.0',
      sink: _hdaOut,
      source: _hdaIn,
    ),
  ),
  'pactl list sources': ProcessOutput(
    exitCode: 0,
    stdout: [
      pactlSource(
        47,
        name: '$_hdaOut.monitor',
        description: 'Monitor of Built-in Audio Analog Stereo',
        monitorOf: _hdaOut,
      ),
      pactlSource(
        48,
        name: _hdaIn,
        description: 'Built-in Audio Analog Stereo',
      ),
    ].join('\n'),
  ),
  'pactl list sinks': ProcessOutput(
    exitCode: 0,
    stdout: pactlSink(
      46,
      name: _hdaOut,
      description: 'Built-in Audio Analog Stereo',
    ),
  ),
};

const _usbOut =
    'alsa_output.usb-C-Media_Electronics_Inc._USB_Audio_Device-00.analog-stereo';
const _usbIn =
    'alsa_input.usb-C-Media_Electronics_Inc._USB_Audio_Device-00.mono-fallback';
const _hdmiOut = 'alsa_output.platform-3510000.hda.hdmi-stereo';

/// Jetson Orin Nano devkit on L4T R36.4: PulseAudio 15.99 (no PipeWire
/// properties), a USB headset as the default, and the HDMI output.
final jetsonPulseAudio = {
  'pactl info': ProcessOutput(
    exitCode: 0,
    stdout: pactlInfo(
      server: 'pulseaudio',
      version: '15.99.1',
      sink: _usbOut,
      source: _usbIn,
    ),
  ),
  'pactl list sources': ProcessOutput(
    exitCode: 0,
    stdout: [
      pactlSource(
        0,
        name: '$_hdmiOut.monitor',
        description: 'Monitor of Jetson Orin Nano HDA Digital Stereo (HDMI)',
        monitorOf: _hdmiOut,
        pipewire: false,
      ),
      pactlSource(
        1,
        name: '$_usbOut.monitor',
        description: 'Monitor of USB Audio Device Analog Stereo',
        monitorOf: _usbOut,
        pipewire: false,
      ),
      pactlSource(
        2,
        name: _usbIn,
        description: 'USB Audio Device Mono',
        pipewire: false,
      ),
    ].join('\n'),
  ),
  'pactl list sinks': ProcessOutput(
    exitCode: 0,
    stdout: [
      pactlSink(
        0,
        name: _hdmiOut,
        description: 'Jetson Orin Nano HDA Digital Stereo (HDMI)',
      ),
      pactlSink(
        1,
        name: _usbOut,
        description: 'USB Audio Device Analog Stereo',
      ),
    ].join('\n'),
  ),
};
