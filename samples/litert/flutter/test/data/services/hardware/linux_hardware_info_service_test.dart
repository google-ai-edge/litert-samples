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

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/services/hardware/linux_hardware_info_service.dart';
import 'package:litert_edge_demos/domain/hardware/accelerator_inference.dart';
import 'package:litert_edge_demos/domain/hardware/device_summary.dart';
import 'package:litert_edge_demos/domain/hardware/diagnostics_report.dart';
import 'package:litert_edge_demos/domain/models/accelerator_evidence.dart';
import 'package:litert_edge_demos/domain/models/hardware_profile.dart';

import '../../../support/hardware_trees.dart';

Future<HardwareProfile> _probe(HardwareTree tree) => LinuxHardwareInfoService(
  files: tree.systemFiles,
  processes: tree.processes,
  libcVersion: () => tree.glibc,
  architecture: tree.architecture,
).probe();

void main() {
  test(
    'T4 VM: Xeon, Ubuntu, glibc, the T4 from /proc/driver/nvidia (not '
    'twice), Vulkan T4 + llvmpipe; a GPU load is inferred on the T4',
    () async {
      final p = await _probe(t4Vm);
      expect(p.platform, HostPlatform.linux);
      expect(p.os, 'Ubuntu 22.04.5 LTS');
      expect(p.kernel, '6.8.0-1069-gcp');
      expect(p.libc, 'glibc 2.35');
      expect(p.machine, 'Google Compute Engine');
      expect(p.cpu.model, 'Intel(R) Xeon(R) CPU @ 2.00GHz');
      expect(p.cpu.cores, 4);
      expect(p.cpu.presentCores, isNull);
      expect(p.cpu.architecture, 'x86_64');
      expect(p.memory!.totalBytes, 15358316 * 1024);
      expect(p.memory!.availableBytes, 14309088 * 1024);
      expect(p.gpus, hasLength(1), reason: 'PCI 00:04.0 is the same T4');
      expect(p.gpus.single.name, 'Tesla T4');
      expect(p.gpus.single.driver, '580.82.07');
      expect(p.gpus.single.kind, GpuKind.discrete);
      expect(p.gpus.single.inferred, isFalse);
      expect(
        [for (final d in p.vulkan!) d.name],
        ['Tesla T4', 'llvmpipe (LLVM 15.0.7, 256 bits)'],
      );
      expect([for (final d in p.vulkan!) d.isSoftware], [false, true]);
      expect(p.vulkan!.first.apiVersion, '1.4.312');
      expect(p.jetson, isNull);
      expect(p.notes, [
        contains('TFLite interpreter needs x86_64 and glibc 2.38'),
        startsWith(
          'No sound server: PulseAudio/PipeWire is not running (pactl info: '
          'Connection failure: Connection refused)',
        ),
      ], reason: 'glibc 2.35: the detector check uses LiteRT CPU');
      expect(p.audio!.server, isNull);
      expect(p.audio!.alsaCards, 0);

      final e = inferEvidence(requested: 'gpu', reported: 'gpu', hardware: p);
      expect(e.api, 'WebGPU/Vulkan');
      expect(e.apiSource, EvidenceSource.inferred);
      expect(e.adapter, 'Tesla T4');
      expect(e.adapterSource, EvidenceSource.inferred);
      expect(e.softwareGpu, isFalse);
    },
  );

  test('without the NVIDIA driver the PCI scan names the T4 from the id '
      'table', () async {
    final files = Map.of(t4Vm.files)
      ..removeWhere((path, _) => path.startsWith('/proc/driver/nvidia'));
    final p = await _probe(
      HardwareTree(files: files, glibc: '2.35', architecture: 'x86_64'),
    );
    expect(p.gpus.single.name, 'NVIDIA Tesla T4');
    expect(p.gpus.single.source, 'PCI 10de:1eb8');
    expect(p.vulkan, isNull);
    expect(p.vulkanNote, contains('vulkaninfo not installed'));
    expect(p.vulkanNote, contains('lvp_icd.x86_64.json, nvidia_icd.json'));
  });

  test('Jetson Orin Nano: device-tree model, tegra234, L4T 36.4.0 → JetPack '
      '6.1, nvpmodel 15W, 6× Cortex-A78AE, the Orin iGPU', () async {
    final p = await _probe(jetsonOrinNano);
    expect(p.machine, 'NVIDIA Jetson Orin Nano Developer Kit');
    expect(p.cpu.model, '6× Cortex-A78AE');
    expect(p.cpu.cores, 6);
    expect(p.cpu.architecture, 'arm64');
    final j = p.jetson!;
    expect(j.model, 'NVIDIA Jetson Orin Nano Developer Kit');
    expect(j.soc, 'tegra234');
    expect(j.l4tRelease, '36.4.0');
    expect(j.jetpack, '6.1');
    expect(j.powerMode, '15W');
    expect(p.gpus.single.name, 'NVIDIA Orin iGPU (Ampere)');
    expect(p.gpus.single.inferred, isTrue);
    expect(p.gpus.single.kind, GpuKind.integrated);
    expect(p.vulkan!.single.name, 'NVIDIA Tegra Orin (nvgpu)');
    expect(p.memory!.totalBytes, 7802384 * 1024);
    expect(p.notes, [contains('TFLite interpreter')]);
    expect(p.audio!.server!.name, 'pulseaudio');
    expect(p.audio!.defaultSource!.name, 'USB Audio Device Mono');
    expect(p.audio!.defaultSink!.name, 'USB Audio Device Analog Stereo');
    expect(p.audio!.alsaCards, 2);

    final e = inferEvidence(requested: 'gpu', reported: 'gpu', hardware: p);
    expect(e.adapter, 'NVIDIA Tegra Orin (nvgpu)', reason: 'Vulkan decides');
  });

  test('Jetson in a low-power mode: cores offline, nvpmodel missing → its '
      'status file; an unknown L4T keeps the major JetPack line', () async {
    final files = Map.of(jetsonOrinNano.files)
      ..['/proc/cpuinfo'] = jetsonOrinNano.files['/proc/cpuinfo']!
          .split('\n\n')
          .take(4)
          .join('\n\n')
      ..['/var/lib/nvpmodel/status'] = 'pmode:0001 fmode:quiet\n'
      ..['/etc/nv_tegra_release'] =
          '# R36 (release), REVISION: 5.0, GCID: 1, BOARD: generic\n';
    final p = await _probe(
      HardwareTree(files: files, glibc: '2.35', architecture: 'arm64'),
    );
    expect(p.cpu.cores, 4);
    expect(p.cpu.presentCores, 6);
    expect(p.jetson!.powerMode, 'mode 1');
    expect(p.jetson!.l4tRelease, '36.5.0');
    expect(p.jetson!.jetpack, '6.x');
    expect(p.vulkan, isNull);
    expect(p.gpus.single.name, 'NVIDIA Orin iGPU (Ampere)');
    final e = inferEvidence(requested: 'gpu', reported: 'gpu', hardware: p);
    expect(e.adapter, 'NVIDIA Orin iGPU (Ampere)');
  });

  test('arm64 cloud VM: 4× Neoverse-N1, no GPU, no vulkaninfo; notes say '
      'so', () async {
    final p = await _probe(arm64CloudVm);
    expect(p.cpu.model, '4× Neoverse-N1');
    expect(p.machine, 'Google Compute Engine');
    expect(p.gpus, isEmpty, reason: 'the NVMe controller is not a GPU');
    expect(p.vulkan, isNull);
    expect(p.vulkanNote, contains('no ICD files'));
    expect(p.jetson, isNull);
    expect(p.notes, [
      contains('TFLite interpreter'),
      startsWith('No GPU found'),
      startsWith('pactl did not run: install pulseaudio-utils'),
    ]);
    final e = inferEvidence(requested: 'cpu', reported: 'cpu', hardware: p);
    expect(e.api, 'CPU');
    expect(e.adapter, isNull);
    expect(e.isError, isFalse);
  });

  test('llvmpipe only: flagged as a software GPU; a GPU load inferred on it '
      'is an error', () async {
    final p = await _probe(llvmpipeOnlyVm);
    expect(p.os, 'Ubuntu 24.04.3 LTS');
    expect(p.libc, 'glibc 2.39');
    expect(p.vulkan!.single.isSoftware, isTrue);
    expect(p.gpus, isEmpty, reason: 'a software Vulkan device is no GPU');
    expect(p.notes, [
      startsWith('Vulkan lists only a software device (llvmpipe'),
      startsWith('No GPU found'),
    ], reason: 'glibc 2.39 on x86_64: no TFLite note');
    final e = inferEvidence(requested: 'gpu', reported: 'gpu', hardware: p);
    expect(e.adapter, 'llvmpipe (LLVM 19.1.1, 256 bits)');
    expect(e.softwareGpu, isTrue);
    expect(e.isError, isTrue);
  });

  test(
    'no vulkaninfo and no GPU: the adapter stays unknown (a software GPU '
    'cannot be ruled out); only lavapipe installed → software, inferred',
    () async {
      final arm = await _probe(arm64CloudVm);
      expect(arm.vulkanIcds, isEmpty);
      expect(arm.notes, contains(contains('unknown without vulkaninfo')));
      final unknown = inferEvidence(
        requested: 'gpu',
        reported: 'gpu',
        hardware: arm,
      );
      expect(unknown.adapter, isNull);
      expect(unknown.adapterUnknown, isTrue);
      expect(unknown.softwareGpu, isFalse, reason: 'no evidence either way');

      final lvpOnly = await _probe(
        HardwareTree(
          files: llvmpipeOnlyVm.files,
          glibc: '2.39',
          architecture: 'x86_64',
        ),
      );
      expect(lvpOnly.vulkan, isNull);
      expect(lvpOnly.vulkanIcds, ['lvp_icd.x86_64.json']);
      expect(lvpOnly.notes.first, startsWith('Only software Vulkan drivers'));
      final e = inferEvidence(
        requested: 'gpu',
        reported: 'gpu',
        hardware: lvpOnly,
      );
      expect(e.adapter, contains('software Vulkan only'));
      expect(e.softwareGpu, isTrue);
    },
  );

  test(
    'Arduino VENTUNO Q (QCS8275): no PCI or DRM GPU ids, so the Adreno 623 '
    'comes from Vulkan (Mesa turnip), marked inferred; llvmpipe never counts',
    () async {
      final p = await _probe(ventunoQ);
      expect(p.os, 'Ubuntu 24.04.4 LTS');
      expect(p.kernel, '6.8.0-1080-qcom');
      expect(p.libc, 'glibc 2.39');
      expect(p.machine, isNull, reason: 'no DMI on the board, not a Jetson');
      expect(p.cpu.model, '4× Cortex-A78C + 4× Cortex-A55');
      expect(p.cpu.cores, 8);
      expect(formatBytes(p.memory!.totalBytes), '14.93 GB');
      expect(
        [for (final d in p.vulkan!) d.name],
        ['Adreno623', 'llvmpipe (LLVM 20.1.2, 128 bits)'],
      );

      final gpu = p.gpus.single;
      expect(gpu.name, 'Adreno623');
      expect(gpu.source, 'vulkaninfo');
      expect(gpu.inferred, isTrue);
      expect(gpu.kind, GpuKind.integrated);
      expect(gpu.driver, 'Mesa 25.2.8-0ubuntu0.24.04.2');
      expect(gpu.api, 'Vulkan', reason: 'the version is on the Vulkan line');
      expect(p.notes, [
        contains('TFLite interpreter'),
        startsWith('pactl did not run: install pulseaudio-utils'),
      ], reason: 'a GPU was found: no "No GPU found" note');

      // The report and the "This device" card say where the name came from.
      expect(deviceLines(p).where((l) => l.startsWith('gpu ')), [
        'gpu        Adreno623 · Vulkan · driver '
            'Mesa 25.2.8-0ubuntu0.24.04.2 · inferred from vulkaninfo',
      ]);
      final device = buildDeviceSummary(
        hardware: p,
        hardwareNote: null,
        models: const [],
      ).device;
      final card = device.where((l) => l.label == 'GPU').single;
      expect(card.value, 'Adreno623 · Vulkan');
      expect(card.tag, 'inferred');
      expect(card.tone, SummaryTone.caution);
      expect(
        [for (final l in device.where((l) => l.label == 'Vulkan')) l.value],
        ['Adreno623 · 1.3.318'],
        reason: 'the version once, on the Vulkan line; llvmpipe left out',
      );

      final e = inferEvidence(requested: 'gpu', reported: 'gpu', hardware: p);
      expect(e.adapter, 'Adreno623');
      expect(e.softwareGpu, isFalse);
    },
  );

  test('musl: no gnu_get_libc_version → not glibc, with a note', () async {
    final p = await LinuxHardwareInfoService(
      files: arm64CloudVm.systemFiles,
      processes: arm64CloudVm.processes,
      libcVersion: () => null,
      architecture: 'arm64',
    ).probe();
    expect(p.libc, 'not glibc');
    expect(p.notes.first, contains('not glibc'));
  });

  group('parsers', () {
    test('arm parts: known names, unknown spelled out', () {
      expect(armCpuName(0x41, 0xd42), 'Cortex-A78AE');
      expect(armCpuName(0x41, 0xd0c), 'Neoverse-N1');
      expect(armCpuName(0x4e, 0x004), 'Carmel');
      expect(armCpuName(0x41, 0xfff), 'CPU part 0xfff (implementer 0x41)');
    });

    test('big.LITTLE is grouped in order', () {
      final text = [
        for (final part in [0xd05, 0xd05, 0xd0b, 0xd0b])
          'processor\t: 0\nCPU implementer\t: 0x41\nCPU part\t: 0x${part.toRadixString(16)}\n',
      ].join('\n');
      expect(parseCpuInfo(text).model, '2× Cortex-A55 + 2× Cortex-A76');
    });

    test('CPU lists, os-release fallbacks, driver versions', () {
      expect(countCpuList('0-5\n'), 6);
      expect(countCpuList('0-3,6-7'), 6);
      expect(countCpuList('0'), 1);
      expect(
        parseOsRelease('NAME="Debian GNU/Linux"\nVERSION_ID="12"\n'),
        'Debian GNU/Linux 12',
      );
      expect(
        parseNvidiaDriverVersion(
          'NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  '
          '570.133.20  Release Build',
        ),
        '570.133.20',
      );
    });

    test('old vulkaninfo prints apiVersion as a number and the version', () {
      final devices = parseVulkanSummary(
        'Devices:\n========\nGPU0:\n'
        '\tapiVersion     = 4206847 (1.3.255)\n'
        '\tdeviceType     = PHYSICAL_DEVICE_TYPE_CPU\n'
        '\tdeviceName     = llvmpipe (LLVM 15.0.7, 256 bits)\n',
      );
      expect(devices.single.apiVersion, '1.3.255');
      expect(devices.single.isSoftware, isTrue);
    });

    test('L4T → JetPack', () {
      String? jp(String rev) =>
          parseTegraRelease('# R36 (release), REVISION: $rev, GCID: 1')
              ?.jetpack;
      expect(jp('4.0'), '6.1');
      expect(jp('4.3'), '6.2');
      expect(jp('3.0'), '6.0');
      expect(
        parseTegraRelease('# R35 (release), REVISION: 4.1, GCID: 1')!.jetpack,
        '5.1.2',
      );
      expect(parseTegraRelease('garbage'), isNull);
    });

    group('gpusFromVulkan', () {
      const adreno = VulkanDevice(
        name: 'Adreno623',
        type: 'INTEGRATED_GPU',
        apiVersion: '1.3.318',
        driver: 'Mesa 25.2.8-0ubuntu0.24.04.2',
      );
      const discrete = VulkanDevice(name: 'AMD Radeon', type: 'DISCRETE_GPU');
      const llvmpipe = VulkanDevice(
        name: 'llvmpipe (LLVM 20.1.2, 128 bits)',
        type: 'CPU',
      );
      const swiftShader = VulkanDevice(
        name: 'SwiftShader Device (Subzero)',
        type: 'OTHER',
      );

      test('no GPU found: each hardware Vulkan device, inferred, with its '
          'kind, driver and API', () {
        final gpus = gpusFromVulkan(const [], const [adreno, discrete]);
        expect([for (final g in gpus) g.name], ['Adreno623', 'AMD Radeon']);
        expect(gpus.every((g) => g.inferred && g.source == 'vulkaninfo'), true);
        expect(
          [for (final g in gpus) g.kind],
          [GpuKind.integrated, GpuKind.discrete],
        );
        expect(gpus.first.driver, 'Mesa 25.2.8-0ubuntu0.24.04.2');
        expect([for (final g in gpus) g.api], ['Vulkan', 'Vulkan']);
      });

      test('a GPU listed twice under the same name counts once', () {
        const proprietary = VulkanDevice(
          name: 'Adreno623',
          type: 'INTEGRATED_GPU',
          driver: 'Qualcomm',
        );
        final gpus = gpusFromVulkan(const [], const [adreno, proprietary]);
        expect(gpus.single.driver, 'Mesa 25.2.8-0ubuntu0.24.04.2');
      });

      test('software devices never count (llvmpipe by type, SwiftShader by '
          'name)', () {
        expect(
          gpusFromVulkan(const [], const [llvmpipe, swiftShader]),
          isEmpty,
        );
      });

      test('nothing when a real GPU was found or vulkaninfo did not run', () {
        final t4 = pciGpu(0x10de, 0x1eb8);
        expect(gpusFromVulkan([t4], const [adreno]), isEmpty);
        expect(gpusFromVulkan(const [], null), isEmpty);
      });

      test(
        'a VM display adapter is no GPU: the Vulkan device still counts',
        () {
          final qemu = pciGpu(0x1234, 0x1111);
          expect(
            [
              for (final g in gpusFromVulkan([qemu], const [adreno])) g.name,
            ],
            ['Adreno623'],
          );
        },
      );
    });

    test('VM display adapters are not GPUs for compute', () {
      final qemu = pciGpu(0x1234, 0x1111);
      expect(qemu.kind, GpuKind.virtualDisplay);
      expect(pciGpu(0x10de, 0x27b8).name, 'NVIDIA L4');
      expect(pciGpu(0x1002, 0x1638).name, 'AMD GPU 0x1638');
    });
  });
}
