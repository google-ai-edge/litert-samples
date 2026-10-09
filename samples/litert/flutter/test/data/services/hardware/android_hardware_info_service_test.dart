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
import 'package:litert_edge_demos/data/services/hardware/android_hardware_info_service.dart';
import 'package:litert_edge_demos/data/services/hardware/android_props.dart';
import 'package:litert_edge_demos/domain/hardware/accelerator_inference.dart';
import 'package:litert_edge_demos/domain/hardware/diagnostics_report.dart';
import 'package:litert_edge_demos/domain/hardware/native_log_parser.dart';
import 'package:litert_edge_demos/domain/models/accelerator_evidence.dart';
import 'package:litert_edge_demos/domain/models/hardware_profile.dart';
import 'package:litert_edge_demos/domain/models/npu_availability.dart';
import 'package:litert_edge_demos/domain/skills/app_intent_handlers.dart';

import '../../../support/android_devices.dart';

const _fastRpc = NpuAvailable(soc: 'QTI SM8650');
const _noFastRpc = NpuUnavailable(
  'this device has no Qualcomm FastRPC (libcdsprpc.so did not open)',
);

void main() {
  test('Galaxy S24 (Snapdragon): SoC, model, Android version, RAM, the '
      'Adreno inferred from the table, Hexagon from the FastRPC check', () {
    final p = androidProfileFrom(
      propsOf(kGalaxyS24Snapdragon),
      files: phoneFiles(),
      npu: _fastRpc,
      cores: 8,
    );
    expect(p.platform, HostPlatform.android);
    expect(p.os, 'Android 14 (API 34)');
    expect(p.machine, 'Samsung SM-S921U');
    expect(p.kernel, '6.1.75-android14-11-g1234');
    expect(p.soc?.label, 'Snapdragon 8 Gen 3 (SM8650)');
    expect(p.cpu.model, 'Snapdragon 8 Gen 3 (SM8650)');
    expect(p.cpu.cores, 8);
    expect(p.cpu.architecture, 'arm64');
    expect(p.memory?.totalBytes, 7620344 * 1024);
    expect(p.memory?.availableBytes, 2903456 * 1024);
    final gpu = p.gpus.single;
    expect(gpu.name, 'Adreno 750');
    expect(gpu.inferred, isTrue);
    expect(gpu.kind, GpuKind.integrated);
    expect(gpu.source, contains('SM8650'));
    expect(p.npuHints.single.text, startsWith('Qualcomm Hexagon V75'));
    expect(p.npuHints.single.text, contains('libcdsprpc.so opens'));
    expect(p.notes.single, contains('named from the SoC, not queried'));
  });

  test('Galaxy S24 (Exynos): Xclipse inferred; the Samsung NPU is a hint '
      'that it is not used; no Qualcomm complaint', () {
    final p = androidProfileFrom(
      propsOf(kGalaxyS24Exynos),
      files: phoneFiles(),
      npu: _noFastRpc,
      cores: 10,
    );
    expect(p.machine, 'Samsung SM-S921B');
    expect(p.cpu.model, 'Exynos 2400 (s5e9945)');
    expect(p.gpus.single.name, 'Xclipse 940');
    expect(p.npuHints.single.text, startsWith('Samsung NPU (not used'));
    expect(p.notes.where((n) => n.contains('Qualcomm')), isEmpty);
  });

  test('a Qualcomm SoC whose FastRPC does not open says so in the notes', () {
    final p = androidProfileFrom(
      propsOf(kGalaxyS24Snapdragon),
      files: phoneFiles(),
      npu: _noFastRpc,
      cores: 8,
    );
    expect(p.npuHints, isEmpty);
    expect(
      p.notes,
      contains(
        'Qualcomm SoC, but the NPU is unavailable: this device has no '
        'Qualcomm FastRPC (libcdsprpc.so did not open).',
      ),
    );
  });

  test('Android 11: the SoC from the board, with a note saying so', () {
    final p = androidProfileFrom(
      propsOf(kAndroid11Kona),
      files: phoneFiles(),
      npu: _noFastRpc,
      cores: 8,
    );
    expect(p.os, 'Android 11 (API 30)');
    expect(p.machine, 'OnePlus IN2023');
    expect(p.cpu.model, 'Snapdragon 865/865+/870 (SM8250)');
    expect(p.gpus.single.name, 'Adreno 650');
    expect(p.notes.first, contains('ro.board.platform kona'));
  });

  test('a SoC the table does not know: named as Android names it, no GPU, '
      'and the note says why', () {
    final p = androidProfileFrom(
      propsOf(kMediatekPhone),
      files: phoneFiles(),
      npu: _noFastRpc,
      cores: 8,
    );
    expect(p.cpu.model, 'Mediatek MT6989');
    expect(p.machine, 'Xiaomi 24117RK2CG');
    expect(p.gpus, isEmpty);
    expect(p.npuHints, isEmpty);
    expect(
      p.notes.single,
      "SoC Mediatek MT6989 is not in the app's table: the GPU is not named "
      '(no OpenCL query).',
    );
  });

  test('getprop failed: unknown chip, the OS from dart:io, and the reason; '
      'RAM still from /proc/meminfo', () {
    final p = androidProfileFrom(
      const MapAndroidProperties({}, error: '/system/bin/sh could not start'),
      files: phoneFiles(),
      npu: _noFastRpc,
      cores: 8,
      osFallback: '14',
    );
    expect(p.cpu.model, 'unknown');
    expect(p.os, 'Android 14');
    expect(p.machine, isNull);
    expect(p.soc, isNull);
    expect(p.gpus, isEmpty);
    expect(p.memory, isNotNull);
    expect(p.notes.single, startsWith('getprop did not run (/system/bin/sh'));
  });

  test('unreadable /proc/meminfo: no RAM, and a note', () {
    final p = androidProfileFrom(
      propsOf(kGalaxyS24Snapdragon),
      files: phoneFiles(meminfo: null),
      npu: _fastRpc,
      cores: 8,
    );
    expect(p.memory, isNull);
    expect(p.notes, contains('/proc/meminfo is not readable.'));
  });

  test('a model that repeats the manufacturer is not doubled', () {
    final p = androidProfileFrom(
      propsOf(
        '[ro.product.manufacturer]: [Google]\n'
        '[ro.product.model]: [Google Pixel 8]\n'
        '[ro.soc.manufacturer]: [Google]\n'
        '[ro.soc.model]: [Tensor G3]\n',
      ),
      files: phoneFiles(),
      npu: _noFastRpc,
      cores: 9,
    );
    expect(p.machine, 'Google Pixel 8');
    expect(p.cpu.model, 'Google Tensor G3');
    expect(p.gpus.single.name, 'Mali-G715');
    expect(p.npuHints.single.text, startsWith('Google TPU (not used'));
  });

  test("the diagnostics device block shows the SoC and where it came from; "
      "Gemma's adapter is inferred from the table's GPU", () {
    final p = androidProfileFrom(
      propsOf(kGalaxyS24Snapdragon),
      files: phoneFiles(),
      npu: _fastRpc,
      cores: 8,
    );
    final lines = deviceLines(p);
    expect(lines, contains(matches(r'^os\s+Android 14 \(API 34\) · kernel 6')));
    expect(lines, contains(matches(r'^machine\s+Samsung SM-S921U$')));
    expect(
      lines,
      contains(
        matches(
          r'^soc\s+Snapdragon 8 Gen 3 \(SM8650\) · QTI SM8650 from '
          r'ro\.soc\.model$',
        ),
      ),
    );
    expect(lines, contains(matches(r'^gpu\s+Adreno 750 · inferred from')));
    expect(lines, contains(matches(r'^npu\s+Qualcomm Hexagon V75')));

    final evidence = inferEvidence(
      requested: 'gpu',
      reported: 'gpu',
      hardware: p,
      log: parseNativeLog(const []),
    );
    expect(evidence.api, 'OpenCL');
    expect(evidence.adapter, 'Adreno 750');
    expect(evidence.adapterSource, EvidenceSource.inferred);
  });

  test('device_info speaks the SoC by name, the RAM and the GPU from this '
      'profile', () {
    final p = androidProfileFrom(
      propsOf(kGalaxyS24Snapdragon),
      files: phoneFiles(),
      npu: _fastRpc,
      cores: 8,
    );
    expect(
      describeHardware(hardware: p, models: const []),
      'This phone has a Snapdragon 8 Gen 3 (SM8650) with seven point three '
      'gigabytes of memory and an Adreno 750 GPU (inferred from the chip). No model is loaded yet.',
    );
  });
}
