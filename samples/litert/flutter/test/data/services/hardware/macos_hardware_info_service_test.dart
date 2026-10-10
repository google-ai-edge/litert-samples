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

import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/services/hardware/macos_hardware_info_service.dart';
import 'package:litert_edge_demos/data/services/hardware/macos_sysctl.dart';
import 'package:litert_edge_demos/data/services/hardware/memory_probe.dart';
import 'package:litert_edge_demos/domain/hardware/accelerator_inference.dart';
import 'package:litert_edge_demos/domain/models/accelerator_evidence.dart';
import 'package:litert_edge_demos/domain/models/hardware_profile.dart';

/// sysctl values from a map (the names as measured on an M4 Pro).
final class _MapSysctl implements Sysctl {
  const _MapSysctl(this.values);

  final Map<String, Object> values;

  @override
  String? string(String name) => values[name] as String?;

  @override
  int? integer(String name) => values[name] as int?;
}

const _m4Pro = _MapSysctl({
  'machdep.cpu.brand_string': 'Apple M4 Pro',
  'hw.model': 'Mac16,8',
  'hw.optional.arm64': 1,
  'hw.memsize': 25769803776,
  'hw.nperflevels': 2,
  'hw.perflevel0.physicalcpu': 10,
  'hw.perflevel1.physicalcpu': 4,
  'hw.logicalcpu': 14,
  'kern.osproductversion': '26.5.1',
  'kern.osversion': '25F80',
});

void main() {
  test('M4 Pro: chip, model, 10P+4E, RAM; the GPU is the chip, inferred, '
      'Metal; the ANE is a hint', () {
    final p = macProfileFrom(_m4Pro, availableBytes: 9 << 30);
    expect(p.platform, HostPlatform.macos);
    expect(p.os, 'macOS 26.5.1 (25F80)');
    expect(p.machine, 'Mac16,8');
    expect(p.cpu.model, 'Apple M4 Pro');
    expect(p.cpu.cores, 14);
    expect(p.cpu.performanceCores, 10);
    expect(p.cpu.efficiencyCores, 4);
    expect(p.cpu.architecture, 'arm64');
    expect(p.memory!.totalBytes, 24 << 30);
    expect(p.memory!.availableBytes, 9 << 30);
    final gpu = p.gpus.single;
    expect(gpu.name, 'Apple M4 Pro');
    expect(gpu.inferred, isTrue);
    expect(gpu.api, 'Metal');
    expect(p.npuHints.single.text, contains('Neural Engine'));
    expect(p.notes, isEmpty);

    final e = inferEvidence(requested: 'gpu', reported: 'gpu', hardware: p);
    expect(e.backendSource, EvidenceSource.api);
    expect(e.api, 'Metal');
    expect(e.apiSource, EvidenceSource.inferred);
    expect(e.adapter, 'Apple M4 Pro');
  });

  test('an Intel Mac or unreadable values: nulls with notes, no guesses', () {
    final p = macProfileFrom(
      const _MapSysctl({
        'machdep.cpu.brand_string': 'Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz',
        'hw.optional.arm64': 0,
        'hw.ncpu': 16,
      }),
    );
    expect(p.cpu.architecture, 'x86_64');
    expect(p.cpu.cores, 16);
    expect(p.cpu.performanceCores, isNull);
    expect(p.gpus, isEmpty);
    expect(p.npuHints, isEmpty);
    expect(p.memory, isNull);
    expect(p.notes, [
      'hw.memsize is not readable.',
      contains('host_statistics64'),
      contains('Intel Mac'),
    ]);
  });

  test('the real sysctl and host_statistics64 answer on this Mac', () {
    const sysctl = FfiSysctl();
    expect(sysctl.string('machdep.cpu.brand_string'), isNotEmpty);
    expect(sysctl.integer('hw.memsize'), greaterThan(1 << 30));
    expect(sysctl.string('no.such.sysctl'), isNull);
    final available = MacMemoryProbe().availableBytes();
    expect(available, isNotNull);
    expect(available, lessThan(sysctl.integer('hw.memsize')!));
    expect(available, greaterThan(0));
  }, skip: Platform.isMacOS ? false : 'macOS only');
}
