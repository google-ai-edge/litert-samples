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
import 'package:litert_edge_demos/data/services/hardware/macos_sysctl.dart';
import 'package:litert_edge_demos/data/services/hardware/memory_probe.dart';
import 'package:litert_edge_demos/domain/models/hardware_profile.dart';

import '../../../support/hardware_trees.dart';

final class _Sysctl implements Sysctl {
  const _Sysctl(this.values);

  final Map<String, int> values;

  @override
  int? integer(String name) => values[name];

  @override
  String? string(String name) => null;
}

void main() {
  test('the physical RAM: MemTotal on Linux and Android, hw.memsize on '
      'macOS and iOS, none elsewhere; null when unreadable', () {
    const meminfo =
        'MemTotal:        7654321 kB\nMemFree:  1000 kB\n'
        'MemAvailable:    3000000 kB\n';
    expect(
      const LinuxMemoryProbe(FakeSystemFiles({'/proc/meminfo': meminfo}))
          .totalBytes(),
      7654321 * 1024,
    );
    expect(const LinuxMemoryProbe(FakeSystemFiles({})).totalBytes(), isNull);
    expect(
      MacMemoryProbe(sysctl: const _Sysctl({'hw.memsize': 24 << 30}))
          .totalBytes(),
      24 << 30,
    );
    expect(MacMemoryProbe(sysctl: const _Sysctl({})).totalBytes(), isNull);
    expect(const ProcessMemoryProbe().totalBytes(), isNull);
  });

  test('the probe per platform: iOS reads hw.memsize like macOS (its RAM '
      'gates the visual memory there too); Windows and others RSS only', () {
    expect(memoryProbeFor(HostPlatform.linux), isA<LinuxMemoryProbe>());
    expect(memoryProbeFor(HostPlatform.android), isA<LinuxMemoryProbe>());
    expect(memoryProbeFor(HostPlatform.macos), isA<MacMemoryProbe>());
    expect(memoryProbeFor(HostPlatform.ios), isA<MacMemoryProbe>());
    expect(memoryProbeFor(HostPlatform.windows), isA<ProcessMemoryProbe>());
    expect(memoryProbeFor(HostPlatform.other), isA<ProcessMemoryProbe>());
  });
}
