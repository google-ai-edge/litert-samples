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

import 'package:litert_edge_demos/data/services/hardware/hardware_info_service.dart';
import 'package:litert_edge_demos/data/services/hardware/memory_probe.dart';
import 'package:litert_edge_demos/data/services/hardware/native_log_tap.dart';
import 'package:litert_edge_demos/domain/models/hardware_profile.dart';

/// An M4 Pro as `MacHardwareInfoService` reports it.
const kFakeMacProfile = HardwareProfile(
  platform: HostPlatform.macos,
  os: 'macOS 26.5.1 (25F80)',
  machine: 'Mac16,8',
  cpu: CpuInfo(
    model: 'Apple M4 Pro',
    cores: 14,
    performanceCores: 10,
    efficiencyCores: 4,
    architecture: 'arm64',
  ),
  memory: MemoryInfo(totalBytes: 24 << 30, availableBytes: 9 << 30),
  gpus: [
    GpuInfo(
      name: 'Apple M4 Pro',
      source: 'chip name (sysctl)',
      inferred: true,
      kind: GpuKind.integrated,
      api: 'Metal',
    ),
  ],
  npuHints: [NpuHint('Apple Neural Engine (not used)')],
);

/// [HardwareInfoService] that returns [profile], or throws [error].
final class FakeHardwareInfoService implements HardwareInfoService {
  FakeHardwareInfoService([this.profile = kFakeMacProfile, this.error]);

  final HardwareProfile profile;
  final Error? error;
  int probes = 0;

  @override
  Future<HardwareProfile> probe() async {
    probes++;
    if (error case final e?) throw e;
    return profile;
  }
}

/// Memory that drops by [stepBytes] per sample: available falls as models
/// load, RSS and peak rise.
final class FakeMemoryProbe implements MemoryProbe {
  FakeMemoryProbe({this.stepBytes = 100 << 20});

  final int stepBytes;
  int _samples = 0;

  @override
  int? availableBytes() => (9 << 30) - _samples * stepBytes;

  @override
  MemorySnapshot snapshot() {
    final available = availableBytes();
    _samples++;
    return MemorySnapshot(
      availableBytes: available,
      rssBytes: (100 << 20) + _samples * stepBytes,
      peakRssBytes: (110 << 20) + _samples * stepBytes,
    );
  }
}

/// A native log that tests append to; [mark]/[since] work like the file tap.
final class FakeNativeLogTap implements NativeLogTap {
  final List<String> lines = [];

  @override
  String get description => 'fake native log';

  @override
  int mark() => lines.length;

  @override
  List<String> since(int mark) => lines.sublist(mark);
}
