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

import 'dart:io' show Platform;

import '../../../domain/models/hardware_profile.dart';
import 'android_hardware_info_service.dart';
import 'linux_hardware_info_service.dart';
import 'macos_hardware_info_service.dart';

/// Probes what the device is. One implementation per platform; tests use a
/// fake.
abstract interface class HardwareInfoService {
  /// Never throws for a missing fact: it is left null and explained in
  /// [HardwareProfile.notes]. Throws only on a bug.
  Future<HardwareProfile> probe();
}

/// The service for the running platform. iOS (and any platform without its own
/// probe) gets the OS and core count only, and says so.
HardwareInfoService hardwareInfoServiceForPlatform() =>
    switch (HostPlatform.fromOperatingSystem(Platform.operatingSystem)) {
      HostPlatform.linux => LinuxHardwareInfoService(),
      HostPlatform.macos => MacHardwareInfoService(),
      HostPlatform.android => AndroidHardwareInfoService(),
      final other => BasicHardwareInfoService(other),
    };

/// OS and core count from `dart:io`; no chip, GPU or RAM.
final class BasicHardwareInfoService implements HardwareInfoService {
  const BasicHardwareInfoService(this._platform);

  final HostPlatform _platform;

  @override
  Future<HardwareProfile> probe() async => HardwareProfile(
    platform: _platform,
    os: '${Platform.operatingSystem} ${Platform.operatingSystemVersion}',
    cpu: CpuInfo(model: 'unknown', cores: Platform.numberOfProcessors),
    notes: ['The chip, GPU and RAM are not probed on ${_platform.name}.'],
  );
}
