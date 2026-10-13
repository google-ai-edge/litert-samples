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

import 'package:flutter/foundation.dart';

import '../../../domain/audio/audio_device_checks.dart';
import '../../../domain/models/audio_devices.dart';
import '../../../domain/models/hardware_profile.dart' show HostPlatform;
import '../hardware/linux_audio_probe.dart';
import '../hardware/macos_core_audio.dart';
import '../hardware/system_access.dart';

/// What the OS says about the input `record` opens and the sound server the
/// output needs. One implementation per platform; tests use a fake.
abstract interface class AudioDeviceService {
  /// The input a capture records from (the system default), or what to do.
  /// Never throws.
  Future<DeviceCheck> checkInput();

  /// Linux: the sound server and its devices (`pactl`); null on the other
  /// platforms. Never throws.
  Future<AudioSystem?> audioSystem();
}

/// The service for [platform].
AudioDeviceService audioDeviceServiceFor(HostPlatform platform) =>
    switch (platform) {
      HostPlatform.linux => const LinuxAudioDeviceService(),
      HostPlatform.macos => const MacAudioDeviceService(),
      final other => BasicAudioDeviceService(other),
    };

/// Linux: `parecord --version` (record_linux needs it) and the `pactl`
/// probe.
final class LinuxAudioDeviceService implements AudioDeviceService {
  const LinuxAudioDeviceService({
    this._processes = const LocalProcessRunner(),
    this._files = const LocalSystemFiles(),
  });

  final ProcessRunner _processes;
  final SystemFiles _files;

  Future<AudioSystem> _probe() =>
      LinuxAudioProbe(processes: _processes, files: _files).probe();

  @override
  Future<AudioSystem?> audioSystem() => _probe();

  @override
  Future<DeviceCheck> checkInput() async {
    final parecord = await _processes.run('parecord', const [
      '--version',
    ], timeout: kPactlTimeout);
    return linuxInputCheck(await _probe(), parecordFound: parecord != null);
  }
}

/// macOS: the Core Audio default input's name (record opens
/// `AVCaptureDevice.default(for: .audio)`, the same device).
final class MacAudioDeviceService implements AudioDeviceService {
  const MacAudioDeviceService({this._coreAudio = const CoreAudioDefaults()});

  final CoreAudioDefaults _coreAudio;

  @override
  Future<AudioSystem?> audioSystem() async => null;

  @override
  Future<DeviceCheck> checkInput() async {
    try {
      final name = _coreAudio.defaultDeviceName(input: true);
      if (name == null) {
        return const DeviceUnavailable(
          '$kNoRecordingDevice: macOS has no default input (System Settings › '
          'Sound › Input)',
        );
      }
      return DeviceReady(name, detail: 'Core Audio default input');
    } catch (e) {
      debugPrint('[Audio] Core Audio input query failed: $e');
      return DeviceReady(
        'system default input',
        detail: 'name unavailable: $e',
        caution: true,
      );
    }
  }
}

/// iOS and Android: the system picks the route; the name is not queried.
final class BasicAudioDeviceService implements AudioDeviceService {
  const BasicAudioDeviceService(this._platform);

  final HostPlatform _platform;

  @override
  Future<AudioSystem?> audioSystem() async => null;

  @override
  Future<DeviceCheck> checkInput() async => DeviceReady(
    'system default input',
    detail: 'the name is not queried on ${_platform.name}',
  );
}
