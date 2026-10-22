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
import 'package:litert_edge_demos/data/services/audio/audio_device_service.dart';
import 'package:litert_edge_demos/domain/audio/audio_device_checks.dart';
import 'package:litert_edge_demos/domain/models/audio_devices.dart';
import 'package:litert_edge_demos/domain/models/hardware_profile.dart';

/// Words that belong to Linux's audio stack only.
final _linuxOnly = RegExp('pavucontrol|parecord|pactl|PulseAudio|PipeWire');

void main() {
  test('Windows gets no pre-check and no Linux words: the default input, '
      'named as not queried, and no sound server', () async {
    final service = audioDeviceServiceFor(HostPlatform.windows);
    expect(service, isA<BasicAudioDeviceService>());
    expect(await service.audioSystem(), isNull);
    final input = await service.checkInput();
    expect(
      input,
      isA<DeviceReady>()
          .having((d) => d.name, 'name', 'system default input')
          .having((d) => d.caution, 'caution', isFalse),
    );
    expect(
      deviceCheckText(input),
      'system default input · the name is not queried on windows',
    );
    expect(deviceCheckText(input), isNot(contains(_linuxOnly)));
  });

  test('the service per platform', () {
    expect(
      audioDeviceServiceFor(HostPlatform.linux),
      isA<LinuxAudioDeviceService>(),
    );
    expect(
      audioDeviceServiceFor(HostPlatform.macos),
      isA<MacAudioDeviceService>(),
    );
    for (final platform in [
      HostPlatform.windows,
      HostPlatform.ios,
      HostPlatform.android,
      HostPlatform.other,
    ]) {
      expect(
        audioDeviceServiceFor(platform),
        isA<BasicAudioDeviceService>(),
        reason: platform.name,
      );
    }
  });
}
