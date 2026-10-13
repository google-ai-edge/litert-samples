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
import 'package:litert_edge_demos/data/services/audio/audio_session_service.dart';
import 'package:litert_edge_demos/domain/models/hardware_profile.dart';

void main() {
  test('Linux gets the explicit no-op session, which says so', () async {
    expect(
      audioSessionServiceFor(HostPlatform.linux),
      isA<NoAudioSessionService>(),
    );
    final lines = <String>[];
    await NoAudioSessionService(log: lines.add).configureHalfDuplex();
    expect(lines, ['[AudioSession] none on Linux (PulseAudio/PipeWire)']);
  });

  test('the other platforms keep audio_session', () {
    for (final platform in [
      HostPlatform.macos,
      HostPlatform.ios,
      HostPlatform.android,
    ]) {
      expect(
        audioSessionServiceFor(platform),
        isA<PlatformAudioSessionService>(),
        reason: platform.name,
      );
    }
  });
}
