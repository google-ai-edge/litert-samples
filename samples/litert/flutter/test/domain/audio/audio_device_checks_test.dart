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

import 'dart:io' show ProcessException;

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/domain/audio/audio_device_checks.dart';
import 'package:litert_edge_demos/domain/models/audio_devices.dart';
import 'package:litert_edge_demos/domain/models/hardware_profile.dart';

const _server = SoundServer(
  name: 'PulseAudio (on PipeWire 1.0.5)',
  defaultSource: 'alsa_input.analog',
  defaultSink: 'alsa_output.analog',
);
const _mic = AudioDevice(
  id: 'alsa_input.analog',
  name: 'Built-in Audio Analog Stereo',
  isDefault: true,
);
const _monitor = AudioDevice(
  id: 'alsa_output.analog.monitor',
  name: 'Monitor of Built-in Audio Analog Stereo',
  isMonitor: true,
);

void main() {
  group('mic start errors', () {
    test('a missing parecord (record_linux ProcessException) names the '
        'package', () {
      final error = const ProcessException(
        'parecord',
        ['--raw', '--format=s16le'],
        'No such file or directory',
        2,
      );
      expect(
        micStartErrorMessage(error, HostPlatform.linux),
        'Microphone unavailable: install pulseaudio-utils (parecord)',
      );
    });

    test('anything else keeps its own text', () {
      expect(
        micStartErrorMessage(
          const ProcessException('ffmpeg', [], 'not found', 2),
          HostPlatform.linux,
        ),
        startsWith('The microphone did not start: ProcessException'),
      );
      expect(
        micStartErrorMessage(
          const ProcessException('parecord', [], 'not found', 2),
          HostPlatform.macos,
        ),
        startsWith('The microphone did not start'),
        reason: 'parecord is a Linux matter',
      );
      expect(
        micStartErrorMessage(StateError('busy'), HostPlatform.linux),
        'The microphone did not start: Bad state: busy',
      );
    });
  });

  group('Linux input check', () {
    test('parecord missing comes first', () {
      expect(
        linuxInputCheck(
          const AudioSystem(server: _server, sources: [_mic]),
          parecordFound: false,
        ),
        isA<DeviceUnavailable>().having(
          (d) => d.message,
          'message',
          kParecordMissing,
        ),
      );
    });

    test('no sound server: the probe\'s message (what to start)', () {
      const problem =
          'No sound server: PulseAudio/PipeWire is not running (pactl info: '
          'Connection failure: Connection refused). Start it: …';
      expect(
        linuxInputCheck(
          const AudioSystem(problem: problem),
          parecordFound: true,
        ),
        isA<DeviceUnavailable>().having((d) => d.message, 'message', problem),
      );
    });

    test('an empty device list, or monitors only: no recording device', () {
      for (final sources in [
        const <AudioDevice>[],
        const [_monitor],
      ]) {
        final check = linuxInputCheck(
          AudioSystem(server: _server, sources: sources),
          parecordFound: true,
        );
        expect(
          check,
          isA<DeviceUnavailable>().having(
            (d) => d.message,
            'message',
            startsWith(
              'No recording device: PulseAudio (on PipeWire 1.0.5) lists no '
              'microphone',
            ),
          ),
          reason: '${sources.length} source(s)',
        );
      }
    });

    test('microphones but no default: say where to choose one', () {
      final check = linuxInputCheck(
        const AudioSystem(
          server: SoundServer(name: 'pulseaudio'),
          sources: [AudioDevice(id: 'usb', name: 'USB Mic')],
        ),
        parecordFound: true,
      );
      expect(
        (check as DeviceUnavailable).message,
        contains('has no default input'),
      );
    });

    test('the default microphone by its description; a monitor as the '
        'default is usable but flagged', () {
      expect(
        linuxInputCheck(
          const AudioSystem(server: _server, sources: [_monitor, _mic]),
          parecordFound: true,
        ),
        isA<DeviceReady>()
            .having((d) => d.name, 'name', 'Built-in Audio Analog Stereo')
            .having((d) => d.detail, 'detail', _server.name)
            .having((d) => d.caution, 'caution', isFalse),
      );
      final monitorDefault = linuxInputCheck(
        const AudioSystem(
          server: _server,
          sources: [
            AudioDevice(
              id: 'virt.monitor',
              name: 'Monitor of Virtual Sink',
              isDefault: true,
              isMonitor: true,
            ),
            AudioDevice(id: 'alsa_input.analog', name: 'Built-in'),
          ],
        ),
        parecordFound: true,
      );
      expect(monitorDefault, isA<DeviceReady>());
      expect((monitorDefault as DeviceReady).caution, isTrue);
      expect(monitorDefault.detail, startsWith('a monitor of an output'));
    });
  });

  group('output check (miniaudio null device)', () {
    const nullDevice = AudioDevice(
      id: '0',
      name: 'NULL Playback Device',
      isDefault: true,
    );
    const pulseSink = AudioDevice(
      id: '0',
      name: 'Built-in Audio Analog Stereo',
      isDefault: true,
    );
    const pipewire = AudioSystem(
      server: _server,
      sinks: [
        AudioDevice(
          id: 'alsa_output.analog',
          name: 'Built-in Audio Analog Stereo',
          isDefault: true,
        ),
      ],
    );

    test('the exact miniaudio name (miniaudio.h:21062)', () {
      expect(kMiniaudioNullPlaybackDevice, 'NULL Playback Device');
    });

    test('the Null device or no device fails, on Linux and on macOS', () {
      for (final platform in [HostPlatform.linux, HostPlatform.macos]) {
        final onNull = outputCheck(
          platform: platform,
          playback: const [nullDevice],
          linux: pipewire,
        );
        expect(onNull, isA<DeviceUnavailable>(), reason: platform.name);
        expect(
          (onNull as DeviceUnavailable).message,
          contains("miniaudio's silent Null device"),
        );
        final none = outputCheck(
          platform: platform,
          playback: const [],
          linux: pipewire,
        );
        expect(
          (none as DeviceUnavailable).message,
          contains('lists no playback device'),
        );
      }
      expect(
        (outputCheck(
          platform: HostPlatform.linux,
          playback: const [],
        ) as DeviceUnavailable).message,
        startsWith(
          'No audio output (no PulseAudio/PipeWire/ALSA device): the audio '
          'engine lists no playback device',
        ),
      );
    });

    test('a running sound server: its default sink, by description', () {
      expect(
        outputCheck(
          platform: HostPlatform.linux,
          playback: const [
            AudioDevice(id: '1', name: 'HDMI'),
            pulseSink,
          ],
          linux: pipewire,
        ),
        isA<DeviceReady>()
            .having((d) => d.name, 'name', 'Built-in Audio Analog Stereo')
            .having((d) => d.detail, 'detail', _server.name)
            .having((d) => d.caution, 'caution', isFalse),
      );
    });

    test('the dummy sink is usable but flagged: nothing is audible', () {
      final check = outputCheck(
        platform: HostPlatform.linux,
        playback: const [
          AudioDevice(id: '0', name: 'Dummy Output', isDefault: true),
        ],
        linux: const AudioSystem(
          server: SoundServer(name: 'pulseaudio', defaultSink: 'auto_null'),
          sinks: [
            AudioDevice(id: 'auto_null', name: 'Dummy Output', isDefault: true),
          ],
        ),
      );
      expect((check as DeviceReady).caution, isTrue);
      expect(check.detail, contains('nothing is audible'));
    });

    test('Linux without a server: ALSA lists its configured "default" even '
        'with no card while the device lands on Null — no card fails, a card '
        'is ALSA (inferred, flagged)', () {
      const alsaList = [
        AudioDevice(
          id: '0',
          name:
              'Discard all samples (playback) or generate zero samples '
              '(capture)',
        ),
        AudioDevice(id: '1', name: 'Default Audio Device', isDefault: true),
      ];
      const problem =
          'No sound server: PulseAudio/PipeWire is not running (pactl info: '
          'Connection failure: Connection refused). Start it: …';
      final noCard = outputCheck(
        platform: HostPlatform.linux,
        playback: alsaList,
        linux: const AudioSystem(problem: problem, alsaCards: 0),
      );
      expect(noCard, isA<DeviceUnavailable>());
      expect(
        (noCard as DeviceUnavailable).message,
        allOf(
          startsWith(
            'No audio output (no PulseAudio/PipeWire/ALSA device): no sound '
            'server answers and /proc/asound/cards lists no card',
          ),
          contains('"Default Audio Device"'),
          endsWith(problem),
        ),
      );
      expect(
        outputCheck(
          platform: HostPlatform.linux,
          playback: alsaList,
          linux: const AudioSystem(problem: problem),
        ),
        isA<DeviceUnavailable>(),
        reason: 'no /proc/asound: no ALSA driver, no card',
      );

      final withCard = outputCheck(
        platform: HostPlatform.linux,
        playback: alsaList,
        linux: const AudioSystem(problem: problem, alsaCards: 1),
      );
      expect(withCard, isA<DeviceReady>());
      expect((withCard as DeviceReady).caution, isTrue);
      expect(withCard.detail, startsWith('ALSA, no sound server (inferred'));
    });

    test('iOS and Android are not listed (the listing would reset the iOS '
        'audio session; Android has no Null fallback): said so', () {
      expect(canListPlaybackDevices(HostPlatform.ios), isFalse);
      expect(canListPlaybackDevices(HostPlatform.android), isFalse);
      expect(canListPlaybackDevices(HostPlatform.linux), isTrue);
      expect(canListPlaybackDevices(HostPlatform.macos), isTrue);
      expect(
        outputCheck(platform: HostPlatform.ios, playback: null),
        isA<DeviceReady>().having(
          (d) => d.detail,
          'detail',
          contains('would reset the audio session'),
        ),
      );
      expect(
        outputCheck(platform: HostPlatform.android, playback: null),
        isA<DeviceReady>().having(
          (d) => d.detail,
          'detail',
          contains('never the Null device'),
        ),
      );
    });

    test('pactl did not run: the server is unknown, not "down" — a card '
        'plays somewhere (flagged), no card cannot be verified', () {
      const list = [
        AudioDevice(id: '0', name: 'Chrome Remote Desktop', isDefault: true),
      ];
      const problem = 'pactl did not run: install pulseaudio-utils';
      final withCard = outputCheck(
        platform: HostPlatform.linux,
        playback: list,
        linux: const AudioSystem(
          serverUnknown: true,
          problem: problem,
          alsaCards: 1,
        ),
      );
      expect((withCard as DeviceReady).caution, isTrue);
      expect(
        withCard.detail,
        startsWith('PulseAudio or ALSA: pactl did not run'),
      );
      final noCard = outputCheck(
        platform: HostPlatform.linux,
        playback: list,
        linux: const AudioSystem(
          serverUnknown: true,
          problem: problem,
          alsaCards: 0,
        ),
      );
      expect(
        (noCard as DeviceUnavailable).message,
        allOf(
          startsWith('The audio output cannot be verified: pactl did not run'),
          isNot(contains('no sound server answers')),
          endsWith(problem),
        ),
      );
    });

    test('route-based backends mark no default: the first listed device '
        '(AAudio\'s "Default Playback Device", an iOS route)', () {
      expect(
        outputCheck(
          platform: HostPlatform.android,
          playback: const [
            AudioDevice(id: '0', name: 'Default Playback Device'),
          ],
        ),
        isA<DeviceReady>().having(
          (d) => d.name,
          'name',
          'Default Playback Device',
        ),
      );
      expect(
        outputCheck(
          platform: HostPlatform.macos,
          playback: const [
            AudioDevice(id: '0', name: 'MacBook Pro Speakers', isDefault: true),
          ],
        ),
        isA<DeviceReady>().having(
          (d) => d.detail,
          'detail',
          'Core Audio default output',
        ),
      );
    });
  });
}
