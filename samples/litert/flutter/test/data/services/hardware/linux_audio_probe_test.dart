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
import 'package:litert_edge_demos/data/services/hardware/linux_audio_probe.dart';
import 'package:litert_edge_demos/data/services/hardware/system_access.dart';
import 'package:litert_edge_demos/domain/hardware/diagnostics_report.dart';
import 'package:litert_edge_demos/domain/models/audio_devices.dart';

import '../../../support/hardware_trees.dart';

/// Canned output per command line; records each call's environment.
final class _Runner implements ProcessRunner {
  _Runner(this.outputs);

  final Map<String, ProcessOutput> outputs;
  final List<Map<String, String>?> environments = [];

  @override
  Future<ProcessOutput?> run(
    String executable,
    List<String> arguments, {
    Duration timeout = const Duration(seconds: 10),
    Map<String, String>? environment,
  }) async {
    environments.add(environment);
    return outputs[[executable, ...arguments].join(' ')];
  }
}

Future<AudioSystem> _probe(
  Map<String, ProcessOutput> tools, {
  Map<String, String> files = const {},
}) => LinuxAudioProbe(
  processes: _Runner(tools),
  files: FakeSystemFiles(files),
).probe();

void main() {
  test('PipeWire desktop: server, default mic and sink by description, the '
      'monitor flagged, no notes; pactl runs in the C locale', () async {
    final runner = _Runner(pipewireDesktopAudio);
    final a = await LinuxAudioProbe(
      processes: runner,
      files: const FakeSystemFiles({
        '/proc/asound/cards': ' 0 [Intel   ]: HDA-Intel - HDA Intel\n',
      }),
    ).probe();
    expect(a.server!.name, 'PulseAudio (on PipeWire 1.0.5)');
    expect(a.server!.version, '15.0.0');
    expect(a.problem, isNull);
    expect(
      [for (final s in a.sources) (s.name, s.isMonitor, s.isDefault)],
      [
        ('Monitor of Built-in Audio Analog Stereo', true, false),
        ('Built-in Audio Analog Stereo', false, true),
      ],
    );
    expect(
      a.microphones.single.id,
      'alsa_input.pci-0000_00_1b.0.analog-stereo',
    );
    expect(a.defaultSink!.name, 'Built-in Audio Analog Stereo');
    expect(a.alsaCards, 1);
    expect(audioNotes(a), isEmpty);
    expect(runner.environments, everyElement({'LC_ALL': 'C'}));

    expect(audioSystemLines(a), [
      'audio      PulseAudio (on PipeWire 1.0.5) · server 15.0.0 · ALSA cards 1',
      'audio in   Built-in Audio Analog Stereo · default · '
          'alsa_input.pci-0000_00_1b.0.analog-stereo',
      'audio out  Built-in Audio Analog Stereo · default · '
          'alsa_output.pci-0000_00_1b.0.analog-stereo',
    ]);
  });

  test('PulseAudio without PipeWire properties: monitors still found by '
      '"Monitor of Sink"', () async {
    final a = await _probe(jetsonPulseAudio);
    expect(a.server!.name, 'pulseaudio');
    expect([for (final s in a.sources) s.isMonitor], [true, true, false]);
    expect(a.defaultSource!.name, 'USB Audio Device Mono');
    expect(a.sinks, hasLength(2));
  });

  test(
    'no sound server: pactl exits 1; the problem says what to start',
    () async {
      final a = await _probe(
        {
          'pactl info': const ProcessOutput(
            exitCode: 1,
            stdout: '',
            stderr: 'Connection failure: Connection refused\n',
          ),
        },
        files: {'/proc/asound/cards': '--- no soundcards ---\n'},
      );
      expect(a.server, isNull);
      expect(a.serverUnknown, isFalse, reason: 'pactl answered: it is down');
      expect(a.alsaCards, 0);
      expect(
        a.problem,
        'No sound server: PulseAudio/PipeWire is not running (pactl info: '
        'Connection failure: Connection refused). Start it: systemctl --user '
        'start pipewire pipewire-pulse (or pulseaudio --start)',
      );
      expect(audioNotes(a).single, contains('need PulseAudio or PipeWire'));
      expect(
        audioSystemLines(a).single,
        'audio      no sound server (see the note) · ALSA cards 0',
      );
    },
  );

  test('pactl missing: the problem names the package', () async {
    final a = await _probe(const {});
    expect(a.server, isNull);
    expect(a.problem, contains('install pulseaudio-utils'));
    expect(a.serverUnknown, isTrue);
    expect(a.alsaCards, isNull, reason: '/proc/asound/cards unreadable');
  });

  test(
    'only monitors (no microphone) and the dummy sink: notes say so',
    () async {
      final a = await _probe({
        'pactl info': ProcessOutput(
          exitCode: 0,
          stdout: pactlInfo(
            server: 'pulseaudio',
            version: '16.1',
            sink: 'auto_null',
            source: 'auto_null.monitor',
          ),
        ),
        'pactl list sources': ProcessOutput(
          exitCode: 0,
          stdout: pactlSource(
            0,
            name: 'auto_null.monitor',
            description: 'Monitor of Dummy Output',
            monitorOf: 'auto_null',
            pipewire: false,
          ),
        ),
        'pactl list sinks': ProcessOutput(
          exitCode: 0,
          stdout: pactlSink(0, name: 'auto_null', description: 'Dummy Output'),
        ),
      });
      expect(a.microphones, isEmpty);
      expect(audioNotes(a), [
        startsWith('No recording device: pulseaudio lists no microphone'),
        contains('dummy sink (auto_null, "Dummy Output")'),
      ]);
      expect(audioSystemLines(a), [
        'audio      pulseaudio · server 16.1',
        'audio in   no microphone',
        'audio in   Monitor of Dummy Output · default · monitor (not a '
            'microphone) · auto_null.monitor',
        'audio out  Dummy Output · default · auto_null',
      ]);
    },
  );

  test('ALSA card count', () {
    expect(countAlsaCards('--- no soundcards ---\n'), 0);
    expect(
      countAlsaCards(
        ' 0 [PCH            ]: HDA-Intel - HDA Intel PCH\n'
        '                      HDA Intel PCH at 0xf7f10000 irq 32\n'
        ' 1 [NVidia         ]: HDA-Intel - HDA NVidia\n'
        '                      HDA NVidia at 0xf7080000 irq 17\n',
      ),
      2,
    );
  });
}
