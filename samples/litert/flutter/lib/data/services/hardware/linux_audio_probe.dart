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

import '../../../domain/models/audio_devices.dart';
import 'system_access.dart';

/// `pactl` labels are translated: parse them in the C locale.
const _cLocale = {'LC_ALL': 'C'};

/// How long one `pactl` call may take (a hung server must not hang a probe).
const kPactlTimeout = Duration(seconds: 5);

/// What to install when `pactl` or `parecord` is missing.
const kPulseUtilsHint =
    'install pulseaudio-utils (sudo apt install pulseaudio-utils)';

/// Linux's audio stack: the sound server (`pactl info`), its sources and sinks
/// (`pactl list sources|sinks`) and the ALSA cards (`/proc/asound/cards`).
/// record_linux records with `parecord` and flutter_soloud's miniaudio prefers
/// PulseAudio, so both need a PulseAudio-compatible server (PulseAudio itself
/// or pipewire-pulse).
final class LinuxAudioProbe {
  const LinuxAudioProbe({
    this._processes = const LocalProcessRunner(),
    this._files = const LocalSystemFiles(),
  });

  final ProcessRunner _processes;
  final SystemFiles _files;

  /// Never throws: a missing tool or server is an [AudioSystem.problem].
  Future<AudioSystem> probe() async {
    final cards = switch (_files.read('/proc/asound/cards')) {
      final text? => countAlsaCards(text),
      null => null,
    };
    final info = await _pactl(const ['info']);
    if (info == null) {
      return AudioSystem(
        serverUnknown: true,
        problem:
            'pactl did not run: $kPulseUtilsHint (or the sound server did '
            'not answer within ${kPactlTimeout.inSeconds} s)',
        alsaCards: cards,
      );
    }
    if (info.exitCode != 0) {
      return AudioSystem(problem: serverDownMessage(info), alsaCards: cards);
    }
    final server = parsePactlInfo(info.stdout);
    if (server == null) {
      return AudioSystem(
        problem: 'pactl info printed no "Server Name" (exit 0)',
        alsaCards: cards,
      );
    }
    final sources = await _pactl(const ['list', 'sources']);
    final sinks = await _pactl(const ['list', 'sinks']);
    return AudioSystem(
      server: server,
      sources: sources == null || sources.exitCode != 0
          ? const []
          : parsePactlDevices(
              sources.stdout,
              kind: 'Source',
              defaultId: server.defaultSource,
            ),
      sinks: sinks == null || sinks.exitCode != 0
          ? const []
          : parsePactlDevices(
              sinks.stdout,
              kind: 'Sink',
              defaultId: server.defaultSink,
            ),
      alsaCards: cards,
    );
  }

  Future<ProcessOutput?> _pactl(List<String> args) => _processes.run(
    'pactl',
    args,
    timeout: kPactlTimeout,
    environment: _cLocale,
  );
}

/// `pactl info` failed: no server answers. The message names what to start.
String serverDownMessage(ProcessOutput info) {
  final why = info.stderr.trim().split('\n').first.trim();
  return 'No sound server: PulseAudio/PipeWire is not running '
      '(pactl info: ${why.isEmpty ? 'exit ${info.exitCode}' : why}). Start '
      'it: systemctl --user start pipewire pipewire-pulse (or pulseaudio '
      '--start)';
}

/// `Server Name`, `Server Version`, `Default Source` and `Default Sink` from
/// `pactl info` (C locale); null without a server name.
SoundServer? parsePactlInfo(String text) {
  final kv = <String, String>{};
  for (final line in text.split('\n')) {
    final i = line.indexOf(':');
    if (i <= 0) continue;
    kv[line.substring(0, i).trim()] = line.substring(i + 1).trim();
  }
  final name = kv['Server Name'];
  if (name == null || name.isEmpty) return null;
  String? value(String key) => switch (kv[key]) {
    final v? when v.isNotEmpty && v != 'n/a' => v,
    _ => null,
  };
  return SoundServer(
    name: name,
    version: value('Server Version'),
    defaultSource: value('Default Source'),
    defaultSink: value('Default Sink'),
  );
}

/// The `Source #n` (or `Sink #n`) blocks of `pactl list sources|sinks`:
/// their `Name`, `Description` and, for sources, `Monitor of Sink`. Only the
/// block's own one-tab keys count (properties are indented twice).
List<AudioDevice> parsePactlDevices(
  String text, {
  required String kind,
  String? defaultId,
}) {
  final devices = <AudioDevice>[];
  String? id;
  String? name;
  String? monitorOf;
  var monitorClass = false;
  void commit() {
    final i = id;
    if (i != null) {
      devices.add(
        AudioDevice(
          id: i,
          name: name ?? i,
          isDefault: i == defaultId,
          isMonitor:
              monitorClass ||
              i.endsWith('.monitor') ||
              (monitorOf != null && monitorOf != 'n/a'),
        ),
      );
    }
    id = name = monitorOf = null;
    monitorClass = false;
  }

  final key = RegExp(r'^\t([A-Za-z][A-Za-z ]*): (.*)$');
  for (final line in text.split('\n')) {
    if (line.startsWith('$kind #')) {
      commit();
      continue;
    }
    if (line.trim() == 'device.class = "monitor"') monitorClass = true;
    final m = key.firstMatch(line.trimRight());
    if (m == null) continue;
    switch (m.group(1)) {
      case 'Name':
        id = m.group(2)!.trim();
      case 'Description':
        name = m.group(2)!.trim();
      case 'Monitor of Sink':
        monitorOf = m.group(2)!.trim();
    }
  }
  commit();
  return devices;
}

/// Cards listed in `/proc/asound/cards` (` 0 [PCH ]: HDA-Intel - …`); 0 for
/// `--- no soundcards ---`.
int countAlsaCards(String text) =>
    RegExp(r'^\s*\d+\s+\[', multiLine: true).allMatches(text).length;

/// What the profile's notes say about [audio] (one sentence each).
List<String> audioNotes(AudioSystem audio) {
  final server = audio.server;
  if (server == null) {
    return [
      '${audio.problem ?? 'No sound server'}. Voice input and spoken replies '
          'need PulseAudio or PipeWire (pipewire-pulse).',
    ];
  }
  final source = audio.defaultSource;
  final sink = audio.defaultSink;
  return [
    if (audio.microphones.isEmpty)
      'No recording device: ${server.name} lists no microphone (only monitors '
          'of outputs); voice input will not work.',
    if (source != null && source.isMonitor && audio.microphones.isNotEmpty)
      'The default input is a monitor of an output (${source.name}), not a '
          'microphone.',
    if (audio.sinks.isEmpty)
      'No audio output: ${server.name} lists no sink; spoken replies will '
          'not be heard.',
    if (sink != null && sink.id == 'auto_null')
      'The default output is the sound server\'s dummy sink (auto_null, '
          '"${sink.name}"): nothing is audible.',
  ];
}
