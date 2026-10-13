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

/// Audio devices as the voice pipeline meets them: Linux's sound server and its
/// sources and sinks (`pactl`), and the verdict on the input and output the app
/// uses. Immutable.
library;

/// A PulseAudio-compatible sound server (`pactl info`).
final class const SoundServer({
  /// `Server Name`: `PulseAudio (on PipeWire 1.0.5)`, `pulseaudio`.
  required final String name,

  /// `Server Version`: `15.0.0`.
  final String? version,

  /// `Default Source` / `Default Sink`: device names (ids), not descriptions.
  final String? defaultSource,
  final String? defaultSink,
});

/// One source or sink (`pactl list sources|sinks`), or another platform's
/// device.
final class const AudioDevice({
  /// `alsa_input.pci-0000_00_1f.3.analog-stereo`.
  required final String id,

  /// What the sound settings show: `Built-in Audio Analog Stereo`.
  required final String name,
  final bool isDefault = false,

  /// A sink's monitor: it records what the sink plays, not a microphone.
  final bool isMonitor = false,
});

/// Linux's audio stack as `pactl` and `/proc/asound/cards` describe it.
final class const AudioSystem({
  /// Null when no sound server answered ([problem] says why).
  final SoundServer? server,

  /// Why there is no [server], phrased for the user (what to install or
  /// start).
  final String? problem,

  /// `pactl` did not run (not installed, or no answer in time): whether a
  /// server runs is unknown, not "no".
  final bool serverUnknown = false,
  final List<AudioDevice> sources = const [],
  final List<AudioDevice> sinks = const [],

  /// Cards in `/proc/asound/cards`; null when it is unreadable.
  final int? alsaCards,
}) {
  /// Sources that are not monitors.
  List<AudioDevice> get microphones => [
    for (final s in sources)
      if (!s.isMonitor) s,
  ];

  /// The default source (what `parecord` records without `--device`).
  AudioDevice? get defaultSource => _default(sources);

  /// The default sink (where PulseAudio plays a stream by default).
  AudioDevice? get defaultSink => _default(sinks);

  static AudioDevice? _default(List<AudioDevice> devices) {
    for (final d in devices) {
      if (d.isDefault) return d;
    }
    return null;
  }
}

/// The verdict on the input or the output the app uses.
sealed class const DeviceCheck();

/// Not checked yet: no voice demo has started the audio.
final class const DeviceUnchecked() extends DeviceCheck;

/// Usable. [name] is what the sound settings show; [detail] how it was found
/// (`PulseAudio (on PipeWire 1.0.5)`, `Core Audio default`). [caution]: usable
/// but worth a look (a monitor as the input, an inferred ALSA output).
final class const DeviceReady(
  final String name, {
  final String? detail,
  final bool caution = false,
}) extends DeviceCheck;

/// Not usable; [message] says what to do.
final class const DeviceUnavailable(final String message) extends DeviceCheck;

/// The input and output the voice pipeline uses, as last checked.
final class const AudioDeviceStatus({
  final DeviceCheck input = const DeviceUnchecked(),
  final DeviceCheck output = const DeviceUnchecked(),
}) {
  AudioDeviceStatus copyWith({DeviceCheck? input, DeviceCheck? output}) =>
      AudioDeviceStatus(
        input: input ?? this.input,
        output: output ?? this.output,
      );
}
