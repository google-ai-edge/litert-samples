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

/// The verdicts on the app's audio input and output, with the messages that
/// say what to do. Pure: unit-tested.
library;

import 'dart:io' show ProcessException;

import '../models/audio_devices.dart';
import '../models/hardware_profile.dart' show HostPlatform;

/// record_linux streams from `parecord`; without it the start throws
/// `ProcessException`.
const kParecordMissing =
    'Microphone unavailable: install pulseaudio-utils (parecord)';

/// The sound server lists no microphone (only monitors of outputs).
const kNoRecordingDevice = 'No recording device';

/// What the user sees when a capture could not start; [error] is what the
/// microphone service threw. A missing `parecord` on Linux names the package
/// to install.
String micStartErrorMessage(Object error, HostPlatform platform) =>
    switch (error) {
      ProcessException(:final executable)
          when platform == HostPlatform.linux && executable == 'parecord' =>
        kParecordMissing,
      _ => 'The microphone did not start: $error',
    };

/// The input `parecord` records from (the server's default source), from
/// what `pactl` reported. [parecordFound]: whether `parecord` ran.
DeviceCheck linuxInputCheck(AudioSystem audio, {required bool parecordFound}) {
  if (!parecordFound) return const DeviceUnavailable(kParecordMissing);
  final server = audio.server;
  if (server == null) {
    return DeviceUnavailable(audio.problem ?? 'No sound server');
  }
  if (audio.microphones.isEmpty) {
    return DeviceUnavailable(
      '$kNoRecordingDevice: ${server.name} lists no microphone, only '
      'monitors of outputs (plug one in, or enable it in the sound settings)',
    );
  }
  final source = audio.defaultSource;
  if (source == null) {
    return DeviceUnavailable(
      '$kNoRecordingDevice: ${server.name} has no default input (choose one '
      'in the sound settings)',
    );
  }
  if (source.isMonitor) {
    return DeviceReady(
      source.name,
      detail: 'a monitor of an output, not a microphone · ${server.name}',
      caution: true,
    );
  }
  return DeviceReady(source.name, detail: server.name);
}

/// The name of miniaudio's Null backend device (flutter_soloud 4.1.7 bundles
/// miniaudio 0.11.25: `miniaudio.h:21062`, enumerate, and `:21091`, device
/// info). It plays into nothing and reports nothing: playback "succeeds" in
/// silence.
const kMiniaudioNullPlaybackDevice = 'NULL Playback Device';

/// Linux's no-output message: miniaudio tried PulseAudio, ALSA and JACK and
/// fell back to Null (`ma_device_init_ex` walks the backends in enum order,
/// `miniaudio.h:44015-44090`; soloud's Linux path calls it,
/// `soloud_miniaudio.cpp:553`).
const kNoAudioOutput = 'No audio output (no PulseAudio/PipeWire/ALSA device)';

/// Whether listing the playback devices is harmless on [platform].
/// iOS: no. `Player::listPlaybackDevices` creates a miniaudio context with
/// the default config, which sets the AVAudioSession category
/// (PlayAndRecord + DefaultToSpeaker, dropping the app's A2DP option) and
/// activates it (`miniaudio.h:36527-36566`), while `audio_session` owns the
/// session (soloud's own init opts out, `soloud_miniaudio.cpp:475-476`).
/// Android: nothing to catch, soloud opens AAudio or OpenSL only
/// (`soloud_miniaudio.cpp:519`), never Null.
bool canListPlaybackDevices(HostPlatform platform) =>
    platform != HostPlatform.ios && platform != HostPlatform.android;

/// The output soloud plays to, from its device list after init
/// ([playback], `listPlaybackDevices`; null where [canListPlaybackDevices]
/// says no) and, on Linux, the sound server ([linux]). Never pretends:
/// - no device, or the Null device, is unavailable on every platform
///   (Null is compiled in everywhere but the web, `miniaudio.h:6648-6651`);
/// - the device is the one marked default, else the first listed
///   (route-based backends mark none: iOS's route outputs and AAudio's
///   "Default Playback Device", `miniaudio.h:34752-34770, 39560-39575`);
/// - Linux: the list comes from the first backend whose *context* starts
///   (`Player::listPlaybackDevices`, `ma_context_init(NULL…)`), the device
///   from the first whose *device* opens. Without a sound server the list is
///   ALSA's (its configured "default" PCM shows even with no card behind
///   it) while the device may be Null. So: a running server → its default
///   sink; no server but ALSA cards → ALSA (inferred, flagged); neither →
///   unavailable.
/// macOS opens Core Audio on an explicit context and fails init when Core
/// Audio's device does not open (`soloud_miniaudio.cpp:479-488`); it reaches
/// Null only if Core Audio's context itself fails, which the name check
/// catches.
DeviceCheck outputCheck({
  required HostPlatform platform,
  required List<AudioDevice>? playback,
  AudioSystem? linux,
}) {
  if (playback == null) {
    return DeviceReady(
      'system default output',
      detail: platform == HostPlatform.ios
          ? 'not listed on iOS: the listing would reset the audio session'
          : 'not listed on ${platform.name}: soloud opens AAudio/OpenSL only, '
                'never the Null device',
    );
  }
  final isLinux = platform == HostPlatform.linux;
  final head = isLinux ? kNoAudioOutput : 'No audio output device';
  if (playback.isEmpty) {
    return DeviceUnavailable(
      '$head: the audio engine lists no playback device',
    );
  }
  final device = playback.firstWhere(
    (d) => d.isDefault,
    orElse: () => playback.first,
  );
  if (device.name == kMiniaudioNullPlaybackDevice) {
    return DeviceUnavailable(
      '$head: the audio engine fell back to miniaudio\'s silent Null device',
    );
  }
  if (!isLinux) {
    return DeviceReady(
      device.name,
      detail: platform == HostPlatform.macos
          ? 'Core Audio default output'
          : 'default output',
    );
  }
  final audio = linux ?? const AudioSystem();
  final server = audio.server;
  if (server != null) {
    final dummy = audio.defaultSink?.id == 'auto_null';
    return DeviceReady(
      device.name,
      detail: dummy
          ? '${server.name} dummy sink (auto_null): nothing is audible'
          : server.name,
      caution: dummy,
    );
  }
  final cards = audio.alsaCards ?? 0;
  if (audio.serverUnknown) {
    // pactl did not run: a server may be there (soloud would use it) or not.
    if (cards > 0) {
      return DeviceReady(
        device.name,
        detail:
            'PulseAudio or ALSA: pactl did not run, so the sound server is '
            'unknown (install pulseaudio-utils)',
        caution: true,
      );
    }
    return DeviceUnavailable(
      'The audio output cannot be verified: pactl did not run and '
      '/proc/asound/cards lists no card, so the audio engine plays either to '
      'a sound server pactl could not ask or into miniaudio\'s silent Null '
      'device. ${audio.problem ?? 'Install pulseaudio-utils'}',
    );
  }
  if (cards > 0) {
    return DeviceReady(
      device.name,
      detail:
          'ALSA, no sound server (inferred: miniaudio opens default, dmix or '
          'hw:0 of $cards card${cards == 1 ? '' : 's'})',
      caution: true,
    );
  }
  return DeviceUnavailable(
    '$kNoAudioOutput: no sound server answers and /proc/asound/cards lists '
    'no card, so the audio engine plays into miniaudio\'s silent Null device '
    '(the list\'s "${device.name}" is ALSA configuration, no card backs it). '
    '${audio.problem ?? 'pactl info failed'}',
  );
}

/// `name · detail`, `not checked yet`, or the failure, for one line.
String deviceCheckText(DeviceCheck check) => switch (check) {
  DeviceUnchecked() => 'not checked yet (a voice demo starts the audio)',
  DeviceReady(:final name, :final detail) => [name, ?detail].join(' · '),
  DeviceUnavailable(:final message) => message,
};
