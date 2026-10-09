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

import 'env.dart';

/// Push-to-talk turn policy.
final class const VoiceConfig({
  /// A press shorter than this is a slip, not a question: "Didn't catch
  /// that", no STT and no LLM call.
  required final Duration minUtterance,

  /// A 20 ms frame must reach this RMS (dBFS) to count as speech.
  required final double silenceGateDbfs,

  /// ...and also the capture's noise floor + this, capped at
  /// [floorCapDbfs] (see `measureVoice`): the constant noise of a noisy room
  /// is not speech.
  final double aboveFloorDb = 10,
  final double floorCapDbfs = -30,

  /// At least this much voiced audio, or the capture is silence: one click
  /// or bump is a frame or two, and Whisper hallucinates text on
  /// near-silence.
  final Duration minVoiced = const Duration(milliseconds: 160),

  /// The longest capture; the STT window (Whisper pads or cuts to 30 s, so
  /// anything longer would be silently truncated). Hitting it ends the press.
  required final Duration maxUtterance,

  /// Capture format the STT expects: 16 kHz mono PCM16.
  required final int captureSampleRate,
}) {
  /// This policy with captures ended at [window] (the recognizer's: moonshine
  /// reads 5 s and would cut a longer question silently).
  VoiceConfig withMaxUtterance(Duration window) => VoiceConfig(
    minUtterance: minUtterance,
    silenceGateDbfs: silenceGateDbfs,
    aboveFloorDb: aboveFloorDb,
    floorCapDbfs: floorCapDbfs,
    minVoiced: minVoiced,
    maxUtterance: window,
    captureSampleRate: captureSampleRate,
  );
}

const kDefaultSilenceGateDbfs = -45.0;

/// The policy, with `VOICE_GATE_DBFS` applied. Throws a [FormatException]
/// naming the flag when it is not a number: a bad flag fails startup instead
/// of being ignored.
final VoiceConfig kVoiceConfig = voiceConfigFromEnvironment();

VoiceConfig voiceConfigFromEnvironment({String gate = kVoiceGateDbfs}) {
  final trimmed = gate.trim();
  final gateDbfs = trimmed.isEmpty
      ? kDefaultSilenceGateDbfs
      : double.tryParse(trimmed) ??
            (throw FormatException(
              'VOICE_GATE_DBFS must be a number in dBFS (e.g. -40), got '
              '"$gate"',
            ));
  if (gateDbfs >= 0 || gateDbfs < -96) {
    throw FormatException(
      'VOICE_GATE_DBFS must be between -96 and 0 dBFS, got $gateDbfs',
    );
  }
  return VoiceConfig(
    minUtterance: const Duration(milliseconds: 300),
    silenceGateDbfs: gateDbfs,
    maxUtterance: const Duration(seconds: 30),
    captureSampleRate: 16000,
  );
}

/// Seconds of PCM soloud wants before it (re)starts a buffer stream. Its
/// default is 2 s: an underrun between clauses would then pause playback until
/// 2 more seconds arrived.
const kPlaybackBufferingSeconds = 0.2;

/// soloud's output engine settings, explicit because barge-in latency depends
/// on them: a native stop is heard within one output buffer
/// (2048 frames / 44.1 kHz ≈ 46 ms).
const kPlaybackEngineSampleRate = 44100;
const kPlaybackEngineBufferFrames = 2048;

/// After the last chunk, playback must report its end within the audio still
/// queued plus this slack, or the turn stops waiting for it (and logs why).
const kPlaybackDrainSlack = Duration(seconds: 2);

/// How long a responder waits for a stopped turn's generation to drain before
/// asking anyway (after a forced drain the next `ask` would fail as busy).
const kResponderIdleWait = Duration(seconds: 6);

/// Shown when the OS gives the app no microphone audio: permission denied, or
/// a capture that is all digital zeros (macOS TCC hands a blocked app silent
/// buffers instead of an error).
String get kMicAccessMessage => micAccessMessage(defaultTargetPlatform);

/// [kMicAccessMessage] for [platform].
String micAccessMessage(TargetPlatform platform) => switch (platform) {
  TargetPlatform.macOS =>
    'Microphone access is off for this app — System Settings › Privacy & '
        'Security › Microphone',
  TargetPlatform.iOS =>
    'Microphone access is off for this app — Settings › Privacy & Security '
        '› Microphone',
  // Android has no "Privacy & Security" page for one app's permissions.
  TargetPlatform.android =>
    'Microphone access is off for this app — Settings › Apps › '
        'LiteRT Demos › Permissions › Microphone',
  // No permission on Linux: PulseAudio/PipeWire handed over silence (a muted
  // or wrong default input).
  TargetPlatform.linux =>
    'No sound from the microphone — check the default input and its mute in '
        'the sound settings (pavucontrol)',
  _ => 'Microphone access is off for this app',
};
