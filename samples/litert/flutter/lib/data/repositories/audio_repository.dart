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

import '../../domain/models/audio_devices.dart';
import '../../domain/models/audio_interruption.dart';
import '../../domain/models/voice.dart';
import '../../utils/result.dart';

/// The app's microphone and speaker, shared by both demos.
///
/// Every operation that can come late is scoped to a handle: a capture or a
/// playback that a newer one replaced ignores its owner's later calls, so a
/// demo that was left (and disposes asynchronously) can never stop the next
/// demo's audio.
///
/// The platform session is held per turn where the platform shares the audio
/// that way: on Android [activate] takes the audio focus (other apps' audio
/// pauses) and [deactivate] gives it back once the turn has ended. On iOS
/// the session is activated by [prepare] and stays active between turns.
/// When the OS or another app takes the audio ([interruptions]) the capture
/// is cancelled, the playback stopped, and the next turn activates the
/// session again.
abstract interface class AudioRepository {
  /// Mic level 0–1 (RMS of each incoming chunk, −60…0 dBFS), latest value
  /// only; 0 when no capture runs. For a `CustomPaint(repaint:)` meter.
  ValueListenable<double> get inputLevel;

  /// The input and the output as last checked: the names the overlay, the
  /// "This device" card and the diagnostics show, or why one is unusable.
  ValueListenable<AudioDeviceStatus> get devices;

  /// The interruptions of the platform session (a call, Siri, an alarm,
  /// another app's audio; the output unplugged), each reported once the
  /// capture in progress was cancelled and the playback stopped. Broadcast,
  /// delivered synchronously; only iOS and Android have any.
  Stream<AudioInterruption> get interruptions;

  /// Configures the audio session and starts the output engine, once (iOS:
  /// also activates the session). Calling it again after a failure retries;
  /// on iOS, after an interruption or after the app was hidden, it runs
  /// again and activates the session again; after the app was hidden it
  /// also restarts the output engine.
  Future<Result<void>> prepare();

  /// Takes the audio for a turn that records or plays: [prepare]s, then on
  /// Android requests the audio focus (until [deactivate]). Fails with
  /// [AudioFocusException] when the OS refuses (a call holds the audio).
  /// [startCapture] activates by itself.
  Future<Result<void>> activate();

  /// The turn has ended: Android gives the audio focus back, so other apps'
  /// audio resumes. Elsewhere nothing changes (iOS keeps the session active,
  /// which keeps `setActive` off the press path). An [activate] after
  /// this one wins.
  Future<void> deactivate();

  /// The app went out of sight. iOS: another app may take the session while
  /// this one is suspended, and that interruption comes marked as such
  /// (`audio_session` drops it) or not at all (read from the sources, not
  /// observed on a device), so as a precaution a playback still running is
  /// stopped and the next [prepare] activates the session again and
  /// restarts the output engine. Elsewhere nothing changes.
  void appHidden();

  /// Opens the mic for one push-to-talk press ([activate]s first). At most
  /// [maxLength] is kept; reaching it calls [onLimit] once (the press should
  /// end). A capture that is still open is cancelled first. Fails with
  /// [MicAccessException] when the app may not record, with
  /// [AudioFocusException] when the OS keeps the audio from it.
  Future<Result<CaptureHandle>> startCapture({
    required Duration maxLength,
    required void Function() onLimit,
  });

  /// Asks for (or confirms) microphone access now — when a
  /// voice demo opens — so the first press is not interrupted by the OS
  /// dialog. Fails with [MicAccessException] (the platform's settings path)
  /// when the app may not record, or when there is nothing to record from
  /// (no input device; on Linux no `parecord` or no sound server), with what
  /// to do.
  Future<Result<void>> requestMicAccess();

  /// A new playback stream at [sampleRate] for one reply. A playback that is
  /// still running is stopped first.
  Result<PlaybackHandle> beginPlayback(int sampleRate);

  /// Stops any capture and playback and shuts the engines down.
  Future<void> close();
}

/// One open microphone press.
abstract interface class CaptureHandle {
  /// Closes the mic, waits for its stream to end, and returns what was
  /// captured. Fails when the platform changed the format mid-capture, the
  /// stream errored, or a newer capture replaced this one.
  Future<Result<Utterance>> stop();

  /// Closes the mic and drops the audio. Safe to call more than once.
  Future<void> cancel();
}

/// One reply's playback.
abstract interface class PlaybackHandle {
  /// Queues a chunk; the first one starts playback. Throws [PlaybackException]
  /// if the engine rejects it. Ignored once stopped or replaced.
  void enqueue(Uint8List pcm);

  /// No more chunks. Idempotent.
  void end();

  /// Completes when the queued audio has played after [end], when stopped or
  /// replaced, or — if the engine never reports the end — after the queued
  /// duration plus a slack (logged).
  Future<void> get drained;

  /// Silences the playback now. The native stop happens before this
  /// returns its future; the future completes once the engine confirms.
  Future<void> stop();
}

/// The app may not use the microphone, or the OS gives it no audio.
final class MicAccessException implements Exception {
  const MicAccessException(this.message);

  final String message;

  @override
  String toString() => message;
}

/// The OS did not give the app the audio ([kAudioFocusMessage]), or took it
/// back while it was being prepared ([kAudioInterruptedMessage]).
final class AudioFocusException implements Exception {
  const AudioFocusException(this.message);

  final String message;

  @override
  String toString() => message;
}

/// [AudioFocusException.message] when the OS refuses the audio.
const kAudioFocusMessage =
    'Another app is using the audio (a call?); try again once it has ended';

/// [AudioFocusException.message] when an interruption came while the audio
/// was being prepared.
const kAudioInterruptedMessage =
    'The audio was interrupted while it started; try again';

/// The platform delivered another format than 16 kHz mono PCM16.
final class MicFormatException implements Exception {
  const MicFormatException(this.actual);

  final String actual;

  @override
  String toString() =>
      'The microphone switched to $actual; speech recognition needs '
      '16 kHz mono PCM16';
}

/// The output engine failed (start, queue or stop).
final class PlaybackException implements Exception {
  const PlaybackException(this.message);

  final String message;

  @override
  String toString() => 'Playback failed: $message';
}

/// The handle was replaced by a newer capture or playback.
final class AudioSupersededException implements Exception {
  const AudioSupersededException(this.what);

  final String what;

  @override
  String toString() => 'The $what was replaced by a newer one';
}
