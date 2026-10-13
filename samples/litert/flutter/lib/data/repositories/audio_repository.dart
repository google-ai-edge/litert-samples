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
import '../../domain/models/voice.dart';
import '../../utils/result.dart';

/// The app's microphone and speaker, shared by both demos.
///
/// Every operation that can come late is scoped to a handle: a capture or a
/// playback that a newer one replaced ignores its owner's later calls, so a
/// demo that was left (and disposes asynchronously) can never stop the next
/// demo's audio.
abstract interface class AudioRepository {
  /// Mic level 0–1 (RMS of each incoming chunk, −60…0 dBFS), latest value
  /// only; 0 when no capture runs. For a `CustomPaint(repaint:)` meter.
  ValueListenable<double> get inputLevel;

  /// The input and the output as last checked: the names the overlay, the
  /// "This device" card and the diagnostics show, or why one is unusable.
  ValueListenable<AudioDeviceStatus> get devices;

  /// Configures the audio session and starts the output engine, once.
  /// Calling it again after a failure retries.
  Future<Result<void>> prepare();

  /// Opens the mic for one push-to-talk press. At most [maxLength] is kept;
  /// reaching it calls [onLimit] once (the press should end). A capture that
  /// is still open is cancelled first. Fails with [MicAccessException] when
  /// the app may not record.
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
