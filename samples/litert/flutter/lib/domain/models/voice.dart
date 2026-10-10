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

import 'dart:typed_data';

/// Where a push-to-talk or typed turn is.
enum TurnPhase {
  /// Nothing running; the mic and Send are available.
  idle,

  /// The button is held but the capture is still starting (it waits for
  /// the audio warm-up: a cold output start on macOS took up to 2.6 s).
  /// Nothing is recorded yet.
  openingMic,

  /// The mic is open while the button is held: the capture runs.
  listening,

  /// Released: the capture is closing and STT runs.
  transcribing,

  /// The question is committed; the model is generating.
  thinking,

  /// The first clause is playing; text may still be streaming.
  speaking,

  /// The last turn failed; shown until the next action, which starts from
  /// here like from [idle].
  error;

  /// A turn or a capture is in flight.
  bool get isActive => switch (this) {
    openingMic || listening || transcribing || thinking || speaking => true,
    idle || error => false,
  };
}

/// One push-to-talk capture: 16 kHz mono PCM16 and how long the button was
/// held (the too-short check uses the hold, not the bytes, so a mic that
/// delivers nothing is not mistaken for a slip).
final class const Utterance({
  required final Uint8List pcm,
  required final Duration held,
});

/// Why a capture made no LLM call.
enum NotHeardReason {
  /// Released before the minimum hold.
  tooShort,

  /// Released while the mic was still opening (before Listening showed):
  /// nothing was recorded.
  releasedBeforeListening,

  /// The loudest frame stayed under the silence gate.
  silent,

  /// STT returned no words.
  emptyTranscript,
}

/// How a turn ended, for the caller that started it.
enum TurnOutcome {
  completed,

  /// Stopped with the Stop button; the partial reply was kept.
  interrupted,

  /// Too short, silent or no words: no LLM call.
  notHeard,

  /// The mic gave no usable audio (permission, TCC zeros, wrong format).
  micUnavailable,
  failed,

  /// A newer action (barge-in, a typed send, Stop, dispose) took over.
  superseded,

  /// Nothing to do: an empty typed text, or a release with no open mic.
  ignored,
}

/// [TurnOutcome] plus the failure, when there is one.
final class const TurnResult(final TurnOutcome outcome, {final Object? error});

/// Timings of one turn for the overlay and the log.
final class const VoiceTurnMetrics({
  /// Typed turns have no STT.
  required final bool typed,

  /// Release (or Send) to the end of capture close (mic `stop` + its stream's
  /// end); voice turns only.
  final Duration? captureClose,

  /// Whisper's transcription time.
  final Duration? stt,

  /// Release (or Send) to the first reply text.
  final Duration? firstText,

  /// Release (or Send) to the first audio chunk handed to the player.
  final Duration? firstAudio,

  /// The rate of the first chunk.
  final int? sampleRate,

  /// TTS time per clause, in order.
  final List<Duration> ttsClauses = const [],

  /// Release (or Send) to the end of the turn (playback drained).
  final Duration? total,
  final TurnOutcome? outcome,

  /// The silence gate's figures for a voice turn: the loudest 20 ms frame,
  /// the threshold a frame had to reach, and how much audio did.
  final double? peakDbfs,
  final double? gateDbfs,
  final Duration? voiced,
});

/// What one barge-in cost: mic-down to the native stop of the playback
/// (silence, give or take one output buffer), to soloud confirming it, and
/// to the interrupted turn reaching its terminal (VoiceSession's drain).
final class const BargeInMetrics({
  /// Whether anything was playing.
  required final bool wasPlaying,
  final Duration? silenced,
  final Duration? stopConfirmed,
  final Duration? interruptDone,
});

/// The model finished a turn without producing any text: a failure the user
/// must see (wrong prompt role, engine trouble), not an empty bubble.
final class EmptyReplyException implements Exception {
  const EmptyReplyException();

  @override
  String toString() => 'the model returned an empty reply';
}
