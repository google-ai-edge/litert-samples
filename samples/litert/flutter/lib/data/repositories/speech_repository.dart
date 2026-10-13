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

import 'package:flutter_edge_ai/flutter_edge_ai.dart'
    show SpeechRecognizer, SpeechSynthesizer;
import 'package:flutter_edge_ai_speech/flutter_edge_ai_speech.dart';

import '../../domain/models/model_id.dart';
import '../../utils/result.dart';
import '../services/speech/speech_decorators.dart';
import '../services/speech/stt_service.dart';
import '../services/speech/tts_service.dart';

/// A timing the speech pipeline reports during a turn.
sealed class const SpeechTiming();

/// One transcription.
final class const SttTiming(final Duration elapsed) extends SpeechTiming;

/// One synthesized clause ([bytes] of PCM16, 0 for a non-speech clause).
final class const TtsClauseTiming(final Duration elapsed, final int bytes)
    extends SpeechTiming;

/// One running turn: its `VoiceSession` events and its barge-in. Listen to
/// [events] in the same synchronous block that got the turn: the session starts
/// on listen, and an [interrupt] that arrives before anyone listens never
/// resolves.
final class const VoiceTurn({
  required final Stream<VoiceEvent> events,

  /// `VoiceSession.interrupt`: stops the responder, drains (bounded, worst
  /// case ≈10 s), resolves once the turn has reached its terminal. A no-op
  /// once the turn has ended.
  required final Future<void> Function() interrupt,
});

/// Builds one `VoiceSession.custom(streamAudio: true)` per turn over the loaded
/// STT and TTS models. Owns no audio: the caller plays the chunks.
class SpeechRepository {
  SpeechRepository({required this._stt, required this._tts});

  final SttService _stt;
  final TtsService _tts;

  /// Starts a turn from captured [pcm] (16 kHz mono PCM16) or, for a typed
  /// turn, from [typedText] (no STT; the transcript event echoes it). With
  /// [speak] off every clause is silent. [onTiming] gets the STT time and
  /// each clause's TTS time.
  ///
  /// Fails when the models are not loaded. A voice turn while the demo's
  /// recognizer is still being switched in waits for it (the STT time then
  /// includes that wait). [expectedStt]: the demo's model; the transcription
  /// fails when another one is active (never served by the wrong model).
  Result<VoiceTurn> startTurn({
    Uint8List? pcm,
    String? typedText,
    ModelId? expectedStt,
    required VoiceResponder responder,
    required bool speak,
    void Function(SpeechTiming timing)? onTiming,
  }) {
    if (typedText == null && !_stt.isLoaded && !_stt.isSwitching) {
      return const Result.error(SpeechNotReadyException('speech recognizer'));
    }
    if (speak && !_tts.isLoaded) {
      return const Result.error(SpeechNotReadyException('speech synthesizer'));
    }
    final SpeechRecognizer recognizer = typedText != null
        ? TypedTextRecognizer(typedText)
        : InstrumentedRecognizer(
            ActiveRecognizer(_stt, expected: expectedStt),
            onTranscribed: (elapsed, _) => onTiming?.call(SttTiming(elapsed)),
          );
    final SpeechSynthesizer synthesizer = speak
        ? SpokenSynthesizer(
            InstrumentedSynthesizer(
              _tts.synthesizer,
              onSynthesized: (elapsed, _, bytes) =>
                  onTiming?.call(TtsClauseTiming(elapsed, bytes)),
            ),
          )
        : SilentSynthesizer(_tts.sampleRate);
    final session = VoiceSession.custom(
      recognizer: recognizer,
      responder: responder,
      synthesizer: synthesizer,
      streamAudio: true,
    );
    return Result.ok(
      VoiceTurn(
        events: session.runTurn(pcm ?? Uint8List(0)),
        interrupt: session.interrupt,
      ),
    );
  }
}
