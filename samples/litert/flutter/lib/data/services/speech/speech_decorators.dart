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

import '../../../domain/models/model_id.dart';
import '../../../utils/spoken_text.dart';
import 'stt_service.dart';

// Per-turn wrappers around the shared recognizer and synthesizer. VoiceSession
// owns none of its components' lifecycles, and neither do these: `close()`
// never closes the wrapped model (SttService / TtsService own it).

/// Times each transcription for the overlay. Delegates [language].
final class InstrumentedRecognizer implements SpeechRecognizer {
  InstrumentedRecognizer(this._inner, {required this._onTranscribed});

  final SpeechRecognizer _inner;
  final void Function(Duration elapsed, String text) _onTranscribed;

  @override
  String? get language => _inner.language;

  @override
  set language(String? value) => _inner.language = value;

  @override
  Future<String> transcribe(Uint8List pcm16kMono, {String? language}) async {
    final watch = Stopwatch()..start();
    final text = await _inner.transcribe(pcm16kMono, language: language);
    _onTranscribed(watch.elapsed, text);
    return text;
  }

  @override
  void addCloseListener(void Function() listener) =>
      _inner.addCloseListener(listener);

  /// Not the owner: the wrapped recognizer stays open.
  @override
  Future<void> close() async {}
}

/// Whichever recognizer is active when the turn transcribes: it waits for a
/// recognizer switch in progress (a demo just entered) and uses the model
/// that switch loads, never one the switch is closing.
final class ActiveRecognizer implements SpeechRecognizer {
  ActiveRecognizer(this._stt, {this._expected});

  final SttService _stt;

  /// The model this turn must be transcribed by; null accepts any.
  final ModelId? _expected;

  /// The active model's default; VoiceSession never sets one per call.
  @override
  String? get language => _stt.isLoaded ? _stt.recognizer.language : null;

  @override
  set language(String? value) {
    if (_stt.isLoaded) _stt.recognizer.language = value;
  }

  @override
  Future<String> transcribe(Uint8List pcm16kMono, {String? language}) {
    if (language != null) {
      // moonshine throws on any language; Demo 1 sets Whisper's at load.
      throw ArgumentError.value(language, 'language', 'set it at load');
    }
    return _stt.transcribe(pcm16kMono, expected: _expected);
  }

  @override
  void addCloseListener(void Function() listener) {}

  /// Not the owner: the service keeps the recognizer open.
  @override
  Future<void> close() async {}
}

/// A typed turn: "transcribes" to the typed text without touching the STT
/// model, so typed and spoken turns share one VoiceSession path.
final class TypedTextRecognizer implements SpeechRecognizer {
  TypedTextRecognizer(this._text);

  final String _text;

  @override
  String? language;

  @override
  Future<String> transcribe(Uint8List pcm16kMono, {String? language}) async =>
      _text;

  @override
  void addCloseListener(void Function() listener) {}

  @override
  Future<void> close() async {}
}

/// Times each synthesized clause for the overlay.
final class InstrumentedSynthesizer implements SpeechSynthesizer {
  InstrumentedSynthesizer(this._inner, {required this._onSynthesized});

  final SpeechSynthesizer _inner;
  final void Function(Duration elapsed, String text, int bytes) _onSynthesized;

  @override
  int get sampleRate => _inner.sampleRate;

  @override
  Future<Uint8List> synthesize(String text) async {
    final watch = Stopwatch()..start();
    final pcm = await _inner.synthesize(text);
    _onSynthesized(watch.elapsed, text, pcm.length);
    return pcm;
  }

  @override
  void addCloseListener(void Function() listener) =>
      _inner.addCloseListener(listener);

  @override
  Future<void> close() async {}
}

/// Speaks what should be heard, not what is shown: each clause goes through
/// [toSpokenText] (citations, markdown, URLs removed). A clause with nothing
/// speakable left returns zero bytes without calling the model; the
/// assistant treats zero-byte chunks as "nothing to play".
final class SpokenSynthesizer implements SpeechSynthesizer {
  SpokenSynthesizer(this._inner);

  final SpeechSynthesizer _inner;

  @override
  int get sampleRate => _inner.sampleRate;

  @override
  Future<Uint8List> synthesize(String text) async {
    final spoken = toSpokenText(text);
    if (spoken.isEmpty) return Uint8List(0);
    return _inner.synthesize(spoken);
  }

  @override
  void addCloseListener(void Function() listener) =>
      _inner.addCloseListener(listener);

  @override
  Future<void> close() async {}
}

/// "Speak replies" off: every clause is zero bytes, so nothing plays and no
/// TTS time is spent. Keeps the TTS model's rate so the events stay honest.
final class SilentSynthesizer implements SpeechSynthesizer {
  const SilentSynthesizer(this.sampleRate);

  @override
  final int sampleRate;

  @override
  Future<Uint8List> synthesize(String text) async => Uint8List(0);

  @override
  void addCloseListener(void Function() listener) {}

  @override
  Future<void> close() async {}
}
