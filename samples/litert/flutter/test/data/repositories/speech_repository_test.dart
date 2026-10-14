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

import 'package:flutter_edge_ai_speech/flutter_edge_ai_speech.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/repositories/speech_repository.dart';
import 'package:litert_edge_demos/data/services/speech/stt_service.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/utils/result.dart';
import 'package:litert_edge_demos/utils/wav.dart';

import '../../fakes/fake_speech.dart';

/// The speech repository's audio turns, transcript corrections and the
/// transcription outside a turn.
void main() {
  final pcm = Uint8List.fromList(List.generate(6400, (i) => i % 5));

  VoiceResponder responder({
    List<String>? heard,
    List<Uint8List>? wavs,
    bool takesAudio = true,
  }) => VoiceResponder(
    respond: (text) {
      heard?.add(text);
      return Stream.value('Ok.');
    },
    respondToAudio: takesAudio
        ? (wav) {
            wavs?.add(wav);
            return Stream.value('Paris.');
          }
        : null,
    stop: () async {},
  );

  test('an audio turn: the responder gets the recording as a 16 kHz mono '
      'WAV, no STT runs, and the turn completes with an empty '
      'transcript', () async {
    final recognizer = FakeRecognizer();
    final speech = await loadedSpeech(
      recognizer: recognizer,
      synthesizer: RecordingSynth(),
    );
    final wavs = <Uint8List>[];

    final turn = speech.startTurn(
      pcm: pcm,
      audio: true,
      responder: responder(wavs: wavs),
      speak: false,
    );
    final events = await (turn as Ok<VoiceTurn>).value.events.toList();

    expect(recognizer.calls, 0);
    expect(pcm16FromWav(wavs.single), pcm);
    expect(events.whereType<VoiceTranscriptEvent>(), isEmpty);
    final done = events.whereType<VoiceTurnCompleteEvent>().single;
    expect(done.transcript, '');
    expect(done.replyText, 'Paris.');
  });

  test('an audio turn needs no recognizer: none loaded is fine', () async {
    final speech = SpeechRepository(
      stt: fakeSttService(),
      tts: fakeTtsService(),
    );
    expect(
      speech.startTurn(
        pcm: pcm,
        audio: true,
        responder: responder(),
        speak: false,
      ),
      isA<Ok<VoiceTurn>>(),
    );
    expect(
      speech.startTurn(pcm: pcm, responder: responder(), speak: false),
      isA<Error<VoiceTurn>>(),
      reason: 'a transcript turn does',
    );
  });

  test('an audio turn without a responder that takes it, or without the '
      'recording, fails before it starts', () async {
    final speech = await loadedSpeech(
      recognizer: FakeRecognizer(),
      synthesizer: RecordingSynth(),
    );
    expect(
      speech.startTurn(
        pcm: pcm,
        audio: true,
        responder: responder(takesAudio: false),
        speak: false,
      ),
      isA<Error<VoiceTurn>>(),
    );
    expect(
      speech.startTurn(audio: true, responder: responder(), speak: false),
      isA<Error<VoiceTurn>>(),
    );
  });

  test('a transcript turn\'s words pass through the corrector before the '
      'transcript event and the responder', () async {
    final speech = await loadedSpeech(
      recognizer: FakeRecognizer('What is littered?'),
      synthesizer: RecordingSynth(),
    );
    final heard = <String>[];
    final turn = speech.startTurn(
      pcm: pcm,
      responder: responder(heard: heard),
      speak: false,
      correct: (t) => t.replaceAll('littered', 'LiteRT'),
    );
    final events = await (turn as Ok<VoiceTurn>).value.events.toList();
    expect(
      events.whereType<VoiceTranscriptEvent>().single.text,
      'What is LiteRT?',
    );
    expect(heard, ['What is LiteRT?']);
  });

  test('transcribe outside a turn: the words, or the failure as an Error; '
      'the active recognizer is followed', () async {
    final recognizer = FakeRecognizer('Hello there.');
    final speech = await loadedSpeech(
      recognizer: recognizer,
      synthesizer: RecordingSynth(),
    );
    expect(speech.activeStt.value?.id, ModelId.whisperBase);
    expect(
      await speech.transcribe(pcm, expected: ModelId.whisperBase),
      isA<Ok<String>>().having((r) => r.value, 'value', 'Hello there.'),
    );
    expect(recognizer.pcms.single, pcm);
    final wrong = await speech.transcribe(pcm, expected: ModelId.moonshineTiny);
    expect(
      (wrong as Error<String>).error,
      isA<SpeechNotReadyException>(),
      reason: 'never served by another model',
    );
    recognizer.error = StateError('worker died');
    expect(await speech.transcribe(pcm), isA<Error<String>>());
  });
}
