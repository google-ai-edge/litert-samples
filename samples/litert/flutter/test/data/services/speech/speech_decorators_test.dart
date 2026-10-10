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

import 'dart:async';
import 'dart:typed_data';

import 'package:flutter_edge_ai/flutter_edge_ai.dart' show SttModelType;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/services/speech/speech_decorators.dart';
import 'package:litert_edge_demos/data/services/speech/stt_service.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/utils/spoken_text.dart';

import '../../../fakes/fake_speech.dart';

/// A recognizer that records its close listeners and each call's language.
class _ListenedRecognizer extends FakeRecognizer {
  _ListenedRecognizer(super.text);

  final List<void Function()> listeners = [];
  final List<String?> languages = [];

  @override
  Future<String> transcribe(Uint8List pcm16kMono, {String? language}) {
    languages.add(language);
    return super.transcribe(pcm16kMono, language: language);
  }

  @override
  void addCloseListener(void Function() listener) => listeners.add(listener);
}

/// A synthesizer that records its close listeners; [error] fails a call.
class _ListenedSynth extends RecordingSynth {
  _ListenedSynth() : super(sampleRate: 22050);

  final List<void Function()> listeners = [];
  Object? error;

  @override
  Future<Uint8List> synthesize(String text) {
    if (error case final e?) {
      synthesized.add(text);
      return Future.error(e);
    }
    return super.synthesize(text);
  }

  @override
  void addCloseListener(void Function() listener) => listeners.add(listener);
}

void main() {
  final pcm = Uint8List(320);

  group('InstrumentedRecognizer', () {
    test('forwards the audio and the language, reports the time and the '
        'text, returns the text', () async {
      final inner = _ListenedRecognizer('hello');
      final reports = <(Duration, String)>[];
      final recognizer = InstrumentedRecognizer(
        inner,
        onTranscribed: (elapsed, text) => reports.add((elapsed, text)),
      );

      final text = await recognizer.transcribe(pcm, language: 'de');

      expect(text, 'hello');
      expect(inner.pcms.single, same(pcm));
      expect(inner.languages, ['de']);
      expect(reports.single.$2, 'hello');
      expect(reports.single.$1, greaterThanOrEqualTo(Duration.zero));
    });

    test('the language is the wrapped model\'s', () {
      final inner = FakeRecognizer();
      final recognizer = InstrumentedRecognizer(
        inner,
        onTranscribed: (_, _) {},
      );

      expect(recognizer.language, 'en');
      recognizer.language = 'fr';
      expect(inner.language, 'fr');
      expect(recognizer.language, 'fr');
    });

    test('a failed transcription propagates and reports nothing', () async {
      final inner = FakeRecognizer()..error = StateError('stt boom');
      var reports = 0;
      final recognizer = InstrumentedRecognizer(
        inner,
        onTranscribed: (_, _) => reports++,
      );

      await expectLater(recognizer.transcribe(pcm), throwsStateError);
      expect(reports, 0);
    });

    test('close listeners go to the wrapped model; close never closes '
        'it', () async {
      final inner = _ListenedRecognizer('x');
      final recognizer = InstrumentedRecognizer(
        inner,
        onTranscribed: (_, _) {},
      );
      void listener() {}

      recognizer.addCloseListener(listener);
      await recognizer.close();

      expect(inner.listeners, [listener]);
      expect(inner.closeCalls, 0);
    });
  });

  group('ActiveRecognizer', () {
    late FakeRecognizer whisper;
    late FakeRecognizer moonshine;
    late Completer<void>? loadGate;
    late SttService stt;

    Future<void> install(ModelId id) =>
        stt.install(id, source: fakeSttSource(id), onProgress: (_) {});

    setUp(() {
      whisper = FakeRecognizer('whisper heard');
      moonshine = FakeRecognizer('moonshine heard')..language = null;
      loadGate = null;
      stt = SttService(
        install: (config, source, onProgress) async => fakeSttModelId(config),
        load: (config) async {
          await loadGate?.future;
          return config.type == SttModelType.moonshine ? moonshine : whisper;
        },
      );
    });

    tearDown(() => stt.close());

    test('before any model is loaded: no language, setting one is a no-op, '
        'and a transcription fails as not ready', () async {
      final recognizer = ActiveRecognizer(stt);

      expect(recognizer.language, isNull);
      recognizer.language = 'en'; // nothing to set it on
      expect(recognizer.language, isNull);
      await expectLater(
        recognizer.transcribe(pcm),
        throwsA(isA<SpeechNotReadyException>()),
      );
    });

    test('the active model transcribes; its language is read and set '
        'through', () async {
      await install(ModelId.whisperBase);
      await stt.load(ModelId.whisperBase);
      final recognizer = ActiveRecognizer(stt);

      expect(await recognizer.transcribe(pcm), 'whisper heard');
      expect(whisper.pcms.single, same(pcm));
      expect(recognizer.language, 'en');
      recognizer.language = 'de';
      expect(whisper.language, 'de');
    });

    test('a per-call language is refused before anything runs (moonshine '
        'throws on any; Demo 1 sets Whisper\'s at load)', () async {
      await install(ModelId.whisperBase);
      await stt.load(ModelId.whisperBase);
      final recognizer = ActiveRecognizer(stt);

      expect(
        () => recognizer.transcribe(pcm, language: 'en'),
        throwsArgumentError,
      );
      expect(whisper.calls, 0);
    });

    test('the demo\'s model is required: another loaded one is never used '
        '(a failed switch)', () async {
      await install(ModelId.whisperBase);
      await stt.load(ModelId.whisperBase);
      final recognizer = ActiveRecognizer(stt, expected: ModelId.moonshineTiny);

      await expectLater(
        recognizer.transcribe(pcm),
        throwsA(
          isA<SpeechNotReadyException>().having(
            (e) => '$e',
            'message',
            contains('the last switch failed'),
          ),
        ),
      );
      expect(whisper.calls, 0);
    });

    test('a transcription during a switch waits for it and uses the model it '
        'loads', () async {
      await install(ModelId.whisperBase);
      await stt.load(ModelId.whisperBase);
      await install(ModelId.moonshineTiny);
      final gate = loadGate = Completer<void>();
      final switching = stt.load(ModelId.moonshineTiny);
      final recognizer = ActiveRecognizer(stt, expected: ModelId.moonshineTiny);

      var done = false;
      final text = recognizer.transcribe(pcm).whenComplete(() => done = true);
      await pumpEventQueue();
      expect(done, isFalse, reason: 'waits for the switch');

      gate.complete();
      expect(await text, 'moonshine heard');
      expect(whisper.calls, 0);
      await switching;
    });

    test(
      'close and close listeners never touch the service\'s model',
      () async {
        await install(ModelId.whisperBase);
        await stt.load(ModelId.whisperBase);
        final recognizer = ActiveRecognizer(stt)..addCloseListener(() {});

        await recognizer.close();

        expect(whisper.closeCalls, 0);
        expect(stt.isLoaded, isTrue);
      },
    );
  });

  group('TypedTextRecognizer', () {
    test('"transcribes" any audio to the typed text', () async {
      final recognizer = TypedTextRecognizer('What time is it?');

      expect(await recognizer.transcribe(pcm), 'What time is it?');
      expect(await recognizer.transcribe(Uint8List(0)), 'What time is it?');
      expect(
        await recognizer.transcribe(pcm, language: 'de'),
        'What time is it?',
      );
    });

    test(
      'holds a language of its own; close and listeners are no-ops',
      () async {
        final recognizer = TypedTextRecognizer('x')..language = 'fr';
        recognizer.addCloseListener(() => fail('never called'));
        await recognizer.close();

        expect(recognizer.language, 'fr');
        expect(await recognizer.transcribe(pcm), 'x');
      },
    );
  });

  group('InstrumentedSynthesizer', () {
    test('forwards the clause, reports the time, the text and the byte count, '
        'returns the audio', () async {
      final inner = RecordingSynth();
      final reports = <(Duration, String, int)>[];
      final synth = InstrumentedSynthesizer(
        inner,
        onSynthesized: (elapsed, text, bytes) =>
            reports.add((elapsed, text, bytes)),
      );

      final audio = await synth.synthesize('Hello there.');

      expect(audio, hasLength(960));
      expect(inner.synthesized, ['Hello there.']);
      expect(reports.single.$2, 'Hello there.');
      expect(reports.single.$3, 960);
      expect(reports.single.$1, greaterThanOrEqualTo(Duration.zero));
      expect(synth.sampleRate, 24000);
    });

    test('a zero-byte clause is reported with 0 bytes', () async {
      final inner = RecordingSynth()..emptyOn = '(';
      final bytes = <int>[];
      final synth = InstrumentedSynthesizer(
        inner,
        onSynthesized: (_, _, n) => bytes.add(n),
      );

      expect(await synth.synthesize('(laughs)'), isEmpty);
      expect(bytes, [0]);
    });

    test('a failed clause propagates and reports nothing', () async {
      final inner = RecordingSynth()..throwOn = 'boom';
      var reports = 0;
      final synth = InstrumentedSynthesizer(
        inner,
        onSynthesized: (_, _, _) => reports++,
      );

      await expectLater(synth.synthesize('boom'), throwsStateError);
      expect(reports, 0);
    });

    test('close listeners go to the wrapped model; close never closes '
        'it', () async {
      final inner = _ListenedSynth();
      final synth = InstrumentedSynthesizer(inner, onSynthesized: (_, _, _) {});
      void listener() {}

      synth.addCloseListener(listener);
      await synth.close();

      expect(inner.listeners, [listener]);
      expect(inner.closeCalls, 0);
      expect(synth.sampleRate, 22050);
    });
  });

  group('SpokenSynthesizer', () {
    test(
      'speaks what should be heard: the clause through toSpokenText',
      () async {
        final inner = RecordingSynth();
        final synth = SpokenSynthesizer(inner);
        const shown = 'See **the guide** [1] at https://example.com/guide.';

        final audio = await synth.synthesize(shown);

        expect(audio, isNotEmpty);
        expect(inner.synthesized, [toSpokenText(shown)]);
        expect(inner.synthesized.single, isNot(contains('**')));
        expect(inner.synthesized.single, isNot(contains('[1]')));
      },
    );

    test('a clause with nothing speakable left is zero bytes and never '
        'reaches the model', () async {
      final inner = RecordingSynth();
      final synth = SpokenSynthesizer(inner);

      expect(toSpokenText('[1] [2]'), isEmpty, reason: 'the premise');
      expect(await synth.synthesize('[1] [2]'), isEmpty);
      expect(inner.synthesized, isEmpty);
    });

    test('a model failure propagates', () async {
      final inner = _ListenedSynth()..error = StateError('synth boom');
      final synth = SpokenSynthesizer(inner);

      await expectLater(synth.synthesize('Hello.'), throwsStateError);
    });

    test('keeps the model\'s rate; listeners go to it; close never closes '
        'it', () async {
      final inner = _ListenedSynth();
      final synth = SpokenSynthesizer(inner);
      void listener() {}

      synth.addCloseListener(listener);
      await synth.close();

      expect(synth.sampleRate, 22050);
      expect(inner.listeners, [listener]);
      expect(inner.closeCalls, 0);
    });
  });

  group('SilentSynthesizer', () {
    test('every clause is zero bytes at the model\'s rate', () async {
      const synth = SilentSynthesizer(24000);

      expect(await synth.synthesize('Hello there.'), isEmpty);
      expect(await synth.synthesize(''), isEmpty);
      expect(synth.sampleRate, 24000);
      synth.addCloseListener(() => fail('never called'));
      await synth.close();
    });
  });
}
