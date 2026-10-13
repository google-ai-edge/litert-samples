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
import 'package:flutter_edge_ai_speech/flutter_edge_ai_speech.dart'
    show VoiceResponder;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/speech_repository.dart';
import 'package:litert_edge_demos/data/services/speech/stt_service.dart';
import 'package:litert_edge_demos/data/services/speech/tts_service.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../../fakes/fake_speech.dart';

void main() {
  group('SttService', () {
    test(
      'loads with the catalog config and warms up on 0.5 s of silence',
      () async {
        final recognizer = FakeRecognizer('');
        final seen = <SttConfig>[];
        final stt = SttService(
          install: (config, source, onProgress) async => 'whisper_base_30s_i8',
          load: (config) async {
            seen.add(config);
            return recognizer;
          },
        );

        expect(
          await stt.install(
            ModelId.whisperBase,
            source: fakeSttSource(ModelId.whisperBase),
            onProgress: (_) {},
          ),
          isA<Ok<String>>(),
        );
        expect(await stt.load(ModelId.whisperBase), isA<Ok<Duration>>());
        expect(seen.single, same(kWhisperSttConfig));
        expect(await stt.warmUp(), isA<Ok<Duration>>());

        expect(recognizer.pcms.single.length, 16000, reason: '0.5 s at 16 kHz');
        expect(recognizer.pcms.single.every((b) => b == 0), isTrue);
        expect(stt.active.value?.id, ModelId.whisperBase);
        expect(stt.active.value?.warmUpTime, isNotNull);
        await stt.close();
        expect(recognizer.closeCalls, 1);
      },
    );

    test('an install failure is a Result.error', () async {
      final stt = SttService(
        install: (config, source, onProgress) async =>
            throw Exception('offline'),
        load: (config) async => FakeRecognizer(),
      );
      final result = await stt.install(
        ModelId.whisperBase,
        source: fakeSttSource(ModelId.whisperBase),
        onProgress: (_) {},
      );
      expect((result as Error<String>).error.toString(), contains('offline'));
    });

    test('close() during a load closes what the load delivers', () async {
      final gate = Completer<void>();
      final recognizer = FakeRecognizer();
      final stt = fakeSttService(recognizer: recognizer, loadGate: gate);
      // A load re-registers from the files the install recorded.
      await stt.install(
        ModelId.whisperBase,
        source: fakeSttSource(ModelId.whisperBase),
        onProgress: (_) {},
      );

      final loading = stt.load(ModelId.whisperBase);
      await Future<void>.delayed(Duration.zero);
      final closing = stt.close();
      gate.complete();

      expect(await loading, isA<Error<Duration>>());
      await closing;
      expect(recognizer.closeCalls, 1);
      expect(stt.isLoaded, isFalse);
    });

    test('a load before any install fails visibly', () async {
      final stt = fakeSttService();
      addTearDown(stt.close);

      final result = await stt.load(ModelId.whisperBase);

      expect(result, isA<Error<Duration>>());
      expect(
        (result as Error<Duration>).error.toString(),
        contains('Whisper base (STT) was never installed'),
      );
    });

    group('one active recognizer, switched per demo', () {
      late FakeRecognizer whisper;
      late FakeRecognizer moonshine;
      late List<String> log;
      late SttService stt;
      final seen = <SttConfig>[];

      setUp(() async {
        whisper = FakeRecognizer('whisper heard');
        moonshine = FakeRecognizer('moonshine heard');
        log = [];
        seen.clear();
        stt = fakeSttService(
          log: log,
          recognizerFor: (config) {
            seen.add(config);
            return config.type == SttModelType.moonshine ? moonshine : whisper;
          },
        );
        // Setup: both installed, Whisper loaded last.
        await stt.install(
          ModelId.moonshineTiny,
          source: fakeSttSource(ModelId.moonshineTiny),
          onProgress: (_) {},
        );
        await stt.load(ModelId.moonshineTiny);
        await stt.install(
          ModelId.whisperBase,
          source: fakeSttSource(ModelId.whisperBase),
          onProgress: (_) {},
        );
        await stt.load(ModelId.whisperBase);
        log.clear();
      });

      tearDown(() => stt.close());

      test('activate closes the old model, restores the spec, loads the new '
          'one (moonshine without a language)', () async {
        expect(moonshine.closeCalls, 1, reason: 'setup switched once');

        final result = await stt.activate(ModelId.moonshineTiny);

        final active = (result as Ok<ActiveStt>).value;
        expect(active.id, ModelId.moonshineTiny);
        expect(active.modelId, 'moonshine_tiny_5s_f32');
        expect(active.switchTime, isNotNull);
        expect(active.warmUpTime, isNull, reason: 'no warm-up on a switch');
        expect(whisper.closeCalls, 1);
        expect(log, [
          'install moonshine_tiny_5s_f32',
          'load moonshine_tiny_5s_f32',
        ]);
        expect(seen.last.language, isNull, reason: 'moonshine throws on any');
        expect(await stt.transcribe(Uint8List(10)), 'moonshine heard');
      });

      test('activating the active model is a no-op', () async {
        await stt.activate(ModelId.whisperBase);
        expect(log, isEmpty);
        expect(whisper.closeCalls, 0);
      });

      test('a question asked during the switch is transcribed by the new '
          'model, never the one being closed', () async {
        final switching = stt.activate(ModelId.moonshineTiny);
        expect(stt.isSwitching, isTrue);
        final text = await stt.transcribe(Uint8List(10));
        expect(text, 'moonshine heard');
        expect(whisper.calls, 0);
        await switching;
      });

      test('the switch waits for a transcription still running on the old '
          'model before closing it', () async {
        final gate = whisper.gate = Completer<void>();
        final transcribing = stt.transcribe(Uint8List(10));
        await Future<void>.delayed(Duration.zero);
        final switching = stt.activate(ModelId.moonshineTiny);
        await Future<void>.delayed(const Duration(milliseconds: 20));
        expect(whisper.closeCalls, 0, reason: 'still transcribing');

        gate.complete();
        expect(await transcribing, 'whisper heard');
        await switching;
        expect(whisper.closeCalls, 1);
        expect(stt.active.value?.id, ModelId.moonshineTiny);
      });

      test(
        'a switch replaced by a newer one before it ran is skipped',
        () async {
          final gate = whisper.gate = Completer<void>();
          final busy = stt.transcribe(Uint8List(10)); // holds the drain
          await Future<void>.delayed(Duration.zero);
          final toMoonshine = stt.activate(ModelId.moonshineTiny);
          final backToWhisper = stt.activate(ModelId.whisperBase);
          gate.complete();
          await busy;
          await toMoonshine;
          await backToWhisper;
          expect(stt.active.value?.id, ModelId.whisperBase);
          expect(log.where((l) => l.startsWith('load moonshine')), isEmpty);
        },
      );
    });
  });

  group('a failed switch never serves the wrong model', () {
    late FakeRecognizer whisper;
    late FakeRecognizer moonshine;
    late SttService stt;
    var failWhisperRestore = false;

    setUp(() async {
      whisper = FakeRecognizer('whisper heard');
      moonshine = FakeRecognizer('moonshine heard');
      failWhisperRestore = false;
      stt = SttService(
        install: (config, source, onProgress) async {
          if (failWhisperRestore && config.type == SttModelType.whisper) {
            throw Exception('disk full');
          }
          return fakeSttModelId(config);
        },
        load: (config) async =>
            config.type == SttModelType.moonshine ? moonshine : whisper,
      );
      await stt.install(
        ModelId.whisperBase,
        source: fakeSttSource(ModelId.whisperBase),
        onProgress: (_) {},
      );
      await stt.load(ModelId.whisperBase);
      await stt.install(
        ModelId.moonshineTiny,
        source: fakeSttSource(ModelId.moonshineTiny),
        onProgress: (_) {},
      );
      await stt.load(ModelId.moonshineTiny);
    });

    tearDown(() => stt.close());

    test(
      'the spec restore throws: the switch fails, moonshine stays loaded, '
      'and a Demo 1 transcription fails fast instead of using moonshine',
      () async {
        failWhisperRestore = true;
        expect(
          await stt.activate(ModelId.whisperBase),
          isA<Error<ActiveStt>>(),
        );
        expect(stt.active.value?.id, ModelId.moonshineTiny);
        expect(stt.switchError.value, contains('disk full'), reason: 'overlay');

        await expectLater(
          stt.transcribe(Uint8List(10), expected: ModelId.whisperBase),
          throwsA(isA<SpeechNotReadyException>()),
        );
        expect(moonshine.calls, 0, reason: 'never served by the wrong model');
        expect(
          await stt.transcribe(Uint8List(10), expected: ModelId.moonshineTiny),
          'moonshine heard',
        );

        failWhisperRestore = false;
        expect(await stt.activate(ModelId.whisperBase), isA<Ok<ActiveStt>>());
        expect(stt.switchError.value, isNull, reason: 'cleared by a success');
      },
    );

    test('through a voice turn: the turn fails visibly', () async {
      failWhisperRestore = true;
      await stt.activate(ModelId.whisperBase);
      final speech = SpeechRepository(stt: stt, tts: fakeTtsService());
      final turn = speech.startTurn(
        pcm: Uint8List(3200),
        expectedStt: ModelId.whisperBase,
        responder: VoiceResponder(
          respond: (text) => Stream.value('reply to $text'),
          stop: () async {},
        ),
        speak: false,
      );
      final events = (turn as Ok<VoiceTurn>).value.events;
      await expectLater(
        events.toList(),
        throwsA(isA<SpeechNotReadyException>()),
      );
      expect(moonshine.calls, 0);
    });
  });

  group('TtsService', () {
    test('a synthesizer at another rate than the catalog fails the load and '
        'is closed (playing at the wrong rate shifts the pitch)', () async {
      final synth = RecordingSynth(sampleRate: 22050);
      final tts = fakeTtsService(synthesizer: synth);

      final result = await tts.load();

      expect(
        (result as Error<Duration>).error,
        isA<TtsSampleRateException>()
            .having((e) => e.expected, 'expected', kTtsConfig.sampleRate)
            .having((e) => e.actual, 'actual', 22050),
      );
      expect(synth.closeCalls, 1);
      expect(tts.isLoaded, isFalse);
    });

    test('warm-up must produce audio', () async {
      final synth = RecordingSynth()..emptyOn = 'Ready';
      final tts = fakeTtsService(synthesizer: synth);
      await tts.load();

      final result = await tts.warmUp();

      expect((result as Error<Duration>).error, isA<TtsSilentException>());
      expect(synth.synthesized, ['Ready.']);
    });

    test('warm-up synthesizes "Ready." and reports the time', () async {
      final tts = fakeTtsService();
      await tts.load();
      expect(await tts.warmUp(), isA<Ok<Duration>>());
      expect(tts.sampleRate, 24000);
    });
  });

  group('SpeechRepository', () {
    test('a voice turn before the models are loaded fails, a typed turn with '
        'speech off does not need them', () async {
      final speech = SpeechRepository(
        stt: fakeSttService(),
        tts: fakeTtsService(),
      );
      final responder = _noResponder();

      expect(
        speech.startTurn(responder: responder, speak: false),
        isA<Error<VoiceTurn>>(),
      );
      expect(
        speech.startTurn(typedText: 'Hi', responder: responder, speak: true),
        isA<Error<VoiceTurn>>(),
      );
      expect(
        speech.startTurn(typedText: 'Hi', responder: responder, speak: false),
        isA<Ok<VoiceTurn>>(),
      );
    });
  });
}

VoiceResponder _noResponder() => VoiceResponder(
  respond: (_) => const Stream<String>.empty(),
  stop: () async {},
);
