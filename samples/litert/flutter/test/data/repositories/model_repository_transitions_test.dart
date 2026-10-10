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
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_edge_ai/flutter_edge_ai.dart'
    show ModelType, PreferredBackend, SpeechSynthesizer, SttModelType;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/model_repository.dart';
import 'package:litert_edge_demos/data/services/detector/detector_engine.dart'
    show kDetectorFileNotFound;
import 'package:litert_edge_demos/data/services/detector/detector_service.dart';
import 'package:litert_edge_demos/data/services/hardware/native_log_tap.dart';
import 'package:litert_edge_demos/data/services/knowledge/embedder_service.dart';
import 'package:litert_edge_demos/data/services/llm/llm_service.dart';
import 'package:litert_edge_demos/data/services/model_store/bundled_model_files.dart';
import 'package:litert_edge_demos/data/services/speech/stt_service.dart';
import 'package:litert_edge_demos/data/services/speech/tts_service.dart';
import 'package:litert_edge_demos/domain/models/chat_model.dart';
import 'package:litert_edge_demos/domain/models/chat_model_config.dart';
import 'package:litert_edge_demos/domain/models/detection.dart';
import 'package:litert_edge_demos/domain/models/detector_choice.dart';
import 'package:litert_edge_demos/domain/models/detector_spec.dart'
    show kDetModelName;
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/ports/chat_model_planner.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../fakes/fake_bundled_files.dart';
import '../../fakes/fake_knowledge.dart';
import '../../fakes/fake_llm_service.dart';
import '../../fakes/fake_model_files.dart';
import '../../fakes/fake_speech.dart';

/// Characterization of `ModelRepository`'s load pipeline: the exact states
/// every model publishes, in order, for success and for a failure at each
/// step, and what its ready row records (timings, backend, native log).
/// `(same)` is a publish that changed nothing (a state published again).
void main() {
  late Directory dir;
  late Map<String, String> embedderFiles;

  setUpAll(() {
    dir = Directory.systemTemp.createTempSync('model_transitions');
    String file(String name) =>
        (File('${dir.path}/$name')..writeAsBytesSync([1])).path;
    embedderFiles = {
      kBundledEmbedderModel.asset: file('m.tflite'),
      kBundledEmbedderTokenizer.asset: file('t.model'),
    };
  });

  tearDownAll(() => dir.deleteSync(recursive: true));

  _Rig rig({
    String gemmaModelPath = kTestChatModelPath,
    ChatModelPlanner? chatModels,
    _SttScript? stt,
    Exception? ttsInstallError,
    SpeechSynthesizer? synthesizer,
    Exception? embedderInstallError,
    Completer<void>? embedderInstallGate,
    FakeEmbeddingModel? embeddingModel,
    bool bundledEmbedder = true,
    String detectorBackend = '',
    Future<Result<DetectorBackendChoice>> Function()? detectorBackendChoice,
  }) {
    final rig = _Rig(
      gemmaModelPath: gemmaModelPath,
      chatModels: chatModels,
      stt: stt ?? _SttScript(),
      ttsInstallError: ttsInstallError,
      synthesizer: synthesizer,
      embedderInstallError: embedderInstallError,
      embedderInstallGate: embedderInstallGate,
      embeddingModel: embeddingModel,
      bundled: bundledEmbedder
          ? FakeBundledFiles.at(embedderFiles)
          : FakeBundledFiles(),
      detectorBackend: detectorBackend,
      detectorBackendChoice: detectorBackendChoice,
    );
    addTearDown(rig.models.close);
    return rig;
  }

  group('a full setup', () {
    test(
      'publishes each model\'s steps in load order, and nothing else',
      () async {
        final r = rig();

        expect(await r.models.prepareAll(), isA<Ok<void>>());

        expect(r.transitions, [
          ..._chat,
          ..._whisper,
          ..._inflect,
          ..._detector,
          ..._moonshine,
          ..._embedder,
        ]);
        expect(r.log.lines, [
          'llm install',
          'llm load',
          'llm warm-up',
          'stt install whisper',
          'stt load whisper',
          'tts install',
          'tts load',
          'detector load gpu',
          'stt install moonshine',
          'stt load moonshine',
          'embedder install',
          'embedder load',
        ]);
      },
    );

    test('each ready row records its own load\'s timings, backend and native '
        'log window', () async {
      final r = rig();
      await r.models.prepareAll();
      LoadedModelInfo info(ModelId id) =>
          (r.models.states.value[id]! as ModelReady).info;

      final chat = info(ModelId.chat);
      expect(chat.modelId, 'gemma-4-E2B-it');
      expect(chat.backend, 'gpu');
      expect(chat.loadTime, const Duration(milliseconds: 2400));
      expect(chat.warmUpTime, const Duration(milliseconds: 150));
      expect(chat.detail, isNull);
      expect(chat.backendReported, isTrue);
      expect(chat.explicitCpu, isFalse);
      expect(chat.nativeLog, [
        'llm load',
        'llm warm-up',
      ], reason: 'from before the load through the warm-up');
      final facts = chat.chat!;
      expect(facts.name, 'Gemma 4 E2B');
      expect(facts.custom, isFalse);
      expect(facts.source, 'GEMMA_MODEL_PATH=$kTestChatModelPath');
      expect(facts.sha256, isNull);
      expect(facts.checksumMatched, isFalse);
      expect(facts.requestedBackend, 'gpu');
      expect(facts.requestedContext, 8192);
      expect(facts.contextTokens, 8192);
      expect(facts.images, isTrue);
      expect(facts.tools, isTrue);
      expect(facts.modelType, 'gemma4');

      void expectSpeech(
        ModelId id,
        String modelId,
        Duration load,
        Duration warmUp,
      ) {
        final row = info(id);
        expect(row.modelId, modelId, reason: '$id');
        expect(row.backend, 'cpu', reason: '$id');
        expect(row.detail, 'CPU (requested)', reason: '$id');
        expect(row.loadTime, load, reason: '$id');
        expect(row.warmUpTime, warmUp, reason: '$id');
        expect(row.backendReported, isFalse, reason: '$id');
        expect(row.explicitCpu, isFalse, reason: '$id');
        expect(row.nativeLog, isEmpty, reason: '$id: no native log tap');
        expect(row.chat, isNull, reason: '$id');
      }

      expectSpeech(
        ModelId.whisperBase,
        'whisper_base_30s_i8',
        _whisperLoad,
        _sttWarmUp,
      );
      expectSpeech(
        ModelId.moonshineTiny,
        'moonshine_tiny_5s_f32',
        _moonshineLoad,
        _sttWarmUp,
      );
      expectSpeech(ModelId.inflectNano, 'inflect', _ttsLoad, _ttsWarmUp);

      final detector = info(ModelId.yolo26n);
      expect(detector.modelId, kDetModelName);
      expect(detector.backend, 'gpu');
      expect(detector.loadTime, _detCreate + _detVerify);
      expect(detector.warmUpTime, _detFirstRun);
      expect(detector.detail, 'GPU fp32 full');
      expect(detector.explicitCpu, isFalse);
      expect(detector.backendReported, isTrue);
      expect(detector.nativeLog, ['detector load gpu'], reason: 'the load');
      expect(detector.chat, isNull);

      final embedder = info(ModelId.embeddingGemma);
      expect(embedder.modelId, 'embeddinggemma-300M_seq512_mixed-precision');
      expect(embedder.backend, 'cpu');
      expect(embedder.loadTime, _embedderLoad);
      expect(embedder.warmUpTime, _embedderWarmUp);
      expect(embedder.detail, 'CPU · 768-d');
      expect(embedder.backendReported, isTrue);
      expect(embedder.explicitCpu, isFalse);
      expect(embedder.nativeLog, isEmpty);
      expect(embedder.chat, isNull);
    });

    test('a second run publishes nothing: every model is ready', () async {
      final r = rig();
      await r.models.prepareAll();
      final before = r.transitions.length;

      expect(await r.models.prepareAll(), isA<Ok<void>>());

      expect(r.transitions.length, before);
      expect(r.llm.loads, hasLength(1));
    });
  });

  group('the chat model', () {
    test(
      'an install failure fails the slot with its error and stops setup',
      () async {
        final r = rig();
        final error = Exception('the disk is full');
        r.llm.installError = error;

        final result = await r.models.prepareAll();

        expect((result as Error<void>).error, same(error));
        expect(r.transitions, [
          'chat installing',
          'chat failed: Exception: the disk is full',
        ]);
        expect(r.llm.loads, isEmpty);
      },
    );

    test('a load failure fails the slot with its error', () async {
      final r = rig();
      final error = Exception('wrong SoC');
      r.llm.loadError = error;

      final result = await r.models.prepareAll();

      expect((result as Error<void>).error, same(error));
      expect(r.transitions, [
        ..._chat.take(4),
        'chat failed: Exception: wrong SoC',
      ]);
      expect(r.llm.warmUpCalls, 0);
    });

    test(
      'a load on another backend than requested is a failure naming both',
      () async {
        final r = rig();
        r.llm.activeBackend = PreferredBackend.cpu;

        final result = await r.models.prepareAll();

        final error = (result as Error<void>).error;
        expect(error, isA<BackendMismatchException>());
        expect(r.transitions, [..._chat.take(4), 'chat failed: $error']);
        expect(error.toString(), contains('requested gpu'));
        expect(error.toString(), contains('loaded on cpu'));
        expect(r.llm.warmUpCalls, 0);
      },
    );

    test('a warm-up failure fails the slot with its error', () async {
      final r = rig();
      final error = Exception('sampler crashed');
      r.llm.warmUpError = error;

      final result = await r.models.prepareAll();

      expect((result as Error<void>).error, same(error));
      expect(r.transitions, [
        ..._chat.take(5),
        'chat failed: Exception: sampler crashed',
      ]);
      expect(r.models.states.value[ModelId.whisperBase], isA<ModelPending>());
      expect(
        r.llm.unloadCalls,
        1,
        reason: 'the failed model is released, not held until a Retry',
      );
      expect(r.llm.isLoaded, isFalse);
      expect(r.log.lines.sublist(r.log.lines.length - 2), [
        'llm warm-up',
        'llm unload',
      ]);
    });

    test('close() during the warm-up waits for it, then unloads the model '
        'before the services close', () async {
      final r = rig();
      final gate = r.llm.warmUpGate = Completer<void>();

      final setup = r.models.prepareAll();
      await _until(() => r.log.lines.contains('llm warm-up'));
      var closed = false;
      final closing = r.models.close().whenComplete(() => closed = true);
      await _turns(20);
      expect(closed, isFalse, reason: 'close() waits for the warm-up');
      expect(r.llm.closeCalls, 0, reason: 'nothing closes under a warm-up');
      gate.complete();
      await closing;

      expect(await setup, isA<Error<void>>());
      expect(r.log.lines.skip(r.log.lines.indexOf('llm warm-up')), [
        'llm warm-up',
        'llm warm-up done',
        'llm unload',
        'llm close',
      ]);
      expect(r.transitions, [..._chat.take(5)], reason: 'no ready row');
    });

    test(
      'a blocked choice fails the slot (with Retry) and stops setup',
      () async {
        final r = rig(
          gemmaModelPath: '',
          chatModels: _Planner(const ChatPlanBlocked('g3.litertlm is gone')),
        );

        final result = await r.models.prepareAll();

        expect((result as Error<void>).error, isA<ChatModelBlockedException>());
        expect(r.transitions, ['chat failed: g3.litertlm is gone']);
      },
    );

    test(
      'none chosen: the slot is unavailable and the rest still loads',
      () async {
        final r = rig(gemmaModelPath: '');

        expect(await r.models.prepareAll(), isA<Error<void>>());

        expect(r.transitions, [
          'chat unavailable: No chat model yet: choose a .litertlm in the Chat '
              'model card.',
          ..._whisper,
          ..._inflect,
          ..._detector,
          ..._moonshine,
          ..._embedder,
        ]);
      },
    );

    test('refused loads: the slot fails without Retry, setup and reloads are '
        'refused, an unload keeps the refusal on the slot', () async {
      final r = rig();

      r.models.refuseChatModelLoads('a self-test is stuck');
      expect(r.transitions, ['chat failed, no retry: a self-test is stuck']);

      final setup = await r.models.prepareAll();
      expect(
        (setup as Error<void>).error,
        isA<ChatModelBlockedException>().having(
          (e) => e.reason,
          'reason',
          'a self-test is stuck',
        ),
      );
      expect(r.transitions, [
        'chat failed, no retry: a self-test is stuck',
        'chat failed, no retry: a self-test is stuck',
      ]);

      expect(await r.models.reloadChatModel(), isA<Error<void>>());
      expect(r.llm.unloadCalls, 0, reason: 'refused before unloading');

      await r.models.unloadChatModel();
      expect(r.llm.unloadCalls, 1);
      expect(r.transitions, hasLength(2), reason: 'no pending slot');
      expect(r.llm.installs, isEmpty);
    });

    test('close() during the install stops before the load', () async {
      final r = rig();
      final gate = r.llm.installGate = Completer<void>();

      final setup = r.models.prepareAll();
      await _until(() => r.log.lines.contains('llm install'));
      // close() waits for the setup run, which the gated install holds.
      final closing = r.models.close();
      gate.complete();
      await closing;

      expect(await setup, isA<Error<void>>());
      expect(r.llm.loads, isEmpty);
      expect(r.transitions, ['chat installing']);
    });
  });

  group('the required speech models', () {
    test('a Whisper install failure fails its row and stops setup', () async {
      final r = rig(stt: _SttScript()..install.add(SttModelType.whisper));

      final result = await r.models.prepareAll();

      expect((result as Error<void>).error.toString(), contains('whisper'));
      expect(r.transitions, [
        ..._chat,
        'whisperBase installing',
        '(same)',
        'whisperBase failed: Exception: whisper install failed',
      ]);
    });

    test('a Whisper load failure fails its row and stops setup', () async {
      final error = Exception('no Whisper');
      final r = rig(stt: _SttScript()..load[SttModelType.whisper] = error);

      final result = await r.models.prepareAll();

      expect((result as Error<void>).error, same(error));
      expect(r.transitions, [
        ..._chat,
        ..._whisper.take(5),
        'whisperBase failed: Exception: no Whisper',
      ]);
    });

    test('a Whisper warm-up failure fails its row and stops setup; the '
        'recognizer is released', () async {
      final script = _SttScript()..warmUp.add(SttModelType.whisper);
      final r = rig(stt: script);

      expect(await r.models.prepareAll(), isA<Error<void>>());

      expect(r.transitions, [
        ..._chat,
        ..._whisper.take(6),
        'whisperBase failed: Bad state: whisper warm-up failed',
      ]);
      expect(script.loaded.single.closeCalls, 1);
      expect(r.models.activeStt.value, isNull);
    });

    test('a TTS install failure fails its row and stops setup', () async {
      final r = rig(ttsInstallError: Exception('no voice'));

      expect(await r.models.prepareAll(), isA<Error<void>>());

      expect(r.transitions, [
        ..._chat,
        ..._whisper,
        'inflectNano installing',
        '(same)',
        'inflectNano failed: Exception: no voice',
      ]);
    });

    test('a TTS load failure (a wrong sample rate) fails its row', () async {
      final r = rig(synthesizer: RecordingSynth(sampleRate: 22050));

      final result = await r.models.prepareAll();

      final error = (result as Error<void>).error;
      expect(error, isA<TtsSampleRateException>());
      expect(r.transitions, [
        ..._chat,
        ..._whisper,
        ..._inflect.take(4),
        'inflectNano failed: $error',
      ]);
    });

    test('a TTS warm-up failure fails its row; the synthesizer is '
        'released', () async {
      final synth = RecordingSynth()..throwOn = 'Ready';
      final r = rig(synthesizer: synth);

      expect(await r.models.prepareAll(), isA<Error<void>>());

      expect(r.transitions, [
        ..._chat,
        ..._whisper,
        ..._inflect.take(5),
        'inflectNano failed: Bad state: synth boom',
      ]);
      expect(synth.closeCalls, 1, reason: 'released, not held until a Retry');
    });

    test('close() during a speech load stops before the warm-up', () async {
      final gate = Completer<void>();
      final r = rig(stt: _SttScript()..loadGate = gate);

      final setup = r.models.prepareAll();
      await _until(() => r.log.lines.contains('stt load whisper'));
      // close() waits for the STT queue, which the gated load holds.
      final closing = r.models.close();
      gate.complete();
      await closing;

      expect(await setup, isA<Error<void>>());
      expect(r.transitions, [..._chat, ..._whisper.take(5)]);
    });
  });

  group('the optional recognizer (moonshine)', () {
    for (final (step, script, failure) in [
      (
        'install',
        _SttScript()..install.add(SttModelType.moonshine),
        ['moonshineTiny installing', '(same)'],
      ),
      (
        'load',
        _SttScript()..load[SttModelType.moonshine] = Exception('no moonshine'),
        _moonshine.take(5).toList(),
      ),
      (
        'warm-up',
        _SttScript()..warmUp.add(SttModelType.moonshine),
        _moonshine.take(6).toList(),
      ),
    ]) {
      test(
        'a $step failure fails its row (published twice) and setup goes on',
        () async {
          final r = rig(stt: script);

          expect(await r.models.prepareAll(), isA<Ok<void>>());

          final failed =
              r.models.states.value[ModelId.moonshineTiny]! as ModelFailed;
          expect(r.transitions, [
            ..._chat,
            ..._whisper,
            ..._inflect,
            ..._detector,
            ...failure,
            'moonshineTiny failed: ${failed.message}',
            '(same)',
            ..._embedder,
          ]);
          expect(failed.retryable, isTrue);
          expect(failed.backend, isNull);
        },
      );
    }
  });

  group('the detector', () {
    test('a load failure on the GPU names the backend (Demo 3 offers the '
        'other one)', () async {
      final r = rig();
      r.detector.failOn[DetectorBackend.gpu] =
          const DetectorUnavailableException(
            'YOLO26n ran only partly on the GPU',
          );

      expect(await r.models.prepareAll(), isA<Ok<void>>());

      expect(r.transitions, [
        ..._chat,
        ..._whisper,
        ..._inflect,
        'yolo26n loading',
        'yolo26n failed on gpu: YOLO26n ran only partly on the GPU',
        ..._moonshine,
        ..._embedder,
      ]);
    });

    test('a bad file is not a backend failure (no backend offered)', () async {
      final r = rig();
      r.detector.failOn[DetectorBackend.gpu] =
          const DetectorUnavailableException(
            '$kDetectorFileNotFound: /nowhere.tflite',
          );

      await r.models.prepareAll();

      expect(
        r.transitions,
        containsAllInOrder([
          'yolo26n loading',
          'yolo26n failed: $kDetectorFileNotFound: /nowhere.tflite',
        ]),
      );
    });

    test('reloadDetector loads the other backend, then fails visibly back '
        'on the GPU', () async {
      var backend = DetectorBackend.gpu;
      final r = rig(
        detectorBackendChoice: () async => Result.ok(
          DetectorBackendChoice(
            backend: backend,
            source: DetectorChoiceSource.setting,
          ),
        ),
      );
      r.detector.failOn[DetectorBackend.gpu] =
          const DetectorUnavailableException(
            'YOLO26n ran only partly on the GPU',
          );
      await r.models.prepareAll();
      r.transitions.clear();

      backend = DetectorBackend.cpu;
      expect(await r.models.reloadDetector(), isA<Ok<void>>());
      expect(r.transitions, [
        'yolo26n pending',
        'yolo26n loading',
        'yolo26n ready: $kDetModelName on cpu',
      ]);
      final info = (r.models.states.value[ModelId.yolo26n]! as ModelReady).info;
      expect(info.detail, 'CPU (chosen)');
      expect(info.explicitCpu, isTrue);
      expect(info.nativeLog, ['detector load cpu']);
      r.transitions.clear();

      backend = DetectorBackend.gpu;
      final failed = await r.models.reloadDetector();
      expect(
        (failed as Error<void>).error,
        isA<DetectorUnavailableException>().having(
          (e) => e.message,
          'message',
          'YOLO26n ran only partly on the GPU',
        ),
      );
      expect(r.transitions, [
        'yolo26n pending',
        'yolo26n loading',
        'yolo26n failed on gpu: YOLO26n ran only partly on the GPU',
      ]);
    });

    test(
      'an invalid DETECTOR_BACKEND is unavailable; reloading says why',
      () async {
        final r = rig(detectorBackend: 'npu');
        await r.models.prepareAll();
        expect(
          r.transitions,
          contains(
            'yolo26n unavailable: DETECTOR_BACKEND must be gpu or cpu, got '
            '"npu"',
          ),
        );
        r.transitions.clear();

        final result = await r.models.reloadDetector();

        expect(
          (result as Error<void>).error,
          isA<DetectorUnavailableException>().having(
            (e) => e.message,
            'message',
            'DETECTOR_BACKEND must be gpu or cpu, got "npu"',
          ),
        );
        expect(r.transitions, [
          'yolo26n pending',
          'yolo26n unavailable: DETECTOR_BACKEND must be gpu or cpu, got "npu"',
        ]);
        expect(r.detector.sources, isEmpty, reason: 'nothing was loaded');
      },
    );

    test(
      'a step that throws fails the row with the error; setup goes on',
      () async {
        final r = rig(
          detectorBackendChoice: () async => throw StateError('settings gone'),
        );

        expect(await r.models.prepareAll(), isA<Ok<void>>());

        expect(r.transitions, [
          ..._chat,
          ..._whisper,
          ..._inflect,
          'yolo26n failed: Bad state: settings gone',
          ..._moonshine,
          ..._embedder,
        ]);
        final reload = await r.models.reloadDetector();
        expect(
          (reload as Error<void>).error.toString(),
          'Bad state: settings gone',
        );
      },
    );

    test('close() during the load closes what loaded', () async {
      final r = rig();
      final gate = r.detector.gate = Completer<void>();

      final setup = r.models.prepareAll();
      await _until(() => r.log.lines.contains('detector load gpu'));
      final closing = r.models.close();
      gate.complete();
      await closing;

      expect(await setup, isA<Error<void>>());
      expect(
        r.detector.closeCalls,
        2,
        reason: 'the repository\'s close and the orphaned load\'s',
      );
      expect(r.detector.isLoaded, isFalse);
      expect(r.transitions.last, 'yolo26n loading');
    });

    test('close() during a load that fails: nothing is published, the '
        'detector ends closed', () async {
      final r = rig();
      r.detector.failOn[DetectorBackend.gpu] =
          const DetectorUnavailableException(
            'YOLO26n ran only partly on the GPU',
          );
      final gate = r.detector.gate = Completer<void>();

      final setup = r.models.prepareAll();
      await _until(() => r.log.lines.contains('detector load gpu'));
      final closing = r.models.close();
      gate.complete();
      await closing;

      expect(await setup, isA<Error<void>>());
      expect(r.detector.closeCalls, greaterThanOrEqualTo(1));
      expect(r.detector.isLoaded, isFalse);
      expect(r.transitions.last, 'yolo26n loading');
    });
  });

  group('the embedder', () {
    test('a broken bundle fails the row before the install', () async {
      final r = rig(bundledEmbedder: false);

      expect(await r.models.prepareAll(), isA<Ok<void>>());

      expect(r.transitions.skip(r.transitions.length - 2), [
        'embeddingGemma installing',
        'embeddingGemma failed: The built-in EmbeddingGemma: No app bundle in '
            'tests',
      ]);
      expect(r.log.lines, isNot(contains('embedder install')));
    });

    test('an install failure fails the row', () async {
      final r = rig(embedderInstallError: Exception('copy failed'));

      expect(await r.models.prepareAll(), isA<Ok<void>>());

      expect(r.transitions.skip(r.transitions.length - 3), [
        'embeddingGemma installing',
        '(same)',
        'embeddingGemma failed: Exception: copy failed',
      ]);
    });

    test('a load failure (another backend) fails the row', () async {
      final r = rig(
        embeddingModel: FakeEmbeddingModel(backend: PreferredBackend.gpu),
      );

      expect(await r.models.prepareAll(), isA<Ok<void>>());

      expect(r.transitions.skip(r.transitions.length - 2), [
        'embeddingGemma loading',
        'embeddingGemma failed: EmbeddingGemma requested cpu but reports gpu',
      ]);
    });

    test('a warm-up failure fails the row', () async {
      final r = rig(embeddingModel: FakeEmbeddingModel(zeroVectors: true));

      expect(await r.models.prepareAll(), isA<Ok<void>>());

      final failed =
          r.models.states.value[ModelId.embeddingGemma]! as ModelFailed;
      expect(r.transitions.skip(r.transitions.length - 2), [
        'embeddingGemma warming up',
        'embeddingGemma failed: ${failed.message}',
      ]);
      expect(failed.message, contains('all-zero'));
    });

    test('close() during the install stops before the load', () async {
      final gate = Completer<void>();
      final r = rig(embedderInstallGate: gate);

      final setup = r.models.prepareAll();
      await _until(() => r.log.lines.contains('embedder install'));
      // close() waits for the setup run, which the gated install holds.
      final closing = r.models.close();
      gate.complete();
      await closing;

      expect(await setup, isA<Error<void>>());
      expect(r.log.lines, isNot(contains('embedder load')));
      expect(r.transitions.skip(r.transitions.length - 2), [
        'embeddingGemma installing',
        '(same)',
      ]);
    });
  });

  group('the STT switch', () {
    test('a failed switch returns the service\'s error, publishes no state '
        'and says why', () async {
      final script = _SttScript();
      final r = rig(stt: script);
      await r.models.prepareAll();
      final before = List.of(r.transitions);
      final error = Exception('Whisper is gone');
      script.load[SttModelType.whisper] = error;

      final result = await r.models.activateStt(ModelId.whisperBase);

      expect((result as Error<void>).error, same(error));
      expect(r.transitions, before);
      expect(r.models.sttSwitchError.value, error.toString());
      expect(r.models.activeStt.value, isNull);

      script.load.clear();
      expect(await r.models.activateStt(ModelId.whisperBase), isA<Ok<void>>());
      expect(r.models.activeStt.value?.id, ModelId.whisperBase);
      expect(r.models.sttSwitchError.value, isNull);
      expect(r.transitions, before);
    });
  });
}

// The transitions of a full setup over the rig's fakes, per model.

const _chat = [
  'chat installing',
  'chat installing 50%',
  'chat installing 100%',
  'chat loading',
  'chat warming up',
  'chat ready: gemma-4-E2B-it on gpu',
];

const _whisper = [
  'whisperBase installing',
  '(same)',
  'whisperBase installing 50%',
  'whisperBase installing 100%',
  'whisperBase loading',
  'whisperBase warming up',
  'whisperBase ready: whisper_base_30s_i8 on cpu',
];

const _inflect = [
  'inflectNano installing',
  '(same)',
  'inflectNano installing 100%',
  'inflectNano loading',
  'inflectNano warming up',
  'inflectNano ready: inflect on cpu',
];

const _detector = ['yolo26n loading', 'yolo26n ready: $kDetModelName on gpu'];

const _moonshine = [
  'moonshineTiny installing',
  '(same)',
  'moonshineTiny installing 50%',
  'moonshineTiny installing 100%',
  'moonshineTiny loading',
  'moonshineTiny warming up',
  'moonshineTiny ready: moonshine_tiny_5s_f32 on cpu',
  '(same)',
];

const _embedder = [
  'embeddingGemma installing',
  '(same)',
  'embeddingGemma installing 40%',
  'embeddingGemma installing 100%',
  'embeddingGemma loading',
  'embeddingGemma warming up',
  'embeddingGemma ready: embeddinggemma-300M_seq512_mixed-precision on cpu',
];

// Fixed, distinct timings: a row must record its own load's.

const _whisperLoad = Duration(milliseconds: 11);
const _moonshineLoad = Duration(milliseconds: 12);
const _sttWarmUp = Duration(milliseconds: 13);
const _ttsLoad = Duration(milliseconds: 21);
const _ttsWarmUp = Duration(milliseconds: 22);
const _embedderLoad = Duration(milliseconds: 31);
const _embedderWarmUp = Duration(milliseconds: 32);
const _detCreate = Duration(milliseconds: 41);
const _detVerify = Duration(milliseconds: 42);
const _detFirstRun = Duration(milliseconds: 43);

Future<void> _until(bool Function() condition) async {
  for (var i = 0; i < 1000 && !condition(); i++) {
    await Future<void>.delayed(Duration.zero);
  }
  expect(condition(), isTrue, reason: 'condition never held');
}

/// Lets [count] event-loop turns pass (what is not waiting runs meanwhile).
Future<void> _turns(int count) async {
  for (var i = 0; i < count; i++) {
    await Future<void>.delayed(Duration.zero);
  }
}

String _show(ModelState? state) => switch (state) {
  null => 'missing',
  ModelPending() => 'pending',
  ModelInstalling(:final percent) =>
    percent == null ? 'installing' : 'installing $percent%',
  ModelLoading() => 'loading',
  ModelWarmingUp() => 'warming up',
  ModelReady(:final info) => 'ready: ${info.modelId} on ${info.backend}',
  ModelUnavailable(:final reason) => 'unavailable: $reason',
  ModelFailed(:final message, :final backend, :final retryable) =>
    'failed${backend == null ? '' : ' on $backend'}'
        '${retryable ? '' : ', no retry'}: $message',
};

/// A repository over fakes that write each step into one native log, with
/// fixed timings, and every state change it publishes ([transitions]).
final class _Rig {
  _Rig({
    required String gemmaModelPath,
    required ChatModelPlanner? chatModels,
    required _SttScript stt,
    required Exception? ttsInstallError,
    required SpeechSynthesizer? synthesizer,
    required Exception? embedderInstallError,
    required Completer<void>? embedderInstallGate,
    required FakeEmbeddingModel? embeddingModel,
    required BundledModelFiles bundled,
    required String detectorBackend,
    required Future<Result<DetectorBackendChoice>> Function()?
    detectorBackendChoice,
  }) {
    llm = _Llm(log);
    detector = _Detector(log);
    models = ModelRepository(
      gemmaModelPath: gemmaModelPath,
      chatModels: chatModels,
      llm: llm,
      stt: _sttService(log, stt),
      tts: _ttsService(log, ttsInstallError, synthesizer),
      bundled: bundled,
      detector: detector,
      bundledDetector: () async => Uint8List(4),
      detectorBackend: detectorBackend,
      detectorBackendChoice: detectorBackendChoice,
      embedder: _embedderService(
        log,
        installError: embedderInstallError,
        installGate: embedderInstallGate,
        model: embeddingModel,
      ),
      logTap: log,
    );
    var previous = models.states.value;
    models.states.addListener(() {
      final current = models.states.value;
      final changed = [
        for (final id in ModelId.values)
          if (!identical(previous[id], current[id]))
            '${id.name} ${_show(current[id])}',
      ];
      transitions.add(changed.isEmpty ? '(same)' : changed.join(' + '));
      previous = current;
    });
  }

  final _Log log = _Log();
  late final _Llm llm;
  late final _Detector detector;
  late final ModelRepository models;
  final List<String> transitions = [];
}

/// The native log the fakes write into; a mark is a line count.
final class _Log implements NativeLogTap {
  final List<String> lines = [];

  @override
  String get description => 'test log';

  @override
  int mark() => lines.length;

  @override
  List<String> since(int mark) => List.unmodifiable(lines.sublist(mark));
}

/// A planner the test sets by hand.
final class _Planner implements ChatModelPlanner {
  _Planner(this.plan);

  @override
  ChatModelPlan plan;
}

/// [FakeLlmService] that writes each step into the log and can fail its
/// install or warm-up, or hold its install.
final class _Llm extends FakeLlmService {
  _Llm(this._log);

  final _Log _log;
  Exception? installError;
  Exception? warmUpError;
  Completer<void>? installGate;
  Completer<void>? warmUpGate;

  @override
  Future<Result<String>> install({
    required String path,
    ModelType modelType = ModelType.gemma4,
    required void Function(int percent) onProgress,
  }) async {
    _log.lines.add('llm install');
    await installGate?.future;
    if (installError case final error?) return Result.error(error);
    return super.install(
      path: path,
      modelType: modelType,
      onProgress: onProgress,
    );
  }

  @override
  Future<Result<LlmInfo>> load(ChatModelConfig model) {
    _log.lines.add('llm load');
    return super.load(model);
  }

  @override
  Future<Result<Duration>> warmUp(
    SamplerConfig sampler, {
    required bool withImage,
  }) async {
    _log.lines.add('llm warm-up');
    await warmUpGate?.future;
    final result = await super.warmUp(sampler, withImage: withImage);
    if (warmUpGate != null) _log.lines.add('llm warm-up done');
    if (warmUpError case final error?) return Result.error(error);
    return result;
  }

  @override
  Future<void> unload() {
    _log.lines.add('llm unload');
    return super.unload();
  }

  @override
  Future<void> close() {
    _log.lines.add('llm close');
    return super.close();
  }
}

/// What the speech recognizers do, by type. Mutable: a test can break one
/// after setup.
final class _SttScript {
  final Set<SttModelType> install = {};
  final Map<SttModelType, Exception> load = {};
  final Set<SttModelType> warmUp = {};
  Completer<void>? loadGate;

  /// Every recognizer a load made, in order.
  final List<FakeRecognizer> loaded = [];
}

SttService _sttService(_Log log, _SttScript script) => _TimedStt(
  install: (config, source, onProgress) async {
    final name = config.type.name;
    log.lines.add('stt install $name');
    if (script.install.contains(config.type)) {
      throw Exception('$name install failed');
    }
    onProgress(50);
    onProgress(100);
    return fakeSttModelId(config);
  },
  load: (config) async {
    final name = config.type.name;
    log.lines.add('stt load $name');
    await script.loadGate?.future;
    if (script.load[config.type] case final error?) throw error;
    final recognizer = FakeRecognizer('')
      ..error = script.warmUp.contains(config.type)
          ? StateError('$name warm-up failed')
          : null;
    script.loaded.add(recognizer);
    return recognizer;
  },
);

/// [SttService] that reports fixed load and warm-up times.
final class _TimedStt extends SttService {
  _TimedStt({required super.install, required super.load});

  @override
  Future<Result<Duration>> load(ModelId id) async =>
      switch (await super.load(id)) {
        Ok() => Result.ok(
          id == ModelId.whisperBase ? _whisperLoad : _moonshineLoad,
        ),
        final Error<Duration> failed => failed,
      };

  @override
  Future<Result<Duration>> warmUp() async => switch (await super.warmUp()) {
    Ok() => const Result.ok(_sttWarmUp),
    final Error<Duration> failed => failed,
  };
}

TtsService _ttsService(
  _Log log,
  Exception? installError,
  SpeechSynthesizer? synthesizer,
) => _TimedTts(
  install: (config, directory, onProgress) async {
    log.lines.add('tts install');
    if (installError case final error?) throw error;
    onProgress(100);
    return 'inflect';
  },
  load: (config) async {
    log.lines.add('tts load');
    return synthesizer ?? RecordingSynth();
  },
);

/// [TtsService] that reports fixed load and warm-up times.
final class _TimedTts extends TtsService {
  _TimedTts({required super.install, required super.load});

  @override
  Future<Result<Duration>> load() async => switch (await super.load()) {
    Ok() => const Result.ok(_ttsLoad),
    final Error<Duration> failed => failed,
  };

  @override
  Future<Result<Duration>> warmUp({String text = 'Ready.'}) async =>
      switch (await super.warmUp(text: text)) {
        Ok() => const Result.ok(_ttsWarmUp),
        final Error<Duration> failed => failed,
      };
}

EmbedderService _embedderService(
  _Log log, {
  required Exception? installError,
  required Completer<void>? installGate,
  required FakeEmbeddingModel? model,
}) => _TimedEmbedder(
  install: (config, source, onProgress) async {
    log.lines.add('embedder install');
    await installGate?.future;
    if (installError case final error?) throw error;
    onProgress(40);
    onProgress(40); // a repeated percent is published once
    onProgress(100);
    return 'embeddinggemma-300M_seq512_mixed-precision';
  },
  load: (config) async {
    log.lines.add('embedder load');
    return model ?? FakeEmbeddingModel();
  },
);

/// [EmbedderService] that reports fixed load and warm-up times.
final class _TimedEmbedder extends EmbedderService {
  _TimedEmbedder({required super.install, required super.load});

  @override
  Future<Result<EmbedderInfo>> load() async => switch (await super.load()) {
    Ok(:final value) => Result.ok(
      EmbedderInfo(
        modelId: value.modelId,
        backend: value.backend,
        dimension: value.dimension,
        loadTime: _embedderLoad,
      ),
    ),
    final Error<EmbedderInfo> failed => failed,
  };

  @override
  Future<Result<Duration>> warmUp() async => switch (await super.warmUp()) {
    Ok() => const Result.ok(_embedderWarmUp),
    final Error<Duration> failed => failed,
  };
}

/// A detector without a worker: loads at once with fixed timings, or fails
/// with [failOn]'s error for that backend; [gate] holds a load.
final class _Detector extends DetectorService {
  _Detector(this._log);

  final _Log _log;
  final Map<DetectorBackend, Exception> failOn = {};
  final List<DetectorModelSource> sources = [];
  Completer<void>? gate;
  int closeCalls = 0;
  bool _loaded = false;

  @override
  bool get isLoaded => _loaded;

  @override
  Future<Result<DetectorInfo>> load({
    required DetectorModelSource source,
    required DetectorBackend backend,
  }) async {
    _log.lines.add('detector load ${backend.name}');
    sources.add(source);
    _loaded = false;
    await gate?.future;
    if (failOn[backend] case final error?) return Result.error(error);
    _loaded = true;
    return Result.ok(
      DetectorInfo(
        backend: backend,
        fullyAccelerated: backend == DetectorBackend.gpu,
        verifyAbsolute: 1.8e-3,
        verifyRelative: 2e-6,
        verifyReference: VerifyReference.interpreter,
        createTime: _detCreate,
        verifyTime: _detVerify,
        firstRunTime: _detFirstRun,
      ),
    );
  }

  @override
  Future<void> close() async {
    closeCalls++;
    _loaded = false;
    await super.close();
  }
}
