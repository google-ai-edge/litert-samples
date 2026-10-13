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

import 'package:flutter_edge_ai/flutter_edge_ai.dart' show PreferredBackend;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/live_camera_settings_repository.dart';
import 'package:litert_edge_demos/data/repositories/model_repository.dart';
import 'package:litert_edge_demos/data/services/detector/detector_engine.dart'
    show kDetectorNotRawHead;
import 'package:litert_edge_demos/data/services/detector/detector_service.dart';
import 'package:litert_edge_demos/data/services/knowledge/embedder_service.dart';
import 'package:litert_edge_demos/data/services/settings/typed_settings.dart';
import 'package:litert_edge_demos/domain/models/detection.dart';
import 'package:litert_edge_demos/domain/models/detector_choice.dart';
import 'package:litert_edge_demos/domain/models/frame_source_spec.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../fakes/fake_bundled_files.dart';
import '../../fakes/fake_detector_runtime.dart';
import '../../fakes/fake_llm_service.dart';
import '../../fakes/fake_model_files.dart';
import '../../fakes/fake_settings_store.dart';
import '../../fakes/fake_speech.dart';
import '../../support/until.dart';

void main() {
  late FakeLlmService llm;
  late ModelRepository repo;

  setUp(() {
    llm = FakeLlmService();
    repo = ModelRepository(
      gemmaModelPath: kTestChatModelPath,
      bundled: FakeBundledFiles(),
      detector: fakeDetectorService(),
      bundledDetector: fakeBundledDetector,
      llm: llm,
      stt: fakeSttService(),
      tts: fakeTtsService(),
      embedder: EmbedderService(),
    );
  });

  tearDown(() => repo.close());

  test('prepares the LLM with kLlmConfig and publishes every state', () async {
    final seen = <Type>[];
    repo.states.addListener(
      () => seen.add(repo.states.value[ModelId.chat].runtimeType),
    );

    expect(await repo.prepareAll(), isA<Ok<void>>());

    expect(llm.loads, [same(kLlmConfig)]);
    expect(llm.warmUpsWithImage, [
      kLlmConfig.supportImage,
    ], reason: 'the warm-up includes an image when the model supports images');
    expect(seen.toSet(), {
      ModelInstalling,
      ModelLoading,
      ModelWarmingUp,
      ModelReady,
    });
    final state = repo.states.value[ModelId.chat];
    expect(state, isA<ModelReady>());
    expect((state! as ModelReady).info.backend, 'gpu');
    expect(repo.requiredReady, isTrue);
  });

  test('a CPU fallback is a blocking failure, and Retry recovers', () async {
    llm.activeBackend = PreferredBackend.cpu;

    final failed = await repo.prepareAll();

    expect(failed, isA<Error<void>>());
    final state = repo.states.value[ModelId.chat];
    expect(state, isA<ModelFailed>());
    expect((state! as ModelFailed).message, contains('requested gpu'));
    expect((state as ModelFailed).message, contains('loaded on cpu'));
    expect(repo.requiredReady, isFalse);

    llm.activeBackend = PreferredBackend.gpu;
    expect(await repo.prepareAll(), isA<Ok<void>>());
    expect(repo.requiredReady, isTrue);
    expect(llm.loads, hasLength(2));
  });

  test('close() during an in-flight load stops setup before warm-up', () async {
    final gate = llm.loadGate = Completer<void>();

    final preparing = repo.prepareAll();
    await Future<void>.delayed(Duration.zero);
    // close() waits for the setup run, which the gated load holds.
    final closing = repo.close();
    gate.complete();
    await closing;
    final result = await preparing;

    expect(result, isA<Error<void>>());
    expect(llm.warmUpCalls, 0, reason: 'no warm-up on an orphaned model');
    expect(llm.closeCalls, greaterThanOrEqualTo(1));
  });

  test('close() waits for a stuck load only so long, then closes the models '
      'anyway', () async {
    final stuck = ModelRepository(
      gemmaModelPath: kTestChatModelPath,
      bundled: FakeBundledFiles(),
      detector: fakeDetectorService(),
      bundledDetector: fakeBundledDetector,
      llm: llm,
      stt: fakeSttService(),
      tts: fakeTtsService(),
      closeWait: const Duration(milliseconds: 50),
      embedder: EmbedderService(),
    );
    final gate = llm.loadGate = Completer<void>();
    addTearDown(() {
      if (!gate.isCompleted) gate.complete();
    });

    final preparing = stuck.prepareAll();
    await Future<void>.delayed(Duration.zero);
    final watch = Stopwatch()..start();
    await stuck.close();

    expect(
      watch.elapsed,
      greaterThanOrEqualTo(const Duration(milliseconds: 50)),
    );
    expect(llm.closeCalls, 1, reason: 'closed while the load still runs');
    gate.complete();
    expect(await preparing, isA<Error<void>>());
  });

  test('the built-in detector loads (from the asset bundle, no '
      'download)', () async {
    expect(await repo.prepareAll(), isA<Ok<void>>());

    expect(repo.requiredReady, isTrue);
    final detector = repo.states.value[ModelId.yolo26n];
    expect(detector, isA<ModelReady>());
    expect((detector! as ModelReady).info.backend, 'gpu');
  });

  group('detector step (real DetectorService and worker, fake runtime)', () {
    ({ModelRepository models, DetectorService detector}) build({
      FakeDetectorRuntime runtime = const FakeDetectorRuntime(),
      String backend = 'gpu',
      Future<Result<DetectorBackendChoice>> Function()? choice,
      Future<Uint8List> Function() bundled = fakeBundledDetector,
    }) {
      final detector = DetectorService(
        runtime: runtime,
        expectedModelBytes: 64,
      );
      final models = ModelRepository(
        gemmaModelPath: kTestChatModelPath,
        bundled: FakeBundledFiles(),
        bundledDetector: bundled,
        llm: FakeLlmService(),
        stt: fakeSttService(),
        tts: fakeTtsService(),
        detector: detector,
        detectorBackend: backend,
        detectorBackendChoice: choice,
        embedder: EmbedderService(),
      );
      addTearDown(models.close);
      return (models: models, detector: detector);
    }

    test('an empty DETECTOR_BACKEND (the default build) is the GPU', () async {
      final (:models, :detector) = build(backend: '');

      await models.prepareAll();

      final info = (models.states.value[ModelId.yolo26n]! as ModelReady).info;
      expect(info.backend, 'gpu');
    });

    group("Demo 3's Detector setting (runtime choice)", () {
      late InMemorySettingsStore store;
      late LiveCameraSettingsRepository settings;

      setUp(() {
        store = InMemorySettingsStore();
        settings = LiveCameraSettingsRepository(
          settings: TypedSettings(store: store),
          environment: const Result.ok(CameraSourceSpec()),
          backendDefine: '',
        );
      });

      test('a saved CPU choice loads the CPU, labelled CPU (chosen)', () async {
        await settings.saveBackend(DetectorBackend.cpu);
        final (:models, :detector) = build(
          runtime: const FakeDetectorRuntime(fullyAccelerated: false),
          choice: settings.readBackend,
        );

        await models.prepareAll();

        final info = (models.states.value[ModelId.yolo26n]! as ModelReady).info;
        expect(info.backend, 'cpu');
        expect(info.detail, 'CPU (chosen)');
        expect(info.explicitCpu, isTrue);
      });

      test('the define wins over the setting', () async {
        await store.setString('detector.backend', 'cpu');
        final locked = LiveCameraSettingsRepository(
          settings: TypedSettings(store: store),
          environment: const Result.ok(CameraSourceSpec()),
          backendDefine: 'gpu',
        );
        final (:models, :detector) = build(choice: locked.readBackend);

        await models.prepareAll();

        final info = (models.states.value[ModelId.yolo26n]! as ModelReady).info;
        expect(info.backend, 'gpu');
      });

      test('a strict GPU failure names the backend; choosing the CPU and '
          'reloading loads it (no silent fallback in between)', () async {
        final (:models, :detector) = build(
          runtime: const FakeDetectorRuntime(fullyAccelerated: false),
          choice: settings.readBackend,
        );
        await models.prepareAll();
        final failed = models.states.value[ModelId.yolo26n]! as ModelFailed;
        expect(failed.backend, 'gpu');
        expect(failed.message, contains('only partly on the GPU'));
        expect(detector.isLoaded, isFalse);

        expect(
          await settings.saveBackend(DetectorBackend.cpu),
          isA<Ok<void>>(),
        );
        final seen = <Type>[];
        models.states.addListener(
          () => seen.add(models.states.value[ModelId.yolo26n].runtimeType),
        );
        expect(await models.reloadDetector(), isA<Ok<void>>());

        expect(
          seen,
          containsAllInOrder([ModelPending, ModelLoading, ModelReady]),
        );
        final info = (models.states.value[ModelId.yolo26n]! as ModelReady).info;
        expect(info.detail, 'CPU (chosen)');
        expect(detector.info?.backend, DetectorBackend.cpu);

        // And back to the GPU: it fails again, visibly.
        await settings.saveBackend(DetectorBackend.gpu);
        expect(await models.reloadDetector(), isA<Error<void>>());
        expect(
          (models.states.value[ModelId.yolo26n]! as ModelFailed).backend,
          'gpu',
        );
      });

      test('bytes that are not the raw-head file are not a backend failure (no '
          'backend offered)', () async {
        final (:models, :detector) = build(
          choice: settings.readBackend,
          bundled: () async => Uint8List(10),
        );

        await models.prepareAll();

        final failed = models.states.value[ModelId.yolo26n]! as ModelFailed;
        expect(failed.message, startsWith(kDetectorNotRawHead));
        expect(failed.backend, isNull);
      });

      test('an unreadable setting fails the row with the reason (Demo 3 can '
          'still fix it)', () async {
        await store.setString('detector.backend', 'npu');
        final (:models, :detector) = build(choice: settings.readBackend);

        await models.prepareAll();

        final failed = models.states.value[ModelId.yolo26n]! as ModelFailed;
        expect(failed.message, contains('"npu" is not gpu or cpu'));
      });
    });

    test('loads after the LLM, strict GPU: Ready with the GPU label', () async {
      final (:models, :detector) = build();
      final seen = <Type>[];
      models.states.addListener(
        () => seen.add(models.states.value[ModelId.yolo26n].runtimeType),
      );

      expect(await models.prepareAll(), isA<Ok<void>>());

      expect(seen, contains(ModelLoading));
      final state = models.states.value[ModelId.yolo26n];
      final info = (state! as ModelReady).info;
      expect(info.backend, 'gpu');
      expect(info.detail, 'GPU fp32 full');
      expect(info.explicitCpu, isFalse);
      expect(detector.isLoaded, isTrue);
    });

    test('DETECTOR_BACKEND=cpu is Ready but flagged CPU (chosen)', () async {
      final (:models, :detector) = build(
        runtime: const FakeDetectorRuntime(fullyAccelerated: false),
        backend: 'cpu',
      );

      await models.prepareAll();

      final info = (models.states.value[ModelId.yolo26n]! as ModelReady).info;
      expect(info.explicitCpu, isTrue);
      expect(info.detail, 'CPU (chosen)');
    });

    test('partial GPU acceleration fails the detector row (no silent CPU) '
        'while setup still succeeds', () async {
      final (:models, :detector) = build(
        runtime: const FakeDetectorRuntime(fullyAccelerated: false),
      );

      expect(await models.prepareAll(), isA<Ok<void>>());

      final state = models.states.value[ModelId.yolo26n];
      expect(state, isA<ModelFailed>());
      expect(
        (state! as ModelFailed).message,
        contains('only partly on the GPU'),
      );
      expect(models.requiredReady, isTrue);
      expect(detector.isLoaded, isFalse);
    });

    test(
      'an unknown DETECTOR_BACKEND is unavailable with the reason',
      () async {
        final (:models, :detector) = build(backend: 'npu');

        await models.prepareAll();

        final state = models.states.value[ModelId.yolo26n];
        expect(
          (state! as ModelUnavailable).reason,
          contains('DETECTOR_BACKEND'),
        );
      },
    );

    test('close() closes the detector', () async {
      final (:models, :detector) = build();
      await models.prepareAll();
      expect(detector.isLoaded, isTrue);

      await models.close();

      expect(detector.isLoaded, isFalse);
    });

    test(
      'close() during the detector load leaves no loaded detector',
      () async {
        final (:models, :detector) = build(
          runtime: const FakeDetectorRuntime(
            runDelay: Duration(milliseconds: 150),
          ),
        );
        final preparing = models.prepareAll();
        await untilNotified(
          models.states,
          () => models.states.value[ModelId.yolo26n] is ModelLoading,
          what: 'the detector load to start',
        );

        await models.close();
        final result = await preparing;

        expect(result, isA<Error<void>>());
        expect(detector.isLoaded, isFalse);
      },
    );
  });

  test('a required failure stops setup before the optional models', () async {
    llm.activeBackend = PreferredBackend.cpu;

    expect(await repo.prepareAll(), isA<Error<void>>());

    expect(repo.states.value[ModelId.chat], isA<ModelFailed>());
    expect(repo.states.value[ModelId.yolo26n], isA<ModelPending>());
  });

  test('a required model that does not become ready fails setup instead of '
      'leaving it waiting with no Retry', () async {
    final strict = ModelRepository(
      gemmaModelPath: kTestChatModelPath,
      bundled: FakeBundledFiles(),
      detector: fakeDetectorService(),
      // The built-in detector cannot be read: required here, it fails setup.
      bundledDetector: () async => throw StateError('no asset bundle'),
      llm: FakeLlmService(),
      stt: fakeSttService(),
      tts: fakeTtsService(),
      requiredModels: {ModelId.chat, ModelId.yolo26n},
      embedder: EmbedderService(),
    );
    addTearDown(strict.close);

    final result = await strict.prepareAll();

    expect(result, isA<Error<void>>());
    expect(strict.requiredReady, isFalse);
    expect(strict.states.value[ModelId.chat], isA<ModelReady>());
  });

  test('concurrent prepareAll calls share one run', () async {
    final results = await Future.wait([repo.prepareAll(), repo.prepareAll()]);

    expect(results, everyElement(isA<Ok<void>>()));
    expect(llm.loads, hasLength(1));
  });

  group('speech models (required, after the LLM)', () {
    test('STT then TTS load after Gemma, warm up, and are Ready on the '
        'requested CPU', () async {
      final order = <String>[];
      final recognizer = FakeRecognizer('');
      final synth = RecordingSynth();
      final models = ModelRepository(
        gemmaModelPath: kTestChatModelPath,
        bundled: FakeBundledFiles(),
        detector: fakeDetectorService(),
        bundledDetector: fakeBundledDetector,
        llm: llm,
        stt: fakeSttService(
          recognizer: recognizer,
          install: () async {
            order.add('stt');
            return 'whisper_base_30s_i8';
          },
        ),
        tts: fakeTtsService(
          synthesizer: synth,
          install: () async {
            order.add('tts');
            return 'inflect';
          },
        ),
        embedder: EmbedderService(),
      );
      addTearDown(models.close);
      final sttStates = <Type>[];
      models.states.addListener(
        () =>
            sttStates.add(models.states.value[ModelId.whisperBase].runtimeType),
      );

      expect(await models.prepareAll(), isA<Ok<void>>());

      expect(llm.loads, hasLength(1));
      // Whisper (required), TTS, then moonshine (Demo 3's, optional) after
      // the detector.
      expect(order, ['stt', 'tts', 'stt']);
      expect(models.states.value[ModelId.moonshineTiny], isA<ModelReady>());
      expect(
        models.activeStt.value?.id,
        ModelId.moonshineTiny,
        reason: 'the last loaded stays active; Demo 1 switches on entry',
      );
      expect(
        sttStates.toSet(),
        containsAll([
          ModelInstalling,
          ModelLoading,
          ModelWarmingUp,
          ModelReady,
        ]),
      );
      final stt =
          (models.states.value[ModelId.whisperBase]! as ModelReady).info;
      expect(stt.modelId, 'whisper_base_30s_i8');
      expect(stt.backend, 'cpu');
      expect(stt.detail, 'CPU (requested)');
      final tts =
          (models.states.value[ModelId.inflectNano]! as ModelReady).info;
      expect(tts.modelId, 'inflect');
      expect(recognizer.calls, 2, reason: 'STT warm-up, once per model');
      expect(synth.synthesized, ['Ready.'], reason: 'TTS warm-up');
      expect(models.requiredReady, isTrue);

      await models.close();
      // One fake stands for both recognizers: closed when moonshine
      // replaced Whisper, and at close.
      expect(recognizer.closeCalls, 2);
      expect(synth.closeCalls, 1);
    });

    test('activateStt switches the singleton per demo; a model setup did '
        'not make ready cannot be activated', () async {
      final whisper = FakeRecognizer('w');
      final moonshine = FakeRecognizer('m');
      final models = ModelRepository(
        gemmaModelPath: kTestChatModelPath,
        bundled: FakeBundledFiles(),
        detector: fakeDetectorService(),
        bundledDetector: fakeBundledDetector,
        llm: llm,
        stt: fakeSttService(
          recognizerFor: (config) =>
              config.language == null ? moonshine : whisper,
        ),
        tts: fakeTtsService(),
        embedder: EmbedderService(),
      );
      addTearDown(models.close);
      expect(await models.prepareAll(), isA<Ok<void>>());
      expect(models.activeStt.value?.id, ModelId.moonshineTiny);

      expect(await models.activateStt(ModelId.whisperBase), isA<Ok<void>>());
      expect(models.activeStt.value?.id, ModelId.whisperBase);
      expect(models.activeStt.value?.switchTime, isNotNull);
      expect(moonshine.closeCalls, 1);

      expect(await models.activateStt(ModelId.moonshineTiny), isA<Ok<void>>());
      expect(models.activeStt.value?.id, ModelId.moonshineTiny);

      final failing = ModelRepository(
        gemmaModelPath: kTestChatModelPath,
        bundled: FakeBundledFiles(),
        detector: fakeDetectorService(),
        bundledDetector: fakeBundledDetector,
        llm: FakeLlmService(),
        stt: fakeSttService(),
        tts: fakeTtsService(),
        embedder: EmbedderService(),
      );
      addTearDown(failing.close);
      final result = await failing.activateStt(ModelId.moonshineTiny);
      expect((result as Error<void>).error.toString(), contains('not ready'));
    });

    test('a TTS failure blocks setup with its error, and Retry recovers '
        'without reloading what is ready', () async {
      var ttsInstalls = 0;
      var fail = true;
      var sttInstalls = 0;
      final models = ModelRepository(
        gemmaModelPath: kTestChatModelPath,
        bundled: FakeBundledFiles(),
        detector: fakeDetectorService(),
        bundledDetector: fakeBundledDetector,
        llm: llm,
        stt: fakeSttService(
          install: () async {
            sttInstalls++;
            return 'whisper_base_30s_i8';
          },
        ),
        tts: fakeTtsService(
          install: () async {
            ttsInstalls++;
            if (fail) throw Exception('HF unreachable');
            return 'inflect';
          },
        ),
        embedder: EmbedderService(),
      );
      addTearDown(models.close);

      expect(await models.prepareAll(), isA<Error<void>>());
      final failed = models.states.value[ModelId.inflectNano];
      expect((failed! as ModelFailed).message, contains('HF unreachable'));
      expect(models.requiredReady, isFalse);
      expect(
        models.states.value[ModelId.yolo26n],
        isA<ModelPending>(),
        reason: 'optional models wait for the required ones',
      );

      fail = false;
      expect(await models.prepareAll(), isA<Ok<void>>());
      expect(models.requiredReady, isTrue);
      expect(ttsInstalls, 2);
      expect(
        sttInstalls,
        2,
        reason: 'Whisper was ready already; moonshine waited for the TTS',
      );
      expect(llm.loads, hasLength(1));
    });

    test('a TTS at the wrong sample rate fails its row (no silent pitch '
        'shift)', () async {
      final models = ModelRepository(
        gemmaModelPath: kTestChatModelPath,
        bundled: FakeBundledFiles(),
        detector: fakeDetectorService(),
        bundledDetector: fakeBundledDetector,
        llm: llm,
        stt: fakeSttService(),
        tts: fakeTtsService(synthesizer: RecordingSynth(sampleRate: 22050)),
        embedder: EmbedderService(),
      );
      addTearDown(models.close);

      expect(await models.prepareAll(), isA<Error<void>>());
      expect(
        (models.states.value[ModelId.inflectNano]! as ModelFailed).message,
        contains('22050 Hz'),
      );
    });
  });
}
