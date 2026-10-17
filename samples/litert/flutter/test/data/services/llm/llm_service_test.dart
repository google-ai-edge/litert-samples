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

import 'package:flutter_edge_ai/flutter_edge_ai.dart' hide ModelSpec;
import 'package:flutter_edge_ai_litertlm/flutter_edge_ai_litertlm.dart'
    show
        BackendInitAttemptFailure,
        BackendInitException,
        LiteRtLmBundleException,
        LiteRtLmBundleProblem,
        LiteRtLmEngineCreateException,
        NpuDispatchUnavailableException;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/services/llm/llm_service.dart';
import 'package:litert_edge_demos/domain/models/chat_model_config.dart';
import 'package:litert_edge_demos/utils/result.dart';
import 'package:litert_edge_demos/utils/wav.dart';

import '../../../fakes/fake_inference_model.dart';

void main() {
  group('strict engine', () {
    test('an engine that reports no backend is loaded, and the info says it '
        'reported none', () async {
      final model = FakeInferenceModel(activeBackend: null);
      final llm = LlmService(loadModel: (_) async => model);

      final info = (await llm.load(kDefineChatModel) as Ok<LlmInfo>).value;

      expect(info.backend, isNull);
      expect(llm.model, same(model));
      expect(model.closeCalls, 0);
    });

    test('a load reports the process\'s RSS before and after it', () async {
      final llm = LlmService(loadModel: (_) async => FakeInferenceModel());
      final info = (await llm.load(kDefineChatModel) as Ok<LlmInfo>).value;
      expect(info.rssBeforeBytes, greaterThan(0));
      expect(info.rssAfterBytes, greaterThan(0));
    });

    test('a GPU load passes kLlmConfig through and keeps the model', () async {
      final gpuModel = FakeInferenceModel();
      LlmConfig? requested;
      final llm = LlmService(
        loadModel: (config) async {
          requested = config;
          return gpuModel;
        },
      );

      final result = await llm.load(kDefineChatModel);

      expect(requested, same(kLlmConfig));
      expect((result as Ok<LlmInfo>).value.backend, PreferredBackend.gpu);
      expect(llm.model, same(gpuModel));
      expect(gpuModel.closeCalls, 0);
    });

    test('a loader that throws becomes a Result.error', () async {
      final llm = LlmService(
        loadModel: (_) async => throw StateError('engine create failed'),
      );

      final result = await llm.load(kDefineChatModel);

      expect(result, isA<Error<LlmInfo>>());
      expect(result.toString(), contains('engine create failed'));
    });
  });

  group("a failed load says the plugin's typed reason", () {
    const npuModel = ChatModelConfig(
      name: 'Gemma 4 E2B NPU',
      modelType: ModelType.gemma4,
      llm: LlmConfig(
        maxTokens: 4096,
        backend: PreferredBackend.npu,
        maxNumImages: 1,
      ),
      tools: false,
    );

    Future<ChatModelLoadException> failWith(
      Exception error, {
      ChatModelConfig model = kDefineChatModel,
    }) async {
      var loads = 0;
      final llm = LlmService(
        loadModel: (_) async {
          loads++;
          throw error;
        },
      );
      final result = await llm.load(model);
      expect(loads, 1, reason: 'one attempt: nothing is tried in its place');
      expect(llm.isLoaded, isFalse);
      final failure = (result as Error<LlmInfo>).error;
      expect(failure, isA<ChatModelLoadException>());
      expect((failure as ChatModelLoadException).cause, same(error));
      return failure;
    }

    test('a LiteRtLmBundleException (an NPU build asked for the GPU): its '
        'message and problem', () async {
      final error = await failWith(
        const LiteRtLmBundleException(
          LiteRtLmBundleProblem.wrongBackend,
          '/m/gemma_sm8750.litertlm is compiled for the Qualcomm SM8750 NPU '
          'and runs only there; gpu was asked for',
        ),
      );

      expect(
        '$error',
        'Gemma 4 E2B did not load on gpu: /m/gemma_sm8750.litertlm is '
            'compiled for the Qualcomm SM8750 NPU and runs only there; gpu was '
            'asked for (wrongBackend)',
      );
    });

    test("a BackendInitException: each attempt's backend and LiteRT-LM's own "
        'message, once: the engine-create message has it already', () async {
      final error = await failWith(
        BackendInitException(
          attempts: [
            const BackendInitAttemptFailure(
              backend: PreferredBackend.npu,
              error: LiteRtLmEngineCreateException(
                'Failed to create engine on npu: Failed to retrieve graph. '
                'Model: /m/gemma_sm8750.litertlm. The bundle is compiled for '
                'the Qualcomm SM8750 NPU; if this device is not a Qualcomm '
                'SM8750, that is the cause.',
                nativeCode: 13,
                nativeMessage: 'Failed to retrieve graph',
              ),
              stackTrace: StackTrace.empty,
            ),
          ],
        ),
        model: npuModel,
      );

      expect(
        '$error',
        'Gemma 4 E2B NPU did not load on npu: npu: Failed to create engine on '
            'npu: Failed to retrieve graph. Model: /m/gemma_sm8750.litertlm. '
            'The bundle is compiled for the Qualcomm SM8750 NPU; if this '
            'device is not a Qualcomm SM8750, that is the cause.',
      );
    });

    test("an engine-create message without LiteRT-LM's reason in it (a "
        'declared section_backend_constraint) gets it appended', () async {
      final error = await failWith(
        BackendInitException(
          attempts: [
            const BackendInitAttemptFailure(
              backend: PreferredBackend.npu,
              error: LiteRtLmEngineCreateException(
                'This .litertlm model requires the "gpu" backend — it '
                'declares section_backend_constraint: gpu, but the "npu" '
                'backend was requested. Use PreferredBackend.gpu (or omit '
                'preferredBackend to try GPU first).',
                nativeCode: 13,
                nativeMessage:
                    'Main backend constraint mismatch. Model requires one '
                    'of [gpu] but Main backend is npu',
              ),
              stackTrace: StackTrace.empty,
            ),
          ],
        ),
        model: npuModel,
      );

      expect(
        error.reason,
        endsWith(
          '(LiteRT-LM: Main backend constraint mismatch. Model requires one '
          'of [gpu] but Main backend is npu)',
        ),
      );
    });

    test(
      'an engine create without a native message says the message alone',
      () async {
        final error = await failWith(
          BackendInitException(
            attempts: [
              const BackendInitAttemptFailure(
                backend: PreferredBackend.gpu,
                error: LiteRtLmEngineCreateException('Failed to create engine'),
                stackTrace: StackTrace.empty,
              ),
            ],
          ),
        );

        expect(error.reason, 'gpu: Failed to create engine');
      },
    );

    test('an NPU this device cannot reach: the plugin says so before any '
        'engine (NpuDispatchUnavailableException)', () async {
      final error = await failWith(
        BackendInitException(
          attempts: [
            const BackendInitAttemptFailure(
              backend: PreferredBackend.npu,
              error: NpuDispatchUnavailableException(
                'no NPU dispatch stack ships for macos',
              ),
              stackTrace: StackTrace.empty,
            ),
          ],
        ),
        model: npuModel,
      );

      expect(
        '$error',
        'Gemma 4 E2B NPU did not load on npu: npu: no NPU dispatch stack '
            'ships for macos',
      );
    });

    test('anything else as it was thrown', () async {
      final error = await failWith(Exception('engine create failed'));

      expect(
        '$error',
        'Gemma 4 E2B did not load on gpu: Exception: engine create failed',
      );
    });
  });

  group('custom chat model: exactly the chosen backend', () {
    const npuModel = ChatModelConfig(
      name: 'Gemma 3 1B NPU',
      modelType: ModelType.gemmaIt,
      llm: LlmConfig(
        maxTokens: 1280,
        backend: PreferredBackend.npu,
        maxNumImages: 1,
      ),
      tools: false,
    );

    test(
      'the config goes to the engine as chosen and is kept as loaded',
      () async {
        LlmConfig? requested;
        final model = FakeInferenceModel(activeBackend: PreferredBackend.npu);
        final llm = LlmService(
          loadModel: (config) async {
            requested = config;
            return model;
          },
        );

        final result = await llm.load(npuModel);

        expect((result as Ok<LlmInfo>).value.backend, PreferredBackend.npu);
        expect(result.value.contextTokens, model.maxTokens);
        expect(requested, same(npuModel.llm));
        expect(llm.loaded, same(npuModel));
      },
    );

    test('unload closes the model and the service loads again', () async {
      final first = FakeInferenceModel();
      final second = FakeInferenceModel();
      final models = [first, second];
      final llm = LlmService(loadModel: (_) async => models.removeAt(0));

      await llm.load(kDefineChatModel);
      await llm.unload();

      expect(first.closeCalls, 1);
      expect(llm.isLoaded, isFalse);
      expect(await llm.load(kDefineChatModel), isA<Ok<LlmInfo>>());
      expect(llm.model, same(second));
    });

    test('a second load closes the model it replaces first', () async {
      final first = FakeInferenceModel();
      final models = [first, FakeInferenceModel()];
      final llm = LlmService(loadModel: (_) async => models.removeAt(0));

      await llm.load(kDefineChatModel);
      await llm.load(kDefineChatModel);

      expect(first.closeCalls, 1);
    });
  });

  group('close', () {
    test('a model that arrives after close() is closed, not kept', () async {
      final pending = Completer<InferenceModel>();
      final lateModel = FakeInferenceModel();
      final llm = LlmService(loadModel: (_) => pending.future);

      final loading = llm.load(kDefineChatModel); // first GPU compile in flight
      await llm.close(); // Cmd-Q
      pending.complete(lateModel);
      final result = await loading;

      expect(result, isA<Error<LlmInfo>>());
      expect(lateModel.closeCalls, 1, reason: 'no orphaned engine');
      expect(llm.isLoaded, isFalse);
    });

    test('warm-up after close() fails without touching a model', () async {
      final model = FakeInferenceModel();
      final llm = LlmService(loadModel: (_) async => model);
      await llm.load(kDefineChatModel);
      await llm.close();

      expect(
        await llm.warmUp(kSampler, withImage: true),
        isA<Error<Duration>>(),
      );
      expect(model.created, isEmpty);
    });

    test('close() failures are logged, not thrown', () async {
      final llm = LlmService(loadModel: (_) async => _ThrowingCloseModel());
      await llm.load(kDefineChatModel);

      await expectLater(llm.close(), completes);
    });
  });

  group('warm-up', () {
    /// Waits until the warm-up session exists (the image is made first).
    Future<FakeInferenceSession> warmUpSession(FakeInferenceModel model) async {
      for (var i = 0; i < 100 && model.created.isEmpty; i++) {
        await pumpEventQueue();
      }
      return model.lastSession;
    }

    test('runs one generation with the chat sampler and closes it', () async {
      final model = FakeInferenceModel();
      final llm = LlmService(loadModel: (_) async => model);
      await llm.load(kDefineChatModel);

      final warming = llm.warmUp(kSampler, withImage: false);
      (await warmUpSession(model))
        ..emit('Hi')
        ..finish();

      expect(await warming, isA<Ok<Duration>>());
      expect(model.sessionSettings.single, (
        temperature: kSampler.temperature,
        topK: kSampler.topK,
        maxOutputTokens: 1,
      ));
      expect(model.lastSession.queries.single.hasImage, isFalse);
      expect(model.lastSession.closed, isTrue);
    });

    test('withImage: the one warm-up prompt carries the small PNG, so the '
        "vision encoder's first use happens at setup", () async {
      final model = FakeInferenceModel();
      final png = Uint8List.fromList([0x89, 0x50, 0x4E, 0x47]);
      final llm = LlmService(
        loadModel: (_) async => model,
        warmUpImage: () async => png,
      );
      await llm.load(kDefineChatModel);

      final warming = llm.warmUp(kSampler, withImage: true);
      (await warmUpSession(model))
        ..emit('A')
        ..finish();

      expect(await warming, isA<Ok<Duration>>());
      expect(model.created, hasLength(1), reason: 'one generation, not two');
      final query = model.lastSession.queries.single;
      expect(query.text, 'Hi');
      expect(query.imageBytes, same(png));
      expect(model.sessionSettings.single.maxOutputTokens, 1);
      expect(model.lastSession.closed, isTrue);
    });

    test('withAudio: the one warm-up prompt also carries a 1 s WAV, so the '
        "audio encoder's first use happens at setup", () async {
      final model = FakeInferenceModel();
      final wav = Uint8List.fromList([0x52, 0x49, 0x46, 0x46]);
      final llm = LlmService(
        loadModel: (_) async => model,
        warmUpImage: () async => Uint8List.fromList([0x89]),
        warmUpAudio: () => wav,
      );
      await llm.load(kDefineChatModel);

      final warming = llm.warmUp(kSampler, withImage: true, withAudio: true);
      (await warmUpSession(model))
        ..emit('A')
        ..finish();

      expect(await warming, isA<Ok<Duration>>());
      expect(model.created, hasLength(1), reason: 'still one generation');
      final query = model.lastSession.queries.single;
      expect(query.audioBytes, same(wav));
      expect(query.hasImage, isTrue);
    });

    test('the default warm-up WAV: 1 s of 16 kHz mono PCM16, not silence', () {
      final wav = warmUpWav();
      expect(wavDuration(wav), const Duration(seconds: 1));
      expect(pcm16FromWav(wav).any((b) => b != 0), isTrue);
    });

    test('without withAudio no recording goes out', () async {
      final model = FakeInferenceModel();
      final llm = LlmService(
        loadModel: (_) async => model,
        warmUpAudio: () => throw StateError('must not be made'),
      );
      await llm.load(kDefineChatModel);
      final warming = llm.warmUp(kSampler, withImage: false);
      (await warmUpSession(model))
        ..emit('A')
        ..finish();
      expect(await warming, isA<Ok<Duration>>());
      expect(model.lastSession.queries.single.hasAudio, isFalse);
    });

    test(
      'a warm-up image that cannot be made fails the warm-up visibly',
      () async {
        final model = FakeInferenceModel();
        final llm = LlmService(
          loadModel: (_) async => model,
          warmUpImage: () async => throw StateError('no raster'),
        );
        await llm.load(kDefineChatModel);

        final result = await llm.warmUp(kSampler, withImage: true);

        expect(result, isA<Error<Duration>>());
        expect(result.toString(), contains('no raster'));
      },
    );
  });

  group('towers', () {
    const npuLlm = LlmConfig(
      maxTokens: 4096,
      backend: PreferredBackend.npu,
      maxNumImages: 1,
      supportImage: true,
      supportAudio: true,
    );

    Future<Map<String, Object?>> passed(LlmConfig config) async {
      late Map<String, Object?> args;
      await getActiveModelFor(
        config,
        getActiveModel:
            ({
              int? maxTokens,
              PreferredBackend? preferredBackend,
              PreferredBackend? preferredVisionBackend,
              PreferredBackend? preferredAudioBackend,
              bool? supportImage,
              int? maxNumImages,
              bool? supportAudio,
            }) async {
              args = {
                'maxTokens': maxTokens,
                'backend': preferredBackend,
                'vision': preferredVisionBackend,
                'audio': preferredAudioBackend,
                'supportImage': supportImage,
                'maxNumImages': maxNumImages,
                'supportAudio': supportAudio,
              };
              return FakeInferenceModel(activeBackend: preferredBackend);
            },
      );
      return args;
    }

    test('getActiveModel loads each tower the file has and asks for no tower '
        'backend: the engine places them', () async {
      expect(await passed(npuLlm), {
        'maxTokens': 4096,
        'backend': PreferredBackend.npu,
        'vision': null,
        'audio': null,
        'supportImage': true,
        'maxNumImages': 1,
        'supportAudio': true,
      });
      const textOnly = LlmConfig(
        maxTokens: 4096,
        backend: PreferredBackend.npu,
        maxNumImages: 1,
      );
      final args = await passed(textOnly);
      expect(args['supportImage'], isFalse);
      expect(args['supportAudio'], isFalse);
      // The app's own settings: the vision tower on, no audio.
      final define = await passed(kLlmConfig);
      expect(define['supportImage'], isTrue);
      expect(define['supportAudio'], isFalse);
      expect(define['vision'], isNull);
    });

    test("the load reports where the engine put each tower: the model's "
        'activeVisionBackend and activeAudioBackend', () async {
      final llm = LlmService(
        loadModel: (_) async => FakeInferenceModel(
          activeBackend: PreferredBackend.npu,
          activeVisionBackend: PreferredBackend.npu,
          activeAudioBackend: PreferredBackend.cpu,
        ),
      );
      const model = ChatModelConfig(
        name: 'Gemma 4 E2B NPU',
        modelType: ModelType.gemma4,
        llm: npuLlm,
        tools: false,
      );

      final info = (await llm.load(model) as Ok<LlmInfo>).value;

      expect(info.backend, PreferredBackend.npu);
      expect(info.visionBackend, PreferredBackend.npu);
      expect(info.audioBackend, PreferredBackend.cpu);
    });
  });
}

/// Lets queued microtasks run.
Future<void> pumpEventQueue() => Future<void>.delayed(Duration.zero);

class _ThrowingCloseModel extends FakeInferenceModel {
  @override
  Future<void> close() async => throw StateError('engine_delete failed');
}
