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
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/services/llm/llm_service.dart';
import 'package:litert_edge_demos/domain/models/chat_model_config.dart';
import 'package:litert_edge_demos/domain/models/npu_availability.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../../fakes/fake_inference_model.dart';

void main() {
  group('backend guard', () {
    test('a CPU load is rejected and the CPU model is closed', () async {
      final cpuModel = FakeInferenceModel(activeBackend: PreferredBackend.cpu);
      final llm = LlmService(loadModel: (_) async => cpuModel);

      final result = await llm.load(kDefineChatModel);

      expect(result, isA<Error<LlmInfo>>());
      final error = (result as Error<LlmInfo>).error;
      expect(error, isA<BackendMismatchException>());
      expect(error.toString(), contains('requested gpu'));
      expect(error.toString(), contains('loaded on cpu'));
      expect(cpuModel.closeCalls, 1, reason: 'no CPU model left running');
      expect(llm.isLoaded, isFalse);
      expect(() => llm.model, throwsStateError);
    });

    test('an unknown backend is rejected too', () async {
      final model = FakeInferenceModel(activeBackend: null);
      final llm = LlmService(loadModel: (_) async => model);

      final result = await llm.load(kDefineChatModel);

      expect((result as Error<LlmInfo>).error, isA<BackendMismatchException>());
      expect(model.closeCalls, 1);
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

  group('custom chat model: exactly the chosen backend', () {
    const npuModel = ChatModelConfig(
      name: 'Gemma 3 1B NPU',
      modelType: ModelType.gemmaIt,
      llm: LlmConfig(
        maxTokens: 1280,
        backend: PreferredBackend.npu,
        supportImage: false,
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
          npu: () => const NpuAvailable(soc: 'QTI SM8750'),
        );

        final result = await llm.load(npuModel);

        expect((result as Ok<LlmInfo>).value.backend, PreferredBackend.npu);
        expect(result.value.contextTokens, model.maxTokens);
        expect(requested, same(npuModel.llm));
        expect(llm.loaded, same(npuModel));
      },
    );

    test(
      'an engine that reports another backend is an error naming the '
      'model, the request and why the NPU failed; the model is closed',
      () async {
        final gpuModel = FakeInferenceModel();
        final llm = LlmService(
          loadModel: (_) async {
            // What flutter_edge_ai prints when the NPU attempt fails and it
            // moves on (backend_preference.dart:296).
            Zone.current.print(
              '[flutter_edge_ai] WARNING: [LiteRtLmEngine] npu backend failed, '
              'trying the next candidate: Exception: Failed to create engine. '
              'Model may be invalid: /x/model.litertlm',
            );
            return gpuModel;
          },
          npu: () => const NpuAvailable(),
        );

        final result = await llm.load(npuModel);

        final error = (result as Error<LlmInfo>).error;
        expect(error, isA<BackendMismatchException>());
        final text = error.toString();
        expect(text, contains('Gemma 3 1B NPU'));
        expect(text, contains('requested npu'));
        expect(text, contains('loaded on gpu'));
        expect(text, contains('npu backend failed'));
        expect(text, contains('Failed to create engine'));
        expect(
          gpuModel.closeCalls,
          1,
          reason: 'never kept on the wrong backend',
        );
        expect(llm.isLoaded, isFalse);
        expect(llm.loaded, isNull);
      },
    );

    test('NPU where flutter_edge_ai would not offer it fails before loading, '
        'with its reason', () async {
      var loads = 0;
      final llm = LlmService(
        loadModel: (_) async {
          loads++;
          return FakeInferenceModel();
        },
        npu: () =>
            const NpuUnavailable('no NPU dispatch stack ships for macos'),
      );

      final result = await llm.load(npuModel);

      final error = (result as Error<LlmInfo>).error;
      expect(error, isA<NpuUnavailableException>());
      expect(error.toString(), contains('Gemma 3 1B NPU'));
      expect(
        error.toString(),
        contains('no NPU dispatch stack ships for macos'),
      );
      expect(loads, 0, reason: 'nothing loads on a GPU in its place');
    });

    test(
      'a load that throws names the model, the backend and the cause',
      () async {
        final llm = LlmService(
          loadModel: (_) async {
            Zone.current.print(
              '[flutter_edge_ai] WARNING: [LiteRtLmEngine] npu backend failed, '
              'no candidates are left: wrong SoC',
            );
            throw Exception('BackendInitException: all FFI backends failed');
          },
          npu: () => const NpuAvailable(),
        );

        final result = await llm.load(npuModel);

        final error = (result as Error<LlmInfo>).error;
        expect(error, isA<ChatModelLoadException>());
        final text = error.toString();
        expect(text, startsWith('Gemma 3 1B NPU did not load on npu'));
        expect(text, contains('all FFI backends failed'));
        expect(text, contains('wrong SoC'));
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

  test('only flutter_edge_ai warnings count as native reasons', () {
    expect(
      nativeReasonsIn([
        '[LlmService] something else',
        '[flutter_edge_ai] WARNING: [LiteRtLmEngine] npu was requested, but '
            'no NPU dispatch stack ships for macos',
      ]),
      [
        '[LiteRtLmEngine] npu was requested, but no NPU dispatch stack ships '
            'for macos',
      ],
    );
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
}

/// Lets queued microtasks run.
Future<void> pumpEventQueue() => Future<void>.delayed(Duration.zero);

class _ThrowingCloseModel extends FakeInferenceModel {
  @override
  Future<void> close() async => throw StateError('engine_delete failed');
}
