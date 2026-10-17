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
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_edge_ai/flutter_edge_ai.dart'
    show
        EmbeddingInputTooLongException,
        EmbeddingLoadException,
        EmbeddingModality,
        EmbeddingModel,
        PreferredBackend,
        TaskType;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/knowledge_config.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/services/knowledge/embedder_service.dart';
import 'package:litert_edge_demos/domain/models/knowledge_query.dart';
import 'package:litert_edge_demos/utils/result.dart';
import 'package:litert_edge_demos/utils/wav.dart';

import '../../../fakes/fake_knowledge.dart';

void main() {
  group('install', () {
    late Directory dir;
    final installs = <EmbedderSource>[];

    EmbedderService service() => EmbedderService(
      install: (config, source, onProgress) async {
        installs.add(source);
        onProgress(100);
        return 'embeddinggemma-2-740m';
      },
      load: (config) async => FakeEmbeddingModel(),
    );

    setUp(() {
      dir = Directory.systemTemp.createTempSync('embedder_service_test');
      installs.clear();
    });

    tearDown(() => dir.deleteSync(recursive: true));

    test('a missing file is reported before anything is registered', () async {
      final model = '${dir.path}/${kEmbedderConfig.modelFile}';

      final result = await service().install(
        EmbedderFromFiles(modelPath: model),
        onProgress: (_) {},
      );

      expect(installs, isEmpty);
      final error = (result as Error<String>).error;
      expect(error, isA<EmbedderFilesMissingException>());
      expect((error as EmbedderFilesMissingException).paths, [model]);
      expect(error.toString(), contains('tool/fetch_models.sh'));
    });

    test(
      'the .litertlm there: installed from it alone, and recorded',
      () async {
        expect(kEmbedderConfig.modelFile, endsWith('.litertlm'));
        final model = (File(
          '${dir.path}/${kEmbedderConfig.modelFile}',
        )..writeAsStringSync('x')).path;
        final embedder = service();

        final result = await embedder.install(
          EmbedderFromFiles(modelPath: model),
          onProgress: (_) {},
        );

        expect(result, isA<Ok<String>>());
        expect(
          installs.single,
          isA<EmbedderFromFiles>().having((s) => s.modelPath, 'model', model),
        );
        expect(embedder.modelId, 'embeddinggemma-2-740m');
        expect(embedder.installedModelPath, model);
      },
    );
  });

  group('load', () {
    Future<(Result<EmbedderInfo>, FakeEmbeddingModel)> loadWith(
      FakeEmbeddingModel model,
    ) async {
      final embedder = EmbedderService(
        install: (config, source, onProgress) async => 'eg',
        load: (config) async => model,
      );
      final files = Directory.systemTemp.createTempSync('embedder_load');
      addTearDown(() => files.deleteSync(recursive: true));
      await embedder.install(
        EmbedderFromFiles(
          modelPath: (File('${files.path}/m.litertlm')..createSync()).path,
        ),
        onProgress: (_) {},
      );
      return (await embedder.load(), model);
    }

    test('the CPU, 768 dimensions, audio and 1024 tokens: ready, with what '
        'it reports', () async {
      final (result, model) = await loadWith(FakeEmbeddingModel());

      final info = (result as Ok<EmbedderInfo>).value;
      expect(info.backend, PreferredBackend.cpu);
      expect(info.dimension, 768);
      expect(info.modelId, 'eg');
      expect(info.modalities, contains(EmbeddingModality.audio));
      expect(info.maxInputTokens, kEmbedderMaxInputTokens);
      expect(info.rssBeforeBytes, isPositive);
      expect(info.rssAfterBytes, isPositive);
      expect(model.closeCalls, 0);
    });

    test('setup declares audio only (images are added for the visual '
        'memory: 70 vision tokens) and the 1024-token limit', () {
      expect(kEmbedderConfig.media, {EmbeddingModality.audio});
      expect(kEmbedderMemoryMedia, {
        EmbeddingModality.audio,
        EmbeddingModality.image,
      });
      expect(kEmbedderVisionTokensPerImage, 70);
      expect(kEmbedderConfig.maxInputTokens, 1024);
    });

    test(
      'a model without the vision tower loads (setup does not ask for it)',
      () async {
        final (result, model) = await loadWith(
          FakeEmbeddingModel(
            modalities: const {EmbeddingModality.text, EmbeddingModality.audio},
          ),
        );

        expect(result, isA<Ok<EmbedderInfo>>());
        expect(model.closeCalls, 0);
      },
    );

    test('a model without the audio encoder fails and closes', () async {
      final (result, model) = await loadWith(
        FakeEmbeddingModel(modalities: const {EmbeddingModality.text}),
      );

      expect(
        (result as Error<EmbedderInfo>).error,
        isA<EmbedderMismatchException>(),
      );
      expect(result.error.toString(), contains('asked for audio'));
      expect(model.closeCalls, 1);
    });

    test(
      'another input limit fails and closes (the caps derive from it)',
      () async {
        final (result, model) = await loadWith(
          FakeEmbeddingModel(maxInputTokens: 512),
        );

        expect(result.toString(), contains('512 input tokens'));
        expect(model.closeCalls, 1);
        final (unlimited, _) = await loadWith(
          FakeEmbeddingModel(maxInputTokens: null),
        );
        expect(unlimited.toString(), contains('unlimited'));
      },
    );

    test('another backend fails and closes the model', () async {
      final (result, model) = await loadWith(
        FakeEmbeddingModel(backend: PreferredBackend.gpu),
      );

      expect(
        (result as Error<EmbedderInfo>).error,
        isA<EmbedderMismatchException>(),
      );
      expect(result.error.toString(), contains('reports gpu'));
      expect(model.closeCalls, 1);
    });

    test('an unknown backend is not taken for the CPU', () async {
      final (result, model) = await loadWith(FakeEmbeddingModel(backend: null));

      expect(result.toString(), contains('an unknown backend'));
      expect(model.closeCalls, 1);
    });

    test(
      'a load that wrote no weight cache (requireWeightCache) fails '
      "the load with the plugin's EmbeddingLoadException, passed on as it is",
      () async {
        // The opening of flutter_edge_ai_litertlm 1.14.0's message
        // (`litert_lm_embedding_pass.dart`); its list of causes is left out.
        const refusal = EmbeddingLoadException(
          'LiteRT-LM wrote no weight cache to /cache while loading '
          '/m/embeddinggemma-2-740m.litertlm, and '
          'LiteRtEmbeddingBackend(requireWeightCache: true) refuses to run it '
          'without one (slower to load; about 290 MB more memory for the '
          '270m). The cache file path is about 271 characters long.',
          engine: 'LiteRT-LM',
        );
        final embedder = EmbedderService(
          install: (config, source, onProgress) async => 'eg',
          load: (config) async => throw refusal,
        );

        final result = await embedder.load();

        expect((result as Error<EmbedderInfo>).error, same(refusal));
        expect('${result.error}', contains('wrote no weight cache'));
        expect(embedder.isLoaded, isFalse);
      },
    );

    test('another dimension fails and closes the model', () async {
      final (result, model) = await loadWith(
        FakeEmbeddingModel(dimension: 512),
      );

      expect(result.toString(), contains('512 dimensions'));
      expect(model.closeCalls, 1);
    });

    test('close() during the load closes what it delivers', () async {
      final gate = Completer<EmbeddingModel>();
      final model = FakeEmbeddingModel();
      final embedder = EmbedderService(
        install: (config, source, onProgress) async => 'eg',
        load: (config) => gate.future,
      );

      final loading = embedder.load();
      await embedder.close();
      gate.complete(model);

      expect(
        (await loading as Error<EmbedderInfo>).error,
        isA<EmbedderServiceClosedException>(),
      );
      expect(model.closeCalls, 1);
      expect(embedder.isLoaded, isFalse);
    });
  });

  group('visual memory', () {
    test('a frame is embedded as an image with the document task type; a '
        'question as raw text (TaskType.none)', () async {
      final model = FakeEmbeddingModel();
      final embedder = await loadedEmbedder(model);
      // Setup's load has no image encoder: the camera demo widens it first.
      expect(embedder.imageUnavailableReason, isNotNull);
      expect(await embedder.enableImages(), isA<Ok<EmbedderImagesLoad>>());
      expect(embedder.imageUnavailableReason, isNull);
      final png = Uint8List.fromList(List.filled(40, 7));

      final image = await embedder.embedImage(png);
      final text = await embedder.embedText('sign about the wifi password');

      expect((image as Ok<List<double>>).value, hasLength(768));
      expect((text as Ok<List<double>>).value, hasLength(768));
      final (imageInput, imageTask) = model.inputs[model.inputs.length - 2];
      expect(imageInput.images.single, png);
      expect(imageInput.text, isNull);
      expect(imageTask, TaskType.retrievalDocument);
      final (textInput, textTask) = model.inputs.last;
      expect(textInput.text, 'sign about the wifi password');
      expect(textTask, TaskType.none);
    });

    test('without the vision tower, or before the load, images are refused '
        'with the reason, never sent', () async {
      final noVision = await loadedEmbedder(
        FakeEmbeddingModel(
          modalities: const {EmbeddingModality.text, EmbeddingModality.audio},
        ),
        config: const EmbedderConfig(
          modelFile: kEmbedderModelFile,
          dimension: 768,
          backend: PreferredBackend.cpu,
          media: {EmbeddingModality.audio},
          maxInputTokens: kEmbedderMaxInputTokens,
        ),
      );
      expect(
        noVision.imageUnavailableReason,
        'EmbeddingGemma was loaded without images',
      );
      final refused = await noVision.embedImage(Uint8List.fromList([1, 2]));
      expect(refused.toString(), contains('without images'));

      final unloaded = EmbedderService();
      expect(unloaded.imageUnavailableReason, 'EmbeddingGemma is not loaded');
      expect(
        (await unloaded.embedImage(Uint8List.fromList([1]))).toString(),
        contains('before load()'),
      );
      expect(await unloaded.embedText('mug'), isA<Error<List<double>>>());
    });

    test('an engine failure is an Error, never a throw', () async {
      final model = FakeEmbeddingModel();
      final embedder = await loadedEmbedder(model);
      expect(await embedder.enableImages(), isA<Ok<EmbedderImagesLoad>>());
      model.queryError = StateError('engine');
      final image = await embedder.embedImage(Uint8List.fromList([1, 2, 3]));
      final text = await embedder.embedText('mug');
      expect(image.toString(), contains('engine'));
      expect(text.toString(), contains('engine'));
    });
  });

  group('image encoder (enableImages)', () {
    const textAudio = {EmbeddingModality.text, EmbeddingModality.audio};
    late List<EmbedderConfig> loads;
    late FakeEmbeddingModel old;
    late FakeEmbeddingModel widened;
    late Completer<void>? loadGate;
    late StateError? failWith;
    late bool closeOldOnLoad;

    /// A service loaded with [old] (text and audio); its next loads act like
    /// flutter_edge_ai's cache: a wider request closes [old] first (when
    /// [closeOldOnLoad]), then builds [widened] — or fails with [failWith]
    /// — and a request for setup's media builds a fresh text and audio
    /// model.
    Future<EmbedderService> service() async {
      final files = Directory.systemTemp.createTempSync('embedder_images');
      addTearDown(() => files.deleteSync(recursive: true));
      final embedder = EmbedderService(
        install: (config, source, onProgress) async => 'eg',
        load: (config) async {
          loads.add(config);
          if (loads.length == 1) return old;
          if (config.media.contains(EmbeddingModality.image)) {
            if (closeOldOnLoad) await old.close();
            await loadGate?.future;
            if (failWith case final error?) throw error;
            return widened;
          }
          return FakeEmbeddingModel(modalities: textAudio);
        },
      );
      await embedder.install(
        EmbedderFromFiles(
          modelPath: (File('${files.path}/m.litertlm')..createSync()).path,
        ),
        onProgress: (_) {},
      );
      expect(await embedder.load(), isA<Ok<EmbedderInfo>>());
      return embedder;
    }

    setUp(() {
      loads = [];
      old = FakeEmbeddingModel(modalities: textAudio);
      widened = FakeEmbeddingModel(
        modalities: const {
          EmbeddingModality.text,
          EmbeddingModality.audio,
          EmbeddingModality.image,
        },
      );
      loadGate = null;
      failWith = null;
      closeOldOnLoad = true;
    });

    test(
      'reloads with audio and image, swaps the model, and images embed',
      () async {
        final embedder = await service();
        expect(
          embedder.imageUnavailableReason,
          'EmbeddingGemma was loaded without images',
        );
        final result = await embedder.enableImages();
        final load = (result as Ok<EmbedderImagesLoad>).value;
        expect(load.alreadyLoaded, isFalse);
        expect(load.rssBeforeBytes, isPositive);
        expect(load.rssAfterBytes, isPositive);
        expect(loads.last.media, {
          EmbeddingModality.audio,
          EmbeddingModality.image,
        });
        expect(embedder.imageUnavailableReason, isNull);
        expect(embedder.modalities, contains(EmbeddingModality.image));
        expect(
          await embedder.embedImage(Uint8List.fromList([1, 2])),
          isA<Ok<List<double>>>(),
        );
        expect(widened.inputs.last.$1.images, isNotEmpty);
        expect(widened.queries.first.$1, 'What is LiteRT?', reason: 'probe');

        // Loaded already: nothing to do.
        final again = await embedder.enableImages();
        expect((again as Ok<EmbedderImagesLoad>).value.alreadyLoaded, isTrue);
        expect(loads, hasLength(2));
      },
    );

    test('concurrent calls share one reload', () async {
      final embedder = await service();
      loadGate = Completer();
      final a = embedder.enableImages();
      final b = embedder.enableImages();
      loadGate!.complete();
      expect(await a, isA<Ok<EmbedderImagesLoad>>());
      expect(await b, isA<Ok<EmbedderImagesLoad>>());
      expect(loads, hasLength(2));
    });

    test('a knowledge-base question during the reload waits and runs on the '
        'new model: it never fails for it', () async {
      final embedder = await service();
      loadGate = Completer();
      final widening = embedder.enableImages();
      await pumpEventQueue();
      var answered = false;
      final query = embedder
          .embedQuery(const KnowledgeQuery.text('What is LiteRT?'))
          .whenComplete(() => answered = true);
      await pumpEventQueue();
      expect(answered, isFalse, reason: 'waits for the swap');

      loadGate!.complete();
      await widening;
      expect(await query, isA<Ok<List<double>>>());
      expect(old.queries, isEmpty, reason: 'nothing ran on the old model');
      expect(widened.queries.map((q) => q.$1), contains('What is LiteRT?'));
    });

    test('a call already running finishes on the old model before the '
        'reload starts', () async {
      final embedder = await service();
      old.gate = Completer();
      final documents = embedder.embedDocuments(['a', 'b']);
      await pumpEventQueue();
      final widening = embedder.enableImages();
      await pumpEventQueue();
      expect(loads, hasLength(1), reason: 'the reload waits for the call');

      old.gate!.complete();
      expect(await documents, isA<Ok<List<List<double>>>>());
      expect(await widening, isA<Ok<EmbedderImagesLoad>>());
      expect(loads, hasLength(2));
    });

    test('a failed reload that took the embedder down loads text and audio '
        'again: the knowledge base goes on, images stay off', () async {
      final embedder = await service();
      failWith = StateError('vision tower missing');
      final result = await embedder.enableImages();
      final error = (result as Error<EmbedderImagesLoad>).error;
      expect(
        error,
        isA<EmbedderImagesException>().having(
          (e) => e.restored,
          'restored',
          isTrue,
        ),
      );
      expect(error.toString(), contains('vision tower missing'));
      expect(loads.last.media, {EmbeddingModality.audio});
      expect(
        embedder.imageUnavailableReason,
        'EmbeddingGemma was loaded without images',
      );
      expect(
        await embedder.embedQuery(const KnowledgeQuery.text('LiteRT?')),
        isA<Ok<List<double>>>(),
      );
    });

    test('a reload refused before the old model was closed keeps it (no '
        'second load)', () async {
      final embedder = await service();
      closeOldOnLoad = false;
      failWith = StateError('refused');
      final result = await embedder.enableImages();
      expect(
        (result as Error<EmbedderImagesLoad>).error,
        isA<EmbedderImagesException>().having(
          (e) => e.restored,
          'restored',
          isTrue,
        ),
      );
      expect(loads, hasLength(2));
      expect(
        await embedder.embedQuery(const KnowledgeQuery.text('LiteRT?')),
        isA<Ok<List<double>>>(),
      );
      expect(old.queries.map((q) => q.$1), contains('LiteRT?'));
    });

    test('a widened model without the vision tower is closed, and text and '
        'audio come back', () async {
      widened = FakeEmbeddingModel(modalities: textAudio);
      final embedder = await service();
      final result = await embedder.enableImages();
      expect(result.toString(), contains('asked for audio+image'));
      expect(widened.closeCalls, 1);
      expect(embedder.isLoaded, isTrue);
    });

    test('when text and audio cannot come back either, nothing is loaded and '
        'it says so', () async {
      final embedder = EmbedderService(
        install: (config, source, onProgress) async => 'eg',
        load: (config) async {
          loads.add(config);
          if (loads.length == 1) return old;
          await old.close();
          throw StateError('out of memory');
        },
      );
      final files = Directory.systemTemp.createTempSync('embedder_images');
      addTearDown(() => files.deleteSync(recursive: true));
      await embedder.install(
        EmbedderFromFiles(
          modelPath: (File('${files.path}/m.litertlm')..createSync()).path,
        ),
        onProgress: (_) {},
      );
      await embedder.load();
      final result = await embedder.enableImages();
      final error = (result as Error<EmbedderImagesLoad>).error;
      expect(
        error,
        isA<EmbedderImagesException>().having(
          (e) => e.restored,
          'restored',
          isFalse,
        ),
      );
      expect(error.toString(), contains('could not be loaded again'));
      expect(embedder.isLoaded, isFalse);
      expect(
        await embedder.embedQuery(const KnowledgeQuery.text('LiteRT?')),
        isA<Error<List<double>>>(),
      );
    });
  });

  group('warm-up and documents', () {
    test('the warm-up embeds one query', () async {
      final model = FakeEmbeddingModel();
      final embedder = await loadedEmbedder(model);

      expect(await embedder.warmUp(), isA<Ok<Duration>>());
      expect(model.queries.single.$2, TaskType.retrievalQuery);
    });

    test(
      'with a clip, the warm-up also embeds it as audio (no prefix), so '
      'the first spoken question does not pay for the audio encoder',
      () async {
        final model = FakeEmbeddingModel();
        final clip = wavFromPcm16(Uint8List(32000), sampleRate: 16000);
        final embedder = await loadedEmbedder(
          model,
          warmUpAudio: () async => clip,
        );

        expect(await embedder.warmUp(), isA<Ok<Duration>>());
        expect(model.inputs, hasLength(2));
        expect(model.inputs.first.$1.text, isNotNull);
        expect(model.inputs.last.$1.audio, clip);
        expect(model.inputs.last.$2, TaskType.none);
        expect(embedder.isLoaded, isTrue);
      },
    );

    test('a clip that cannot be read fails the warm-up and unloads the '
        'model (fail fast)', () async {
      final model = FakeEmbeddingModel();
      final embedder = await loadedEmbedder(
        model,
        warmUpAudio: () async => throw StateError('no asset'),
      );

      final result = await embedder.warmUp();

      expect(result, isA<Error<Duration>>());
      expect(result.toString(), contains('no asset'));
      expect(model.closeCalls, 1);
      expect(embedder.isLoaded, isFalse);
    });

    test('a clip whose embedding is all zeros fails the warm-up too', () async {
      final model = FakeEmbeddingModel(zeroVectors: true);
      final embedder = await loadedEmbedder(
        model,
        warmUpAudio: () async => Uint8List(0),
      );
      expect(await embedder.warmUp(), isA<Error<Duration>>());
      expect(embedder.isLoaded, isFalse);
    });

    test('a model loaded without audio skips the clip', () async {
      final model = FakeEmbeddingModel();
      final embedder = EmbedderService(
        config: const EmbedderConfig(
          modelFile: kEmbedderModelFile,
          dimension: 768,
          backend: PreferredBackend.cpu,
          maxInputTokens: kEmbedderMaxInputTokens,
        ),
        install: (config, source, onProgress) async => 'id',
        load: (config) async => model,
        warmUpAudio: () async => throw StateError('must not be read'),
      );
      final dir = Directory.systemTemp.createTempSync('embedder_noaudio');
      await embedder.install(
        EmbedderFromFiles(
          modelPath: (File('${dir.path}/m.litertlm')..createSync()).path,
        ),
        onProgress: (_) {},
      );
      expect(await embedder.load(), isA<Ok<EmbedderInfo>>());
      expect(await embedder.warmUp(), isA<Ok<Duration>>());
      expect(model.inputs, hasLength(1));
    });

    test('a typed knowledge-base question is embedded as text with the '
        'retrievalQuery task type (its prefix), checked like the '
        'warm-up\'s', () async {
      final model = FakeEmbeddingModel();
      final embedder = await loadedEmbedder(model);

      final vector = await embedder.embedQuery(
        const KnowledgeQuery.text('What is LiteRT?'),
      );

      expect((vector as Ok<List<double>>).value, hasLength(768));
      expect(model.queries.last, ('What is LiteRT?', TaskType.retrievalQuery));
      final (input, _) = model.inputs.last;
      expect(input.audio, isNull);

      final zero = await loadedEmbedder(FakeEmbeddingModel(zeroVectors: true));
      expect(
        '${(await zero.embedQuery(const KnowledgeQuery.text('q')) as Error).error}',
        contains('all-zero'),
      );
    });

    test('a spoken question is embedded as audio, the WAV as it is, with no '
        'prefix (TaskType.none: a task type reaches text only)', () async {
      final model = FakeEmbeddingModel();
      final embedder = await loadedEmbedder(model);
      final wav = wavFromPcm16(Uint8List(32000), sampleRate: 16000);

      final vector = await embedder.embedQuery(KnowledgeQuery.audio(wav));

      expect((vector as Ok<List<double>>).value, hasLength(768));
      final (input, taskType) = model.inputs.last;
      expect(input.text, isNull);
      expect(input.audio, wav);
      expect(taskType, TaskType.none);
      expect(model.queries, isEmpty, reason: 'no text was embedded');
    });

    group('over-long questions: a typed failure, never a cut', () {
      test('text over kKbMaxTextQueryUtf8Bytes is refused before embedding, '
          'counted in UTF-8 bytes', () async {
        final model = FakeEmbeddingModel();
        final embedder = await loadedEmbedder(model);
        final atCap = 'a' * kKbMaxTextQueryUtf8Bytes;
        // 3 bytes each: a third as many characters reach the cap.
        final cjk = '語' * (kKbMaxTextQueryUtf8Bytes ~/ 3 + 1);

        expect(
          await embedder.embedQuery(KnowledgeQuery.text(atCap)),
          isA<Ok<List<double>>>(),
        );
        final over = await embedder.embedQuery(
          KnowledgeQuery.text('${atCap}b'),
        );
        final wide = await embedder.embedQuery(KnowledgeQuery.text(cjk));

        for (final result in [over, wide]) {
          final error = (result as Error<List<double>>).error;
          expect(error, isA<KnowledgeQueryTooLongException>());
          expect(
            (error as KnowledgeQueryTooLongException).kind,
            KnowledgeQueryKind.text,
          );
          expect(error.limit, '$kKbMaxTextQueryUtf8Bytes UTF-8 bytes');
        }
        expect(
          '${(over as Error).error}',
          contains('${kKbMaxTextQueryUtf8Bytes + 1} UTF-8 bytes'),
        );
        expect(utf8.encode(cjk).length, greaterThan(kKbMaxTextQueryUtf8Bytes));
        expect(model.queries, hasLength(1), reason: 'only the one at the cap');
      });

      test('audio over kKbMaxAudioQuery is refused before embedding', () async {
        final model = FakeEmbeddingModel();
        final embedder = await loadedEmbedder(model);
        final long = wavFromPcm16(
          Uint8List((kKbMaxAudioQuery.inSeconds + 1) * 32000),
          sampleRate: 16000,
        );

        final result = await embedder.embedQuery(KnowledgeQuery.audio(long));

        final error = (result as Error<List<double>>).error;
        expect(error, isA<KnowledgeQueryTooLongException>());
        expect(
          (error as KnowledgeQueryTooLongException).kind,
          KnowledgeQueryKind.audio,
        );
        expect(error.length, '31.0 s');
        expect(model.inputs, isEmpty);
      });

      test('the embedder refusing an input as too long becomes the same '
          'typed failure, with its token counts', () async {
        final model = FakeEmbeddingModel()
          ..tooLong = const EmbeddingInputTooLongException(
            'too long',
            maxInputTokens: 1024,
            actualTokens: 1100,
          );
        final embedder = await loadedEmbedder(model);

        final result = await embedder.embedQuery(
          const KnowledgeQuery.text('q'),
        );

        final error = (result as Error<List<double>>).error;
        expect(error, isA<KnowledgeQueryTooLongException>());
        expect((error as KnowledgeQueryTooLongException).length, '1100 tokens');
        expect(error.limit, '1024 tokens');
        expect(error.toString(), contains('ask a shorter one'));
      });

      test('audio that is not a 16 kHz mono PCM16 WAV, or has no samples, '
          'is a FormatException', () async {
        final embedder = await loadedEmbedder(FakeEmbeddingModel());

        final notWav = await embedder.embedQuery(
          KnowledgeQuery.audio(Uint8List.fromList([1, 2, 3, 4])),
        );
        final empty = await embedder.embedQuery(
          KnowledgeQuery.audio(wavFromPcm16(Uint8List(0), sampleRate: 16000)),
        );

        expect((notWav as Error).error, isA<FormatException>());
        expect((empty as Error).error, isA<FormatException>());
      });
    });

    test('an all-zero vector fails the warm-up', () async {
      final embedder = await loadedEmbedder(
        FakeEmbeddingModel(zeroVectors: true),
      );

      final result = await embedder.warmUp();

      expect(result.toString(), contains('all-zero'));
    });

    test('a rejected warm-up vector closes the model and unloads it, like a '
        'mismatch at load', () async {
      final model = FakeEmbeddingModel(zeroVectors: true);
      final embedder = await loadedEmbedder(model);

      final result = await embedder.warmUp();

      expect(
        (result as Error<Duration>).error,
        isA<EmbedderMismatchException>(),
      );
      expect(model.closeCalls, 1, reason: 'the ~0.3 GB model is released');
      expect(embedder.isLoaded, isFalse);
      expect(
        await embedder.embedDocuments(['a']),
        isA<Error<List<List<double>>>>(),
      );
    });

    test('a warm-up that throws also closes the model', () async {
      final model = FakeEmbeddingModel()
        ..queryError = StateError('worker died');
      final embedder = await loadedEmbedder(model);

      final result = await embedder.warmUp();

      expect(result.toString(), contains('worker died'));
      expect(model.closeCalls, 1);
      expect(embedder.isLoaded, isFalse);
    });

    test('documents are embedded with the document prefix, in order', () async {
      final model = FakeEmbeddingModel();
      final embedder = await loadedEmbedder(model);

      final result = await embedder.embedDocuments(['a', 'b']);

      final vectors = (result as Ok<List<List<double>>>).value;
      expect(vectors, [model.vectorFor('a'), model.vectorFor('b')]);
      final (texts, taskType) = model.batches.single;
      expect(texts, ['a', 'b']);
      expect(taskType, TaskType.retrievalDocument);
    });

    test('a worker failure is an Error, not a throw', () async {
      final model = FakeEmbeddingModel()..failFromBatch = 1;
      final embedder = await loadedEmbedder(model);

      final result = await embedder.embedDocuments(['a']);

      expect(result.toString(), contains('embedding worker died'));
    });

    test('close() closes the model once and stops further use', () async {
      final model = FakeEmbeddingModel();
      final embedder = await loadedEmbedder(model);

      await embedder.close();
      await embedder.close();

      expect(model.closeCalls, 1);
      expect(
        (await embedder.embedDocuments(['a']) as Error).error,
        isA<EmbedderServiceClosedException>(),
      );
    });
  });
}
