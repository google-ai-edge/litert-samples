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

import 'package:flutter_edge_ai/flutter_edge_ai.dart'
    show EmbeddingModel, PreferredBackend, TaskType;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/services/knowledge/embedder_service.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../../fakes/fake_knowledge.dart';

void main() {
  group('install', () {
    late Directory dir;
    final installs = <EmbedderSource>[];

    EmbedderService service() => EmbedderService(
      install: (config, source, onProgress) async {
        installs.add(source);
        onProgress(100);
        return 'embeddinggemma-300M_seq512_mixed-precision';
      },
      load: (config) async => FakeEmbeddingModel(),
    );

    setUp(() {
      dir = Directory.systemTemp.createTempSync('embedder_service_test');
      installs.clear();
    });

    tearDown(() => dir.deleteSync(recursive: true));

    test('a missing file is reported before anything is registered', () async {
      final tokenizer = File('${dir.path}/${kEmbedderConfig.tokenizerFile}')
        ..writeAsStringSync('');
      final model = '${dir.path}/${kEmbedderConfig.modelFile}';

      final result = await service().install(
        EmbedderFromFiles(modelPath: model, tokenizerPath: tokenizer.path),
        onProgress: (_) {},
      );

      expect(installs, isEmpty);
      final error = (result as Error<String>).error;
      expect(error, isA<EmbedderFilesMissingException>());
      expect((error as EmbedderFilesMissingException).paths, [model]);
      expect(error.toString(), contains('tool/fetch_models.sh'));
    });

    test('both files there: installed from them, and recorded', () async {
      final [model, tokenizer] = [
        for (final name in [
          kEmbedderConfig.modelFile,
          kEmbedderConfig.tokenizerFile,
        ])
          (File('${dir.path}/$name')..writeAsStringSync('x')).path,
      ];
      final embedder = service();

      final result = await embedder.install(
        EmbedderFromFiles(modelPath: model, tokenizerPath: tokenizer),
        onProgress: (_) {},
      );

      expect(result, isA<Ok<String>>());
      expect(
        installs.single,
        isA<EmbedderFromFiles>().having((s) => s.modelPath, 'model', model),
      );
      expect(embedder.modelId, 'embeddinggemma-300M_seq512_mixed-precision');
      expect(embedder.installedFiles, (model: model, tokenizer: tokenizer));
    });
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
          modelPath: (File('${files.path}/m.tflite')..createSync()).path,
          tokenizerPath: (File('${files.path}/t.model')..createSync()).path,
        ),
        onProgress: (_) {},
      );
      return (await embedder.load(), model);
    }

    test('the CPU and 768 dimensions: ready, with what it reports', () async {
      final (result, model) = await loadWith(FakeEmbeddingModel());

      final info = (result as Ok<EmbedderInfo>).value;
      expect(info.backend, PreferredBackend.cpu);
      expect(info.dimension, 768);
      expect(info.modelId, 'eg');
      expect(model.closeCalls, 0);
    });

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

  group('warm-up and documents', () {
    test('the warm-up embeds one query', () async {
      final model = FakeEmbeddingModel();
      final embedder = await loadedEmbedder(model);

      expect(await embedder.warmUp(), isA<Ok<Duration>>());
      expect(model.queries.single.$2, TaskType.retrievalQuery);
    });

    test('a knowledge-base query is embedded with the retrievalQuery task '
        'type (its prefix), checked like the warm-up\'s', () async {
      final model = FakeEmbeddingModel();
      final embedder = await loadedEmbedder(model);

      final vector = await embedder.embedQuery('What is LiteRT?');

      expect((vector as Ok<List<double>>).value, hasLength(768));
      expect(model.queries.last, ('What is LiteRT?', TaskType.retrievalQuery));

      final zero = await loadedEmbedder(FakeEmbeddingModel(zeroVectors: true));
      expect(
        '${(await zero.embedQuery('q') as Error).error}',
        contains('all-zero'),
      );
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
