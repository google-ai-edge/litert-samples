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

// Knowledge-base retrieval on a real target (macOS first): the built-in
// EmbeddingGemma on the CPU (its files found as setup finds them), the real
// sqlite-vec store and the real KnowledgeRepository indexing the bundled
// assets/kb, then the golden set (test_assets/kb_golden.json: 19 on-topic,
// 12 off-topic questions).
//
//   flutter test integration_test/kb_retrieval_test.dart -d macos
//
// For honest latencies keep the test app out of App Nap first:
//   defaults write com.google.ai.edge.examples.litertEdgeDemos NSAppSleepDisabled -bool YES
//
// 1. A fresh index embedded on this device (its own directory, not the
//    app's; the prebuilt index kept out): every chunk embedded and stored,
//    the marker written.
// 2. A second repository over the same files (the next launch): reused, no
//    embedding. The golden set's answers from it are the reference.
// 3. The prebuilt index (assets/kb_index, tool/build_kb_index.sh) installed
//    into an empty directory, as on a first launch: nothing embedded, and the
//    golden set gets the same answers: the same top document for every
//    on-topic question, the same gate decision (top hit at or above
//    kKbMinSimilarity, or not) for every question, and scores within 1e-2 of
//    the device's. The prebuilt vectors come from the build machine's CPU, so
//    on another architecture near-equal chunks can swap places (Linux arm64:
//    same_top=24/31, max_delta=3.90e-3); identical ranked lists are printed
//    as information only.
// 4. On-topic (prebuilt index): the expected document in the top 3 and the
//    top hit at or above the gate for ≥ 17/19 (9/10, scaled).
//    Off-topic: the top hit below the gate for ≥ 10/12 (8/10, scaled).
//
// Prints `KB_PREBUILT install=<ms>ms device_index=<ms>ms same_top=<n>/31
// same_doc=<n>/19 same_gate=<n>/31 max_delta=<x>`, the gate sweep 0.30–0.70 (step 0.025) and
// `KB index=<ms>ms chunks=<n> reuse=<ms>ms prebuilt=<ms>ms query_p50=<ms>ms
// on=<a>/19 off=<b>/12`.

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:litert_edge_demos/config/bootstrap.dart';
import 'package:litert_edge_demos/config/knowledge_config.dart';
import 'package:litert_edge_demos/data/repositories/knowledge_repository.dart';
import 'package:litert_edge_demos/data/services/knowledge/embedder_digests.dart';
import 'package:litert_edge_demos/data/services/knowledge/embedder_service.dart';
import 'package:litert_edge_demos/data/services/knowledge/kb_documents.dart';
import 'package:litert_edge_demos/data/services/knowledge/kb_prebuilt_index.dart';
import 'package:litert_edge_demos/data/services/knowledge/vector_store_service.dart';
import 'package:litert_edge_demos/data/services/model_store/bundled_model_files.dart';
import 'package:litert_edge_demos/domain/models/knowledge.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/utils/result.dart';
import 'package:path_provider/path_provider.dart';

typedef _Answer = ({List<Passage> top, Duration latency});

/// The golden set's answers from one index.
typedef _Golden = ({
  List<({String q, String doc, List<Passage> top})> on,
  List<({String q, List<Passage> top})> off,
  List<Duration> latencies,
});

Future<_Answer> _ask(KnowledgeRepository knowledge, String question) async {
  switch (await knowledge.retrieve(question)) {
    case Ok(:final value):
      expect(
        value.outcome,
        isNot(RetrievalOutcome.unavailable),
        reason: 'the knowledge base must be ready: ${value.detail}',
      );
      return (top: value.candidates, latency: value.latency!);
    case Error(:final error):
      fail('retrieval failed for "$question": $error');
  }
}

String _fmt(List<Passage> top) =>
    top.map((p) => '${p.doc}(${p.similarity.toStringAsFixed(3)})').join(', ');

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('KB: index, reuse, and the golden set clears the gate', (
    tester,
  ) async {
    // The files built into the app, as setup finds them (in place on
    // desktop and iOS, extracted and verified on Android).
    final EmbedderSource source;
    final bundled = BundledModelFiles();
    final model = await bundled.pathOf(kBundledEmbedderModel);
    final tokenizer = await bundled.pathOf(kBundledEmbedderTokenizer);
    switch ((model, tokenizer)) {
      case (Ok(value: final m), Ok(value: final t)):
        source = EmbedderFromFiles(modelPath: m, tokenizerPath: t);
        debugPrint('KB_SOURCE bundled $m');
      case _:
        fail('The built-in EmbeddingGemma: $model / $tokenizer');
    }
    await initEdgeAi();

    // The embedder, exactly as setup loads it.
    final embedder = EmbedderService();
    final installed = await embedder.install(source, onProgress: (_) {});
    expect(installed, isA<Ok<String>>(), reason: '$installed');
    final EmbedderInfo info;
    switch (await embedder.load()) {
      case Ok(:final value):
        info = value;
      case Error(:final error):
        fail('embedder load failed: $error');
    }
    expect(await embedder.warmUp(), isA<Ok<Duration>>());
    final models = ValueNotifier<Map<ModelId, ModelState>>({
      ModelId.embeddingGemma: ModelReady(
        LoadedModelInfo(
          modelId: info.modelId,
          backend: info.backend.name,
          loadTime: info.loadTime,
          warmUpTime: Duration.zero,
        ),
      ),
    });
    // One store per launch, as in the app: a repository's close closes its
    // store for good (a closed store refuses to open again).
    VectorStoreService newStore() =>
        VectorStoreService(embedQuery: embedder.embedQuery);
    final dir = Directory(
      '${(await getApplicationSupportDirectory()).path}/kb_test',
    );
    if (dir.existsSync()) dir.deleteSync(recursive: true);

    // 1. A fresh index embedded on this device (the prebuilt one is kept out
    //    of it: this is the reference the prebuilt index is compared to).
    final first = KnowledgeRepository(
      embedder: embedder,
      store: newStore(),
      models: models,
      indexDir: () async => dir,
      prebuilt: const NoPrebuiltKbIndex('kb_retrieval_test: the reference'),
      digests: FileEmbedderDigests(cacheDir: () async => dir),
      documents: AssetKbDocumentSource(),
    );
    final KnowledgeReady indexed;
    switch (await first.ensureIndexed()) {
      case Ok(:final value):
        indexed = value;
      case Error(:final error):
        fail('indexing failed: $error');
    }
    expect(indexed.reused, isFalse);
    expect(indexed.origin, KnowledgeOrigin.device);
    expect(File('${dir.path}/index.json').existsSync(), isTrue);
    await first.close();

    // 2. The next launch: same files, same model.
    final device = KnowledgeRepository(
      embedder: embedder,
      store: newStore(),
      models: models,
      indexDir: () async => dir,
      digests: FileEmbedderDigests(cacheDir: () async => dir),
      documents: AssetKbDocumentSource(),
      prebuilt: AssetPrebuiltKbIndex(),
    );
    final KnowledgeReady reused;
    switch (await device.ensureIndexed()) {
      case Ok(:final value):
        reused = value;
      case Error(:final error):
        fail('reopening the index failed: $error');
    }
    expect(reused.reused, isTrue);
    expect(reused.chunks, indexed.chunks);

    // The golden set on the on-device index: the reference.
    final golden = jsonDecode(
      await rootBundle.loadString('test_assets/kb_golden.json'),
    ) as Map<String, Object?>;
    final onTopic = [
      for (final entry in golden['on_topic']! as List<Object?>)
        if (entry case {'q': final String q, 'expect_doc': final String doc})
          (q: q, doc: doc)
        else
          throw FormatException('Unexpected golden entry', entry),
    ];
    final offTopic = [
      for (final q in golden['off_topic']! as List<Object?>) q! as String,
    ];
    Future<_Golden> answer(KnowledgeRepository knowledge) async {
      final latencies = <Duration>[];
      final on = <({String q, String doc, List<Passage> top})>[];
      for (final (:q, :doc) in onTopic) {
        final (:top, :latency) = await _ask(knowledge, q);
        latencies.add(latency);
        on.add((q: q, doc: doc, top: top));
      }
      final off = <({String q, List<Passage> top})>[];
      for (final q in offTopic) {
        final (:top, :latency) = await _ask(knowledge, q);
        latencies.add(latency);
        off.add((q: q, top: top));
      }
      return (on: on, off: off, latencies: latencies);
    }

    final onDevice = await answer(device);
    await device.close();

    // 3. The prebuilt index built into the app (assets/kb_index), installed
    //    into an empty directory as on a first launch: nothing embedded.
    final prebuiltDir = Directory('${dir.path}_prebuilt');
    if (prebuiltDir.existsSync()) prebuiltDir.deleteSync(recursive: true);
    final knowledge = KnowledgeRepository(
      embedder: embedder,
      store: newStore(),
      models: models,
      indexDir: () async => prebuiltDir,
      digests: FileEmbedderDigests(cacheDir: () async => prebuiltDir),
      documents: AssetKbDocumentSource(),
      prebuilt: AssetPrebuiltKbIndex(),
    );
    final KnowledgeReady fromPrebuilt;
    switch (await knowledge.ensureIndexed()) {
      case Ok(:final value):
        fromPrebuilt = value;
      case Error(:final error):
        fail('installing the prebuilt index failed: $error');
    }
    expect(
      fromPrebuilt.origin,
      KnowledgeOrigin.prebuilt,
      reason: 'prebuilt index not used: ${fromPrebuilt.prebuiltSkipped}',
    );
    expect(fromPrebuilt.reused, isFalse);
    expect(fromPrebuilt.chunks, indexed.chunks);
    final prebuilt = await answer(knowledge);
    final onAnswers = prebuilt.on;
    final offAnswers = prebuilt.off;
    final latencies = prebuilt.latencies;

    // The same answers: the prebuilt vectors were embedded on the build
    // machine's CPU, these on this device's, so near-equal chunks may swap
    // places on another architecture. What the app does with the result must
    // not change: the same top document for on-topic questions and the same
    // gate decision for every question, with scores within 1e-2.
    bool gateOn(List<Passage> top) =>
        top.isNotEmpty && top.first.similarity >= kKbMinSimilarity;
    final pairs = [
      for (var i = 0; i < onTopic.length; i++)
        (q: onTopic[i].q, a: onDevice.on[i].top, b: onAnswers[i].top, on: true),
      for (var i = 0; i < offTopic.length; i++)
        (
          q: offTopic[i],
          a: onDevice.off[i].top,
          b: offAnswers[i].top,
          on: false,
        ),
    ];
    var sameTop = 0;
    var sameDoc = 0;
    var sameGate = 0;
    var maxDelta = 0.0;
    final problems = <String>[];
    for (final (:q, :a, :b, :on) in pairs) {
      final identical =
          a.length == b.length &&
          [for (var i = 0; i < a.length; i++) a[i].id == b[i].id]
              .every((x) => x);
      if (identical) sameTop++;
      // Scores: the top hit's (what the gate sees), and every chunk both
      // lists hold.
      if (a.isNotEmpty && b.isNotEmpty) {
        maxDelta = math.max(
          maxDelta,
          (a.first.similarity - b.first.similarity).abs(),
        );
      }
      final scores = {for (final p in b) p.id: p.similarity};
      for (final p in a) {
        if (scores[p.id] case final other?) {
          maxDelta = math.max(maxDelta, (p.similarity - other).abs());
        }
      }
      final docAgrees =
          a.isNotEmpty && b.isNotEmpty && a.first.doc == b.first.doc;
      if (on && docAgrees) sameDoc++;
      final gateAgrees = gateOn(a) == gateOn(b);
      if (gateAgrees) sameGate++;
      if (!identical) {
        debugPrint(
          'KB_PREBUILT differs q="$q" device=[${_fmt(a)}] prebuilt=[${_fmt(b)}]',
        );
      }
      if ((on && !docAgrees) || !gateAgrees) {
        problems.add(
          '${on && !docAgrees ? 'top document' : 'gate'} differs: q="$q"',
        );
      }
    }
    debugPrint(
      'KB_PREBUILT install=${fromPrebuilt.elapsed.inMilliseconds}ms '
      'device_index=${indexed.elapsed.inMilliseconds}ms '
      'same_top=$sameTop/${pairs.length} '
      'same_doc=$sameDoc/${onTopic.length} '
      'same_gate=$sameGate/${pairs.length} '
      'max_delta=${maxDelta.toStringAsExponential(2)}',
    );
    expect(problems, isEmpty, reason: 'the prebuilt index answers differently');
    expect(maxDelta, lessThanOrEqualTo(1e-2), reason: 'scores within 1e-2');

    // 4. The golden thresholds, on the prebuilt index (what the app uses).
    bool onPasses(({String q, String doc, List<Passage> top}) a, double g) =>
        a.top.take(3).any((p) => p.doc == a.doc) &&
        a.top.isNotEmpty &&
        a.top.first.similarity >= g;
    bool offPasses(({String q, List<Passage> top}) a, double g) =>
        a.top.isEmpty || a.top.first.similarity < g;

    for (final a in onAnswers) {
      final rank = a.top.indexWhere((p) => p.doc == a.doc);
      debugPrint(
        '${onPasses(a, kKbMinSimilarity) ? 'ok  ' : 'MISS'} on  '
        'rank=${rank < 0 ? '-' : rank + 1} expect=${a.doc} '
        'top=[${_fmt(a.top)}] q="${a.q}"',
      );
    }
    for (final a in offAnswers) {
      debugPrint(
        '${offPasses(a, kKbMinSimilarity) ? 'ok  ' : 'MISS'} off '
        'top=[${_fmt(a.top)}] q="${a.q}"',
      );
    }

    debugPrint(
      'KB sweep gate  on/${onAnswers.length}  off/${offAnswers.length}',
    );
    for (var i = 0; i <= 16; i++) {
      final g = 0.30 + i * 0.025;
      final on = onAnswers.where((a) => onPasses(a, g)).length;
      final off = offAnswers.where((a) => offPasses(a, g)).length;
      debugPrint(
        'KB sweep ${g.toStringAsFixed(3)}  $on  $off'
        '${(g - kKbMinSimilarity).abs() < 1e-9 ? '  <- kKbMinSimilarity' : ''}',
      );
    }

    final sorted = [...latencies]..sort();
    final p50 = sorted[sorted.length ~/ 2];
    final onPass = onAnswers.where((a) => onPasses(a, kKbMinSimilarity));
    final offPass = offAnswers.where((a) => offPasses(a, kKbMinSimilarity));
    debugPrint(
      'KB index=${indexed.elapsed.inMilliseconds}ms '
      'chunks=${indexed.chunks} reuse=${reused.elapsed.inMilliseconds}ms '
      'prebuilt=${fromPrebuilt.elapsed.inMilliseconds}ms '
      'query_p50=${p50.inMilliseconds}ms '
      'on=${onPass.length}/${onAnswers.length} '
      'off=${offPass.length}/${offAnswers.length} '
      'gate=$kKbMinSimilarity backend=${info.backend.name} dim=${info.dimension}',
    );

    expect(onAnswers, hasLength(19));
    expect(offAnswers, hasLength(12));
    expect(onPass.length, greaterThanOrEqualTo(17), reason: 'on-topic');
    expect(offPass.length, greaterThanOrEqualTo(10), reason: 'off-topic');

    await knowledge.close();
    models.dispose();
    await embedder.close();
  }, timeout: const Timeout(Duration(minutes: 10)));
}
