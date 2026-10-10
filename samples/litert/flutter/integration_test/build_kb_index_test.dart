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

// Builds the prebuilt knowledge-base index (assets/kb_index/): the built-in
// EmbeddingGemma embeds assets/kb on the CPU exactly as the app does on a
// device — the app's own KnowledgeRepository, chunker and sqlite-vec store —
// and the store's kb.db is written out with its manifest (what it was built
// from). Run it through the wrapper, which reads the store's version from
// pubspec.lock and copies the result out of the macOS sandbox:
//
//   tool/build_kb_index.sh
//
// Rebuild whenever assets/kb, the chunker (MarkdownChunker.version or its
// budget), the embedder files or flutter_edge_ai_sqlite change:
// test/data/services/knowledge/kb_prebuilt_asset_test.dart fails until the
// index matches again.
// Prints `KB_INDEX_OUT=<dir>` and
// `KB_INDEX_BUILT chunks=<n> db=<bytes>B embed=<ms>ms backend=<b>`.

import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:litert_edge_demos/config/bootstrap.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/knowledge_repository.dart';
import 'package:litert_edge_demos/data/services/knowledge/embedder_digests.dart';
import 'package:litert_edge_demos/data/services/knowledge/embedder_service.dart';
import 'package:litert_edge_demos/data/services/knowledge/kb_documents.dart';
import 'package:litert_edge_demos/data/services/knowledge/kb_index_key.dart';
import 'package:litert_edge_demos/data/services/knowledge/kb_prebuilt_index.dart';
import 'package:litert_edge_demos/data/services/knowledge/vector_store_service.dart';
import 'package:litert_edge_demos/data/services/model_store/bundled_model_files.dart';
import 'package:litert_edge_demos/data/services/model_store/model_file_ops.dart';
import 'package:litert_edge_demos/domain/models/knowledge.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/utils/result.dart';
import 'package:path_provider/path_provider.dart';

/// `flutter_edge_ai_sqlite <version>` from pubspec.lock (the wrapper).
const kStore = String.fromEnvironment('KB_INDEX_STORE');

Future<String> _bundledPath(BundledModelFiles bundled, BundledFile file) async {
  switch (await bundled.pathOf(file)) {
    case Ok(:final value):
      return value;
    case Error(:final error):
      fail('The built-in ${file.name}: $error');
  }
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('build the prebuilt knowledge-base index', (tester) async {
    if (kStore.isEmpty) {
      fail('Pass --dart-define=KB_INDEX_STORE=... (tool/build_kb_index.sh)');
    }
    // 1. The built-in embedder files, hashed for real: the app takes their
    //    digests from the build's constants, so prove those here.
    final bundled = BundledModelFiles();
    final modelPath = await _bundledPath(bundled, kBundledEmbedderModel);
    final tokenizerPath = await _bundledPath(
      bundled,
      kBundledEmbedderTokenizer,
    );
    const ops = ModelFileOps();
    expect(
      await ops.sha256OfFile(modelPath, onProgress: (_, _) {}),
      kBundledEmbedderModel.sha256,
    );
    expect(
      await ops.sha256OfFile(tokenizerPath, onProgress: (_, _) {}),
      kBundledEmbedderTokenizer.sha256,
    );

    // 2. The embedder, exactly as setup loads it.
    await initEdgeAi();
    final embedder = EmbedderService();
    final installed = await embedder.install(
      EmbedderFromFiles(modelPath: modelPath, tokenizerPath: tokenizerPath),
      onProgress: (_) {},
    );
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

    // 3. A fresh index on this machine, through the app's own repository.
    final root = Directory(
      '${(await getApplicationSupportDirectory()).path}/kb_index_build',
    );
    if (root.existsSync()) root.deleteSync(recursive: true);
    final indexDir = Directory('${root.path}/index');
    final repository = KnowledgeRepository(
      embedder: embedder,
      store: VectorStoreService(embedQuery: embedder.embedQuery),
      models: models,
      indexDir: () async => indexDir,
      prebuilt: const NoPrebuiltKbIndex('building the prebuilt index'),
      digests: FileEmbedderDigests(cacheDir: () async => root),
      documents: AssetKbDocumentSource(),
    );
    final KnowledgeReady ready;
    switch (await repository.ensureIndexed()) {
      case Ok(:final value):
        ready = value;
      case Error(:final error):
        fail('indexing failed: $error');
    }
    expect(ready.reused, isFalse);
    expect(ready.origin, KnowledgeOrigin.device);

    // 4. kb.db and the manifest. The marker carries the key the repository
    //    computed; the app compares the manifest's against its own.
    final marker = jsonDecode(
      File('${indexDir.path}/index.json').readAsStringSync(),
    ) as Map<String, Object?>;
    final key = KbIndexKey.fromJson(marker['key']);
    expect(key.modelSha256, kBundledEmbedderModel.sha256);
    expect(key.tokenizerSha256, kBundledEmbedderTokenizer.sha256);
    expect(key.dim, kEmbedderConfig.dimension);
    final db = File('${indexDir.path}/kb.db');
    final bytes = db.readAsBytesSync();
    final out = Directory('${root.path}/out')..createSync(recursive: true);
    File('${out.path}/kb.db').writeAsBytesSync(bytes, flush: true);
    final manifest = PrebuiltKbManifest(
      key: key,
      chunks: ready.chunks,
      dbBytes: bytes.length,
      dbSha256: sha256.convert(bytes).toString(),
      store: kStore,
      embedderBackend: info.backend.name,
      builtOn: Platform.operatingSystem,
      builtAt: DateTime.now(),
    );
    File('${out.path}/manifest.json').writeAsStringSync(
      '${const JsonEncoder.withIndent('  ').convert(manifest.toJson())}\n',
      flush: true,
    );
    debugPrint('KB_INDEX_OUT=${out.path}');
    debugPrint(
      'KB_INDEX_BUILT chunks=${ready.chunks} db=${bytes.length}B '
      'embed=${ready.elapsed.inMilliseconds}ms backend=${info.backend.name} '
      'store="$kStore"',
    );

    await repository.close();
    models.dispose();
    await embedder.close();
  }, timeout: const Timeout(Duration(minutes: 15)));
}
