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

import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/repositories/diagnostics_repository.dart';
import 'package:litert_edge_demos/domain/models/diagnostics_snapshot.dart';
import 'package:litert_edge_demos/domain/models/knowledge.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/ui/core/debug_overlay.dart';

import '../../fakes/fake_knowledge.dart';

const _embedder = LoadedModelInfo(
  modelId: 'embeddinggemma-300M_seq512_mixed-precision',
  backend: 'cpu',
  loadTime: Duration(milliseconds: 1144),
  warmUpTime: Duration(milliseconds: 149),
  detail: 'CPU · 768-d',
);

/// The overlay's knowledge-base block — the embedder, the index
/// status and progress, and the last retrieval's time, top similarity and
/// "below gate".
void main() {
  List<String> kbLines(DiagnosticsSnapshot s) => debugOverlayLines(s)
      .where(
        (l) =>
            l.startsWith('KB') ||
            l.startsWith('RAG') ||
            l.startsWith('embeddinggemma') ||
            l.startsWith('EmbeddingGemma'),
      )
      .toList();

  test('indexing progress and the embedder line', () {
    final lines = kbLines(
      const DiagnosticsSnapshot(
        models: {ModelId.embeddingGemma: _embedder},
        knowledge: KnowledgeIndexing(done: 120, total: 290),
      ),
    );

    expect(lines, [
      'embeddinggemma-300M_seq512_mixed-precision · CPU · 768-d · '
          'load 1144 ms · warm 149 ms',
      'KB indexing 41% (120/290 chunks)',
    ]);
  });

  test('a turn below the gate shows its time and top similarity', () {
    final lines = kbLines(
      DiagnosticsSnapshot(
        models: const {ModelId.embeddingGemma: _embedder},
        knowledge: const KnowledgeReady(
          chunks: 290,
          reused: true,
          elapsed: Duration(milliseconds: 35),
        ),
        lastRetrieval: Retrieval(
          outcome: RetrievalOutcome.belowGate,
          candidates: [passage(1, similarity: 0.214)],
          gate: 0.4,
          latency: const Duration(milliseconds: 128),
        ),
      ),
    );

    expect(lines.sublist(1), [
      'KB ready · 290 chunks · reused in 35 ms',
      'RAG 128 ms · top 0.21 · below gate 0.40',
    ]);
  });

  test('a skill question skipped retrieval, and the overlay says '
      'which route', () {
    final lines = kbLines(
      const DiagnosticsSnapshot(
        models: {ModelId.embeddingGemma: _embedder},
        knowledge: KnowledgeReady(
          chunks: 290,
          reused: true,
          elapsed: Duration(milliseconds: 35),
        ),
        lastRetrieval: Retrieval(
          outcome: RetrievalOutcome.skipped,
          detail: 'deviceFacts (device:accelerator+now)',
        ),
      ),
    );

    expect(
      lines.last,
      'RAG skipped: skill question deviceFacts (device:accelerator+now)',
    );
  });

  test('a used retrieval, an unavailable knowledge base', () {
    final used = kbLines(
      DiagnosticsSnapshot(
        knowledge: const KnowledgeReady(
          chunks: 290,
          reused: false,
          elapsed: Duration(milliseconds: 48900),
        ),
        lastRetrieval: Retrieval(
          outcome: RetrievalOutcome.used,
          passages: [passage(1, similarity: 0.62), passage(2)],
          candidates: [passage(1, similarity: 0.62), passage(2)],
          gate: 0.4,
          latency: const Duration(milliseconds: 131),
        ),
      ),
    );
    expect(used, [
      'EmbeddingGemma 300M (knowledge base): not loaded',
      'KB ready · 290 chunks · indexed in 48900 ms',
      'RAG 131 ms · top 0.62 · 2 excerpts (gate 0.40)',
    ]);

    final unavailable = kbLines(
      const DiagnosticsSnapshot(
        knowledge: KnowledgeUnavailable(
          'the built-in EmbeddingGemma is missing',
        ),
        lastRetrieval: Retrieval(
          outcome: RetrievalOutcome.unavailable,
          detail: 'the built-in EmbeddingGemma is missing',
        ),
      ),
    );
    expect(unavailable.sublist(1), [
      'KB unavailable: the built-in EmbeddingGemma is missing',
      'RAG unavailable: the built-in EmbeddingGemma is missing',
    ]);
  });

  test('the prebuilt index: installed, reused; and why it was not used while '
      'indexing on the device', () {
    String kbLine(KnowledgeStatus status) =>
        kbLines(DiagnosticsSnapshot(knowledge: status))
            .singleWhere((l) => l.startsWith('KB'));

    expect(
      kbLine(
        const KnowledgeReady(
          chunks: 290,
          reused: false,
          elapsed: Duration(milliseconds: 97),
          origin: KnowledgeOrigin.prebuilt,
        ),
      ),
      'KB ready · 290 chunks · prebuilt index installed in 97 ms',
    );
    expect(
      kbLine(
        const KnowledgeReady(
          chunks: 290,
          reused: true,
          elapsed: Duration(milliseconds: 27),
          origin: KnowledgeOrigin.prebuilt,
        ),
      ),
      'KB ready · 290 chunks · reused (prebuilt) in 27 ms',
    );
    const why =
        'it was built from another knowledge-base documents (1a2b3c4d ≠ '
        '987e8d5a); rebuild it with tool/build_kb_index.sh';
    expect(
      kbLine(
        const KnowledgeIndexing(done: 8, total: 290, prebuiltSkipped: why),
      ),
      'KB indexing 2% (8/290 chunks) · prebuilt index not used: $why',
    );
    expect(
      kbLine(
        const KnowledgeReady(
          chunks: 290,
          reused: false,
          elapsed: Duration(milliseconds: 216000),
          prebuiltSkipped: why,
        ),
      ),
      'KB ready · 290 chunks · indexed in 216000 ms · prebuilt index not '
      'used: $why',
    );
  });

  test('no knowledge base wired: no block, and the model lines are as '
      'before', () {
    final lines = debugOverlayLines(const DiagnosticsSnapshot());

    expect(lines.where((l) => l.startsWith('KB')), isEmpty);
    expect(lines.where((l) => l.contains('EmbeddingGemma')), isEmpty);
    expect(lines[4], 'moonshine-tiny (STT, Live camera): not loaded');
    expect(lines[5], startsWith('TTFT'));
  });

  testWidgets('the repository records retrievals and follows the knowledge '
      'status', (tester) async {
    final models = ValueNotifier<Map<ModelId, ModelState>>(const {});
    final knowledge = ValueNotifier<KnowledgeStatus>(const KnowledgeWaiting());
    final repo = DiagnosticsRepository(
      models: models,
      knowledge: knowledge,
      minInterval: Duration.zero,
      sampleRss: false,
    );

    expect(repo.latest.knowledge, isA<KnowledgeWaiting>());
    knowledge.value = const KnowledgeIndexing(done: 8, total: 290);
    expect(repo.latest.knowledge, isA<KnowledgeIndexing>());
    repo.recordRetrieval(
      const Retrieval(
        outcome: RetrievalOutcome.failed,
        detail: 'database is locked',
      ),
    );
    await tester.pump(const Duration(milliseconds: 10));

    expect(repo.snapshot.value.lastRetrieval?.outcome, RetrievalOutcome.failed);
    expect(repo.snapshot.value.knowledge, isA<KnowledgeIndexing>());

    repo.dispose();
    knowledge.dispose();
    models.dispose();
  });

  test('a failed recognizer switch says so, not "switching"', () {
    final failed = debugOverlayLines(
      const DiagnosticsSnapshot(sttSwitchError: 'Exception: disk full'),
    );
    expect(failed, contains('STT switch failed: Exception: disk full'));
    expect(failed.where((l) => l.contains('switching')), isEmpty);

    final kept = debugOverlayLines(
      const DiagnosticsSnapshot(
        activeStt: ActiveSttInfo(
          id: ModelId.moonshineTiny,
          modelId: 'moonshine_tiny_5s_f32',
        ),
        sttSwitchError: 'Exception: disk full',
      ),
    );
    expect(kept, contains('STT active: moonshine_tiny_5s_f32'));
    expect(kept, contains('STT switch failed: Exception: disk full'));

    expect(
      debugOverlayLines(const DiagnosticsSnapshot()),
      contains('STT active: none (switching…)'),
    );
  });
}
