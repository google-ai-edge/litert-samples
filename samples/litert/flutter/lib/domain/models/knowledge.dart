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

/// One knowledge-base hit, with the metadata a citation shows.
final class const Passage({
  /// `doc#n`.
  required final String id,

  /// The document's file name, e.g. `litert-overview.md`.
  required final String doc,
  required final String title,

  /// `H2` or `H2 › H3`.
  required final String section,

  /// The stored chunk: `Title › Section`, a blank line, the body. Goes into
  /// the prompt as is.
  required final String content,

  /// Cosine similarity to the question, 1 = identical.
  required final double similarity,
  final String? source,
}) {
  /// What a citation chip says before the similarity.
  String get label => '$title › $section';
}

/// What retrieval did for one turn.
enum RetrievalOutcome {
  /// At least one hit cleared the gate; those hits went into the prompt.
  used,

  /// The best hit was below the gate; the prompt is the plain question.
  belowGate,

  /// The knowledge base cannot answer now (no embedder, still indexing, the
  /// index failed); the prompt is the plain question and the reply says so
  /// with a chip.
  unavailable,

  /// The search itself failed; the prompt is the plain question and the
  /// reply shows the error as a chip.
  failed,

  /// Not searched: a live skill question (device facts, the time)
  /// on an agent chat; the prompt is the plain
  /// question, so the model sees only its skills. [Retrieval.detail] names
  /// the route.
  skipped,
}

/// One turn's retrieval: its outcome, the hits and the timing.
final class const Retrieval({
  required final RetrievalOutcome outcome,

  /// The hits that cleared the gate, best first: these are the excerpts in
  /// the prompt, numbered from 1 in this order.
  final List<Passage> passages = const [],

  /// Every hit the search returned (at most top-k), best first, gated or
  /// not. The overlay's top similarity and the golden-set test read it.
  final List<Passage> candidates = const [],

  /// The similarity gate in effect.
  final double? gate,

  /// Query embedding plus vector search.
  final Duration? latency,

  /// Why the knowledge base was unavailable, or what failed.
  final String? detail,
}) {
  /// The best hit's similarity, gated or not; null without hits.
  double? get topSimilarity =>
      candidates.isEmpty ? null : candidates.first.similarity;
}

/// The knowledge base's state, for the home tile, the overlay and each
/// turn's retrieval.
sealed class const KnowledgeStatus();

/// The embedder is not ready yet (setup is still loading it).
final class const KnowledgeWaiting() extends KnowledgeStatus;

/// No embedder in this build or configuration, or it failed to load; chat
/// works without the knowledge base, and says so.
final class const KnowledgeUnavailable(final String reason)
    extends KnowledgeStatus;

/// Chunking and embedding the documents on this device: [done] of [total]
/// chunks stored.
final class const KnowledgeIndexing({
  required final int done,
  required final int total,

  /// Why the index built into the app was not used (it was built from other
  /// documents, another chunker or other embedder files; or this build has
  /// none). Shown next to the progress.
  final String? prebuiltSkipped,
}) extends KnowledgeStatus {
  int get percent => total == 0 ? 0 : (done * 100 / total).floor();
}

/// Where a ready index's vectors came from.
enum KnowledgeOrigin {
  /// The index built into the app (`assets/kb_index`, made by
  /// `tool/build_kb_index.sh`), copied into place: nothing was embedded.
  prebuilt,

  /// Every chunk embedded on this device.
  device,
}

/// Searchable. [reused]: the index from an earlier launch matched the
/// documents, the chunker and the embedder files, so nothing was done.
/// [origin]: where its vectors came from (on the launch that made it).
final class const KnowledgeReady({
  required final int chunks,
  required final bool reused,
  required final Duration elapsed,
  final KnowledgeOrigin origin = KnowledgeOrigin.device,

  /// Why the prebuilt index was not used, when [origin] is
  /// [KnowledgeOrigin.device].
  final String? prebuiltSkipped,
}) extends KnowledgeStatus;

/// Indexing failed; [message] is shown as is. No marker was written, so the
/// next launch indexes again.
final class const KnowledgeFailed(final String message) extends KnowledgeStatus;

/// What a committed reply shows about the knowledge base: the turn's
/// retrieval and which excerpts the reply cited.
final class const ReplyKnowledge({
  required final Retrieval retrieval,

  /// 1-based excerpt numbers that appear as `[n]` in the reply.
  final Set<int> cited = const {},
});
