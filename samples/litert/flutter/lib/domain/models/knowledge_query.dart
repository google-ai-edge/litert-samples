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

import 'dart:typed_data';

/// The shape of a knowledge-base question: each has its own similarity gate
/// (`kKbMinSimilarityText`, `kKbMinSimilarityAudio`).
enum KnowledgeQueryKind { text, audio }

/// One knowledge-base question, as the domain asks it. The data layer maps
/// it onto the embedder's own input type; nothing above it sees that type.
sealed class const KnowledgeQuery() {
  /// A typed or transcribed question (embedded with the `retrievalQuery`
  /// prefix).
  const factory KnowledgeQuery.text(String text) = KnowledgeTextQuery;

  /// A spoken question: 16 kHz mono PCM16 in a WAV container (a 44-byte
  /// RIFF header, as `wavFromPcm16` writes it), embedded as audio with no
  /// prefix (a task type reaches text only).
  const factory KnowledgeQuery.audio(Uint8List wav) = KnowledgeAudioQuery;

  KnowledgeQueryKind get kind;
}

final class const KnowledgeTextQuery(final String text) extends KnowledgeQuery {
  @override
  KnowledgeQueryKind get kind => KnowledgeQueryKind.text;

  @override
  String toString() => 'KnowledgeQuery.text(${text.length} chars)';
}

final class const KnowledgeAudioQuery(final Uint8List wav)
    extends KnowledgeQuery {
  @override
  KnowledgeQueryKind get kind => KnowledgeQueryKind.audio;

  @override
  String toString() => 'KnowledgeQuery.audio(${wav.length} bytes)';
}

/// The question is longer than the embedder takes: it was not searched, and
/// nothing was cut to make it fit. Either caught before embedding (text over
/// its UTF-8 cap, audio over its duration cap) or refused by the embedder
/// itself (more than its token limit). Shown as the turn's failed retrieval.
final class KnowledgeQueryTooLongException implements Exception {
  const KnowledgeQueryTooLongException({
    required this.kind,
    required this.length,
    required this.limit,
  });

  final KnowledgeQueryKind kind;

  /// What was measured, in words: `1210 UTF-8 bytes`, `34.2 s`, `1100
  /// tokens`.
  final String length;

  /// The limit in the same unit: `992 UTF-8 bytes`, `30 s`, `1024 tokens`.
  final String limit;

  @override
  String toString() => switch (kind) {
    KnowledgeQueryKind.text =>
      'The question is too long to search the knowledge base ($length, at '
          'most $limit): ask a shorter one',
    KnowledgeQueryKind.audio =>
      'The spoken question is too long to search the knowledge base '
          '($length, at most $limit): ask a shorter one',
  };
}
