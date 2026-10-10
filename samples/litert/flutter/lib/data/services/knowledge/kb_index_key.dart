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

import 'dart:convert';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';

import '../../../utils/markdown_chunker.dart';
import 'kb_documents.dart';

/// Everything the vectors of a knowledge-base index depend on. Two indexes
/// with equal keys hold the same chunks embedded by the same model files, so
/// one can stand in for the other.
///
/// The on-device marker (`kb/index.json`) and the prebuilt index's manifest
/// (`assets/kb_index/manifest.json`) both carry one.
final class const KbIndexKey({
  /// [kbDocumentsHash] of the bundled `assets/kb/*.md`.
  required final String documents,

  /// [chunkerId]: the chunker's version and budget.
  required final String chunker,

  /// The text the embedder puts before every chunk
  /// (`TaskType.retrievalDocument.prefix`).
  required final String documentPrefix,

  /// The installed embedder's id (its file name without the extension).
  required final String modelId,

  /// SHA-256 of the embedder's `.tflite` and of its `sentencepiece.model`.
  required final String modelSha256,
  required final String tokenizerSha256,

  /// The vector size.
  required final int dim,

  /// [format] when this app made the key; what a marker or manifest says
  /// otherwise. Part of [digest], the JSON and [differencesFrom], so a bump
  /// invalidates on-device indexes and the shipped prebuilt index alike.
  final String keyFormat = format,
}) {
  /// Bump it when the key's meaning changes.
  static const format = 'kb-index/2';

  /// One SHA-256 over every field (length-prefixed): the marker's `hash`.
  String get digest {
    final input = BytesBuilder(copy: false);
    for (final field in [
      keyFormat,
      documents,
      chunker,
      documentPrefix,
      modelId,
      modelSha256,
      tokenizerSha256,
      '$dim',
    ]) {
      final bytes = utf8.encode(field);
      input
        ..add((ByteData(4)..setUint32(0, bytes.length)).buffer.asUint8List())
        ..add(bytes);
    }
    return sha256.convert(input.takeBytes()).toString();
  }

  /// What differs from [other], in words, for the log and the status line:
  /// e.g. `knowledge-base documents (1a2b3c4d ≠ 5e6f7a8b)`. Empty when the
  /// keys are equal.
  List<String> differencesFrom(KbIndexKey other) {
    // A SHA-256 by its first 8 hex digits; anything else as it is.
    String short(String value) => RegExp(r'^[0-9a-f]{64}$').hasMatch(value)
        ? value.substring(0, 8)
        : '"$value"';
    return [
      if (keyFormat != other.keyFormat)
        'key format ("$keyFormat" ≠ "${other.keyFormat}")',
      if (documents != other.documents)
        'knowledge-base documents '
            '(${short(documents)} ≠ ${short(other.documents)})',
      if (chunker != other.chunker)
        'chunker (${short(chunker)} ≠ ${short(other.chunker)})',
      if (documentPrefix != other.documentPrefix)
        'document prefix '
            '(${short(documentPrefix)} ≠ ${short(other.documentPrefix)})',
      if (modelId != other.modelId)
        'embedder id (${short(modelId)} ≠ ${short(other.modelId)})',
      if (modelSha256 != other.modelSha256)
        'embedder model file '
            '(${short(modelSha256)} ≠ ${short(other.modelSha256)})',
      if (tokenizerSha256 != other.tokenizerSha256)
        'embedder tokenizer '
            '(${short(tokenizerSha256)} ≠ ${short(other.tokenizerSha256)})',
      if (dim != other.dim) 'dimension ($dim ≠ ${other.dim})',
    ];
  }

  Map<String, Object?> toJson() => {
    'format': keyFormat,
    'documents': documents,
    'chunker': chunker,
    'documentPrefix': documentPrefix,
    'modelId': modelId,
    'modelSha256': modelSha256,
    'tokenizerSha256': tokenizerSha256,
    'dim': dim,
  };

  /// Throws [FormatException] unless [json] has every field.
  static KbIndexKey fromJson(Object? json) {
    if (json case {
      'format': final String keyFormat,
      'documents': final String documents,
      'chunker': final String chunker,
      'documentPrefix': final String documentPrefix,
      'modelId': final String modelId,
      'modelSha256': final String modelSha256,
      'tokenizerSha256': final String tokenizerSha256,
      'dim': final int dim,
    }) {
      return KbIndexKey(
        documents: documents,
        chunker: chunker,
        documentPrefix: documentPrefix,
        modelId: modelId,
        modelSha256: modelSha256,
        tokenizerSha256: tokenizerSha256,
        dim: dim,
        keyFormat: keyFormat,
      );
    }
    throw FormatException('Not a knowledge-base index key', json);
  }
}

/// SHA-256 over each document's asset path and bytes (length-prefixed), in
/// the order given (the sources return them sorted by path).
String kbDocumentsHash(List<KbDocument> documents) {
  final input = BytesBuilder(copy: false);
  void field(List<int> bytes) {
    input
      ..add((ByteData(4)..setUint32(0, bytes.length)).buffer.asUint8List())
      ..add(bytes);
  }

  field(utf8.encode('kb-docs/1'));
  for (final document in documents) {
    field(utf8.encode(document.path));
    field(document.bytes);
  }
  return sha256.convert(input.takeBytes()).toString();
}

/// `md-chunker-1/1400/300`: [MarkdownChunker.version] and the budget.
String chunkerId(MarkdownChunker chunker) =>
    '${MarkdownChunker.version}/${chunker.maxCost}/${chunker.overlapMaxChars}';

/// The prebuilt index built into the app (`assets/kb_index/`): the sqlite-vec
/// database `kb.db` exactly as the app's store wrote it on the build machine,
/// and this manifest saying what it was built from.
final class const PrebuiltKbManifest({
  required final KbIndexKey key,
  required final int chunks,

  /// Size and SHA-256 of the shipped `kb.db`.
  required final int dbBytes,
  required final String dbSha256,

  /// The store that wrote it, e.g. `flutter_edge_ai_sqlite 2.0.0`: the
  /// database file is that release's vec0 layout.
  required final String store,

  /// The embedder's backend on the build machine (`cpu`).
  required final String embedderBackend,

  /// The build machine's OS (`macos`).
  required final String builtOn,
  required final DateTime builtAt,
}) {
  static const format = 'kb-prebuilt/1';

  Map<String, Object?> toJson() => {
    'format': format,
    'key': key.toJson(),
    'chunks': chunks,
    'dbBytes': dbBytes,
    'dbSha256': dbSha256,
    'store': store,
    'embedderBackend': embedderBackend,
    'builtOn': builtOn,
    'builtAt': builtAt.toUtc().toIso8601String(),
  };

  /// Throws [FormatException] for another format or a missing field.
  static PrebuiltKbManifest fromJson(Object? json) {
    if (json case {
      'format': final String format,
      'key': final Object key,
      'chunks': final int chunks,
      'dbBytes': final int dbBytes,
      'dbSha256': final String dbSha256,
      'store': final String store,
      'embedderBackend': final String embedderBackend,
      'builtOn': final String builtOn,
      'builtAt': final String builtAt,
    }) {
      if (format != PrebuiltKbManifest.format) {
        throw FormatException(
          'Prebuilt index format "$format", this build reads '
          '"${PrebuiltKbManifest.format}"',
        );
      }
      return PrebuiltKbManifest(
        key: KbIndexKey.fromJson(key),
        chunks: chunks,
        dbBytes: dbBytes,
        dbSha256: dbSha256,
        store: store,
        embedderBackend: embedderBackend,
        builtOn: builtOn,
        builtAt: DateTime.parse(builtAt),
      );
    }
    throw FormatException('Not a prebuilt index manifest', json);
  }
}
