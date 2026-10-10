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
import 'dart:isolate';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';

import '../../../utils/markdown_chunker.dart';
import 'kb_documents.dart';

// The knowledge base's CPU work that would drop frames on the main isolate,
// each on a short-lived isolate. Top-level functions, so a closure cannot
// capture its caller (the knowledge repository and its unsendable embedder
// ports).

/// Every document's chunks, in order (~50 ms for the bundled KB, a dropped
/// frame as home appears otherwise). A chunker error (a malformed document)
/// crosses the isolate as it was thrown.
Future<List<KbChunk>> chunkInBackground(
  MarkdownChunker chunker,
  List<KbDocument> documents,
) => Isolate.run(
  () => [
    for (final document in documents)
      ...chunker.chunk(utf8.decode(document.bytes), docId: document.name),
  ],
  debugName: 'kb-chunker',
);

/// SHA-256 of a few MB as lowercase hex (tens of ms on a phone, a dropped
/// frame on the home screen otherwise).
Future<String> sha256InBackground(Uint8List bytes) => Isolate.run(
  () => sha256.convert(bytes).toString(),
  debugName: 'kb-prebuilt-sha256',
);
