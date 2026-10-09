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

/// Knowledge-base tunables.
library;

/// Retrieval gate: a hit enters the prompt only at or above this
/// cosine similarity, checked per hit. On the golden set every gate from
/// 0.25 to 0.525 kept 19/19 on-topic and rejected 12/12 off-topic
/// questions; re-tune with `integration_test/kb_retrieval_test.dart`.
const kKbMinSimilarity = 0.40;

/// Excerpts per turn: ~650–950 prompt tokens at the chunker's sizes.
const kKbTopK = 3;

/// The bundled documents (`assets/kb/*.md`).
const kKbAssetPrefix = 'assets/kb/';

/// Chunks embedded per call while indexing. Batching is not faster; it sets
/// how often progress is published.
const kKbEmbedBatch = 8;

/// The prebuilt index built into the app (`tool/build_kb_index.sh`): the
/// sqlite-vec database as the store wrote it on the build machine, and what
/// it was built from. Used on the first launch when its key matches the
/// running app's (documents, chunker, embedder files), instead of embedding
/// every chunk on the device (216 s on a Galaxy S24's CPU).
const kKbPrebuiltManifestAsset = 'assets/kb_index/manifest.json';
const kKbPrebuiltDatabaseAsset = 'assets/kb_index/kb.db';

/// The embedding space of every knowledge-base index: the built-in
/// EmbeddingGemma-300M files (by digest: model ad09e815…, tokenizer
/// d6daa52d…), the package's retrieval prefixes, 768 dimensions.
/// flutter_edge_ai_rag binds it to the database file and refuses vectors of
/// another profile there; change the id when any of these change (and
/// rebuild assets/kb_index with tool/build_kb_index.sh).
const kKbEmbeddingProfileId =
    'embeddinggemma-300m-seq512-mp-ad09e815-sp-d6daa52d-retrieval-prefix-v1';

/// EmbeddingGemma-300M's vector length (the catalog's dimension).
const kKbEmbeddingDimension = 768;
