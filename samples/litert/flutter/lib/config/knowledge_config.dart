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

import '../domain/models/knowledge_query.dart';

/// Retrieval gate for a text question (typed, or transcribed by Whisper): a
/// hit enters the prompt only at or above this cosine similarity, checked
/// per hit. Sits between the best-match scores of the golden set's on-topic
/// and off-topic questions (test_assets/kb_golden.json, checked by
/// integration_test/kb_retrieval_test.dart), about the middle of the gates
/// that keep every on-topic question and reject every off-topic one. Through
/// Whisper (with the voice chat's corrections) it still sits between the
/// on-topic questions' expected documents and the golden file's
/// `everyday_chat` lines ("Can you help me?", "What did I just say?", …).
/// Re-tune with that test.
const kKbMinSimilarityText = 0.70;

/// Retrieval gate for a spoken question embedded as audio (no transcript):
/// audio scores lower than its own words as text. Sits between the scores of
/// the spoken golden set (test_assets/kb_golden.json, spoken by the app's own
/// TTS into memory by integration_test/kb_retrieval_test.dart): above the
/// best hit of every off-topic question and every everyday-chat line (which
/// score higher than the off-topic questions), below the expected document
/// of every on-topic question. About the middle of that range, rounded up:
/// an off-topic hit puts unrelated excerpts in the prompt. Re-tune with that
/// test.
const kKbMinSimilarityAudio = 0.69;

/// The gate for a question of [kind].
double kbMinSimilarityFor(KnowledgeQueryKind kind) => switch (kind) {
  KnowledgeQueryKind.text => kKbMinSimilarityText,
  KnowledgeQueryKind.audio => kKbMinSimilarityAudio,
};

/// EmbeddingGemma 2's input limit in tokens, task prefix, BOS and EOS
/// included (the bundle's metadata, set explicitly by
/// flutter_edge_ai_litertlm 1.14.0). The embedder must report exactly this at
/// load, or setup fails: [kKbMaxTextQueryUtf8Bytes] is derived from it.
const kEmbedderMaxInputTokens = 1024;

/// The longest text question searched, in UTF-8 bytes; a longer one fails
/// with `KnowledgeQueryTooLongException` before anything is embedded (the
/// engine refuses an over-long input only after paying for all of it, and
/// nothing is ever cut to fit). Derived from [kEmbedderMaxInputTokens] with
/// the tokenizer's worst case: the engine tokenizes the `retrievalQuery`
/// prefix plus the question and adds BOS and EOS, and a byte-fallback
/// SentencePiece token covers at least one UTF-8 byte, so the input has at
/// most `bytes(prefix + question) + 2` tokens (+1 for a dummy-prefix piece).
/// 1024 − 29 (the prefix `task: search result | query: `) − 3 = 992; 32
/// bytes more are kept spare: 960 bytes, about 950 characters of English or
/// 320 of a 3-byte script. test/config/knowledge_config_test.dart checks the
/// arithmetic.
const kKbMaxTextQueryUtf8Bytes = 960;

/// The longest spoken question searched as audio: the voice capture's cap
/// (Whisper's 30 s window); about 25 tokens per second, so ~750 tokens.
const kKbMaxAudioQuery = Duration(seconds: 30);

/// The clip the embedder's audio encoder is warmed up with at load, so the
/// first spoken question does not pay for its first use: the self-test's
/// spoken "What time is it?" (2 s, 16 kHz mono PCM16 WAV, the app's own TTS).
const kKbAudioWarmUpAsset = 'assets/selftest/what_time_is_it.wav';

/// Excerpts per turn, at most: ~650–950 prompt tokens at the chunker's sizes.
/// A turn whose excerpts do not all fit the chat model's context window beside
/// it (with the reply allowance and the headroom: a 1280-token NPU build has
/// room for a prompt of ~600) keeps the best that do, lowest score dropped
/// first (`ChatTurnResponder`), and its retrieval says so ("2 of 3 excerpts
/// (context)").
const kKbTopK = 3;

/// The bundled documents (`assets/kb/*.md`).
const kKbAssetPrefix = 'assets/kb/';

/// Chunks embedded per call while indexing. Batching is not faster; it sets
/// how often progress is published.
const kKbEmbedBatch = 8;

/// The prebuilt index built into the app (`tool/build_kb_index.sh`): the
/// sqlite-vec database as the store wrote it on the build machine, and what
/// it was built from. Used on the first launch when its key matches the
/// running app's (documents, chunker, embedder file), instead of embedding
/// every chunk on the device.
const kKbPrebuiltManifestAsset = 'assets/kb_index/manifest.json';
const kKbPrebuiltDatabaseAsset = 'assets/kb_index/kb.db';

/// The embedding space of every knowledge-base index: the built-in
/// EmbeddingGemma 2 740m bundle
/// (litert-community/embeddinggemma-2-740m-litert-lm at revision 24d962e9,
/// SHA-256 e7a8a220…), text embedded with the package's retrieval prefixes,
/// 768 dimensions. flutter_edge_ai_rag binds it to the database file and
/// refuses vectors of another profile there (an index of another embedder is
/// refused, deleted and replaced); change the id when
/// any of these change (and rebuild assets/kb_index with
/// tool/build_kb_index.sh). Free of control characters, as the package
/// requires.
const kKbEmbeddingProfileId =
    'embeddinggemma-2-740m-litertlm-24d962e9-e7a8a220-text-retrieval-prefix-v1';

/// EmbeddingGemma 2's vector length (the catalog's dimension).
const kKbEmbeddingDimension = 768;
