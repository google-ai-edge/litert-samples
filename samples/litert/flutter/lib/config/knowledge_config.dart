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
/// per hit. EmbeddingGemma 2 740m on the golden set
/// (`integration_test/kb_retrieval_test.dart`, 2026-10-13): the best match of
/// the 19 on-topic questions scores 0.789–0.874, of the 12 off-topic ones
/// 0.573–0.623; every gate from 0.625 to 0.775 keeps 19/19 and rejects 12/12,
/// and 0.70 is the middle of that range. Through Whisper (with the voice
/// chat's corrections) the on-topic questions' expected documents score
/// 0.740–0.848 and the 12 lines of everyday chat (the golden file's
/// `everyday_chat`: "Can you help me?", "What did I just say?", …) 0.600–0.680,
/// so 0.70 still sits in the clean range (0.680, 0.740]. Re-tune with that
/// test.
const kKbMinSimilarityText = 0.70;

/// Retrieval gate for a spoken question embedded as audio (no transcript):
/// audio scores lower than its own words as text. The spoken golden set
/// (spoken by the app's own TTS into memory by
/// `integration_test/kb_retrieval_test.dart`; on an M4 Pro Mac's CPU,
/// 2026-10-13): the expected document of the 19 on-topic questions scores
/// 0.728–0.839, the best hit of the 12 off-topic ones 0.537–0.647, so every
/// gate in (0.647, 0.728] answers 19/19 and rejects 12/12. 0.69 is the middle
/// of that range (0.6875) and of the spike's two macOS voices (on-topic
/// 0.73–0.85, off-topic 0.54–0.64: 0.685), rounded up: an off-topic hit puts
/// unrelated excerpts in the prompt. Everyday chat does score higher than
/// those off-topic questions: its 12 lines, spoken the same way, best-match
/// 0.575–0.665 ("Explain what a neural network is." 0.665, "Can you help
/// me?" 0.664), which narrows the clean range to (0.665, 0.728] (middle
/// 0.697); 0.69 stays inside it and rejects all 24 negatives. Re-tune with
/// that test.
const kKbMinSimilarityAudio = 0.69;

/// The gate for a question of [kind].
double kbMinSimilarityFor(KnowledgeQueryKind kind) => switch (kind) {
  KnowledgeQueryKind.text => kKbMinSimilarityText,
  KnowledgeQueryKind.audio => kKbMinSimilarityAudio,
};

/// EmbeddingGemma 2's input limit in tokens, task prefix, BOS and EOS
/// included (the bundle's metadata, set explicitly by
/// flutter_edge_ai_litertlm 1.13.2). The embedder must report exactly this at
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
/// running app's (documents, chunker, embedder file), instead of embedding
/// every chunk on the device (29 s on an M4 Pro Mac's CPU, minutes on a
/// phone).
const kKbPrebuiltManifestAsset = 'assets/kb_index/manifest.json';
const kKbPrebuiltDatabaseAsset = 'assets/kb_index/kb.db';

/// The embedding space of every knowledge-base index: the built-in
/// EmbeddingGemma 2 740m bundle
/// (litert-community/embeddinggemma-2-740m-litert-lm at revision 24d962e9,
/// SHA-256 e7a8a220…), text embedded with the package's retrieval prefixes,
/// 768 dimensions. flutter_edge_ai_rag binds it to the database file and
/// refuses vectors of another profile there (an index of the earlier
/// EmbeddingGemma-300M is refused, deleted and replaced); change the id when
/// any of these change (and rebuild assets/kb_index with
/// tool/build_kb_index.sh). Free of control characters, as the package
/// requires.
const kKbEmbeddingProfileId =
    'embeddinggemma-2-740m-litertlm-24d962e9-e7a8a220-text-retrieval-prefix-v1';

/// EmbeddingGemma 2's vector length (the catalog's dimension).
const kKbEmbeddingDimension = 768;
