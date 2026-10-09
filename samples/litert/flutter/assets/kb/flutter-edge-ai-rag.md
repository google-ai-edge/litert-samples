---
title: On-device RAG and embeddings with flutter_edge_ai
source: https://pub.dev/packages/flutter_edge_ai_rag/versions/1.0.0 (README.md, lib/src/) ; https://pub.dev/packages/flutter_edge_ai/versions/2.1.1 (skills/flutter-edge-ai-rag/SKILL.md, lib/core/embedding/, example/lib/models/embedding_model.dart) ; https://pub.dev/packages/flutter_edge_ai_sqlite/versions/2.0.0 (README.md) ; https://pub.dev/packages/flutter_edge_ai_embeddings/versions/2.2.2 (README.md) ; https://pub.dev/packages/flutter_edge_ai_litertlm/versions/1.10.1 (README.md) ; https://github.com/DenisovAV/flutter_edge_ai/blob/8ab3ca0d53dd4fab353d5880635d95e9d888472b/website/content/docs/embeddings-and-rag.md
license: MIT
---

# On-device RAG and embeddings with flutter_edge_ai

This document explains retrieval-augmented generation (RAG) and semantic search fully on-device with flutter_edge_ai (formerly flutter_gemma): the embedding models and tokenizers, the flutter_edge_ai_rag package with its RagIndex and embedding profiles, the sqlite-vec and qdrant-edge storage providers, metadata filters, persistence, web setup and the common traps.

## The pieces of on-device RAG in flutter_edge_ai

On-device RAG in flutter_edge_ai combines an embedding model with a vector store. The packages are flutter_edge_ai (core, which owns the active embedder), flutter_edge_ai_litertlm (it provides LiteRtEmbeddingBackend), flutter_edge_ai_embeddings (the tokenizers), flutter_edge_ai_rag (the RAG orchestration: FlutterEdgeAiRag and RagIndex), and one storage provider: flutter_edge_ai_sqlite or flutter_edge_ai_qdrant. A typical install is `flutter pub add flutter_edge_ai flutter_edge_ai_litertlm flutter_edge_ai_embeddings flutter_edge_ai_rag flutter_edge_ai_sqlite path path_provider`.

RAG is not set up by FlutterEdgeAi.initialize. The app creates its own FlutterEdgeAiRag instance with the storage providers it uses and opens a RagIndex — addText, searchText, flush — which embeds documents and queries with the correct task types. Add inferenceEngines to FlutterEdgeAi.initialize as well when the app also generates answers from the results.

Two storage packages implement the same RAG contract: flutter_edge_ai_qdrant (qdrant-edge, native — fastest on Android, iOS and desktop) and flutter_edge_ai_sqlite (in-SQLite sqlite-vec/vec0 KNN, portable across all six platforms including Web, since qdrant-edge can't target WASM). Switching storage changes the registered provider and VectorStoreSpec.providerId, not the retrieval code.

## Rules for using RAG with flutter_edge_ai

1. Use the independent flutter_edge_ai_rag package and an app-owned FlutterEdgeAiRag instance; FlutterEdgeAi.initialize does not configure RAG.
2. One persistent location belongs to one stable EmbeddingProfile, whose ID versions the weights, tokenizer, pooling, normalization and document/query prefixes. A mutable URL or file path is not an identity.
3. Keep one live RagIndex per location, make opening single-flight, and share it across widgets.
4. Declare every field used in a filter in VectorStoreSpec.filterSchema before the SQLite index is created. A condition on an undeclared field is dropped, never rejected.
5. On native, give the index an absolute location in a writable directory — a database file for SQLite, a directory for qdrant. On Web a plain name is enough.
6. Install and activate the embedder with getActiveEmbedder() before opening the index; open() pins the embedder that is active at that moment.
7. Dispose indexes before FlutterEdgeAi.dispose(). An index owns its vector store but only borrows its embedder.
8. Android needs minSdk 30 — for the LiteRT embedding runtime, not the vector store.

## Registering the embedding backend and tokenizer

FlutterEdgeAi.initialize takes two embedding lists: embeddingBackends, for example LiteRtEmbeddingBackend() from flutter_edge_ai_litertlm, and embeddingTokenizers, for example GemmaEmbeddingTokenizers() from flutter_edge_ai_embeddings. The same call takes an optional huggingFaceToken for the gated EmbeddingGemma download.

```dart
await FlutterEdgeAi.initialize(
  embeddingBackends: const [LiteRtEmbeddingBackend()],
  embeddingTokenizers: const [GemmaEmbeddingTokenizers()],
  huggingFaceToken: hfToken.isEmpty ? null : hfToken,
);
```

There are two lists because they answer different questions. The backend is the engine that turns token ids into a vector; the tokenizer is what turns text into those ids, and which one a model needs is a property of the model — EmbeddingGemma is SentencePiece whether LiteRT or ONNX Runtime runs it. Keeping them apart is why neither engine package depends on flutter_edge_ai_embeddings, and why an app that never embeds anything resolves neither. Forget the second list and the first embedding throws a StateError naming the package to add — it never silently falls back to a tokenizer with the wrong convention.

## What flutter_edge_ai_embeddings contains

flutter_edge_ai_embeddings holds the embedding tokenizers for flutter_edge_ai: Gemma SentencePiece and BERT-family WordPiece, plus the task-type prefixing and the routing that picks between them, on Android, iOS, macOS, Linux, Windows and Web. The seam an engine implements, the background-isolate worker and the pooling live in flutter_edge_ai itself. The package is pure Dart with no native or FFI code; the concrete backend and its native library are owned by whichever engine package you add.

There are three tokenizer profiles, picked by the model you load, because a model's special-token convention is not negotiable and using the wrong one corrupts the vector silently rather than failing. Gemma (SentencePiece) uses BOS 2, EOS 1 and a TaskType prefix. WordPiece (BERT, MiniLM) uses [CLS] … [SEP]. The SigLIP2 text tower uses no BOS, one trailing EOS, lowercasing and a fixed 64-token width; it is not selected automatically, and the loader refuses a SigLIP2 tokenizer.json rather than embedding it wrongly.

## Installing EmbeddingGemma as the embedder

An embedding model is installed with FlutterEdgeAi.installEmbedder(), giving the model file and its tokenizer, and activated with FlutterEdgeAi.getActiveEmbedder(). Pin the download to an immutable revision, so the bytes behind the embedding profile never change:

```dart
const revision = '29888fcee3216acadc7e844906e5fe0d79a61875';
const base = 'https://huggingface.co/litert-community/embeddinggemma-300m/resolve/$revision';
await FlutterEdgeAi.installEmbedder()
    .modelFromNetwork('$base/embeddinggemma-300M_seq256_mixed-precision.tflite')
    .tokenizerFromNetwork('$base/sentencepiece.model')
    .install();
final EmbeddingModel embedder = await FlutterEdgeAi.getActiveEmbedder();
```

EmbeddingGemma is a gated repo: the token's Hugging Face account must have accepted the Gemma licence, and the token ships inside the app — on web inside main.dart.js. seq256 in the file name is the input window in tokens; seq512, seq1024 and seq2048 variants sit in the same repo.

## Text embedding models and their sequence lengths

All of these embedding models generate 768-dimensional vectors. The numbers in names (64, 256, 512, 1024, 2048) indicate the maximum input sequence length in tokens, not the embedding dimension.

| Model | Parameters | Max sequence length | Size | Hugging Face token needed |
|---|---|---|---|---|
| Gecko 64 | 110M | 64 tokens | 110MB | no |
| Gecko 256 | 110M | 256 tokens | 114MB | no |
| Gecko 512 | 110M | 512 tokens | 116MB | no |
| EmbeddingGemma 256 | 300M | 256 tokens | 179MB | yes |
| EmbeddingGemma 512 | 300M | 512 tokens | 179MB | yes |
| EmbeddingGemma 1024 | 300M | 1024 tokens | 183MB | yes |
| EmbeddingGemma 2048 | 300M | 2048 tokens | 196MB | yes |

Gecko comes from litert-community/Gecko-110m-en and EmbeddingGemma from litert-community/embeddinggemma-300m.

## Choosing between Gecko and EmbeddingGemma

Gecko has about a third of EmbeddingGemma's parameters and needs no Hugging Face token, which suits short queries and fast search. EmbeddingGemma, with 300M parameters, is the higher-quality choice for semantic search. Pick the window by the length of your chunks: a 64-token Gecko suits short queries, while the 1024- and 2048-token EmbeddingGemma variants suit long documents. Every model and window defines its own embedding space, so each needs its own profile ID and index location.

## Opening a RAG index with an embedding profile

FlutterEdgeAiRag(providers: [...]) holds the storage providers; rag.open(spec:, ...) returns a RagIndex. VectorStoreSpec names the providerId ('sqlite' for SqliteVectorStoreProvider.providerId, 'qdrant' for qdrant), the location and the filterSchema. For text RAG with the active core embedder, pass activeEmbedderProfileId: a stable, versioned ID that describes the exact model bytes and preprocessing, for example 'embeddinggemma-300m-seq256-mp-rev-29888fcee321-retrieval-prefix-meanpool-l2-v1'.

```dart
final rag = FlutterEdgeAiRag(providers: const [SqliteVectorStoreProvider()]);
final index = await rag.open(
  spec: VectorStoreSpec(
    providerId: SqliteVectorStoreProvider.providerId,
    location: databasePath, // absolute path on native, a plain name on Web
    filterSchema: FilterSchema(fields: [
      FilterField(name: 'lang', type: FilterFieldType.string),
      FilterField(name: 'year', type: FilterFieldType.number),
    ]),
  ),
  activeEmbedderProfileId: embeddingProfileId,
);
```

On native, build the location from getApplicationDocumentsDirectory() in path_provider. A store that fails to open on a phone is usually a bare name such as 'rag.db', because it resolves against the process working directory, which is not writable on Android or iOS. rag.canOpen(spec) tells whether a registered provider can open a spec on this platform.

## Indexing documents with addText and chunk size limits

index.addText takes an id, the content text and an optional metadata JSON string, for example metadata: jsonEncode({'lang': 'en', 'year': 2024}). It embeds the text with the document task type. Adding an existing id replaces that document. index.remove(id:) deletes one, and index.clear() empties the store but keeps its profile binding: one location stays one embedding space for its lifetime.

For higher throughput you can batch-embed yourself with embedder.generateEmbeddings(texts, taskType: TaskType.retrievalDocument) and feed the pre-computed vectors through index.addVector(id:, content:, embedding:, metadata:).

Splitting documents, chunk size and overlap are the app's to decide. Keep each chunk within the embedding model's window — 256 tokens for seq256; the forward pass truncates a longer one without an error.

## Searching with searchText

index.searchText takes the question as text and embeds it with the query task type, plus topK (default 5), threshold (default 0.0) and an optional filter. Each RetrievalResult has id, content, similarity and metadata. index.searchVector does the same for a pre-computed query vector.

```dart
final hits = await index.searchText(
  query: question,
  topK: 5,
  threshold: 0.3,
  filter: Filter(must: [FieldEquals(key: 'lang', value: 'en')]),
);
```

searchText returns cosine similarity (1 = identical, higher = better), sorted descending and filtered by threshold. That is the same contract on the sqlite store and the qdrant store: vec0 returns a distance, and the sqlite store converts it to 1 minus distance at the boundary.

## Query and document task types

Query and document embeddings are trained asymmetrically. addText uses the document embedding path and searchText the query path, but when you embed by hand, generateEmbedding defaults to TaskType.retrievalQuery, so text embedded for indexing without a task type gets the query prefix and retrieval quality drops. The fix is to pass TaskType.retrievalDocument when indexing by hand and then store the vector with addVector.

TaskType has two values, and both prefixes are non-empty, so no caller can embed without one. That is the intended contract for Gemma and Gecko. The SigLIP2 profile drops the prefix, because a CLIP-family vision side encodes an image with no prefix at all.

## Metadata filters and the filterSchema

Filter supports must, should and mustNot lists of FieldEquals, FieldRange (gte, lte) and FieldMatchAny conditions. Both storage providers honor it: qdrant-edge natively, and the sqlite-vec store on all platforms including Web.

Declare every filterable field in VectorStoreSpec.filterSchema, for example FilterSchema(fields: [FilterField(name: 'lang', type: FilterFieldType.string), FilterField(name: 'year', type: FilterFieldType.number)]); field types are string, number and bool. A condition on an undeclared field is dropped, never rejected: with no schema at all the search comes back completely unfiltered, identical to filter: null. So the symptom of a missing declaration is results that ignore the filter, with no error.

The sqlite store filters KNN only on declared, typed metadata columns, not arbitrary JSON. It promotes the declared fields out of each document's metadata JSON into real columns and translates the Filter into a vec0 WHERE clause. Supported operators are =, !=, >, >=, <, <=, BETWEEN and IN, with at most 16 declared columns. Changing that physical schema later needs a new, schema-versioned location and a re-index.

## Filter field name rules in the sqlite store

With flutter_edge_ai_sqlite, a FilterField name must match ^[A-Za-z][A-Za-z0-9_]*$ and must not be a name vec0 already declares: id, embedding, content, metadata, and the hidden distance and k. rag.open() throws an ArgumentError otherwise, at open rather than at the first write. The name becomes a real vec0 column, and sqlite-vec's DDL grammar accepts no quoted identifier form, so a name outside that set is unrepresentable rather than merely unescaped.

qdrant accepts most of these names, so a schema written for qdrant may be refused by the sqlite store; the sqlite set is the portable one. A duplicate or empty name is rejected on any store.

## Embedding profiles, mismatches and existing stores

One RagIndex pins one EmbeddingProfile, and the provider stores that binding beside the vectors and refuses to overwrite it, so the rule survives app restarts. A profile mismatch means the model or preprocessing changed, or the wrong location was opened: choose the correct profile or location rather than bypassing it. Never mix vectors from different models in one location, even when their dimensions match.

For vector-only use, open a new location with an explicit embeddingProfile: EmbeddingProfile(id: ..., dimension: 768); the first addVector on an empty store needs it, because raw numbers cannot identify their embedding space. A custom text pipeline implements RagEmbedder (profile, embedDocument, embedQuery) and passes it as embedder: to open().

A nonempty store written before flutter_edge_ai_rag has no profile metadata and is never adopted silently. Prefer a new profile-versioned location and a re-index. Only when the exact old embedding pipeline is known may the app open it with an explicit embeddingProfile and VectorStoreSpec(allowLegacyProfileAdoption: true); the RAG layer checks the vector dimension before binding the profile.

## Embeddings run on the CPU

LiteRT embeddings always run on CPU on native, and so do ONNX ones. getActiveEmbedder(preferredBackend:) is accepted for symmetry with getActiveModel and never applied; core logs one line per isolate saying so, in debug builds only. Read EmbeddingModel.activeBackend when it matters — that answer exists in release builds too, and it is cpu on native and null on web.

CPU is the correct answer rather than a fallback: LiteRT's GPU delegate returns all-zero vectors for EmbeddingGemma's int4 weights, and the ONNX client appends no execution provider.

On web the LiteRT embedder asks for the WebGPU accelerator and recompiles for WASM when the browser has none, and a model that is not fully accelerated can be partly delegated to WASM. window.getLiteRtEmbeddingFullyAccelerated() says whether the graph landed entirely on the requested accelerator, and window.getLiteRtEmbeddingAccelerator() names where the output lived after the first embedding.

## The sqlite-vec vector store

flutter_edge_ai_sqlite is a first-class SQLite storage provider for flutter_edge_ai_rag. KNN runs inside SQLite via sqlite-vec (the vec0 virtual table) — no Dart brute-force, no in-memory index. Register SqliteVectorStoreProvider once and it selects the implementation: on native (Android, iOS, macOS, Linux, Windows) SqliteVectorStore uses package:sqlite3 over dart:ffi plus the per-platform vec0 loadable extension; on web, WebSqliteVectorStore uses package:sqlite3/wasm.dart driving a custom sqlite3.wasm with sqlite-vec statically linked. Both arms speak the same vec0 SQL dialect, so KNN and Filter behave identically across all six platforms. A vec0 table declares an id TEXT PRIMARY KEY, so KNN returns the document id directly — no JOIN, no rowid bridge. The package needs Flutter 3.47 or newer.

## Native setup and the sqlite-vec extension download

Native needs no setup: the vec0 loadable extension is fetched per platform by the package's Native Assets hook, SHA256-verified, and loaded automatically before any database is opened. The loadables come from the repository's native-sqlite-vec GitHub Release, so the first build of each platform needs github.com reachable. The library is cached under ~/.cache/flutter_gemma/native/ (~/Library/Caches/… on macOS, %LOCALAPPDATA%\… on Windows) — the cache keeps its original name — and later builds do not go out again. In an air-gapped environment, pre-populate that cache directory.

## Persisting the index with flush

Call index.flush() after a write batch. The qdrant store keeps new documents in memory until the store is flushed or closed, so an index built without either is lost when the process ends; an Android app killed in the background is the ordinary case.

On native SQLite, flush() is a no-op: the connection autocommits, so a statement that returned is on disk. On web it drains the IndexedDB storage and waits for it. sqlite3 3.4.0 through 3.5.2 returned early over a write batch already in flight, which is why the package requires sqlite3 3.6.0 and, with it, Flutter 3.47. When neither OPFS nor IndexedDB is available the store runs in memory, and flush() throws VectorStoreException; such an in-memory store cannot hold a durable embedding profile, so rag.open() rejects it.

## One index per location and the shutdown order

RagIndex owns and closes its vector store; the embedder is always borrowed. Normal reads and writes may overlap, while flush, clear and dispose are exclusive barriers, and dispose rejects new work at once and waits for work already accepted.

Keep one app-owned index per location. On Web the SQLite store enforces this with an exclusive Web Lock held for its whole lifetime, so another tab, worker or store instance fails at once with a VectorStoreException; make open() single-flight and share the returned future rather than opening from several widgets. At shutdown, dispose every RagIndex first, then call FlutterEdgeAi.dispose() or dispose a custom embedder that the indexes borrowed.

## The qdrant-edge vector store

flutter_edge_ai_qdrant is a storage provider for flutter_edge_ai_rag backed by qdrant-edge, which runs Qdrant's vector search inside the app. qdrant's HNSW index makes it the fastest native RAG store — roughly 5 to 11 times faster search than the in-SQLite sqlite-vec store at 1k to 10k documents. It is native only: Android, iOS, macOS, Linux and Windows. For web, or when exact KNN with identical results across platforms matters more than peak speed, use flutter_edge_ai_sqlite.

Register QdrantVectorStoreProvider() in FlutterEdgeAiRag and pass the providerId string 'qdrant' — the provider has no static constant. The location is a directory, not a file, because qdrant creates its shard files inside it; keep it apart from any SQLite database. A store written by flutter_gemma_rag_qdrant 1.2 or earlier makes open() throw QdrantLegacyStoreException: its format cannot be read, so delete the files its message names and re-index.

## RAG on the web

On web, copy web/rag/sqlite3.wasm from the flutter_edge_ai_sqlite package into the app as web/rag/sqlite3.wasm, next to index.html — that's the URL WasmSqlite3.loadFromUrl fetches. OPFS persistence and SharedArrayBuffer require the web server to send the cross-origin isolation headers Cross-Origin-Opener-Policy: same-origin and Cross-Origin-Embedder-Policy: require-corp.

Web embeddings need four module files side by side in the app's web folder, all four from flutter_edge_ai_litertlm/web/: litert_embeddings.js, sentencepiece.js, litert.js and tensorflow.js. The first imports the other three by relative path, so three files alone give a 404 and an embedder that never initialises. Load litert_embeddings.js from web/index.html as a module, next to the model storage helpers cache_api.js and opfs_helper.js that every flutter_edge_ai web app loads.

## Moving from 1.x core RAG to 2.0

In flutter_edge_ai 2.0 RAG left core: the FlutterEdgeAi.rag calls and the vectorStore: and filterSchema: parameters of FlutterEdgeAi.initialize are gone. Add flutter_edge_ai_rag and a storage package, remove those two parameters, and open a RagIndex:

| 1.x (`FlutterEdgeAi.rag`) | 2.0 `RagIndex` |
|---|---|
| `initialize(vectorStore: ...)` on core | `FlutterEdgeAiRag(providers: [...])` |
| `rag.initialize(location)` | `rag.open(spec: VectorStoreSpec(...))` |
| `addDocument(...)` | `addText(...)` |
| `addDocumentWithEmbedding(...)` | `addVector(...)` |
| `searchSimilar(query: ...)` | `searchText(query: ...)` |
| `removeDocument(id: ...)` | `remove(id: ...)` |
| `stats()`, `flush()`, `clear()` | the same methods on the index |

`No vector-store provider can handle providerId "…"` means the provider for that ID was not passed to FlutterEdgeAiRag(providers: ...), or cannot run on this platform (qdrant on Web). A nonempty store written by 1.x is usable only through the legacy-adoption path or after re-indexing into a new location.

## Troubleshooting a missing native embedding library

flutter_edge_ai_litertlm is the sole owner of the shared native library and bundles it via its build hook. A stale Native-Assets cache after a native version bump can leave the library unbundled, surfacing as an opaque dlopen "no such file" error for libLiteRtLm on the first embedding call. Fix it with a clean rebuild: run flutter clean, delete the cached native folder (~/Library/Caches/flutter_gemma/native on macOS, ~/.cache/flutter_gemma/native on Linux, %LOCALAPPDATA%\flutter_gemma\native on Windows), then run flutter pub get.
