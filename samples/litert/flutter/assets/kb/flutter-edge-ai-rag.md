---
title: On-device RAG and embeddings with flutter_edge_ai
source: https://pub.dev/packages/flutter_edge_ai_rag/versions/1.1.1 (README.md, CHANGELOG.md, lib/src/) ; https://pub.dev/packages/flutter_edge_ai/versions/2.4.0 (skills/flutter-edge-ai-rag/SKILL.md, lib/src/embedding/, lib/src/api/, example/lib/models/embedding_model.dart) ; https://pub.dev/packages/flutter_edge_ai_sqlite/versions/2.1.0 (README.md) ; https://pub.dev/packages/flutter_edge_ai_qdrant/versions/2.1.0 (README.md) ; https://pub.dev/packages/flutter_edge_ai_embeddings/versions/2.3.1 (README.md) ; https://pub.dev/packages/flutter_edge_ai_litertlm/versions/1.14.0 (README.md: Embeddings, EmbeddingGemma 2 on a Qualcomm NPU, Embeddings on web) ; https://github.com/DenisovAV/flutter_edge_ai/blob/d35c575ab91e078c47201131800cf33dd7630833/website/content/docs/embeddings-and-rag.md ; https://github.com/DenisovAV/flutter_edge_ai/blob/d35c575ab91e078c47201131800cf33dd7630833/website/content/docs/models.md
license: MIT
---

# On-device RAG and embeddings with flutter_edge_ai

This document explains retrieval-augmented generation (RAG) and semantic search fully on-device with flutter_edge_ai (formerly flutter_gemma): the embedding models (EmbeddingGemma 2, EmbeddingGemma 1 and Gecko) and their tokenizers, the flutter_edge_ai_rag package with its RagIndex and embedding profiles, the sqlite-vec and qdrant-edge storage providers, metadata filters, where embeddings run, persistence, web setup and the common traps.

## The pieces of on-device RAG in flutter_edge_ai

On-device RAG in flutter_edge_ai combines an embedding model with a vector store. The packages are flutter_edge_ai (core, which owns the active embedder), flutter_edge_ai_litertlm (it provides LiteRtEmbeddingBackend), flutter_edge_ai_embeddings (the tokenizers, needed for a model that comes with a separate tokenizer file, such as the `.tflite` EmbeddingGemma 1 and Gecko), flutter_edge_ai_rag (the RAG orchestration: FlutterEdgeAiRag and RagIndex), and one storage provider: flutter_edge_ai_sqlite or flutter_edge_ai_qdrant. A typical install with EmbeddingGemma 2 is `flutter pub add flutter_edge_ai flutter_edge_ai_litertlm flutter_edge_ai_rag flutter_edge_ai_sqlite path path_provider`; a `.tflite` model also needs flutter_edge_ai_embeddings.

RAG is not set up by FlutterEdgeAi.initialize. The app creates its own FlutterEdgeAiRag instance with the storage providers it uses and opens a RagIndex — add, search, flush — which embeds documents and queries, passed as EmbeddingInput.text(...), with the correct task types. flutter_edge_ai_rag re-exports EmbeddingInput, EmbeddingModality, TaskType and the EmbeddingException family, so RAG code needs no core import for them. Add inferenceEngines to FlutterEdgeAi.initialize as well when the app also generates answers from the results.

Two storage packages implement the same RAG contract: flutter_edge_ai_qdrant (qdrant-edge, native — fastest on Android, iOS and desktop) and flutter_edge_ai_sqlite (in-SQLite sqlite-vec/vec0 KNN, portable across all six platforms including Web, since qdrant-edge can't target WASM). Switching storage changes the registered provider and VectorStoreSpec.providerId, not the retrieval code.

## Rules for using RAG with flutter_edge_ai

1. Use the independent flutter_edge_ai_rag package and an app-owned FlutterEdgeAiRag instance; FlutterEdgeAi.initialize does not configure RAG.
2. One persistent location belongs to one stable EmbeddingProfile and one document modality (VectorStoreSpec.documents: text, image or audio). The profile ID versions the weights, tokenizer, pooling, normalization and document/query prefixes. A mutable URL or file path is not an identity.
3. Keep one live RagIndex per location, make opening single-flight, and share it across widgets.
4. Declare every field used in a filter in VectorStoreSpec.filterSchema before the SQLite index is created. Through RagIndex, a filter on an undeclared field throws ArgumentError.
5. On native, give the index an absolute location in a writable directory — a database file for SQLite, a directory for qdrant. On Web a plain name is enough.
6. Install the embedder before opening the index: open() resolves the embedder's profile, which loads the model, and binds the location at once. FlutterEdgeAiActiveEmbedder pins the embedding model that is active when the adapter is created.
7. Dispose indexes before FlutterEdgeAi.dispose(). An index owns its vector store but only borrows its embedder.
8. Android needs minSdk 30 — for the LiteRT embedding runtime, not the vector store.

## Registering the embedding backend and tokenizer

FlutterEdgeAi.initialize takes two embedding lists: embeddingBackends, for example LiteRtEmbeddingBackend() from flutter_edge_ai_litertlm, and embeddingTokenizers, for example GemmaEmbeddingTokenizers() from flutter_edge_ai_embeddings. A `.tflite` model (EmbeddingGemma 1, Gecko) needs both lists. An EmbeddingGemma 2 `.litertlm` bundle carries its own tokenizer and needs only the backend list. The same call takes an optional huggingFaceToken for the gated EmbeddingGemma 1 download.

```dart
await FlutterEdgeAi.initialize(
  embeddingBackends: const [LiteRtEmbeddingBackend()],
  embeddingTokenizers: const [GemmaEmbeddingTokenizers()], // .tflite models only
  huggingFaceToken: hfToken.isEmpty ? null : hfToken,
);
```

There are two lists because they answer different questions. The backend is the engine that turns token ids into a vector; the tokenizer is what turns text into those ids, and which one a model needs is a property of the model — EmbeddingGemma 1 is SentencePiece whether LiteRT or ONNX Runtime runs it. Keeping them apart is why neither engine package depends on flutter_edge_ai_embeddings, and why an app that never embeds anything resolves neither. Forget the second list with a `.tflite` model and getActiveEmbedder() fails with an EmbeddingLoadException, its cause a StateError, naming the package to add — it never silently falls back to a tokenizer with the wrong convention.

## What flutter_edge_ai_embeddings contains

flutter_edge_ai_embeddings holds the embedding tokenizers for flutter_edge_ai: Gemma SentencePiece and BERT-family WordPiece, plus the task-type prefixing and the routing that picks between them, on Android, iOS, macOS, Linux, Windows and Web. The seam an engine implements, the background-isolate worker and the pooling live in flutter_edge_ai itself. The package is pure Dart with no native or FFI code; the concrete backend and its native library are owned by whichever engine package you add. An app that embeds only with an EmbeddingGemma 2 bundle needs neither this package nor embeddingTokenizers.

There are three tokenizer profiles, picked by the model you load, because a model's special-token convention is not negotiable and using the wrong one corrupts the vector silently rather than failing. Gemma (SentencePiece) uses BOS 2, EOS 1 and a TaskType prefix. WordPiece (BERT, MiniLM) uses [CLS] … [SEP]. The SigLIP2 text tower uses no BOS, one trailing EOS, lowercasing and a fixed 64-token width; it is not selected automatically, and the loader refuses a SigLIP2 tokenizer.json rather than embedding it wrongly.

## Installing EmbeddingGemma as the embedder

An embedding model is installed with FlutterEdgeAi.installEmbedder() and activated with FlutterEdgeAi.getActiveEmbedder(). EmbeddingGemma 2 is one self-contained `.litertlm` file: install the model alone, with no tokenizerFrom…() call, since a `.litertlm` given a tokenizer is refused. Its Hugging Face repositories are not gated. Pin the download to an immutable revision and give it a revision-qualified file name, so the bytes behind the embedding profile never change:

```dart
const revision = '9be6e8b90982095dc05c2bd162e4b954ee4dbac7';
await FlutterEdgeAi.installEmbedder()
    .modelFromNetwork(
      'https://huggingface.co/litert-community/'
      'embeddinggemma-2-text-270m-litert-lm/resolve/$revision/'
      'embeddinggemma-2-text-270m.litertlm',
      filename: 'embeddinggemma-2-text-270m__rev-9be6e8b90982.litertlm',
    )
    .install();
final EmbeddingModel embedder = await FlutterEdgeAi.getActiveEmbedder();
```

A `.tflite` model, EmbeddingGemma 1 or Gecko, is two files, the model and its tokenizer, and install() refuses the model without a tokenizer:

```dart
const revision = '29888fcee3216acadc7e844906e5fe0d79a61875';
const base = 'https://huggingface.co/litert-community/embeddinggemma-300m/resolve/$revision';
await FlutterEdgeAi.installEmbedder()
    .modelFromNetwork('$base/embeddinggemma-300M_seq256_mixed-precision.tflite',
        filename: 'embeddinggemma-300m-seq256-mp__rev-29888fcee321.tflite')
    .tokenizerFromNetwork('$base/sentencepiece.model',
        filename: 'embeddinggemma-300m__sentencepiece__rev-29888fcee321.model')
    .install();
```

EmbeddingGemma 1 is a gated repo: the token's Hugging Face account must have accepted the Gemma licence, and the token ships inside the app — on web inside main.dart.js. seq256 in the file name is the input window in tokens; seq512, seq1024 and seq2048 variants sit in the same repo. Instead of the network, modelFromAsset, modelFromBundled and modelFromFile take the model from a Flutter asset, a bundled native resource or a file on disk.

## Text embedding models and their sequence lengths

All of these embedding models generate 768-dimensional vectors. In the Gecko and EmbeddingGemma 1 names the numbers (64, 256, 512, 1024, 2048) indicate the maximum input sequence length in tokens, not the embedding dimension; the EmbeddingGemma 2 numbers (270m, 440m, 740m) are model sizes in parameters.

| Model | Parameters | Max sequence length | Size | Hugging Face token needed |
|---|---|---|---|---|
| EmbeddingGemma 2 text | 270M | 1024 tokens | 165MB | no |
| EmbeddingGemma 2 text + vision | 440M | 1024 tokens | 388MB | no |
| EmbeddingGemma 2 text + vision + audio | 740M | 1024 tokens | 485MB | no |
| Gecko 64 | 110M | 64 tokens | 110MB | no |
| Gecko 256 | 110M | 256 tokens | 114MB | no |
| Gecko 512 | 110M | 512 tokens | 116MB | no |
| EmbeddingGemma 1, 256 | 300M | 256 tokens | 179MB | yes |
| EmbeddingGemma 1, 512 | 300M | 512 tokens | 179MB | yes |
| EmbeddingGemma 1, 1024 | 300M | 1024 tokens | 183MB | yes |
| EmbeddingGemma 1, 2048 | 300M | 2048 tokens | 196MB | yes |

Gecko comes from litert-community/Gecko-110m-en, EmbeddingGemma 1 from litert-community/embeddinggemma-300m, and EmbeddingGemma 2 from litert-community/embeddinggemma-2-text-270m-litert-lm, embeddinggemma-2-text-vision-440m-litert-lm and embeddinggemma-2-740m-litert-lm. The three EmbeddingGemma 2 bundles share one text tower and give identical text vectors; the 440m adds images and the 740m images and audio. On native, EmbeddingGemma 2 also writes weight caches into the app's cache directory at load, also for text-only use: 71 MB for the 270m, 218 MB more for the 440m, and 385 MB in all for the 740m.

## Choosing between Gecko and EmbeddingGemma

Gecko has about a third of EmbeddingGemma 1's parameters and needs no Hugging Face token, which suits short queries and fast search: on a Pixel 8 the package's models page lists Gecko 64 at about 109 ms per document against about 286 ms for EmbeddingGemma 1 at seq256, which it calls the more accurate of the two. EmbeddingGemma 2 is one ungated file with its tokenizer inside and a 1024-token window; for text alone its 270m bundle is the one to use, because a 440m loaded for text still loads its vision tower, and the 440m and 740m write their media weight caches also for text-only use. Pick the window by the length of your chunks: a 64-token Gecko suits short queries, while EmbeddingGemma 2 and the 1024- and 2048-token EmbeddingGemma 1 variants suit long documents. Gecko, EmbeddingGemma 1 and EmbeddingGemma 2 are different embedding spaces, although all three produce 768-dimensional vectors, so each needs its own profile ID and index location: the dimension check alone cannot tell them apart, and an EmbeddingGemma 2 NPU bundle's index gets its own profile ID as well. The three EmbeddingGemma 2 bundles share one text space, so one text index can take vectors from any of them under one explicit profile ID; FlutterEdgeAiActiveEmbedder.forActiveModel keys on the file name and still gives each bundle its own ID.

## Opening a RAG index with an embedding profile

FlutterEdgeAiRag(providers: [...]) holds the storage providers; rag.open(spec:, embedder:) returns a RagIndex. VectorStoreSpec names the providerId ('sqlite' for SqliteVectorStoreProvider.providerId, 'qdrant' for qdrant), the location, the filterSchema and the documents modality, text by default. For RAG with the active core embedder, pass embedder: FlutterEdgeAiActiveEmbedder.forActiveModel(namespace: 'knowledge'), which derives the profile ID from the installed file names ('<namespace>@<files joined by +>'), or FlutterEdgeAiActiveEmbedder(profileId: ...) with a stable, versioned ID that describes the exact model bytes and preprocessing, for example 'embeddinggemma-300m-seq256-mp-rev-29888fcee321-retrieval-prefix-meanpool-l2-v1'. A derived ID follows file names, not bytes, which is why the install gives revision-qualified file names.

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
  embedder: FlutterEdgeAiActiveEmbedder.forActiveModel(namespace: 'knowledge'),
);
```

With FlutterEdgeAiActiveEmbedder, rag.open() loads the model to resolve the profile, then binds the location to that profile and the document modality: an unbound location is bound at once, and a location bound to anything else is refused with a StateError. FlutterEdgeAiActiveEmbedder resolves the model again on every operation, so it follows core's embedder cache; after the app switches embedders it throws StateError, and a new embedder and a new location are needed. A request that failed because core replaced the model before it ran (EmbedderClosedException) is retried once.

On native, build the location from getApplicationDocumentsDirectory() in path_provider. A store that fails to open on a phone is usually a bare name such as 'rag.db', because it resolves against the process working directory, which is not writable on Android or iOS. rag.canOpen(spec) tells whether a registered provider can open a spec on this platform.

## Indexing documents with add and chunk size limits

index.add takes an id, the document as an EmbeddingInput, for example document: EmbeddingInput.text(chunk), an optional content and an optional metadata JSON string, for example metadata: jsonEncode({'lang': 'en', 'year': 2024}). It embeds the document with the document task type and stores the chunk's text as the hit's content, unless content: gives something else. Adding an existing id replaces that document. index.remove(id:) deletes one, and index.clear() empties the store but keeps its binding: one location stays one embedding space and one document modality for its lifetime.

For precomputed batches, embed yourself with embedder.embedAll([for (final t in texts) EmbeddingInput.text(t)], taskType: TaskType.retrievalDocument) and feed the vectors through index.addVector(id:, content:, embedding:, metadata:). embedAll keeps the input order, fails on the first error and takes a cancelToken. The older generateEmbedding and generateEmbeddings still compile as deprecated forwarders and are removed in flutter_edge_ai 2.5.0.

Splitting documents, chunk size and overlap are the app's to decide. Keep each chunk within the embedder's maxInputTokens, task prefix, BOS and EOS included: 256 for an EmbeddingGemma 1 seq256 file, and 1024 for EmbeddingGemma 2, where chunks of 512 tokens or fewer are recommended because memory grows with the input length. A longer chunk or query is never truncated: add and search fail with EmbeddingInputTooLongException, whose maxInputTokens and actualTokens say by how much; older releases truncated it silently. With EmbeddingGemma 2 an over-long input still costs a full embedding before it is refused, so split before embedding.

## Searching with search

index.search takes the question as an EmbeddingInput, query: EmbeddingInput.text(question), and embeds it with the query task type, plus topK (default 5), threshold (default 0.0) and an optional filter. Each RetrievalResult has id, content, similarity and metadata. index.searchVector does the same for a pre-computed query vector.

```dart
final hits = await index.search(
  query: EmbeddingInput.text(question),
  topK: 5,
  threshold: 0.3,
  filter: Filter(must: [FieldEquals(key: 'lang', value: 'en')]),
);
```

index.search returns cosine similarity (1 = identical, higher = better), sorted descending and filtered by threshold. That is the same contract on the sqlite store and the qdrant store: vec0 returns a distance, and the sqlite store converts it to 1 minus distance at the boundary. Every error of a RagIndex method that returns a Future, argument errors included, arrives as a failed future.

## Query and document task types

Query and document embeddings are trained asymmetrically. index.add uses the document path (TaskType.retrievalDocument) and index.search the query path: TaskType.retrievalQuery for a text query over text documents, or another task passed as queryTask: to FlutterEdgeAiActiveEmbedder, for example TaskType.questionAnswering. When you embed by hand, embed and embedAll require a taskType: pass TaskType.retrievalDocument when indexing, then store the vector with addVector. The deprecated generateEmbedding and generateEmbeddings still default to TaskType.retrievalQuery, so text indexed through them without a task type gets the query prefix and retrieval quality drops.

TaskType has nine values: retrievalQuery and retrievalDocument for search, none for no prefix, and questionAnswering, factChecking, codeRetrieval, classification, clustering and sentenceSimilarity. Every value but TaskType.none prepends its prompt to the text part of an input, the same on native and web; images and audio never get one, and FlutterEdgeAiActiveEmbedder sends a text query over an image or audio index raw, with TaskType.none. The SigLIP2 profile drops the prefix, because a CLIP-family vision side encodes an image with no prefix at all.

## Metadata filters and the filterSchema

Filter supports must, should and mustNot lists of FieldEquals, FieldRange (gte, lte) and FieldMatchAny conditions. Both storage providers honor it: qdrant-edge natively, and the sqlite-vec store on all platforms including Web.

Declare every filterable field in VectorStoreSpec.filterSchema, for example FilterSchema(fields: [FilterField(name: 'lang', type: FilterFieldType.string), FilterField(name: 'year', type: FilterFieldType.number)]); field types are string, number and bool. Through RagIndex, a filter that names an undeclared field throws ArgumentError naming that field and the declared ones, so a missing declaration fails loudly instead of returning unfiltered results; the fix is to add the field to filterSchema. The sqlite store called directly, outside RagIndex, still treats a condition on an undeclared field as a no-op, and with no filterSchema at all it ignores filters entirely: there, results that ignore the filter, with no error, are the symptom of a missing declaration.

The sqlite store filters KNN only on declared, typed metadata columns, not arbitrary JSON. It promotes the declared fields out of each document's metadata JSON into real columns and translates the Filter into a vec0 WHERE clause. Supported operators are =, !=, >, >=, <, <=, BETWEEN and IN, with at most 16 declared columns. Changing that physical schema later needs a new, schema-versioned location and a re-index.

## Filter field name rules in the sqlite store

With flutter_edge_ai_sqlite, a FilterField name must match ^[A-Za-z][A-Za-z0-9_]*$ and must not be a name vec0 already declares: id, embedding, content, metadata, and the hidden distance and k. rag.open() and rag.openVectors() throw an ArgumentError otherwise, at open rather than at the first write. The name becomes a real vec0 column, and sqlite-vec's DDL grammar accepts no quoted identifier form, so a name outside that set is unrepresentable rather than merely unescaped.

qdrant accepts most of these names, so a schema written for qdrant may be refused by the sqlite store; the sqlite set is the portable one. A duplicate or empty name is rejected on any store.

## Embedding profiles, mismatches and existing stores

One RagIndex pins one EmbeddingProfile and one document modality, and the provider stores that binding beside the vectors and refuses to overwrite it, so the rule survives app restarts. A mismatch, a StateError saying the vector store is bound to another profile, means the model or preprocessing changed, a different model is installed, or the wrong location was opened: choose the correct profile or location rather than bypassing it. Never mix vectors from different models in one location, even when their dimensions match.

For vector-only use, open with rag.openVectors(spec:, profile: EmbeddingProfile(id: ..., dimension: 768)): raw numbers cannot identify their embedding space, so the profile is required, and such an index has no embedder, so add and search throw StateError while addVector and searchVector work. A custom pipeline implements RagEmbedder — resolveProfile(), modalities, embedDocument(EmbeddingInput) and embedQuery(EmbeddingInput, documentModality:) — and passes it as embedder: to rag.open(). EmbeddingProfile has no const constructor: it validates its id and dimension.

A nonempty store written before flutter_edge_ai_rag has no profile metadata and is never adopted silently. Prefer a new profile-versioned location and a re-index. Only when the exact old embedding pipeline is known may the app open it with rag.openVectors(spec: VectorStoreSpec(..., allowLegacyProfileAdoption: true), profile: ...), for text locations only; the RAG layer checks the vector dimension before binding the profile. Stores written by flutter_edge_ai_rag 1.0 open unchanged as text locations.

## Where embeddings run: CPU, NPU and WebGPU

On native, LiteRT embeddings run on the CPU, and so do ONNX ones, with one exception: an EmbeddingGemma 2 per-SoC bundle, compiled for one chip's NPU, runs on that NPU when asked with getActiveEmbedder(preferredBackend: PreferredBackend.npu). flutter_edge_ai_litertlm runs the Qualcomm bundles on Qualcomm Android and Qualcomm Linux arm64, in an app whose pubspec sets qualcomm_npu: true under hooks: user_defines: flutter_edge_ai_litertlm:, and the Intel Panther Lake bundles on Windows x64. Any other preferredBackend is not applied, and core logs one line per isolate saying so, in debug builds only. Read EmbeddingModel.activeBackend when it matters — that answer exists in release builds too: cpu on native, npu for a per-SoC bundle, gpu for EmbeddingGemma 2 on web, and null for the other web embedders.

For EmbeddingGemma 2 nothing switches the backend by itself: a per-SoC bundle without npu, the generic bundle with npu, npu on iOS, macOS, Linux x64 or web, and a bundle the device's NPU cannot run each fail with EmbeddingLoadException. NPU vectors are close to the CPU bundle's but not equal, so an NPU index needs its own profile ID; FlutterEdgeAiActiveEmbedder.forActiveModel separates them by file name, and FlutterEdgeAiActiveEmbedder takes preferredBackend: as well, because it loads the embedder for every operation.

For `.tflite` and ONNX embedders the CPU is the correct answer rather than a fallback: LiteRT's GPU delegate returns all-zero vectors for EmbeddingGemma's int4 weights, and the ONNX client appends no execution provider.

On web, EmbeddingGemma 2 runs on WebGPU through @litert-lm/core's EmbeddingEngine and nowhere else: there is no CPU fallback, and a browser without WebGPU (no navigator.gpu, or no GPU adapter) fails getActiveEmbedder() with an EmbeddingLoadException that says so. For EmbeddingGemma 1 and Gecko on web, LiteRT.js asks for WebGPU and can fall back to WASM: window.getLiteRtEmbeddingAccelerator() names where the output lived after the first embedding, and window.getLiteRtEmbeddingFullyAccelerated() says whether the graph landed entirely on the requested accelerator — false also when LiteRT silently recompiled a WebGPU request for WASM.

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

Web embeddings with a `.tflite` model (EmbeddingGemma 1, Gecko) need four module files side by side in the app's web folder, all four from flutter_edge_ai_litertlm/web/: litert_embeddings.js, sentencepiece.js, litert.js and tensorflow.js. The first imports the other three by relative path, so three files alone give a 404 and an embedder that never initialises. Load litert_embeddings.js from web/index.html as a module, next to the model storage helpers cache_api.js and opfs_helper.js that every flutter_edge_ai web app loads, and copy the four again after upgrading the package: a stale litert_embeddings.js fails getActiveEmbedder() with an EmbeddingLoadException that names it. EmbeddingGemma 2 needs none of these four files, only window.EmbeddingEngine = m.EmbeddingEngine; next to window.Engine = m.Engine; in the @litert-lm/core module script of web/index.html, and a browser with WebGPU.

## Moving from 1.x core RAG to 2.0

In flutter_edge_ai 2.0 RAG left core: the FlutterEdgeAi.rag calls and the vectorStore: and filterSchema: parameters of FlutterEdgeAi.initialize are gone. Add flutter_edge_ai_rag and a storage package, remove those two parameters, and open a RagIndex:

| 1.x (`FlutterEdgeAi.rag`) | `RagIndex` (flutter_edge_ai_rag 1.1) |
|---|---|
| `initialize(vectorStore: ...)` on core | `FlutterEdgeAiRag(providers: [...])` |
| `rag.initialize(location)` | `rag.open(spec: VectorStoreSpec(...), embedder: ...)` |
| `addDocument(...)` | `add(id:, document: EmbeddingInput.text(...))` |
| `addDocumentWithEmbedding(...)` | `addVector(...)` |
| `searchSimilar(query: ...)` | `search(query: EmbeddingInput.text(...))` |
| `removeDocument(id: ...)` | `remove(id: ...)` |
| `stats()`, `flush()`, `clear()` | the same methods on the index |

`No vector-store provider can handle providerId "…"` means the provider for that ID was not passed to FlutterEdgeAiRag(providers: ...), or cannot run on this platform (qdrant on Web). A nonempty store written by 1.x is usable only through the legacy-adoption path or after re-indexing into a new location.

## Moving from flutter_edge_ai_rag 1.0 to 1.1

flutter_edge_ai_rag 1.1 changed the index API. index.addText(id:, content: c) became index.add(id:, document: EmbeddingInput.text(c)), and index.searchText(query: q) became index.search(query: EmbeddingInput.text(q)). open(spec:, activeEmbedderProfileId: id) became open(spec:, embedder: FlutterEdgeAiActiveEmbedder(profileId: id)), or .forActiveModel(namespace:), and open(spec:, embeddingProfile: p) for vectors only became openVectors(spec:, profile: p). open() now resolves the profile and binds the location at once, so the embedder is installed first. A RagEmbedder implements resolveProfile(), modalities, embedDocument(EmbeddingInput) and embedQuery(EmbeddingInput, documentModality:) instead of a profile getter and String methods, and a filter on an undeclared field, once ignored, throws ArgumentError.

flutter_edge_ai_rag 1.1 needs flutter_edge_ai 2.3.0 and the 2.1.0 stores, so every flutter_edge_ai package is upgraded together. Stores written by flutter_edge_ai_rag 1.0 open unchanged as text locations.

## Troubleshooting a missing native embedding library

flutter_edge_ai_litertlm is the sole owner of the shared native library and bundles it via its build hook. A stale Native-Assets cache after a native version bump can leave the library unbundled, surfacing as an opaque dlopen "no such file" error for libLiteRtLm on the first embedding call. Fix it with a clean rebuild: run flutter clean, delete the cached native folder (~/Library/Caches/flutter_gemma/native on macOS, ~/.cache/flutter_gemma/native on Linux, %LOCALAPPDATA%\flutter_gemma\native on Windows), then run flutter pub get.
