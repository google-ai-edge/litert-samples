---
title: EmbeddingGemma 2 embedding model
source: https://huggingface.co/google/embeddinggemma-2/blob/914f7f89142e33e77833254d9c9b90c3cef7303b/README.md ; https://huggingface.co/litert-community/embeddinggemma-2-740m-litert-lm/blob/24d962e906c7d332c6428e71c9676855024569e2/README.md ; https://huggingface.co/litert-community/embeddinggemma-2-740m-litert-lm/tree/24d962e906c7d332c6428e71c9676855024569e2 ; https://huggingface.co/litert-community/embeddinggemma-2-text-vision-440m-litert-lm/blob/e301f74d5551b0c2641bd5cb4652a76239d5c5f8/README.md ; https://huggingface.co/litert-community/embeddinggemma-2-text-270m-litert-lm/blob/9be6e8b90982095dc05c2bd162e4b954ee4dbac7/README.md ; https://pub.dev/packages/flutter_edge_ai_litertlm/versions/1.17.0 (README.md, "Embeddings", the EmbeddingGemma 2 sections, Google Tensor NPU, MediaTek NPU)
license: Apache-2.0 (google/embeddinggemma-2 and the litert-community cards) ; MIT (flutter_edge_ai_litertlm)
---

# EmbeddingGemma 2 embedding model

This document summarizes EmbeddingGemma 2, Google's open multimodal embedding model: what goes in and what comes out, the three LiteRT-LM bundles (270m for text, 440m with images, 740m with images and audio), how its vectors can be shortened, the prefixes that tell it whether it is looking at a query or a document, what images and audio cost in tokens, its benchmark scores, its measured speed and memory, the CPU and per-SoC NPU bundles, and how the flutter_edge_ai packages run it.

## What EmbeddingGemma 2 is

EmbeddingGemma 2 is an open multimodal embedding model built by Google DeepMind. It maps text (code included), images, video and audio, and combinations of them, into one shared 768-dimensional vector space, so a text can find a photo or a spoken sentence. The model has 740M parameters in total: a 270M text model (a 130M transformer and a 140M embedder) with a 170M vision encoder and a 300M audio encoder that can be loaded separately. It builds on Gemma 4.

It understands more than 100 languages and scores about 14% better on code tasks than its predecessor, EmbeddingGemma (version 1). It is designed for consumer hardware such as phones and laptops, for search, retrieval-augmented generation (RAG), classification and clustering. The licence is Apache 2.0, and the Hugging Face repositories are not gated: no account or licence acceptance is needed to download them.

## Model architecture of EmbeddingGemma 2

| Property | Value |
| --- | --- |
| Parameters | 740M total: backbone 130M, embedder 140M, vision encoder 170M, audio encoder 300M |
| Layers | 24 |
| Model dimension | 512 |
| Hidden dimension | 2048 |
| Sliding window | 1024 tokens |
| Vocabulary size | 262,144 |
| Attention heads | 4; KV heads 2 local / 1 global; local:global 5:1; GQA/MQA |
| Activation | gated FFN with GELU |
| Pooling | mean pooling, then a 512→768 projection |
| Context window | 8,192 tokens |
| Native output dimension | 768; Matryoshka truncation to 512, 256 and 128 |

The vision and audio encoders are independent parts. Loading only the ones a task needs gives an effective size of 270M for text only, 440M for text and images, 570M for text and audio, and 740M for everything.

## The three LiteRT-LM bundles: 270m, 440m and 740m

The litert-community organization on Hugging Face publishes EmbeddingGemma 2 for LiteRT-LM as one self-contained `.litertlm` file per variant, with the tokenizer inside:

| | Text 270M | Text Vision 440M | 740M |
| --- | --- | --- | --- |
| Repository | `litert-community/embeddinggemma-2-text-270m-litert-lm` | `litert-community/embeddinggemma-2-text-vision-440m-litert-lm` | `litert-community/embeddinggemma-2-740m-litert-lm` |
| Modalities | text | text, images | text, images, video, audio |
| Download size (CPU/GPU file) | 165 MB | 388 MB | 485 MB |
| Quantization | transformer and embeddings int4 per-channel (QAT) | as 270M, plus vision encoder int8 per-channel (QAT) | as 440M, plus audio encoder mixed int2/int4/int8 per-channel (QAT) |
| On-demand modality loading | no | yes | yes |
| Input sizes | text signatures of 128, 256, 512, 1024, 2048 or 8192 tokens | text, plus 70 or 140 image soft tokens | text and images, plus audio |
| Output sizes | 128, 256, 512 or 768 values | same | same |

The 440m carries the 270m's text tower, so its text vectors are the same; the 740m carries the 440m's text and vision towers, so its text and image vectors are the same as the 440m's. For text alone the 270m is the one to use: a 440m loaded for text still loads its vision tower (flutter_edge_ai_litertlm README). The 740m accepts only the media declared when the embedder is created, although the engine writes the weight caches of all its towers, also for text-only use (flutter_edge_ai_litertlm README). This sample app declares audio at setup and adds images the first time the camera's visual memory runs, which reloads the model. Measured for this sample on an M4 Pro Mac in a test process holding only the embedder, declaring images at load raised that process's memory after load and warm-up from 648 MB to 1055 MB and the load time from 1.4 s to 2.2 s; a re-run of that test the next day measured the widening reload the app does at 1.9 s, adding 327 MB.

## Inputs, outputs and Matryoshka embedding sizes

- In: text, an image, a video, an audio clip, or several of them in one input. In the reference model all parts share one 8,192-token context. The flutter_edge_ai_litertlm package embeds at most 1024 tokens per input (task prefix, BOS and EOS included) and never truncates: a longer input fails with `EmbeddingInputTooLongException`, which carries the exact token count, after a full embedding has been paid for. Split long text before embedding.
- Out: one vector of 768 numbers (3072 bytes as float32).

EmbeddingGemma 2 is trained with Matryoshka Representation Learning (MRL): keep only the first 512, 256 or 128 values of the 768 and the shorter vector still works, with up to 6 times less storage and faster comparisons. Two rules: normalize the shortened vector again before computing cosine similarity (skipping this degrades ranking silently), and give queries and documents the same dimension. Quality stays close to lossless down to 256 dimensions. The card calls 128 dimensions best suited to text-only workloads: it degrades multimodal quality substantially (MMEB v2 overall drops from 59.01 at 768 dimensions to 45.65) and should be validated against your own workload before adoption.

EmbeddingGemma 1 and EmbeddingGemma 2 both produce 768-dimensional vectors, but in different embedding spaces: never mix them in one vector store.

The model card says to run the reference model in `bfloat16` or `float32`, never in `float16`: its activations exceed the `float16` range, and the result is NaN or silently degraded vectors rather than an error.

## Task prefixes for queries and documents

EmbeddingGemma 2 is trained with short task prefixes in front of text inputs. The right prefix improves quality; leaving it out still works but is less precise. Prefixes apply to text only: images, video and audio are passed without a prefix.

- Asymmetric tasks, such as retrieval, use a query prefix for queries and a document prefix for the indexed items.
- Symmetric tasks, such as classification and similarity, give every compared input the same prefix.

| Use case | Query prefix | Document prefix |
| --- | --- | --- |
| Web or document search | `task: search result \| query: {query}` | `title: {title} \| text: {content}` |
| Question answering | `task: question answering \| query: {question}` | `title: {title} \| text: {passage}` |
| Fact checking | `task: fact checking \| query: {claim}` | `title: {title} \| text: {evidence}` |
| Code search | `task: code retrieval \| query: {query}` | `title: {title or filename} \| text: {code}` |
| Text classification | `task: classification \| query: {content}` | none (symmetric) |
| Clustering | `task: clustering \| query: {content}` | none (symmetric) |
| Measuring similarity | `task: sentence similarity \| query: {content}` | none (symmetric) |

A document without a real title gets `title: none`. In flutter_edge_ai the prefix comes from the required `taskType` argument of `embed`: `retrievalQuery` and `retrievalDocument` for search, `none` for no prefix, and `questionAnswering`, `factChecking`, `codeRetrieval`, `classification`, `clustering` and `sentenceSimilarity`. A text query searched against images or audio works best raw, with `TaskType.none`.

## Image inputs and the vision token budget

The reference model spends 280 tokens per image by default and accepts vision budgets from 70 to 1120 soft tokens: a higher budget captures finer detail at the cost of time and tokens. Video is sampled at one frame per second, 140 tokens per frame by default.

The LiteRT-LM bundles take 70 or 140 image soft tokens, and LiteRT-LM scales and patchifies images of any size to fit the vision encoder. In flutter_edge_ai_litertlm the default is 140 tokens, the closest to the reference model (minimum cosine 0.985 on the package's test images); `LiteRtEmbeddingBackend(visionTokensPerImage: 70)`, what the model card's benchmarks and Google AI Edge Gallery use, is about twice as fast with slightly lower fidelity (minimum cosine 0.970). Any other value fails at load. Text vectors do not depend on the budget; image vectors do, so an image index should record the budget it was built with.

The engine decodes JPEG, PNG, GIF and BMP; HEIC, WebP, TIFF and AVIF fail with `EmbeddingFormatNotSupportedException`. LiteRT-LM ignores EXIF orientation, so a portrait phone photo stored sideways is embedded lying on its side (cosine 0.89 against the upright picture), and it decodes the file at full size. Pass images upright and downscaled, for example at most 1024 pixels on the long side.

## Audio inputs and their length

Audio should be mono at 16 kHz and costs about 25 tokens per second, so the reference model's 8,192-token context holds about 327 seconds. Within flutter_edge_ai_litertlm's 1024-token cap a clip of up to about 40 seconds fits; a 60-second clip fails with `EmbeddingInputTooLongException(maxInputTokens: 1024, actualTokens: 1504)`.

The engine decodes WAV, MP3 and FLAC at any sample rate and channel count, mixing down to mono and resampling to 16 kHz; Ogg and M4A fail before anything is sent. One clip goes into one vector, unless it is part of a combined input.

Audio vectors are not bit-portable: the audio encoder quantizes its activations to 8 bits, so a last-bit difference (another CPU kernel, or one sample changed by 1) can move a clip's vector by up to cosine 0.05. The package measured its native build and Google's `litert-lm` Python wheel within cosine 0.97 of each other for audio, with identical text vectors. Retrieval holds across builds: a text finds its clip on every build measured.

## Combined inputs of text, images and audio

One input may mix text with images, video and audio, and the model returns a single vector for all of it. In the reference model the text marks each media position with a placeholder token (`<|image|>`, `<|video|>`, `<|audio|>`). In flutter_edge_ai, `EmbeddingInput(text:, images:, audio:)` puts the text with its task prefix first, then the images in order, then the audio, whatever order the arguments are written in. Every medium sent must be declared when the embedder is created, and the 1024-token cap counts all parts together (about 140 tokens per image at the default budget).

## Benchmark results of EmbeddingGemma 2

Google evaluated the full-precision checkpoint at 768 dimensions:

| Modality | Benchmark | Metric | EmbeddingGemma 2 | EmbeddingGemma 1 |
| --- | --- | --- | --- | --- |
| Text | MTEB multilingual v2 | Mean (Task) | 61.36 | 61.15 |
| Text | MTEB code v1 | Mean (Task), NDCG@10 | 78.68 | 68.76 |
| Image | MIEB lite | Mean (TaskType) | 64.64 | - |
| Image | MMEB v2 Image | Mean (Task), Hit@1 | 57.28 | - |
| Image | MMEB v2 VisDoc | Mean (Task), NDCG@5 | 67.84 | - |
| Video | MMEB v2 Video | Mean (Task), Hit@1 | 50.67 | - |
| Audio | MSEB Retrieval | Mean (Task), MRR@10 | 69.54 | - |
| Audio | MAEB | Mean (Task) | 49.39 | - |

### Benchmark results with Matryoshka truncation

| Output dimension | Compression | MTEB multilingual v2 | MTEB English v2 | MTEB code v1 | MIEB lite | MMEB v2 overall | MSEB Retrieval | MAEB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 768 | 1:1 | 61.36 | 68.46 | 78.68 | 64.64 | 59.01 | 69.54 | 49.39 |
| 512 | 1:1.5 | 61.17 | 68.41 | 77.24 | 64.32 | 58.38 | 69.18 | 49.21 |
| 256 | 1:3 | 60.41 | 67.78 | 76.18 | 63.13 | 56.24 | 66.76 | 48.91 |
| 128 | 1:6 | 57.89 | 65.68 | 71.41 | 59.06 | 45.65 | 56.71 | 46.92 |

## EmbeddingGemma 2 speed on LiteRT-LM

The litert-community 740m card measured text with the 128-token text signature, images with the 70-token vision signature, a 10-second clip for text plus audio and a 5-second clip for all three, averaged over 5 runs. Latency per input:

| Device | Backend | Text | Text + image | Text + audio | Text + image + audio |
| --- | --- | --- | --- | --- | --- |
| Samsung S26 Ultra | CPU | 27.1 ms | 175 ms | 305 ms | 322 ms |
| Samsung S26 Ultra | GPU | 25.9 ms | 119 ms | 350 ms | 324 ms |
| Google Pixel 11 Pro | TPU | 8.3 ms | 49 ms | 193.5 ms | 149 ms |
| iPhone 18 Pro | CPU | 41.8 ms | 191 ms | 445 ms | 400 ms |
| iPhone 18 Pro | GPU | 11.6 ms | 69.8 ms | 165 ms | 150 ms |
| MacBook Pro M5 | CPU | 31.3 ms | 151 ms | 314 ms | 432 ms |
| MacBook Pro M5 | GPU | 9.5 ms | 37.3 ms | 195 ms | 131 ms |
| Dell XPS 16 (Intel Core Ultra Series 3) | Intel OpenVINO NPU | 13.3 ms | 49.8 ms | 124.3 ms | 108.6 ms |
| Raspberry Pi 5 16GB | CPU | 161 ms | 1761 ms | 946 ms | 2304 ms |
| Jetson Orin Nano | GPU | 88 ms | 485 ms | 1195 ms | 1069 ms |
| Arduino VENTUNO Q | NPU | 13.6 ms | 135 ms | not supported | not supported |

The 270m and 440m cards report the same text and text + image numbers. Audio is not supported on the VENTUNO Q's NPU (IQ-8275).

## EmbeddingGemma 2 memory use on LiteRT-LM

CPU memory from the same 740m card, taken at the second load (which reads the weight caches); memory at the first load may vary. Each platform uses its own metric (`ru_maxrss` on Android and Linux, `phys_footprint` on iOS and macOS), so the numbers are not comparable across operating systems, and accelerator memory is not included except in `phys_footprint`.

| Device | Backend | Text | Text + image | Text + audio | Text + image + audio |
| --- | --- | --- | --- | --- | --- |
| Samsung S26 Ultra | CPU | 334 MB | 694 MB | 465 MB | 811 MB |
| Samsung S26 Ultra | GPU | 333 MB | 427 MB | 407 MB | 493 MB |
| iPhone 18 Pro | CPU | 84 MB | 196 MB | 98 MB | 196 MB |
| MacBook Pro M5 | CPU | 165 MB | 310 MB | 201 MB | 336 MB |
| Raspberry Pi 5 16GB | CPU | 282 MB | 619 MB | 408 MB | 736 MB |
| Arduino VENTUNO Q | NPU | 206 MB | 460 MB | not supported | not supported |

flutter_edge_ai_litertlm adds its own measurements on the CPU: memory grows with the input length (on an M4 Pro Mac about +170 MB at 128 tokens, +340 MB at 512 and +560 MB at 1024, so chunks of 512 tokens or fewer are recommended); the first image adds about 385 MB of RAM on an M4 Pro Mac and about 500 MB on a Pixel 8; an image at 140 tokens takes about 0.35 s on an Apple-silicon Mac and about 2 s on an Android flagship, and 70 tokens about half; a 3-second clip takes about 0.2–0.3 s on an M4 Pro Mac.

## CPU, GPU and per-SoC NPU bundles

Each of the three repositories holds the generic `.litertlm` for the CPU and GPU and the same model compiled ahead of time for one chip's NPU: Google Tensor G5 and G6, Intel Panther Lake (`_Intel_PTL`), MediaTek MT6991 and MT6993, and Qualcomm QCS8275, SM8550, SM8650, SM8750 and SM8850, for example `embeddinggemma-2-740m_Qualcomm_SM8650.litertlm`. A per-SoC bundle is compiled for that chip's NPU.

flutter_edge_ai_litertlm runs the generic bundle on the CPU natively (`activeBackend` is `cpu` on Android, iOS and desktop) and on WebGPU in the browser. With `PreferredBackend.npu` it runs the Qualcomm per-SoC bundles on Qualcomm Android and Linux arm64 (verified on SM8850, SM8650 and QCS8275, with `qualcomm_npu: true` in the app's build hooks), the Google Tensor G5 and G6 ones on Pixels with that Tensor generation in apps built with `google_tensor_npu: true` (the G6 270m and 440m ran on a Pixel 11), and the Intel Panther Lake ones on Windows x64, with no flag (the 270m loads in about 0.9 s there). The MediaTek MT6991 and MT6993 bundles are accepted in apps built with `mediatek_npu: true`, but have not run on such a phone yet. On an SM8650 (Galaxy S24) a text takes about 9 ms on the NPU against 53 ms on its CPU, and an image about 0.31 s. The 740m's per-SoC bundles embed audio with their audio encoder on the CPU (verified on SM8850, a Galaxy S26+).

NPU and CPU vectors are close but not equal: cosine 0.79–0.86 for sentences of a few words, 0.92 at about 50 words and 0.99 from about 200 words, and 0.86–0.88 for images. Retrieval within the NPU's own space holds, but an index built with one bundle must not take queries from the other: give an NPU index its own profile id. Nothing switches backend by itself; every mismatch fails at load with `EmbeddingLoadException`.

## Running EmbeddingGemma 2 with flutter_edge_ai

The bundle carries its own tokenizer, so it needs no tokenizer file and no `flutter_edge_ai_embeddings` package. Register `LiteRtEmbeddingBackend` from `flutter_edge_ai_litertlm`, install the `.litertlm`, then ask for an embedder that declares the media you will send:

```dart
await FlutterEdgeAi.initialize(
  embeddingBackends: const [LiteRtEmbeddingBackend()],
);
await FlutterEdgeAi.installEmbedder()
    .modelFromNetwork(
      'https://huggingface.co/litert-community/'
      'embeddinggemma-2-740m-litert-lm/resolve/$revision/'
      'embeddinggemma-2-740m.litertlm',
    )
    .install();
final embedder = await FlutterEdgeAi.getActiveEmbedder(supportAudio: true);
final clip = await embedder.embed(
  EmbeddingInput.audio(wavBytes),
  taskType: TaskType.retrievalDocument, // prefixes reach text only
);
```

`supportImage: true` on the 270m fails with `EmbeddingLoadException` naming the 440m, and `supportAudio: true` on the 270m or the 440m fails naming the 740m; natively both are refused before an engine or a cache is made, the image case from the bundle's model info and the audio case from its header.

Disk: the engine writes weight caches into the app's cache directory at the first load, and again whenever they are missing, also for text-only use: 71 MB for the 270m, plus a 218 MB vision cache for the 440m, and 385 MB in total for the 740m (text 71 MB, vision 218 MB, audio 95 MB). The cache is never backed up; if the OS clears it, the next load writes it again. When the engine cannot create a tower's cache file (a path too long for the platform, a directory the app cannot write to), that tower runs anyway, slower to load and, measured by the package on the 270m, with about 290 MB more memory, and the package only prints a warning. `LiteRtEmbeddingBackend(requireWeightCache: true)` turns a load that wrote no weight cache at all into an `EmbeddingLoadException`; one tower's cache missing beside the others stays a printed warning, and the check runs on the CPU, and on the NPU only for a load with declared audio. A disk that fills while the cache is being written aborts the process in native code.

## Intended uses of EmbeddingGemma 2

- Retrieval: semantic search across text, code, images, video and audio, such as document search, RAG over a knowledge base, code search from plain-language questions, or spoken-query search over audio archives.
- Classification: sorting inputs into preset labels, for example sentiment, content moderation, audio event detection or image categories.
- Clustering: grouping inputs by meaning, such as document collections, customer feedback by theme, or near-duplicate content across modalities.
- Semantic similarity: pairwise similarity for recommendations, duplicate detection, paraphrase identification or cross-lingual alignment.
- Fact verification: retrieving evidence documents for a claim.

## Training data, limitations and safety of EmbeddingGemma 2

The pre-training data, with a cutoff of January 2025, covers web documents in more than 140 languages, code, images, video, speech recordings in many languages with environmental sounds, and paired text, image, video and audio samples that align the modalities. Google filtered it for CSAM, certain personal and sensitive information, and low-quality or unsafe content.

EmbeddingGemma 2 is a pre-trained embedding model with no post-training alignment, safety tuning or output moderation; its risks show up in how its vectors are used downstream, so applications need their own safeguards, such as retrieval filtering and fairness testing. Its quality depends on the training data: performance may not be equal across its 100+ languages, subtle wording, sarcasm and figurative language can be misread, and omitting the task prefix gives weaker text vectors. Use must follow the Gemma Prohibited Use Policy.
