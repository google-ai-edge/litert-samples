---
title: EmbeddingGemma-300M embedding model
source: https://huggingface.co/google/embeddinggemma-300m/blob/57c266a740f537b4dc058e1b0cda161fd15afa75/README.md ; https://huggingface.co/litert-community/embeddinggemma-300m/blob/29888fcee3216acadc7e844906e5fe0d79a61875/README.md ; https://huggingface.co/litert-community/embeddinggemma-300m/tree/29888fcee3216acadc7e844906e5fe0d79a61875 ; https://github.com/google-ai-edge/litert-samples/blob/367ddbdee22a8d73fba6e519c4a3f18d19008b05/samples/litert/semantic_similarity/build_from_source/README.md
license: Apache-2.0
---

# EmbeddingGemma-300M embedding model

This document summarizes EmbeddingGemma, Google's open text embedding model with about 300 million parameters: what goes in and what comes out, how its vectors can be shortened, the prefixes that tell it whether it is looking at a query or a document, its benchmark scores, the quantized variants, and the LiteRT files for Android and iOS with their measured speed on the CPU, GPU and NPU.

## What EmbeddingGemma is

EmbeddingGemma turns a piece of text into a vector of numbers, so that texts with a similar meaning end up close to each other. Google DeepMind built it on Gemma 3, initialized from T5Gemma, with research it shares with the Gemini models; among models of its size (300M parameters) it scores at the top. Typical jobs are search and retrieval, classification, clustering and semantic similarity. Its training text covers more than 100 spoken languages.

Because it is small, it is meant for places where memory and compute are tight: phones, laptops and desktops. The paper that describes it is "EmbeddingGemma: Powerful and Lightweight Text Representations" (arXiv 2509.20354). The weights are released under the Gemma Terms of Use, and Hugging Face asks you to accept Google's licence before it lets you download them.

## Inputs, outputs and Matryoshka embedding sizes

- In: one string, such as a question, a prompt or a document, of up to 2048 tokens.
- Out: one vector of 768 numbers.

The model was trained with Matryoshka Representation Learning (MRL), which puts most of the meaning into the start of the vector. Keep only the first 512, 256 or 128 values, normalize the shorter vector again, and it still works for search, with less storage and faster comparisons.

The model's activations cannot run in `float16`. Run it in `float32`, or in `bfloat16` where the hardware supports that.

## Prompt instructions for queries and documents

EmbeddingGemma expects a short prefix in front of every input, saying what kind of text it is and what the vector will be used for. A query gets `task: {task description} | query: `; without a specific task the description is `search result`. A document gets `title: {title | "none"} | text: `, with `none` unless the document has a real title. Passing the real title usually gives better document vectors, but you have to format it yourself. Many frameworks already include these prefixes in their EmbeddingGemma configuration.

### Recommended prompt for each task

- Retrieval (query): `task: search result | query: {content}`
- Retrieval (document): `title: {title | "none"} | text: {content}`
- Question answering: `task: question answering | query: {content}`
- Fact verification: `task: fact checking | query: {content}`
- Classification: `task: classification | query: {content}`
- Clustering: `task: clustering | query: {content}`
- Semantic similarity: `task: sentence similarity | query: {content}`
- Code retrieval: `task: code retrieval | query: {content}`

The sentence-similarity prefix is tuned for comparing two texts with each other, not for search: use the retrieval prefixes to search. For code search, the question in plain language (for example "sort an array") gets the code-retrieval prefix, and the code blocks themselves are embedded with the document prefix.

## Using EmbeddingGemma with Sentence Transformers

The original weights (not the LiteRT files) are meant for the Sentence Transformers library, which runs the Gemma 3 implementation of Hugging Face Transformers underneath. The library has one method for queries and another for documents, so each side gets its own prefix, and a similarity call scores every document against the query. One query becomes a vector of shape (768,); four documents become a matrix of shape (4, 768).

```python
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("google/embeddinggemma-300m")
query = embedder.encode_query("How do I run a model on the phone's GPU?")
passages = embedder.encode_document(texts)
scores = embedder.similarity(query, passages)
```

The model card's own example asks which planet is called the Red Planet. The passage about Mars scores 0.6359, clearly ahead of the passages about Jupiter (0.4930), Saturn (0.4889) and Venus (0.3011).

## Benchmark results by embedding size and quantization

Google evaluated the full-precision model on MTEB. Mean (Task) scores for each vector length:

| Dimensionality | MTEB Multilingual v2 | MTEB English v2 | MTEB Code v1 |
| --- | --- | --- | --- |
| 768d | 61.15 | 69.67 | 68.76 |
| 512d | 60.71 | 69.18 | 68.48 |
| 256d | 59.68 | 68.37 | 66.74 |
| 128d | 58.23 | 66.66 | 62.96 |

At 768 dimensions the Mean (TaskType) scores are 54.31 for multilingual, 65.11 for English and 68.76 for code.

The quantization-aware-trained (QAT) checkpoints were scored after quantization, all at 768 dimensions. Mean (Task) scores:

| Quant config | MTEB Multilingual v2 | MTEB English v2 | MTEB Code v1 |
| --- | --- | --- | --- |
| Q4_0 | 60.62 | 69.31 | 67.99 |
| Q8_0 | 60.93 | 69.49 | 68.70 |
| Mixed precision | 60.69 | 69.32 | 68.03 |

"Mixed precision" means per-channel quantization with int4 for the embedding, feed-forward and projection layers and int8 for attention, written e4_a8_f4_p4. The LiteRT files described below use this scheme.

## Training data and development of EmbeddingGemma

The training set holds about 320 billion tokens: web pages in more than 100 languages, code and technical documents, and synthetic and task-specific data, including data put together for retrieval, classification and sentiment analysis. Google filtered it to take out CSAM, certain personal and sensitive information, and low-quality or unsafe content. Training ran on TPU v5e hardware with JAX and ML Pathways.

## Intended uses of EmbeddingGemma

- Semantic similarity: scoring how alike two texts are, for recommendations or for finding duplicates.
- Classification: sorting texts into fixed labels, for example sentiment or spam.
- Clustering: grouping related texts, to organize documents, for market research or to spot anomalies.
- Retrieval: vectors for the articles, books or web pages you index, for search queries, and for plain-language questions that should find code.
- Question answering: question vectors tuned to find the passages that answer them, as a chatbot needs.
- Fact verification: statement vectors tuned to find the passages that support or contradict them.

## Limitations and risks of EmbeddingGemma

How well the model handles a text depends on its training data: material that is biased or missing there shows up as skewed or weaker vectors, and subjects outside that data are covered less well. Subtle wording, sarcasm and figurative language can still be misread. Google recommends ongoing monitoring and de-biasing, points to the Gemma Prohibited Use Policy for the uses that are not allowed, and asks developers to follow privacy law and to use privacy-preserving techniques.

## LiteRT builds of EmbeddingGemma

The Hugging Face repository `litert-community/embeddinggemma-300m` holds EmbeddingGemma converted for LiteRT on Android and iOS; on Android the Google AI Edge RAG Library can use it too. Its tokenizer is the SentencePiece model `sentencepiece.model` (about 4.7 MB) in the same repository.

There is one generic file for each maximum sequence length, `embeddinggemma-300M_seq256_mixed-precision.tflite` and its `seq512`, `seq1024` and `seq2048` versions, and for each length a set of NPU builds compiled for one chip each: Google Tensor G5 and G6, MediaTek MT6991 and MT6993, and Qualcomm SM8550, SM8650, SM8750 and SM8850. The generic `seq512` file is about 179 MB. The RAG Library SDK is published on Maven, with an Android guide and a sample app. The Google Tensor builds were compiled with the standard settings, so they need no extra compiler flags.

## EmbeddingGemma performance on a Samsung S25 Ultra

The repository's measurements, all taken on a Samsung S25 Ultra with the mixed-precision files:

| Backend | Max sequence length | Init time (ms) | Inference time (ms) | Memory, RSS (MB) | Model size (MB) |
| --- | --- | --- | --- | --- | --- |
| NPU | 256 | 206 | 7.8 | 224 | 182 |
| NPU | 512 | 241 | 18 | 231 | 184 |
| NPU | 1024 | 272 | 57 | 263 | 195 |
| NPU | 2048 | 468 | 169 | 332 | 220 |
| GPU | 256 | 1175 | 64 | 762 | 179 |
| GPU | 512 | 1445 | 119 | 762 | 179 |
| GPU | 1024 | 1545 | 241 | 771 | 183 |
| GPU | 2048 | 1707 | 683 | 786 | 196 |
| CPU | 256 | 17.6 | 66 | 110 | 179 |
| CPU | 512 | 24.9 | 169 | 123 | 179 |
| CPU | 1024 | 35.4 | 549 | 169 | 183 |
| CPU | 2048 | 35.8 | 2455 | 333 | 196 |

Init time is paid once, when the app sets the model up; later calls skip it. Memory is the peak resident set size, and model size is the size of the `.tflite` file. The CPU runs use LiteRT's XNNPACK delegate with 4 threads. The numbers were taken with the cache enabled and already warm, so the very first run can differ.

## Semantic similarity sample with EmbeddingGemma in C++

The litert-samples repository has a C++ program that runs EmbeddingGemma on the LiteRT runtime and prints the cosine similarity of two sentences. It takes the tokenizer (`sentencepiece.model`), the embedder `.tflite` file, the two sentences and a sequence length such as 256. On Linux it is built with Bazel and runs on the CPU. For Android, a script builds it, copies the files to a connected device with `adb` and runs it there, by default with `embeddinggemma-300M_seq256_mixed-precision.tflite` on the `cpu` accelerator. With `--accelerator "npu"` and `--soc_man` set to Qualcomm (the HTP, with `QNN_SDK_ROOT` pointing at the QAIRT SDK), MediaTek (the APU) or Google (the Tensor TPU), the script also gathers the NPU libraries the program needs. The script wants a space between a flag and its value; `--flag=value` does not work.
