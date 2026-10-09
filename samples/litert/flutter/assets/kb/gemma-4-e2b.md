---
title: Gemma 4 E2B model card
source: https://huggingface.co/google/gemma-4-E2B-it/blob/3e22461f65e89153144f8adb70e3b8c2cc9845a7/README.md ; https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/blob/b3ca0d2f076785a8f4b2219ddbd2bdb99954eae1/README.md
license: Apache-2.0
---

# Gemma 4 E2B model card

This document describes Gemma 4 E2B, the smallest model of Google DeepMind's Gemma 4 family: its architecture and per-layer embeddings, multimodal capabilities (text, image and audio), benchmarks, recommended settings for thinking, images and audio, intended uses and limitations, and the LiteRT-LM build with its on-device memory use and performance.

## The Gemma 4 model family

Gemma is a family of open models built by Google DeepMind. Gemma 4 models are multimodal, handling text and image input (with audio supported on E2B, E4B, and 12B) and generating text output. The release includes open-weights models in both pre-trained and instruction-tuned variants. Gemma 4 features a context window of up to 256K tokens and maintains multilingual support in over 140 languages. The license is Apache 2.0.

Featuring both Dense and Mixture-of-Experts (MoE) architectures, Gemma 4 is well-suited for tasks like text generation, coding, and reasoning. The models are available in five sizes: E2B, E4B, 12B, 26B A4B, and 31B, deployable in environments ranging from high-end phones to laptops and servers. Gemma 4 targets mobile and edge devices with E2B and E4B, and consumer GPUs and workstations with 12B, 26B A4B and 31B.

## Gemma 4 capability and architecture advancements

- Reasoning: all models in the family are designed as highly capable reasoners, with configurable thinking modes.
- Extended multimodalities: text and image with variable aspect ratio and resolution support (all models), video, and audio (featured natively on the E2B, E4B, and 12B models).
- Diverse and efficient architectures: Dense and Mixture-of-Experts variants of different sizes.
- Optimized for on-device: smaller models are specifically designed for efficient local execution on laptops and mobile devices.
- Increased context window: the small models feature a 128K context window, while the medium models support 256K.
- Enhanced coding and agentic capabilities: notable improvements in coding benchmarks alongside native function-calling support.
- Native system prompt support: Gemma 4 introduces native support for the `system` role, enabling more structured and controllable conversations.

The models employ a hybrid attention mechanism that interleaves local sliding window attention with full global attention, ensuring the final layer is always global. To optimize memory for long contexts, global layers feature unified Keys and Values and apply Proportional RoPE (p-RoPE).

## What the E in E2B means: per-layer embeddings

The "E" in E2B and E4B stands for "effective" parameters. The smaller models incorporate Per-Layer Embeddings (PLE) to maximize parameter efficiency in on-device deployments. Rather than adding more layers or parameters to the model, PLE gives each decoder layer its own small embedding for every token. These embedding tables are large but are only used for quick lookups, which is why the effective parameter count is much smaller than the total.

## Gemma 4 E2B and E4B specifications

| Property | E2B | E4B |
| --- | --- | --- |
| Total parameters | 2.3B effective (5.1B with embeddings) | 4.5B effective (8B with embeddings) |
| Layers | 35 | 42 |
| Sliding window | 512 tokens | 512 tokens |
| Context length | 128K tokens | 128K tokens |
| Vocabulary size | 262K | 262K |
| Supported modalities | Text, Image, Audio | Text, Image, Audio |
| Vision encoder parameters | ~150M | ~150M |
| Audio encoder parameters | ~300M | ~300M |

For comparison, the 12B Unified model has an encoder-free architecture that projects raw image patches and audio waveforms directly into the LLM's embedding space, and the 26B A4B Mixture-of-Experts model activates only 3.8B of its 25.2B parameters during inference.

## Benchmark results for Gemma 4 E2B and E4B

Evaluation results are for instruction-tuned models.

| Benchmark | Gemma 4 E4B | Gemma 4 E2B |
| --- | --- | --- |
| MMLU Pro | 69.4% | 60.0% |
| AIME 2026 no tools | 42.5% | 37.5% |
| LiveCodeBench v6 | 52.0% | 44.0% |
| Codeforces ELO | 940 | 633 |
| GPQA Diamond | 58.6% | 43.4% |
| Tau2 (average over 3) | 42.2% | 24.5% |
| BigBench Extra Hard | 33.1% | 21.9% |
| MMMLU | 76.6% | 67.4% |
| MMMU Pro (vision) | 52.6% | 44.2% |
| MATH-Vision | 59.5% | 52.4% |
| CoVoST (audio) | 35.54 | 33.47 |
| FLEURS (audio, lower is better) | 0.08 | 0.09 |
| MRCR v2 8 needle 128k (average) | 25.4% | 19.1% |

For reference, Gemma 3 27B without thinking scores 67.6% on MMLU Pro and 20.8% on AIME 2026, and the largest Gemma 4 model, 31B, scores 85.2% and 89.2%.

## Core capabilities of Gemma 4

- Thinking: a built-in reasoning mode that lets the model think step-by-step before answering.
- Long context: context windows of up to 128K tokens (E2B/E4B) and 256K tokens (12B, 26B A4B/31B).
- Image understanding: object detection, document and PDF parsing, screen and UI understanding, chart comprehension, OCR (including multilingual), handwriting recognition, and pointing. Images can be processed at variable aspect ratios and resolutions.
- Video understanding: analyze video by processing sequences of frames.
- Interleaved multimodal input: freely mix text and images in any order within a single prompt.
- Function calling: native support for structured tool use, enabling agentic workflows.
- Coding: code generation, completion, and correction.
- Multilingual: out-of-the-box support for 35+ languages, pre-trained on 140+ languages.
- Audio (E2B, E4B, and 12B only): automatic speech recognition (ASR) and speech-to-translated-text translation across multiple languages.

## Recommended sampling parameters and thinking mode

For the best performance, the model card recommends one standardized sampling configuration across all use cases: `temperature=1.0`, `top_p=0.95` and `top_k=64`.

Gemma 4 uses the standard `system`, `assistant` and `user` roles. Thinking is enabled by including the `<|think|>` token at the start of the system prompt; to disable thinking, remove the token. When thinking is enabled, the model outputs its internal reasoning followed by the final answer, using the structure `<|channel>thought\n` [internal reasoning] `<channel|>`. For all models except the E2B and E4B variants, if thinking is disabled the model still generates the tags but with an empty thought block. Many libraries, such as Transformers and llama.cpp, handle the complexities of the chat template for you.

In multi-turn conversations, the historical model output should only include the final response. Thoughts from previous model turns must not be added before the next user turn begins, with the exception of tool call turns, where thinking content should be preserved.

## Ordering images and audio and choosing image resolution

For optimal performance with multimodal inputs, place image content before the text in your prompt, and audio content after the text.

Aside from variable aspect ratios, Gemma 4 supports variable image resolution through a configurable visual token budget, which controls how many tokens are used to represent an image. A higher budget preserves more visual detail at the cost of additional compute, while a lower budget enables faster inference. The supported token budgets are 70, 140, 280, 560, and 1120. Use lower budgets for classification, captioning, or video understanding, where faster inference and processing many frames outweigh fine-grained detail, and higher budgets for tasks like OCR, document parsing, or reading small text.

## Audio prompts and length limits

All models support image inputs and can process videos as frames, whereas the E2B, E4B, and 12B models also support audio inputs. Audio supports a maximum length of 30 seconds. Video supports a maximum of 60 seconds, assuming the images are processed at one frame per second.

For speech recognition, the model card recommends the prompt: "Transcribe the following speech segment in {LANGUAGE} into {LANGUAGE} text. Follow these specific instructions for formatting the answer: Only output the transcription, with no newlines. When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three."

For speech translation: "Transcribe the following speech segment in {SOURCE_LANGUAGE}, then translate it into {TARGET_LANGUAGE}. When formatting the answer, first output the transcription in {SOURCE_LANGUAGE}, then one newline, then output the string '{TARGET_LANGUAGE}: ', then the translation in {TARGET_LANGUAGE}."

## Training data and safety evaluation

The pre-training dataset is a large-scale, diverse collection of data encompassing web documents, code, mathematics, images and audio, with a cutoff date of January 2025. The web documents include content in over 140 languages. Data preprocessing applied rigorous CSAM filtering at multiple stages, automated filtering of certain personal information and other sensitive data, and filtering based on content quality and safety.

Gemma 4 undergoes the same rigorous safety evaluations as Google's proprietary Gemini models, with automated and human evaluations covering child safety, dangerous content, sexually explicit content, hate speech and harassment. The card reports major improvements in all categories of content safety relative to previous Gemma models, with Gemma 4 significantly outperforming Gemma 3 and 3n while keeping unjustified refusals low. All testing was conducted without safety filters.

## Intended uses of Gemma 4

The model card lists potential uses including text generation (poems, scripts, code, marketing copy, email drafts), chatbots and conversational AI, text summarization, image data extraction, and audio processing and interaction: the E2B, E4B, and 12B models can analyze and interpret audio inputs, enabling voice-driven interactions and transcriptions. For research and education it lists NLP and VLM research, language learning tools, and knowledge exploration.

## Limitations of Gemma 4

- Training data: the quality and diversity of the training data significantly influence the model's capabilities, and biases or gaps can limit responses.
- Context and task complexity: models perform well on tasks with clear prompts and instructions; open-ended or highly complex tasks might be challenging.
- Language ambiguity and nuance: models might struggle with subtle nuances, sarcasm, or figurative language.
- Factual accuracy: models generate responses based on information learned from their training datasets, but they are not knowledge bases, and may generate incorrect or outdated factual statements.
- Common sense: models rely on statistical patterns in language and might lack common sense reasoning in certain situations.

The card identifies risks such as harmful content generation, misuse, privacy violations and perpetuation of biases, and encourages developers to implement content safety safeguards suited to their product.

## The LiteRT-LM build of Gemma 4 E2B and its memory footprint

The Hugging Face repo `litert-community/gemma-4-E2B-it-litert-lm` provides Gemma 4 E2B ready for deployment on Android, iOS, Desktop, IoT and Web, in the `.litertlm` format for the LiteRT-LM framework. This Gemma 4 model is small, so it is ideal for on-device use cases: by running it on device, users can have private access to generative AI technology without even requiring an internet connection.

LiteRT-LM uses a Gemma 4 mobile quantization scheme that uses a mixture of 2-bit, 4-bit and 8-bit weights. For text-only use cases the weight footprint in memory can be as low as 0.8 GB, while the runtime uses memory mapping to support the 1.12 GB of embedding parameters. This gives significant working memory savings on some platforms. Additionally, the vision and audio models are loaded on demand to further reduce memory consumption. The model file on disk is 2583 MB. Web uses a specially optimized model, `gemma-4-E2B-it-web.litertlm` (2008 MB), because of its unique memory constraints; it is currently text-only.

## Gemma 4 E2B performance on phones

All benchmarks used 1024 prefill tokens and 256 decode tokens with a context length of 2048 tokens via LiteRT-LM; the litert-lm card states that the model can support up to 32k context length. CPU inference is accelerated via the LiteRT XNNPACK delegate with 4 threads. Time-to-first-token does not include load time, and benchmarks ran with caches enabled and initialized.

| Device | Backend | Prefill (tokens/s) | Decode (tokens/s) | Time to first token (s) | Memory (MB) |
| --- | --- | --- | --- | --- | --- |
| Samsung S26 Ultra | CPU | 557 | 46.9 | 1.8 | 1733 |
| Samsung S26 Ultra | GPU | 3,808 | 52.1 | 0.3 | 676 |
| iPhone 17 Pro | CPU | 532 | 25.0 | 1.9 | 607 |
| iPhone 17 Pro | GPU | 2,878 | 56.5 | 0.3 | 1450 |

CPU memory was measured with `rusage::ru_maxrss` on Android, Linux and Raspberry Pi, `task_vm_info::phys_footprint` on iOS and MacBook, and `process_memory_counters::PrivateUsage` on Windows. On supported Android devices, Gemma 4 is available through Android AI Core as Gemini Nano, which the card calls the recommended path for production applications.

## Speculative decoding speedups for Gemma 4 E2B

Speculative decoding accelerates LLMs by using a small, fast draft model to quickly predict multiple upcoming tokens, while a larger target model verifies those tokens in parallel. Its effectiveness is task dependent. It is available on CPU and GPU on mobile and desktop; a model downloaded before May 5, 2026 must be re-downloaded to use it. Decode speed on a Samsung S26 Ultra:

| Backend | Task type | Speculative decoding | Decode (tokens/s) | CPU memory (MB) |
| --- | --- | --- | --- | --- |
| CPU | Baseline | No | 40.7 | 1362 |
| CPU | Summarize text | Yes | 47.5 | 1582 |
| CPU | Free form | Yes | 38.1 | 1459 |
| GPU | Baseline | No | 51.5 | 791 |
| GPU | Summarize text | Yes | 91.7 | 817 |
| GPU | Code snippet | Yes | 84.4 | 788 |
| GPU | Rewrite tone | Yes | 87.4 | 762 |
| GPU | Free form | Yes | 66.5 | 804 |

## Gemma 4 E2B performance on desktop, web and IoT

| Device | Backend | Prefill (tokens/s) | Decode (tokens/s) | Time to first token (s) | Memory (MB) |
| --- | --- | --- | --- | --- | --- |
| MacBook Pro M4 Max | CPU | 901 | 41.6 | 1.1 | 736 |
| MacBook Pro M4 Max | GPU | 7,835 | 160.2 | 0.1 | 1623 |
| MacBook Pro M4 Max (web) | WebGPU | 4,853 | 73 | 1.09 | ~1800 GPU |
| Linux, Arm 2.3 and 2.8 GHz | CPU | 260 | 35.0 | 4.0 | 1628 |
| Linux, NVIDIA GeForce RTX 4090 | GPU | 11,234 | 143.4 | 0.1 | 913 |
| Windows, Intel LunarLake | CPU | 435 | 29.8 | 2.39 | 3505 |
| Windows, Intel LunarLake | GPU | 3,751 | 48.4 | 0.29 | 3540 |
| Raspberry Pi 5 16GB | CPU | 133 | 7.6 | 7.8 | 1546 |
| Jetson Orin Nano | GPU | 1,142 | 24.2 | 0.9 | 2739 |
| Qualcomm Dragonwing IQ8 (IQ-8275) | NPU | 3,747 | 31.7 | 0.3 | 1869 |

The NPU model was benchmarked with a 4096 context length and is 2967 MB on disk. Gemma 4 E2B can also run on the web through the MediaPipe LLM Inference Engine with the `gemma-4-E2B-it-web.task` file, but that route is currently in maintenance mode.
