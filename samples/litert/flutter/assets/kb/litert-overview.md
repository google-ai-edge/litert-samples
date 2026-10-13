---
title: LiteRT, Google's on-device AI runtime
source: https://github.com/google-ai-edge/LiteRT/blob/bbb1312f51e37cab695f7ec3a3c74b8a4a9ca13a/README.md ; https://github.com/google-ai-edge/LiteRT/blob/bbb1312f51e37cab695f7ec3a3c74b8a4a9ca13a/litert/README.md ; https://github.com/google-ai-edge/litert-samples/blob/367ddbdee22a8d73fba6e519c4a3f18d19008b05/README.md ; https://github.com/google-ai-edge/litert-samples/blob/367ddbdee22a8d73fba6e519c4a3f18d19008b05/samples/litert/README.md ; https://github.com/google-ai-edge/litert-samples/blob/367ddbdee22a8d73fba6e519c4a3f18d19008b05/samples/litert_model_zoo/README.md ; https://github.com/google-ai-edge/litert-samples/blob/367ddbdee22a8d73fba6e519c4a3f18d19008b05/models/README.md ; https://github.com/google-ai-edge/litert-samples/blob/367ddbdee22a8d73fba6e519c4a3f18d19008b05/skills/README.md
license: Apache-2.0
---

# LiteRT, Google's on-device AI runtime

This document explains what LiteRT is, how it relates to TensorFlow Lite, the key features of LiteRT V2, the path from a trained model to a phone, the platforms and accelerators it supports, and the samples, model recipes and agent skills that Google publishes around it.

## What LiteRT is and where it came from

LiteRT is Google's on-device runtime for high-performance ML and GenAI deployment on edge platforms. LiteRT (short for Lite Runtime), formerly known as TensorFlow Lite, is Google's high-performance runtime for on-device AI. LiteRT continues the legacy of TensorFlow Lite as the trusted, high-performance runtime for on-device AI. Featuring advanced GPU and NPU acceleration, LiteRT delivers superior ML and GenAI performance, making on-device ML inference easier than ever.

You can find ready-to-run LiteRT models for a wide range of ML and AI tasks, or convert and run TensorFlow, PyTorch, and JAX models to the TFLite format using the AI Edge conversion and optimization tools. The model file format is the `.tflite` FlatBuffer; large language models are packaged as `.litertlm` bundles and run through LiteRT-LM.

LiteRT provides nightly builds and targets stable releases on a 6 to 8 week cadence. LiteRT is licensed under the Apache-2.0 License. The official documentation lives at ai.google.dev/edge/litert.

## Key features of LiteRT V2

LiteRT V2 is built around three headline features.

- Compiled Model API, for streamlined development. It features automated accelerator selection (no explicit delegates needed), true asynchronous execution, easy NPU distribution, and highly efficient I/O buffer handling.
- Unified NPU acceleration, for broad silicon support. It gives seamless access to NPUs from major chipset providers through a single, consistent API.
- Faster GPU acceleration via ML Drift, supporting GenAI inference. It leverages state-of-the-art GPU acceleration with new buffer interoperability that minimizes latency across various GPU buffer types.

Developers upgrading from TensorFlow Lite or LiteRT V1.x use the LiteRT Migration Guide to move to LiteRT V2.x.

## What is new in the LiteRT ecosystem

Recent additions to LiteRT cover generative AI, the web, graph authoring and coding agents.

- Superior GenAI inference: deploy LLMs directly on-device using LiteRT-LM.
- High-performance web inference: run secure client-side ML in the browser via WebGPU and WASM with LiteRT.js.
- C++ graph authoring: manipulate high-performance tensors using a lightweight, tensor-centric C++ library via the Tensor API.
- Accelerated agentic coding: streamline AI coding agent workflows using the LiteRT CLI command-line toolkit. LiteRT-CLI (the `litert` command) downloads, converts, quantizes, runs and benchmarks models from one command. Its quick setup creates a Python 3.13 virtual environment with `uv` and installs the `litert-cli-nightly` package, after which `litert --help` lists the commands.

## From a trained model to on-device deployment

LiteRT covers the whole path from model to on-device deployment for PyTorch, TensorFlow, and JAX models. In the reference pipeline, a PyTorch model, or a Hugging Face transformer stored as safetensors, goes into LiteRT Torch (including the LiteRT Torch Generative and Hugging Face export path). LiteRT Torch produces either a classic `.tflite` file or a `.litertlm` bundle for language models.

Either artifact can then pass through the AI Edge Quantizer, which produces an optimized `.tflite` or an optimized `.litertlm`. Optimized `.litertlm` bundles run on LiteRT-LM, which has Python, C++, Kotlin, Swift and JavaScript APIs. Both paths end in the LiteRT runtime, with C++, Kotlin and JavaScript APIs, which executes the model on the CPU through XNNPack, on the GPU through ML Drift, or on supported TPUs and NPUs.

## Supported platforms and accelerators

LiteRT is designed for cross-platform deployment on a wide range of hardware. Every platform has a CPU path.

| Platform | GPU APIs | NPU or hardware accelerators |
| :--- | :--- | :--- |
| Android | OpenCL, OpenGL | Broadcom, Google Tensor, Intel, MediaTek, Qualcomm, S.LSI |
| iOS | Metal | ANE (coming soon) |
| Linux | WebGPU | Broadcom, Intel |
| macOS | WebGPU, Metal | ANE (coming soon) |
| Windows | WebGPU | Intel |
| Web | WebGPU | WebNN (coming soon) |
| IoT | WebGPU | Raspberry Pi (coming soon) |

## Choosing a path: common developer journeys

The LiteRT project suggests a starting point for each goal.

- Upgrade from TensorFlow Lite or LiteRT V1.x: use the LiteRT Migration Guide to upgrade to LiteRT V2.x.
- Run a pretrained model such as image segmentation on mobile: follow the step-by-step Android Studio codelab that builds a real-time segmentation app for CPU, GPU and NPU inference.
- Convert PyTorch models: use the LiteRT Torch Converter for `.tflite` (classic) or the Generative Torch API for `.litertlm` (LLMs).
- Deploy generative AI: optimize and run quantized LLMs or diffusion models on-device using LiteRT-LM.
- Maximize performance: explore the LiteRT API and LiteRT NPU acceleration to leverage the underlying hardware acceleration.
- Run in the browser: deploy secure, client-side web apps leveraging WebGPU and WASM via LiteRT.js.
- Control memory and graph execution: use the LiteRT Tensor API, a tensor-centric C++ library for high-performance tensor manipulation on mobile devices.

## Coding guidance: use the Compiled Model API, not the Interpreter

The LiteRT repository states strict directives for application code that uses LiteRT (they do not apply to LiteRT's own runtime internals).

- Must use: the Compiled Model API for all new inference code in Kotlin, C++, Python, and JavaScript.
- Do not use: the TensorFlow Lite `Interpreter` API (for example `tflite::Interpreter`, `org.tensorflow.lite.Interpreter`, `tf.lite.Interpreter`) or manual delegate creation. TensorFlow Lite packages and `tensorflow/lite/` are in maintenance mode and only receive critical security and stability updates.
- Code examples: take code from the LiteRT documentation, not from older TensorFlow Lite examples. To move existing code over, follow the LiteRT Migration Guide.
- Agent skills: use the LiteRT agent skills for step-by-step workflows (convert, quantize, verify, benchmark, build apps, and migrate from TensorFlow Lite).

## The Google AI Edge ecosystem and the litert-community models

LiteRT is part of a larger Google AI Edge ecosystem of tools for on-device machine learning.

- LiteRT Torch Converter: a tool to convert PyTorch models into the `.tflite` format.
- LiteRT Torch Generative API: a library to reauthor LLMs for efficient conversion and inference.
- AI Edge Quantizer: a quantizer for advanced developers to quantize converted LiteRT models.
- LiteRT-LM: a library to efficiently run LLMs across edge platforms. The samples repository describes it as a specialized orchestration layer for running LLMs with LiteRT.
- LiteRT.js: the WebAI runtime, targeting production web applications.
- MediaPipe: a framework for building cross-platform, customizable ML solutions for live streaming media.
- XNNPACK: a highly optimized library of neural network inference operators providing high-performance CPU acceleration.

Recently added supported models are published to the Hugging Face LiteRT Community (huggingface.co/litert-community). The collections highlighted in the LiteRT README are the Gemma 4 family (various sizes, multimodal), ASR models (various, audio) and image classification models (various, vision). Developers can also contribute their own `.tflite` or `.litertlm` models through the LiteRT Hugging Face community page.

## The LiteRT samples repository

The google-ai-edge/litert-samples repository contains official and community contributed sample applications, model recipes, agent skills and utilities for LiteRT and LiteRT-LM. The samples demonstrate different API paradigms (the LiteRT CompiledModel API and the legacy Interpreter API, the Tensor API, and LiteRT-LM) and provide end-to-end model conversion and deployment pipelines.

The repository is organized in four parts. `samples/` holds runnable apps: `samples/litert/` uses the CompiledModel API, designed for modern hardware acceleration (GPU and NPU) and asynchronous execution; `samples/litert_interpreter/` holds legacy Interpreter API samples; `samples/litert_lm/` holds LLM engine samples; and `samples/tensor_api_playground/` is a browser playground. `models/` holds conversion scripts and export recipes. `utilities/` holds shared Kotlin helpers and a GPU conversion toolkit. `skills/` holds agent skills. Running a sample needs Android Studio, Xcode, Python 3.9+ with `pip install ai-edge-litert`, or a browser with WebGPU and WebAssembly support.

### Sample apps by task

Each sample directory in `samples/litert/` targets one task.

- Speech recognition with Parakeet, Moonshine, Whisper and Qwen3-ASR models through the CompiledModel API on Android.
- Text to speech with Matcha-TTS, voice-cloning text to speech with Qwen3-TTS, and streaming text to speech with KittenTTS nano (a 15M-parameter, 32 MB model with sentence-level streaming playback).
- Semantic similarity of two sentences with litert-community/embeddinggemma-300m in C++ on Linux and Android.
- Multimodal chat with Gemma 4 E2B on the Google Tensor TPU of a Pixel 10, and multimodal chat (text, image, audio) on the Qualcomm NPU.
- PhotoTalk, which classifies a photo with a LiteRT vision model and then chats about it with an LLM through LiteRT-LM.
- Image segmentation on Android (Kotlin on CPU, GPU and NPU, and C++) and iOS, image classification, digit classification, zero-shot text classification, and text-to-image generation with Bonsai Image 4B on iOS and macOS.

## The LiteRT model zoo app

The LiteRT Model Zoo is one Android app that runs 29 on-device tasks (21 vision, 8 audio) through the LiteRT CompiledModel API. Each task compiles its graph with `CompiledModel` (LiteRT 2.2.0), GPU by default; if the GPU compile fails, the task recompiles on CPU and the result card says why. Nothing is bundled: the 29 model sets (2.2 GB in total) download on demand from Hugging Face, and every file is checked against its byte size and SHA-256 before it is committed to storage.

It was tested on a Galaxy S26 (Android 16), and it builds for arm64-v8a on Android 8.0+ (minSdk 26). Measured times on the Galaxy S26 (median of 10 timed runs after warm-up) include RF-DETR Nano object detection on the GPU in 54 ms, PIDNet-S semantic segmentation in 31 ms, PP-OCRv5 OCR in 77 ms, Zipformer-small speech recognition in 80 ms, Matcha-TTS text-to-speech on GPU plus CPU in 580 ms, and Depth Anything 3 Small in 243 ms. The app uses MVVM with Jetpack Compose, and model calls run on single-thread executors because the wrappers reuse native buffers.

## Model recipes and the three artifact kinds

Each directory under `models/<family>/<model>/` in litert-samples is one recipe: the scripts that convert one model to LiteRT or LiteRT-LM, the checks that verify the result, and a README that records the toolchain, the steps and the results. A recipe produces one of three kinds of artifact.

- A `.litertlm` bundle (LLMs) runs on the LiteRT-LM engine. From a terminal: `pip install litert-lm`, then `litert-lm run model.litertlm --prompt "..."`.
- `.tflite` graphs (vision, audio, diffusion) run through the LiteRT CompiledModel API. In Python: `pip install ai-edge-litert`; the app samples have host code in Kotlin, Swift, Python and C++.
- A Tensor API program: the graph is written in C++ with the LiteRT Tensor API, with no converter.

Published recipes include MiniCPM5-2B, Hy-MT2-1.8B and Nemotron-3-Nano-4B as `.litertlm` bundles, and Bonsai Image 4B, Qwen3-TTS, SAM 3, wav2vec2 keyword spotting, Zipformer speech recognition and GLiNER2.5 Small as `.tflite` graphs.

## Agent skills for the LiteRT model lifecycle

The litert-samples `skills/` directory holds self-contained `SKILL.md` playbooks, each covering one stage of taking a model to LiteRT on device. They chain in lifecycle order: convert, quantize, verify, benchmark, build the app.

- litert-conversion-workflow: convert a Hugging Face LLM or vision-language model checkpoint into a `.litertlm` bundle with verified quality.
- gpu-clean-conversion: convert a PyTorch or Hugging Face model into a LiteRT model that runs fully on the GPU via the CompiledModel API.
- accuracy-safe-quantization: shrink a converted model with ai-edge-quantizer (fp16, int8, int4) without losing accuracy.
- on-device-verification: prove a converted or quantized model on the actual device.
- benchmark-on-ddp: measure a `.tflite` model with `benchmark_model` on Developer Device Platform lab phones.
- compiled-model-app-scaffolding: build an Android app (Kotlin, Compose) around a verified model.
- litert-compiled-model-migration: migrate an Android app from TensorFlow Lite to the CompiledModel API.
- litert-runtime and litert-lm: two basic skills in the android/skills form for an Android developer's agent, covering apps that run a `.tflite` model or an open text LLM on the CPU or the GPU.

## Building LiteRT from source

You can build LiteRT artifacts for Linux and Android (via cross-compilation) using Docker: start a Docker daemon and run `build_with_docker.sh` inside the `docker_build/` directory. The repository also documents CMake and Bazel build instructions. Inside the source tree, `c/` holds the stable, ABI-stable C APIs, `cc/` holds the public header-only C++ APIs for app developers, `kotlin/`, `python/` and `js/` hold the language bindings, `runtime/` and `compiler/` hold private implementation code, `tools/` holds benchmark and accuracy tools, and `vendors/` holds code specific to SoC vendors.

## Roadmap and recent LiteRT announcements

The LiteRT team's stated commitment is to make LiteRT the best runtime for any on-device ML deployment. The roadmap has four pillars: hardware acceleration (broadening NPU support and improving performance across all major hardware accelerators), generative AI optimizations (new features tailored for the next wave of on-device generative AI models), developer tools (better utilities for debugging, profiling, and optimizing models), and platform support (enhancing core platform support and exploring emerging ecosystems).

Recent posts from the LiteRT team and partners include "Building real-world on-device AI with LiteRT and NPU" (April 2026), "Arm and Google AI Edge optimization" (May 2026), "LiteRT Support for Intel NPUs via OpenVINO" (May 2026), "Google Tensor SDK Beta with LiteRT" (May 2026), "LiteRT.js, Google's high performance Web AI Inference" (July 2026) and "Mastering Edge AI on Raspberry Pi" with LiteRT and Gemma (August 2026).
