# **Google AI Edge LiteRT Samples**

This repository contains official and community contributed sample applications, model recipes, agent skills and utilities for **[LiteRT](https://github.com/google-ai-edge/litert)** (formerly known as TensorFlow Lite), Google's open source, high-performance on-device machine learning framework and **[LiteRT-LM](https://github.com/google-ai-edge/litert-lm)**, a specialized orchestration layer for running LLMs with LiteRT, unlocking maximum performance and efficiency.

**Note** Please access the interactive web page with a collections of demos there at: [https://google-ai-edge.github.io/litert-samples/](https://google-ai-edge.github.io/litert-samples/)

The samples demonstrate different API paradigms (LiteRT CompiledModel API and legacy Interpreter API, Tensor API, LiteRT-LM) and provide end-to-end model conversion and deployment pipelines.

---

## **🔥 What's New**

* 🐱 **Streaming TTS (KittenTTS nano)**: Added a tiny (15M-param, 32 MB) streaming text-to-speech Android sample — dynamic-length LiteRT graphs, sentence-level streaming playback, live TTFA/RTF metrics ([`samples/litert/text_to_speech_streaming/`](samples/litert/text_to_speech_streaming/)).
* 🎙️ **Speech Recognition (ASR)**: Added end-to-end [Automatic Speech Recognition sample](samples/litert/speech_recognition) using the CompiledModel API.
* 📸 **PhotoTalk Sample App**: Added multimodal sample app combining LiteRT vision processing with LiteRT-LM audio/text generation ([`samples/litert/phototalk_sample_app/`](samples/litert/phototalk_sample_app/)).
* 🗣️ **Qwen3-TTS & Qwen3 ASR**: Added model recipes, conversion scripts, and Tensor API implementations for [Qwen3-TTS](models/qwen/qwen3_tts/) and [Qwen3 ASR](models/qwen/qwen_asr/).
* 🎨 **Bonsai Image 4B**: Added text-to-image diffusion model sample with Python inference and conversion tools ([`models/bonsai/bonsai_image_4b/`](models/bonsai/bonsai_image_4b/)).
* 🤖 **Agent Skills & Utilities**: Added four lifecycle agent skills ([`skills/`](skills/): conversion, quantization, on-device verification, app scaffolding), a GPU conversion toolkit ([`utilities/litert_gpu_toolkit/`](utilities/litert_gpu_toolkit/)), and shared Kotlin helpers ([`utilities/common/`](utilities/common/)).

---

## **📂 Repository Structure**

### **1. `samples/` — Application Samples**

All runnable sample applications and interactive playgrounds are organized under `samples/`:

* **`samples/litert/`**: Standard samples using the **LiteRT CompiledModel API**. Designed for modern hardware acceleration (GPU/NPU) and asynchronous execution.
  * *Samples:* Speech Recognition, PhotoTalk, Text-to-Speech, Image Segmentation, Image Classification, Digit Classification, Qualcomm NPU acceleration (Gemma, MobileNet, Fast VLM), Google TPU sample app.
* **`samples/litert_interpreter/`**: Legacy samples using the **Interpreter API**.
  * *Samples:* Broad compatibility examples for Android, iOS, and Python (Image Classification, Object Detection, Image Segmentation, Audio Classification).
* **`samples/litert_lm/`**: High-level Engine samples for Large Language Models (LLM/SLM).
* **`samples/end_to_end/`**: Complete full-system pipelines (e.g. ImageNet model conversion, preprocessing, and classification).
* **`samples/tensor_api_playground/`**: Interactive Web/WASM playground demonstrating LiteRT Tensor API capabilities directly in the browser (Gemma 3, Image Segmentation, Mandelbrot, Game of Life).

### **2. `models/` — Model Recipes & Export Pipelines**

Contains standalone model conversion scripts, export recipes, and model-specific utilities. Many are working in process.

### **3. `utilities/` — Shared Tools & Helper Scripts**

* **`utilities/common/`**: Shared Kotlin helpers for Android samples (camera pipeline, audio capture, CompiledModel runner, image/tensor and math helpers).
* **`utilities/litert_gpu_toolkit/`**: Pre-conversion patches that rewrite common PyTorch patterns into forms the LiteRT GPU delegate accepts, plus a post-conversion checker.

### **4. `skills/` — Agent Automation & Skills**

Custom AI agent skills that carry a model through the LiteRT deployment lifecycle, in order — see [`skills/README.md`](skills/README.md) for the full index:

* [`gpu-clean-conversion/`](skills/gpu-clean-conversion/): PyTorch / Hugging Face model → GPU-resident LiteRT model.
* [`accuracy-safe-quantization/`](skills/accuracy-safe-quantization/): Quantize (fp16 / int8 / int4) without losing accuracy.
* [`on-device-verification/`](skills/on-device-verification/): Prove the converted model on the actual device.
* [`compiled-model-app-scaffolding/`](skills/compiled-model-app-scaffolding/): Build an Android app around the verified model.

---

## **🛠️ Getting Started**

### **Prerequisites**

* **Android**: Android Studio (latest stable version).
* **iOS**: Xcode (latest version).
* **Python**: Python 3.9+ and `pip install ai-edge-litert`.
* **Web / WASM**: Modern browser with WebGPU / WebAssembly support.

### **Running a Sample**

#### **For Samples Using Compiled Model API**

1. Navigate to `samples/litert/<sample_name>`.
2. Ensure you have a device with a supported NPU/GPU (e.g., modern Pixel, Samsung, or Qualcomm/MediaTek devices).
3. Follow the specific setup instructions in the sample's `README.md`.

#### **For Web / Tensor API Playground**

Visit ["Interactive Web"](https://google-ai-edge.github.io/litert-samples/) tab, or open
 `samples/tensor_api_playground/index.html` or run `index.html` at the repository root
 via a local HTTP server.

---

## **📚 Documentation**

* **LiteRT Overview**: [ai.google.dev/edge/litert](https://ai.google.dev/edge/litert)
* **LiteRT-LM Overview**:   [ai.google.dev/edge/litert-lm](https://ai.google.dev/edge/litert-lm)
* **CompiledModel API Guide**: [LiteRT for Android](https://ai.google.dev/edge/litert/android)
* **Model Conversion**: [Convert models to LiteRT](https://ai.google.dev/edge/litert/conversion/overview)
* **LiteRT CLI**: [CLI for end-to-end journey](https://github.com/google-ai-edge/LiteRT-CLI)

---

## **🤝 Contributing**

Contributions are welcome!

1. Read [CONTRIBUTING.md](https://github.com/google-ai-edge/litert-samples/blob/main/CONTRIBUTING.md).
2. Fork the repo and create a branch.
3. Submit a Pull Request.

---

## **📄 License**

Apache License 2.0. See [LICENSE](https://github.com/google-ai-edge/litert-samples/blob/main/LICENSE) for details.

---

*Disclaimer: This is a sample repository maintained by Google. It is provided "as is" without warranty of any kind.*
