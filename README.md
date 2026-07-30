# **Google AI Edge LiteRT Samples**

This repository contains official sample applications, model recipes, and code examples for **LiteRT** (formerly known as TensorFlow Lite), Google's high-performance on-device machine learning framework.

The samples demonstrate different API paradigms (CompiledModel API, Interpreter API, Tensor API, LiteRT-LM) and provide end-to-end model conversion and deployment pipelines.

**Note:** For Generative AI and Large Language Models (LLMs), please also refer to the [LiteRT-LM repository](https://github.com/google-ai-edge/LiteRT-LM).

---

## **🔥 What's New**

* 🗣️ **Qwen3-TTS & Qwen3 ASR**: Added model recipes, conversion scripts, and Tensor API implementations for [Qwen3-TTS](models/qwen/qwen3_tts/) and [Qwen3 ASR](models/qwen/qwen_asr/).
* 💎 **Gemma 3 & Gemma 4**: Added model recipes, conversion pipelines, and Qualcomm NPU acceleration examples for [Gemma 3](models/gemma/gemma3/) and [Gemma 4](models/gemma/gemma4/).
* 🎨 **Bonsai Image 4B**: Added text-to-image diffusion model sample with Python inference and conversion tools ([`models/bonsai/bonsai_image_4b/`](models/bonsai/bonsai_image_4b/)).
* 📸 **PhotoTalk Sample App**: Added multimodal sample app combining LiteRT vision processing with LiteRT-LM audio/text generation ([`samples/litert/phototalk_sample_app/`](samples/litert/phototalk_sample_app/)).
* 🎙️ **Speech Recognition (ASR)**: Added end-to-end [Automatic Speech Recognition sample](samples/litert/speech_recognition) using the CompiledModel API.
* 🤖 **Agent Skills & Utilities**: Added custom agent skills ([`skills/gpu-clean-conversion/`](skills/gpu-clean-conversion/)) and shared Kotlin helpers ([`utilities/common/`](utilities/common/)).

---

## **📂 Repository Structure**

### **1. `samples/` — Application Samples**

All runnable sample applications and interactive playgrounds are organized under `samples/`:

* **`samples/litert/`**: Standard samples using the **LiteRT CompiledModel API**. Designed for modern hardware acceleration (GPU/NPU) and asynchronous execution.
  * *Samples:* Speech Recognition, PhotoTalk, Text-to-Speech, Image Segmentation, Image Classification, Digit Classification, Qualcomm NPU acceleration (Gemma, MobileNet, Fast VLM), Google TPU sample app.
* **`samples/litert_interpreter/`**: Legacy samples using the **Interpreter API**.
  * *Samples:* Broad compatibility examples for Android, iOS, and Python (Image Classification, Object Detection, Image Segmentation, Audio Classification).
* **`samples/litert_lm/`**: High-level Engine samples for Large Language Models (LLM/SLM) with KV-Cache management and streaming decoding.
* **`samples/end_to_end/`**: Complete full-system pipelines (e.g. ImageNet model conversion, preprocessing, and classification).
* **`samples/tensor_api_playground/`**: Interactive Web/WASM playground demonstrating LiteRT Tensor API capabilities directly in the browser (Gemma 3, Image Segmentation, Mandelbrot, Game of Life).

### **2. `models/` — Model Recipes & Export Pipelines**

Contains standalone model conversion scripts, export recipes, and model-specific utilities:

* **`models/gemma/`**: Model recipes, conversion scripts, and instructions for **Gemma 3** and **Gemma 4**.
* **`models/qwen/`**: Conversion recipes and Tensor API implementations for **Qwen3**, **Qwen3-TTS**, and **Qwen3 ASR**.
* **`models/bonsai/`**: Text-to-image diffusion model recipes and Python execution scripts for **Bonsai Image 4B**.

### **3. `utilities/` — Shared Tools & Helper Scripts**

* **`utilities/common/`**: Shared Kotlin utilities (`CompiledModelRunner`, `RealtimeCameraPipeline`, `ImageTensor`, `AudioCapture`, `MathOps`) and drift check scripts.
* **`utilities/tools/`**: Common conversion utilities and build tools.

### **4. `skills/` — Agent Automation & Skills**

* Custom AI Agent skills and interactive workflows (e.g., `gpu-clean-conversion`) to streamline model conversion and deployment for LiteRT.

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

Open `samples/tensor_api_playground/index.html` or run `index.html` at the repository root via a local HTTP server.

---

## **📚 Documentation**

* **LiteRT Overview**: [ai.google.dev/edge/litert](https://ai.google.dev/edge/litert)
* **CompiledModel API Guide**: [LiteRT for Android](https://ai.google.dev/edge/litert/android)
* **Model Conversion**: [Convert models to LiteRT](https://ai.google.dev/edge/litert/conversion/overview)

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
