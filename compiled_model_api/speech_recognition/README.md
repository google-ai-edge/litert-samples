# Automatic Speech Recognition Samples with LiteRT

This directory contains a collection of samples, tools, and applications
demonstrating how to utilize the **LiteRT** (formerly TensorFlow Lite) runtime
to execute state-of-the-art, open-weight **Automatic Speech Recognition (ASR)**
models on device, leveraging hardware acceleration (CPU, GPU, NPU).

---

## Supported Models

The tools and applications in this directory support several popular
open-weight ASR models:

*   [**Parakeet TDT**](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3):
    Transducer-Delay-Transducer model.
*   [**Parakeet CTC**](https://huggingface.co/nvidia/parakeet-ctc-0.6b):
    Connectionist Temporal Classification model.
*   [**Moonshine**](https://huggingface.co/UsefulSensors/moonshine-tiny):
    A lightweight, low-latency autoregressive ASR model.
*   [**Whisper**](https://huggingface.co/openai/whisper-tiny):
    OpenAI's robust multilingual ASR and translation model.
*   [**Qwen3-ASR**](https://huggingface.co/Qwen/Qwen3-ASR-0.6B):
    Robust multimodal/speech language model capability.

---

## Directory Structure & Components

The workspace is organized into three main components:

```
samples/asr/
├── AndroidApp/       # Android demo application (Java/Kotlin & LiteRT SDK)
└── convert/          # Python conversion and verification pipelines
```

### 1. AndroidApp

A Gradle-based Android demo application demonstrating how to integrate the
LiteRT Android SDK.

*   **Model Support:** On-device execution of supported ASR models.
*   **Accelerators:** Hardware accelerator delegation to execute models on CPU,
    GPU, and NPU (Google Tensor, Qualcomm Snapdragon, or MediaTek MTK).
*   **Feature Processing & Decoding:** Implements spectrogram audio feature
    extractors, custom decoders (e.g., `CtcDecoder`, `TdtDecoder`), and a JNI
    Hugging Face tokenizer.

Notes:

*   NPU runtime libs are commented out in
    [AndroidApp/app/build.gradle.kts](AndroidApp/app/build.gradle.kts#L51).
    Uncomment one of them to enable TPU/NPU support.
*   Only Parakeet TDT supports Google Tensor on Pixel 10.
*   Only Parakeet CTC supports Qualcomm Snapdragon on S23 and S24 families
*   Qwen3-ASR supports only CPU inference.

### 2. convert

A suite of Python utilities to prepare, convert, and verify PyTorch and
Hugging Face model weights for LiteRT inference on edge devices.

*   `convert_to_tflite.py`: Converts PyTorch models to stateful or stateless
    `.tflite` format, supporting Dynamic Range Quantization (DRQ).
*   `compile_for_npu.py`: Compiles converted `.tflite` models into NPU
    binaries targeting Qualcomm Snapdragon/QNN or MediaTek MTK processors.
*   `verify_model.py`: Validates re-authored PyTorch models against reference
    Hugging Face models with sample wav inputs.
*   `verify_tflite.py`: Verifies local `.tflite` execution against the
    reference PyTorch model.
