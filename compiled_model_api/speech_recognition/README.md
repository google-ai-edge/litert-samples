# Automatic Speech Recognition Samples with LiteRT

This directory contains a collection of samples, tools, and applications
demonstrating how to utilize the **LiteRT** (formerly TensorFlow Lite) runtime
to execute state-of-the-art, open-weight **Automatic Speech Recognition (ASR)**
models on device, leveraging hardware acceleration (CPU, GPU, NPU).

## Supported Models

The tools and applications in this directory support several popular
open-weight ASR models. The table below outlines the supported backends for each model:

| Model | CPU | GPU | TPU/NPU | Description |
| :--: | :--: | :--: | :--: | :-- |
| [**Parakeet TDT**](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | ✅ | ✅ | P10 | Transducer-Delay-Transducer model |
| [**Parakeet CTC**](https://huggingface.co/nvidia/parakeet-ctc-0.6b) | ✅ | ✅ | S23/24 | Connectionist Temporal Classification model |
| [**Moonshine**](https://huggingface.co/UsefulSensors/moonshine-tiny) | ✅ | ✅ | | A lightweight, low-latency autoregressive ASR model |
| [**Whisper**](https://huggingface.co/openai/whisper-tiny) | ✅ | ✅ | | OpenAI's robust multilingual ASR and translation model |
| [**Qwen3-ASR**](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | ✅ | | | Robust multimodal/speech language model capability |

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

## Technical Details

### 1. Pipeline

The sample app builds a pipeline to reuse many components in processing audio,
recognizing speech and decoding tokens with multiple models and different
configurations. The pipeline consists of:

1.  [AudioSource](AndroidApp/app/src/main/java/com/google/ai/edge/examples/asr/AudioSource.kt)
    that reads audio data either from
    [File](AndroidApp/app/src/main/java/com/google/ai/edge/examples/asr/FileAudioSource.kt) or
    [Microphone](AndroidApp/app/src/main/java/com/google/ai/edge/examples/asr/MicrophoneAudioSource.kt)
    from microphone
2.  [AudioPreprocessor](AndroidApp/app/src/main/java/com/google/ai/edge/examples/asr/AudioPreprocessor.kt)
    that converts audio to
    [log MelSpectrogram](AndroidApp/app/src/main/java/com/google/ai/edge/examples/asr/MelSpectroProcessor.kt)
    or does [nothing](AndroidApp/app/src/main/java/com/google/ai/edge/examples/asr/DummyAudioProcessor.kt)
3.  [SpeechRecoginizer](AndroidApp/app/src/main/java/com/google/ai/edge/examples/asr/SpeechRecognizer.kt)
    that runs tflite model with
    [LiteRt](AndroidApp/app/src/main/java/com/google/ai/edge/examples/asr/LiteRtRunner.kt)
    and decoders;
    [Default](AndroidApp/app/src/main/java/com/google/ai/edge/examples/asr/LiteRtRunner.kt#L120),
    [TDT](AndroidApp/app/src/main/java/com/google/ai/edge/examples/asr/TdtDecoder.kt),
    [CTC](AndroidApp/app/src/main/java/com/google/ai/edge/examples/asr/CtcDecoder.kt).
4.  [Postprocessor](AndroidApp/app/src/main/java/com/google/ai/edge/examples/asr/Postprocessor.kt)
    that decodes tokens with
    [Huggingface tokenizer](AndroidApp/app/src/main/java/com/google/ai/edge/examples/asr/HuggingfaceTokenizer.kt),
    then [merges texts](AndroidApp/app/src/main/java/com/google/ai/edge/examples/asr/LevenshteinTokenMerger.kt)
    in consecutive sequences

### 2. Sliding window for audio chunks

None of the models supported currently are streaming models. Instead, models get
audio data with overlapped windows.

When the audio data is read from a file, each audio chunk is 5 seconds with 2
seconds overlapped with previous and next chunks. The last chunk may be padded
with zeros as silence.

When the audio data is captured from the microphone, each audio chunk is
5 seconds with 4 seconds overlapped with previous and next chunks. The first 4
chunks are pre-padded with zeros as silence which is to show texts as early as
possible.

### 3. Audio preprocessing

Except moonshine which gets raw audio as input, all models take
[Mel](https://en.wikipedia.org/wiki/Mel_scale)-[Spectrogram](https://en.wikipedia.org/wiki/Spectrogram)
data with small different configurations like number of filter banks. The sample
app defines [LogMelSpectroConfig](AndroidApp/app/src/main/java/com/google/ai/edge/examples/asr/ModelMetadataManager.kt#L24)
in [ModelMetadataConfig](AndroidApp/app/src/main/java/com/google/ai/edge/examples/asr/ModelMetadataManager.kt#L35)
to capture these configurations.

### 4. Zero copy to transfer data

Many models except Parakeet CTC are converted to multiple subgraphs to encode
audio and/then decode tokens. Encoder passes data encoded as hidden states to
decoder. The sample app passes data with zero-copy by passing buffer pointer
instead of data content in
[LiteRtRunner](AndroidApp/app/src/main/java/com/google/ai/edge/examples/asr/LiteRtRunner.kt#L157)

```
val decodeInputBuffers =
  buildList<TensorBuffer> {
    // Avoid copy. Build a list of TensorBuffers directly from the encoder output buffers.
    addAll(encodeOutputBuffers)
    // Add the decoder input buffers at the end of the list.
    add(tokenIdsBuffer)
    add(maskBuffer)
  }
```

Parakeet-TDT caches the LSTM internal states to avoid decode the same token over
and over. Passing the previous internal states to the next decode step also does
zero copy in
[TdtDecoder](AndroidApp/app/src/main/java/com/google/ai/edge/examples/asr/TdtDecoder.kt#L143)

```
private fun getInputBuffers(
  encodeOutputBuffers: List<TensorBuffer>,
  tokenIdsBuffer: TensorBuffer,
) =
  buildList<TensorBuffer> {
    // Avoid copy. Build a list of TensorBuffers directly from the encoder output buffers.
    addAll(encodeOutputBuffers)
    add(tokenIdsBuffer)
    addAll(statesBuffers[inputStatesBuffersIndex])
  }
```

### 5. Downloading models from litert-community in Huggingface

To make it simple, all the models are pre-converted and AOT-compiled for NPUs
and uploaded to [litert-community](https://huggingface.co/litert-community)
in Huggingface. For production app, it would be better to utilize
[AI packs](https://developer.android.com/google/play/on-device-ai)
to deliver models.

### 6. Merging texts in consecutive sequences

As audio chunks are overlapped, the tokens detected by SpeechRecognizer are also
duplicated. To remove duplicates from consecutive token sequences, tokens in a
sequence should be aligned to ones in the previous sequence.

If SpeechRecognizer outputs timestamps along with tokens, like Parakeet-TDT and
Parakeet-CTC, the alignment is straightforward.

If timestamps are not output, a heuristic based on
[Levenshtein edit distance](https://en.wikipedia.org/wiki/Levenshtein_distance)
and the number of words/tokens are used to figure out the best alignment.

Once aligned, words are merged. Assuming that the words close to the start or
the end of the sequence are less correct than others, first N words are from the
previous sequence while last M words are from the next sequence.
