# LiteRT Core Samples

Basic usage samples and end-to-end demonstrations for LiteRT APIs.

## Overview

- **CompiledModel API**: Code examples in C++, C, Python, Kotlin, Swift, and Rust.
- **End-to-End Demos**: Showcases for ASR, TTS, Vision, and other model capabilities.
- **Feature Demos**: Miscellaneous demonstrations highlighting LiteRT features.

## Samples

Each row is one sample directory; the Model column says where the sample gets its model files, with a link for every model hosted in [litert-community](https://huggingface.co/litert-community). Each sample's own README has the build and run steps.

| Sample | Task | API | Platform | Model |
|---|---|---|---|---|
| [`digit_classifier/`](digit_classifier/) | Handwritten digit classification | CompiledModel API | Android | an MNIST `.tflite`, downloaded by Gradle |
| [`google/sample_app_tpu/`](google/sample_app_tpu/) | Multimodal chat with an LLM on the Google Tensor TPU | LiteRT-LM | Android (Pixel 10) | [litert-community/gemma-4-E2B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) (`gemma-4-E2B-it_Google_Tensor_G5.litertlm`) |
| [`image_classification/`](image_classification/) | Image classification from the camera | CompiledModel API | Android | `efficientnet_lite0.tflite` and `efficientnet_lite2.tflite`, downloaded by Gradle |
| [`image_generation/`](image_generation/) | Text to image (Bonsai Image 4B) | CompiledModel API | iOS, macOS | [litert-community/Bonsai-Image-ternary-4B](https://huggingface.co/litert-community/Bonsai-Image-ternary-4B) |
| [`image_segmentation/`](image_segmentation/) | Image segmentation | CompiledModel API | Android (Kotlin on CPU/GPU and NPU; C++), iOS | Kotlin, CPU/GPU and the NPU variants' fallback: [litert-community/MediaPipe-Selfie-Segmentation](https://huggingface.co/litert-community/MediaPipe-Selfie-Segmentation), downloaded by Gradle; Kotlin NPU: per-SoC models through an AI Pack; C++: `models/selfie_multiclass_256x256*.tflite` in the tree; iOS: `selfie_multiclass_256x256.tflite` from [Kaggle](https://www.kaggle.com/models/google/mediapipe/tfLite/selfie-multiclass-256x256) |
| [`phototalk_sample_app/`](phototalk_sample_app/) | Classify a photo, then chat about it with an LLM | CompiledModel API and LiteRT-LM | Android | classifier: `efficientnet_lite0.tflite`, downloaded by Gradle; LLM: [litert-community/gemma-4-E2B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm), [litert-community/Gemma3-1B-IT](https://huggingface.co/litert-community/Gemma3-1B-IT), [litert-community/FastVLM-0.5B](https://huggingface.co/litert-community/FastVLM-0.5B) |
| [`qualcomm/gemma3/cpu_gpu/`](qualcomm/gemma3/cpu_gpu/) | Chat with Gemma 3 1B | LiteRT-LM | Android (CPU, GPU) | [litert-community/Gemma3-1B-IT](https://huggingface.co/litert-community/Gemma3-1B-IT) (`gemma3-1b-it-int4.litertlm`) |
| [`qualcomm/gemma3/npu/`](qualcomm/gemma3/npu/) | Chat with Gemma 3 1B on the Qualcomm NPU | LiteRT-LM | Android (Qualcomm NPU) | [litert-community/Gemma3-1B-IT](https://huggingface.co/litert-community/Gemma3-1B-IT) (`Gemma3-1B-IT_q4_ekv1280_sm8750.litertlm`) |
| [`qualcomm/fast_vlm/`](qualcomm/fast_vlm/) | Chat with FastVLM 0.5B | LiteRT-LM | Android (CPU, GPU, Qualcomm NPU) | `FastVLM-0.5B.litertlm` or `FastVLM-0.5B.sm8850.litertlm`, pushed to `/data/local/tmp/fastvlm` |
| [`qualcomm/gemma4/cpu_gpu/`](qualcomm/gemma4/cpu_gpu/) | Multimodal chat with Gemma | LiteRT-LM | Android | [litert-community/gemma-4-E2B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) (`gemma-4-E2B-it.litertlm`) through the in-app downloader, or `gemma3-1b-it-int4.litertlm` pushed with adb, the README's default |
| [`qualcomm/llm_chatbot_npu/`](qualcomm/llm_chatbot_npu/) | Multimodal chat (text, image, audio) on the Qualcomm NPU | LiteRT-LM | Android (Qualcomm NPU) | [litert-community/gemma-4-E2B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) (`gemma-4-E2B-it_qualcomm_sm8750.litertlm`), [litert-community/FastVLM-0.5B](https://huggingface.co/litert-community/FastVLM-0.5B) (`FastVLM-0.5B.qualcomm.sm8750.litertlm`) |
| [`qualcomm/mobilenet_v2/`](qualcomm/mobilenet_v2/) | Image classification on the Qualcomm NPU (JIT) | CompiledModel API | Android (Galaxy S25 Ultra for the NPU; an emulator for CPU) | Qualcomm's MobileNet-v2 float `.tflite`, downloaded by Gradle ([qualcomm/MobileNet-v2](https://huggingface.co/qualcomm/MobileNet-v2)) |
| [`qualcomm/object_detection/efficientdet/kotlin_npu/`](qualcomm/object_detection/efficientdet/kotlin_npu/) | Object detection from the camera on the Qualcomm NPU | CompiledModel API | Android (Qualcomm NPU) | `efficientdet_lite0_detection.tflite` (TF Hub EfficientDet-Lite0) in the app's assets |
| [`semantic_similarity/`](semantic_similarity/) | Semantic similarity of two sentences | LiteRT (C++) | Linux, Android | [litert-community/embeddinggemma-300m](https://huggingface.co/litert-community/embeddinggemma-300m) |
| [`speech_recognition/`](speech_recognition/) | Speech recognition (Parakeet TDT, CTC and TDT-CTC ja; Moonshine; Whisper; Qwen3-ASR) | CompiledModel API | Android; Python for conversion | [litert-community/parakeet-tdt-0.6b-v3](https://huggingface.co/litert-community/parakeet-tdt-0.6b-v3), [parakeet-ctc-0.6b](https://huggingface.co/litert-community/parakeet-ctc-0.6b), [parakeet-tdt_ctc-0.6b-ja](https://huggingface.co/litert-community/parakeet-tdt_ctc-0.6b-ja), [moonshine-tiny](https://huggingface.co/litert-community/moonshine-tiny), [whisper-tiny](https://huggingface.co/litert-community/whisper-tiny), [Qwen3-ASR-0.6B](https://huggingface.co/litert-community/Qwen3-ASR-0.6B) |
| [`text_classification/`](text_classification/) | Text classification | CompiledModel API | Android | `bert_classifier.tflite` and `average_word_classifier.tflite`, downloaded by Gradle |
| [`text_to_speech/`](text_to_speech/) | Text to speech (Matcha-TTS) | CompiledModel API | Android | [litert-community/Matcha-TTS](https://huggingface.co/litert-community/Matcha-TTS), or built by `conversion/` |
| [`text_to_speech_lm/`](text_to_speech_lm/) | Text to speech with voice cloning (Qwen3-TTS) | CompiledModel API | Python, Android | [litert-community/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/litert-community/Qwen3-TTS-12Hz-0.6B-Base) |
| [`text_to_speech_streaming/`](text_to_speech_streaming/) | Streaming text to speech (KittenTTS nano) | CompiledModel API (G2P, vocoder) and Interpreter API (predictor, prosody) | Android, Python | built by `conversion/` from KittenTTS nano; the G2P graph from [litert-community/Matcha-TTS](https://huggingface.co/litert-community/Matcha-TTS) |

[`colab/`](colab/) holds the AOT compilation tutorial notebook.
