---
title: README
emoji: 🌖
colorFrom: yellow
colorTo: indigo
sdk: static
pinned: false
---

# LiteRT Community

[LiteRT](https://ai.google.dev/edge/litert) is Google's on-device framework for high-performance ML & GenAI deployment on edge platforms. It is the improved successor to TensorFlow Lite. On this community page, you can [find ready-to-run LiteRT models](https://huggingface.co/spaces/litert-community/find-your-models) for a wide range of ML/AI tasks. 

Within this ecosystem, [LiteRT-LM](https://ai.google.dev/edge/litert-lm) specializes in cutting edge GenAI. Recognizing that LLMs now function as complex pipelines of related models rather than single standalone models, LiteRT-LM leverages LiteRT to deliver an optimized solution for running LLMs on-device. 

Both LiteRT and LiteRT-LM can run on a variety of devices including Android, iOS, Windows, macOS, Linux, IoT and Web allow easy deployment and scaling across a diverse device landscape. 

## 🌟 Community's Picks of the Week
* **Laya Series** (`System One`): [laya-LiteRT](https://huggingface.co/litert-community/laya-LiteRT) (the English and multilingual checkpoints in token-id form) / [Laya-English-LiteRT](https://huggingface.co/litert-community/Laya-English-LiteRT) (English, with a typed-decisions fine-tune) / [Laya-Multilingual-LiteRT](https://huggingface.co/litert-community/Laya-Multilingual-LiteRT) (multilingual)
  Convai Innovations' text encoders converted for LiteRT: each answers a question defined at request time (pick an option, score on a scale, yes/no) in one forward pass.
* **[PaddleOCR-VL-1.6](https://huggingface.co/litert-community/PaddleOCR-VL-1.6)** (`Image-Text-to-Text`) High-accuracy vision-language model tailored for robust on-device document understanding and OCR tasks.
* **[decider-2b-vision-LiteRT](https://huggingface.co/litert-community/decider-2b-vision-LiteRT)** (`Image-Text-to-Text`) A compact 2B multimodal vision model converted for real-time visual reasoning and image understanding at the edge.
* **[Spark-X2.5-4B](https://huggingface.co/litert-community/Spark-X2.5-4B)** (`Text Generation`) The most popular edge LLM this week, offering strong instruction-following capabilities within a 4B footprint.
* **[Audio8-TTS-Preview-0.6b](https://huggingface.co/litert-community/Audio8-TTS-Preview-0.6b)** (`Text-to-Speech`) A tiny 0.6B parameter TTS model enabling ultra-fast, low-latency voice synthesis on resource-constrained devices.
* **[Nemotron-3-Diarization-LiteRT](https://huggingface.co/litert-community/Nemotron-3-Diarization-LiteRT)** (`Voice Activity Detection`) Optimized edge diarization model for precise multi-speaker segmentation and voice tracking.

# Community Contributions

Are we missing your favorite model? You can convert and run [PyTorch](https://github.com/google-ai-edge/litert-torch), [TensorFlow](https://ai.google.dev/edge/litert/models/convert_tf), or [JAX](https://ai.google.dev/edge/litert/models/convert_jax) models to the classic TFLite format using the LiteRT conversion and optimization tools. Or for LLMs, you can use the [LiteRT Torch Generative API](https://github.com/google-ai-edge/ai-edge-torch/tree/main/ai_edge_torch/generative). When your model is ready, join the LiteRT community org and upload the model here for others to try!
