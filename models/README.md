# Model recipes

Each directory under `models/<family>/<model>/` is one recipe: the scripts that convert one model to [LiteRT](https://github.com/google-ai-edge/litert) or [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM), the checks that verify the result, and a README that records the toolchain, the steps and the results. The [model list](#model-list) at the end names every recipe with its published weights.

## How to use the models

A recipe produces one of three kinds of artifact; the **Artifact** column of the model list says which.

**`.litertlm` bundle (LLMs)**: runs on the [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) engine.

- Terminal: `pip install litert-lm`, then `litert-lm run model.litertlm --prompt "..."`.
- Apps: [PhotoTalk](../samples/litert/phototalk_sample_app/) (Android) and the [Qualcomm LLM chatbot](../samples/litert/qualcomm/llm_chatbot_npu/) (Android, NPU) under `samples/litert/`, and the samples under [`samples/litert_lm/`](../samples/litert_lm/).

**`.tflite` graphs (vision, audio, diffusion)**: run through the LiteRT CompiledModel API.

- Python: `pip install ai-edge-litert`. Apps: [`samples/litert/`](../samples/litert/) has the host code in Kotlin, Swift, Python and C++.
- Two recipes have their own app: [image generation](../samples/litert/image_generation/) runs Bonsai Image 4B, [text to speech](../samples/litert/text_to_speech_lm/) runs Qwen3-TTS.

**Tensor API program**: the graph is written in C++ with the [LiteRT Tensor API](https://github.com/google-ai-edge/LiteRT/tree/main/tensor), with no converter. Each `tensor_api/` directory is a complete program; its README has the Bazel build and run commands.

[LiteRT-CLI](https://github.com/google-ai-edge/LiteRT-CLI) (`litert`) downloads, converts, quantizes, runs and benchmarks models from one command.

Each model page (for example [`minicpm/minicpm5_2b/`](minicpm/minicpm5_2b/) and [`bonsai/bonsai_image_4b/`](bonsai/bonsai_image_4b/)) has the same sections in the same order: Run, Which file, a section for the model's own switches (Thinking, for MiniCPM5-2B; Steps and seed, for Bonsai Image 4B), Python, Android and iOS, Serve (`.litertlm` bundles only), Tested on, Conversion, References. The commands are the standard LiteRT-LM tools, or the model's own host script, with the model's file names filled in; a new model page copies the sections, and only the file names, the switches and the numbers change.

## Where to find examples

- [`samples/litert/`](../samples/litert/): CompiledModel API apps: speech recognition, text to speech, image generation, image segmentation and classification, PhotoTalk, Qualcomm NPU.
- [`samples/litert_interpreter/`](../samples/litert_interpreter/): Interpreter API apps for Android, iOS and Python: image classification, object detection, segmentation, audio classification.
- [`samples/litert_lm/`](../samples/litert_lm/): LiteRT-LM engine samples.
- [`samples/end_to_end/`](../samples/end_to_end/): a full pipeline from conversion to on-device classification (ImageNet).
- [`samples/web_demos/`](../samples/web_demos/) and [`samples/tensor_api_playground/`](../samples/tensor_api_playground/): browser demos; live at [google-ai-edge.github.io/litert-samples](https://google-ai-edge.github.io/litert-samples/).

## How to convert

- [conversion.md](conversion.md): the step-by-step guide from a Hugging Face checkpoint to a verified `.litertlm` bundle or `.tflite` graph. Read it before converting a new model; the recipes below are its worked examples.
- [`skills/`](../skills/): the same procedure as agent skills. [`litert-conversion-workflow`](../skills/litert-conversion-workflow/) covers LLM and VLM checkpoints; [`gpu-clean-conversion`](../skills/gpu-clean-conversion/), [`accuracy-safe-quantization`](../skills/accuracy-safe-quantization/) and [`on-device-verification`](../skills/on-device-verification/) cover `.tflite` models.
- A new recipe follows the directory layout in [conversion.md §9](conversion.md#9-publish): export, verification and repair as separately re-runnable scripts, plus a README with the versions and the gate results.

## Model list

| Recipe | Task | Artifact | Source model | Converted weights |
|---|---|---|---|---|
| [`minicpm/minicpm5_2b/`](minicpm/minicpm5_2b/) | Chat LLM with optional thinking (MiniCPM5-2B, 2.5B); runs on the CPU and GPU backends | `.litertlm` | [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) | [litert-community/MiniCPM5-2B](https://huggingface.co/litert-community/MiniCPM5-2B) |
| [`bonsai/bonsai_image_4b/`](bonsai/bonsai_image_4b/) | Text to image (Bonsai Image 4B, diffusion); three graphs and a Python host loop | `.tflite` | [prism-ml/bonsai-image-ternary-4B](https://huggingface.co/prism-ml/bonsai-image-ternary-4B) | [litert-community/Bonsai-Image-ternary-4B](https://huggingface.co/litert-community/Bonsai-Image-ternary-4B) |
| [`qwen/qwen3_tts/`](qwen/qwen3_tts/) | Text to speech (Qwen3-TTS 0.6B); three graphs and host tables, plus a Tensor API version | `.tflite`, Tensor API | [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) | [litert-community/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/litert-community/Qwen3-TTS-12Hz-0.6B-Base) |
| [`sam3/sam3_image/converted/`](sam3/sam3_image/converted/) | Detection and segmentation from a text prompt (SAM 3); three GPU-resident graphs | `.tflite` | [facebookresearch/sam3](https://github.com/facebookresearch/sam3) | [mlboydaisuke/SAM3-LiteRT](https://huggingface.co/mlboydaisuke/SAM3-LiteRT) |
| [`wav2vec2/wav2vec2_kws/`](wav2vec2/wav2vec2_kws/) | Keyword spotting (wav2vec2); two graphs | `.tflite` | [superb/wav2vec2-base-superb-ks](https://huggingface.co/superb/wav2vec2-base-superb-ks) | [litert-community/wav2vec2-keyword-spotting](https://huggingface.co/litert-community/wav2vec2-keyword-spotting) |
| [`zipformer/zipformer_ctc/`](zipformer/zipformer_ctc/) | Speech recognition (Zipformer-medium CR-CTC); one graph | `.tflite` | [Zengwei/icefall-asr-librispeech-zipformer-medium-cr-ctc-20241018](https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-medium-cr-ctc-20241018) | [litert-community/Zipformer-medium-CR-CTC-LiteRT](https://huggingface.co/litert-community/Zipformer-medium-CR-CTC-LiteRT) |
| [`sam2/sam2_hiera_tiny_video/tensor_api/`](sam2/sam2_hiera_tiny_video/tensor_api/) | Video tracking (SAM 2.1 Hiera-Tiny) | Tensor API | | |
| [`llada/llada_8b/tensor_api/`](llada/llada_8b/tensor_api/) | Denoise step of a diffusion language model (LLaDA-8B) | Tensor API | | |
| [`gemma/gemma3/`](gemma/gemma3/), [`gemma/gemma4/`](gemma/gemma4/) | Reserved for the Gemma recipes | | | |
