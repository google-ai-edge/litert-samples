---
title: Converting and quantizing models for LiteRT
source: https://github.com/google-ai-edge/litert-torch/blob/5a31633a9118b122c2ada01b136d470a714b3550/README.md ; https://github.com/google-ai-edge/litert-torch/blob/5a31633a9118b122c2ada01b136d470a714b3550/docs/pytorch_converter/README.md ; https://github.com/google-ai-edge/litert-torch/blob/5a31633a9118b122c2ada01b136d470a714b3550/litert_torch/generative/README.md ; https://github.com/google-ai-edge/litert-torch/blob/5a31633a9118b122c2ada01b136d470a714b3550/litert_torch/generative/quantize/README.md ; https://github.com/google-ai-edge/litert-samples/blob/367ddbdee22a8d73fba6e519c4a3f18d19008b05/models/conversion.md ; https://github.com/google-ai-edge/litert-samples/blob/367ddbdee22a8d73fba6e519c4a3f18d19008b05/skills/litert-conversion-workflow/SKILL.md ; https://github.com/google-ai-edge/litert-samples/blob/367ddbdee22a8d73fba6e519c4a3f18d19008b05/skills/gpu-clean-conversion/SKILL.md ; https://github.com/google-ai-edge/litert-samples/blob/367ddbdee22a8d73fba6e519c4a3f18d19008b05/skills/accuracy-safe-quantization/SKILL.md ; https://github.com/google-ai-edge/LiteRT/blob/bbb1312f51e37cab695f7ec3a3c74b8a4a9ca13a/litert/python/tools/mixed_precision/README.md ; https://github.com/google-ai-edge/LiteRT/blob/bbb1312f51e37cab695f7ec3a3c74b8a4a9ca13a/docs/instructions/LARGE_TFLITE_MODELS.md
license: Apache-2.0
---

# Converting and quantizing models for LiteRT

This document covers how models get into LiteRT formats: converting PyTorch models to `.tflite` with LiteRT Torch, authoring LLMs with the Generative API, the workflow for turning a Hugging Face LLM into a verified `.litertlm` bundle, making vision and audio models run cleanly on the GPU, quantization schemes from fp16 to int4, mixed precision, and models larger than the 2 GB FlatBuffer limit.

## Two conversion lanes, .litertlm bundles and .tflite graphs

The litert-samples model conversion cookbook splits models into two lanes by what the model is. A text LLM or vision-language model becomes one `.litertlm` bundle that runs on the LiteRT-LM engine. Anything else (vision, audio, diffusion, encoders) becomes one or more `.tflite` graphs plus a host loop, run through the LiteRT CompiledModel API.

Everything in the cookbook was hit on a real model, and where a rule names a runtime version it is a dated fact to re-test on each release. A conversion is finished when four things hold, in order: the bundle loads and generates through the LiteRT-LM engine, not only through the raw interpreter; output quality is gated against the source model with a floor gate plus a task-level parity check, not a smoke test; it holds on the deployment path it claims, meaning the target backend, the target device and a multi-turn conversation; and the record names the device, the backend, the runtime version and the toolchain versions.

## LiteRT Torch, converting PyTorch models

LiteRT Torch is a Python library that supports converting PyTorch models into the `.tflite` format, which can then be run with LiteRT, enabling applications for Android, iOS and IoT that run models completely on-device. It offers broad CPU coverage, with initial GPU and NPU support, and builds on top of `torch.export()` with good coverage of Core ATen operators. The PyTorch converter is a Beta release, while the Generative API in the same package is an Alpha release.

It requires Python 3.10 or newer (Python 3.11 is highly recommended), Linux, and PyTorch 2.4.0 or newer, and installs with `pip install litert-torch` (or `litert-torch-nightly` for nightly builds).

### Converting a PyTorch model to a .tflite file

`litert_torch.convert()` converts a PyTorch model to an on-device (edge) model. It requires sample inputs for tracing and shape inference, passed as a tuple. The source model must be compliant with `torch.export`; `convert` expects a `torch.nn.Module` whose `forward` function receives tensors as positional arguments and returns tensors, and it does not support keyword arguments, so other interfaces need a small wrapper module. The model should be in evaluation mode.

```python
import litert_torch
edge_model = litert_torch.convert(resnet18.eval(), (torch.randn(1, 3, 224, 224),))
edge_model.export("resnet18.tflite")
```

Before deployment, the outputs of PyTorch and the edge model can be compared in Python as a smoke check, for example with `np.allclose` at an absolute tolerance of 1e-5. A serialized model can be imported back with `litert_torch.load`. Once exported, the model runs on-device through the LiteRT CompiledModel API across CPU, GPU and NPU.

### Signatures, channel-last inputs and debugging conversion failures

Multi-signature conversion puts several PyTorch modules into one edge model, which is useful when components share weights; each signature is then run by name. `litert_torch.to_channel_last_io` wraps a model so that inputs and outputs use the channel-last NHWC layout that mobile image pipelines expect, instead of PyTorch's usual NCHW.

Conversion can fail in `torch.export` (the fix is to make the model source torch-exportable) or while lowering the exported program to an edge model. For the second case, `litert_torch.debug.find_culprits` takes the same arguments as `convert` and generates a minimal PyTorch program that reproduces the failure, suitable for a GitHub issue; it overwrites weights and inputs with random values. Converted models can be inspected visually with Model Explorer (`pip install ai-edge-model-explorer`). LiteRT Torch uses a modern conversion backend, with the legacy Torch XLA backend available through `USE_TORCH_XLA=1` for compatibility issues.

## The LiteRT Torch Generative API for LLMs

The Generative API is a Torch-native library for authoring mobile-optimized PyTorch transformer models such as Gemma, TinyLlama and Phi-2 from building blocks, which can be converted to LiteRT-LM models. It is described as v0.1, an early developer preview whose API is unstable. The workflow is: start with a trained PyTorch LLM; re-author it with the Edge Generative API layers; quantize it, which is critical for reducing model size and achieving reasonable performance; verify quality; convert it to a LiteRT FlatBuffer; and deploy an end-to-end inference pipeline. A converted `.tflite` model can be packaged with a tokenizer into a deployment-ready `.litertlm` container using the `litert-lm-builder` CLI tool.

LLMs are usually exported with two signatures, `prefill` and `decode`, which differ only in argument shapes; multiple prefill signatures named `prefill_{SEQ-LENS}` let the runtime use the one closest to the input length. The Generative API currently supports CPU and GPU, with planned support for NPU. A known issue is that conversion keeps multiple copies of the weights in memory, so it needs a powerful Linux workstation or cloud instance with at least 32 GB of RAM.

### What a full LLM pipeline needs besides the model

The model files typically only perform the core ML computation. A text generation pipeline also needs a tokenizer that converts text to integers, the `prefill` signature that ingests the input tokens, the `decode` signature invoked to obtain logits, a sampler that selects a token from the logits in an autoregressive loop, and a detokenizer that maps generated tokens back to text. Calling LiteRT runtime APIs directly gives the most control, for example for streaming, memory control, constrained grammar decoding or speculative decoding. The high-level MediaPipe LLM Inference API is the alternative that takes care of the pipeline with a prompt-in, prompt-out interface.

Many PyTorch models ship BPE tokenizers rather than SentencePiece model files. A conversion script can build a SentencePiece model from tokenizer config files, but the result does not always output the same token IDs: for Llama 3.2 it mismatched the original BPE tokenizer on 35 of 1000 pairs strictly (3.5 percent) and 9 of 1000 loosely (0.9 percent).

## Converting a Hugging Face LLM into a .litertlm bundle

The litert-conversion-workflow skill and the cookbook begin by classifying the architecture from `config.json` (`model_type`, `layer_types`, MoE fields, size) before running anything. Dense decoders (Llama, Qwen, Phi, Mistral, Gemma, SmolLM and their finetunes) work and take the standard lane. Reasoning models work but need template and thought-channel care. Interleaved hybrids with SSM, linear-attention or short-conv layers run on the CPU backend since litert-lm 0.15 with a state-aware export. Parallel hybrids, mixture-of-experts models and MLA latent-KV models (the DeepSeek family) are currently blocked. Vision-language models work when the decoder is convertible and the vision tower can be made static, with one image per prompt. Pre-quantized MXFP4 or FP8 checkpoints are not a wall, but a GPTQ int4 checkpoint must never be re-quantized.

The toolchain must be pinned and recorded. The MiniCPM5-2B bundles were built on `litert-torch` 0.9.3, `litert-converter` 0.4.0, `ai-edge-quantizer` 0.9.0, `litert-lm-builder` 0.16.1 and `transformers` 5.14.1, with `litert-lm` 0.16 or newer as the runtime.

### The chat template, the most common ship-killer

More conversions have died on the chat template than on any numerical issue. The runtime renders templates with a Rust minijinja build, with no Python, so a vendor template that calls `.get()`, `.strip()`, `.startswith()` or `.split()` imports fine and crashes on the user's first message with "unknown method: map has no method named get". The default is to embed no Jinja at all: export with `use_jinja_template=False` and a minimal ChatML-style template, so the bundle carries plain prefix and suffix markers.

At each message the engine renders the history and requires the new render to extend the previous one as a string, prefilling only the new suffix. A template that rewrites history, typically a reasoning template that strips think blocks from past turns, dies at turn two. Tokenizers need equal care: special tokens added beyond the base vocabulary get dropped by SentencePiece conversion, and every turn-end token must be declared as a stop token, or the literal end-of-turn text leaks into every reply.

### Quality gates before publishing a bundle

First inspect the bundle with `python -m litert_lm_builder.litertlm_peek_main --litertlm_file model.litertlm`, checking the template, stop tokens, thought channel and sections, then confirm the engine answers one prompt on the CPU with `litert-lm run`. The gates then run in order, CPU first and then the shipping backend. The floor gate is eight fixed questions (such as "What is 17 + 25?" expecting 42, and the capital of Japan expecting Tokyo), passing at 6 of 8 with no degenerate answers; it catches collapse but is never a parity verdict. Task parity runs a benchmark such as GSM8K with at least 100 items against the source model, with reasoning models given at least 2048 output tokens. A first-token length sweep catches state-carrying models that corrupt at specific prompt lengths, and a three-turn multi-turn gate catches template contract violations.

On the GPU, the executor keeps activations in fp16 unless the bundle declares `prefer_activation_type = "fp32"` in its `model.toml`; that declaration costs about 15 percent of GPU decode speed and must be decided per file. Gate with `--cache no`, never quote a first-run number, and budget about twice the model size of disk and RAM for the first GPU load.

## Converting vision and audio models for a clean GPU run

For `.tflite` models, the gpu-clean-conversion skill defines done as three things in order: it converts, every node runs on the GPU, and the on-device output matches the source model. Full residency does not imply correct numbers. The loop is to convert plain first with no patches, check the model through the CompiledModel API with `check_gpu_compatibility` from the litert-samples GPU toolkit (which compiles for the GPU, runs every signature on random inputs and compares against a CPU reference), map each failure to a rewrite, and repeat.

Common rewrites include: `GATHER_ND` errors from stride-2 slicing, `grid_sample`, bicubic interpolation or reflect padding; rank-5 and higher tensors from `PixelShuffle`, windowed attention or `einops.rearrange`; `SELECT` from PReLU, ELU or `torch.where` masking, replaced with arithmetic; and runtime `BROADCAST_TO`, as in grouped-query attention's `repeat_kv`, replaced with an exact concatenation. Output that is wrong or NaN usually comes from the fp16 reduction family, triggered when an fp16 accumulator passes 65504; patches exist for safe LayerNorm, RMSNorm and instance norm. `.chunk()` lowers to a GPU-rejected `SPLIT`, so slice directly instead.

## Choosing a quantization scheme

The accuracy-safe-quantization skill uses `ai-edge-quantizer` (`pip install ai-edge-quantizer`) and says to start with the lightest recipe that meets the size budget, moving down only on evidence.

- About 2× smaller with zero risk: fp16 float-casting. Weights are cast to fp16 and compute stays float.
- About 4× smaller, for encoders, conv nets and diffusion blocks: dynamic-range int8 channelwise, with int8 weights and float activations, a shape that rides the GPU delegate.
- When dynamic int8 loses quality: weight-only quantization at the same bits, which inserts an explicit dequantize so the matmul runs in float.
- About 7× smaller, for LLMs and autoregressive decoders: int4 blockwise-32 with OCTAV clipping, with embeddings kept at int8. Never channelwise for a decoder: it looks fine on short outputs and degenerates over long generations.
- When data-free int4 still fails: ingest a calibrated GPTQ checkpoint with dequantized weight recovery.

Full-integer static quantization (`static_wi8_ai8`, quantized activations, calibration data required) is a separate lane aimed at NPU and AOT targets. The quantizer also ships presets such as `dynamic_wi8_afp32()`, `dynamic_wi4b32_afp32()` and `weight_only_wi8_afp32()`.

### Quantizing LLM decoders for phones

For decoders the cookbook recommends int8 dynamic as the safe default, which often beats data-free int4 on quality and prefills faster on the CPU backend. For the phone file up to about 3B parameters it recommends int4 blockwise-32 with OCTAV on linears and an int8 embedding, the fastest GPU decode. Around 4B, or within an iPhone section budget, int4 blockwise-128 gives lighter dequantization and a smaller section, although math and reasoning models want block-32, which is also the faster kernel on Apple GPUs. Decoders under about 0.5B should stay fp16, because int4 and even dynamic int8 have corrupted task output at 0.3B.

Weight-only recipes are CPU-only bundles on the released GPU delegates, which reject the dequantized weight at engine creation. After every blockwise int4 export, scan for zero scales: all-zero weight rows produce scale 0, which the CPU backend refuses to load while the GPU accepts the file silently. CPU and GPU also invert: on the CPU int8 beats int4 on prefill and quality, while on the GPU int4 prefills several times faster at similar decode speed.

### Quantization in the Generative API

The LiteRT Torch Generative API applies quantization through a configuration passed into conversion, with well-supported presets in `quant_recipes.py` such as `full_int8_dynamic_recipe()`. The supported schemes are dynamic INT8 (FP32 activations, INT8 weights, integer computation), weight-only INT8 (floating point computation), FP16 (FP16 weights, FP32 activations) and dynamic INT4 blockwise (FP32 activations, INT4 weights, integer computation).

In blockwise quantization, the last dimension of the weight is sliced into blocks, each with its own scale and zero point; the smallest supported block size is 32, and the last dimension must be divisible by the block size. Smaller blocks give better quality but a bigger, slower model; bigger blocks are faster and smaller with worse quality. Custom recipes can mix schemes per component, for example int8 embeddings, int4 block-32 attention, int4 block-256 feedforward layers and FP16 for everything else.

### Verifying a quantized model

Check size first: fp16 is about half, int8 about a quarter and int4 blockwise about a seventh of fp32, and a file that did not shrink as predicted means the recipe's regex did not match. Then check parity against the float reference with output correlation plus a task-level check. A smoke gate is a floor, not a parity verdict: an LLM can pass most of a handful of chat prompts and still score near zero on a real benchmark. Test long generations specifically, and test on the target device, since host emulation of integer kernels is pessimistic. Bytes are not speed: int4's latency gain depends on the backend's kernels, about 1.5× on one device and barely 1.1× on another. Some small reasoning-distilled decoders are genuinely 4-bit sensitive; then int8 is the quality row and int4 only a speed reference.

## Mixed precision, FP16 with selected FP32 layers

The LiteRT Mixed Precision tool converts a model's weights and activations from float32 to float16 while selectively keeping specific operations or layers in float32, to preserve accuracy where operations are sensitive to FP16 precision loss. It can keep op types (for example `tfl.AddOp` or `tfl.CumsumOp`) or name patterns (for example attention or norm layers) in FP32, and inserts cast operations at the boundaries automatically. Its default conversion already keeps highly sensitive operations such as RMS Norm in FP32, and a flag can clamp add operations after RMS norm to prevent overflow.

It runs as `mixed_precision_main --input_file=model.tflite --output_file=model_fp16.tflite --convert_to_fp16`, with `--fp32_ops` and `--fp32_names` for the exceptions. The typical workflow is to convert fully to FP16, profile accuracy and speed on the device, then refine the FP32 boundaries. At run time the GPU must be configured for FP32 precision, otherwise it may run the whole model in FP16 and ignore the FP32 sections.

## Models larger than the 2 GB FlatBuffer limit

The TFLite FlatBuffer structure must fit below approximately 2 GiB because of 32-bit internal offsets, and embedding all weights inline can exceed that for modern models. LiteRT supports two alternatives. In `buffer_offset` mode the model stays a single `.tflite` file: the graph and metadata sit in a FlatBuffer at the start and constant payloads are appended after it, aligned to 16 bytes, so the total file can exceed 2 GiB. The FlatBuffer exporter enables this mode automatically when the estimated module size exceeds the FlatBuffer limit minus 512 MiB. With external buffers, weights live in separate files or application-provided memory, which suits sharding or targets whose memory cannot load the complete file.

Neither removes the limit on the FlatBuffer graph structure itself or the device's physical memory. The guidance is to default to the same `.tflite` file across web, desktop and mobile and change packaging only for a concrete limitation. On the web, LiteRT.js `loadAndCompile` copies the complete model into Wasm memory and caps it at 2,000,000,000 bytes, while `loadModelAndWeights` streams external weights separately and requires a WebGPU device.
