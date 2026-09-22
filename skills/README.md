# Agent Skills

Custom skills, interactive demos, and automation workflow extensions for AI agents.

Each skill is a self-contained `SKILL.md` playbook covering one stage of taking a model to LiteRT on device. They chain in lifecycle order: convert → quantize → verify → benchmark → build the app.

## Available skills

* [`litert-conversion-workflow/`](litert-conversion-workflow/) — Convert a Hugging Face LLM or vision-language model checkpoint into a `.litertlm` bundle for the LiteRT-LM runtime with verified quality: classify the architecture against known runtime walls, pick the recipe family, export, quantize, gate against the source model, and publish. The human-readable version is the [model conversion cookbook](../models/conversion.md).
* [`gpu-clean-conversion/`](gpu-clean-conversion/) — Convert a PyTorch or Hugging Face model into a LiteRT model that runs fully on the GPU via the CompiledModel API, with verified-correct output, laid out as a model recipe.
* [`accuracy-safe-quantization/`](accuracy-safe-quantization/) — Shrink a converted LiteRT model with ai-edge-quantizer (fp16 / int8 / int4) without losing accuracy, verifying parity against the float source after every step.
* [`on-device-verification/`](on-device-verification/) — Prove a converted or quantized model on the actual device via the CompiledModel API: confirm GPU residency, compare device output against the source model, and diagnose device-only failures.
* [`benchmark-on-ddp/`](benchmark-on-ddp/) — Measure a LiteRT `.tflite` model with `benchmark_model` on Developer Device Platform (DDP) lab phones through `litert benchmark --ddp`, or on the Mac you run it from, and turn the session into rows of the performance leaderboard under `benchmark/leaderboard`: one matrix entry per model, one session per accelerator, collect, rebuild the board, commit the data.
* [`compiled-model-app-scaffolding/`](compiled-model-app-scaffolding/) — Build an Android app (Kotlin, Compose) around a verified LiteRT model using the CompiledModel API: app architecture, inference-layer lifecycle rules, model delivery, and UI traps.
* [`litert-compiled-model-migration/`](litert-compiled-model-migration/) — Rapidly migrate an existing Android application from legacy TensorFlow Lite (TFLite) to the LiteRT CompiledModel API (2.2.0): dependency swap, `CompiledModel` rewrite with buffer reuse, optional NPU JIT acceleration, and an instrumented-test gate.

## Skills in the android/skills form (for app developers' agents)

Two skills written in the form of [android/skills](https://github.com/android/skills) (frontmatter with license and metadata, prerequisites, numbered steps, `references/`), for an Android developer's agent rather than a conversion engineer's:

* [`litert-compiled-model/`](litert-compiled-model/) — Run a `.tflite` model in an Android app with the CompiledModel API on CPU, GPU or NPU: the dependency, accelerator choice, buffers and lifecycle, migration from the Interpreter API, and on-device verification.
* [`litert-lm/`](litert-lm/) — On-device LLM inference with the LiteRT-LM Kotlin API: get a `.litertlm` model from litert-community, initialize the engine off the main thread, stream a conversation, and add images, audio and tools.
