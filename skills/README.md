# Agent Skills

Custom skills, interactive demos, and automation workflow extensions for AI agents.

Each skill is a self-contained `SKILL.md` playbook covering one stage of taking a model to LiteRT on device. They chain in lifecycle order: convert → quantize → verify → build the app.

## Available skills

* [`gpu-clean-conversion/`](gpu-clean-conversion/) — Convert a PyTorch or Hugging Face model into a LiteRT model that runs fully on the GPU via the CompiledModel API, with verified-correct output, laid out as a model recipe.
* [`accuracy-safe-quantization/`](accuracy-safe-quantization/) — Shrink a converted LiteRT model with ai-edge-quantizer (fp16 / int8 / int4) without losing accuracy, verifying parity against the float source after every step.
* [`on-device-verification/`](on-device-verification/) — Prove a converted or quantized model on the actual device via the CompiledModel API: confirm GPU residency, compare device output against the source model, and diagnose device-only failures.
* [`compiled-model-app-scaffolding/`](compiled-model-app-scaffolding/) — Build an Android app (Kotlin, Compose) around a verified LiteRT model using the CompiledModel API: app architecture, inference-layer lifecycle rules, model delivery, and UI traps.
