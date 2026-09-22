# Model conversion cookbook

How to take a model from a Hugging Face checkpoint to a LiteRT artifact that runs on a phone, and how to know it is right before you publish it. This page is the human-readable version of the procedure that the agent skills in [`skills/`](../skills/) encode for coding agents; the recipes under this directory are its worked examples, [MiniCPM5-2B](minicpm/minicpm5_2b/) first among them.

Two lanes, by what the model is:

| Model | Artifact | Runtime | Read |
|---|---|---|---|
| Text LLM or vision-language model | one `.litertlm` bundle | [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) engine | sections 1–11 |
| Anything else (vision, audio, diffusion, encoders) | one or more `.tflite` graphs plus a host loop | LiteRT CompiledModel API | section 12, then the three `.tflite` skills |

Everything below was hit on a real model. Where a rule names a runtime version it is a dated fact: re-test it on each release.

## Contents

1. [What "done" means](#1-what-done-means)
2. [Before you convert: classify the architecture](#2-before-you-convert-classify-the-architecture)
3. [Set up the toolchain and record it](#3-set-up-the-toolchain-and-record-it)
4. [Export](#4-export)
5. [Quantize](#5-quantize)
6. [Inspect the bundle before measuring anything](#6-inspect-the-bundle-before-measuring-anything)
7. [Gate quality against the source model](#7-gate-quality-against-the-source-model)
8. [Gate on the target backend and device](#8-gate-on-the-target-backend-and-device)
9. [Publish](#9-publish)
10. [When something breaks](#10-when-something-breaks)
11. [Worked example: MiniCPM5-2B](#11-worked-example-minicpm5-2b)
12. [Other models: the `.tflite` lane](#12-other-models-the-tflite-lane)
13. [Recipes in this directory](#13-recipes-in-this-directory)
14. [Going deeper](#14-going-deeper)

## 1. What "done" means

A conversion is finished when four things hold, in this order:

1. The bundle loads and generates through the LiteRT-LM engine, not only through the raw interpreter.
2. Output quality is gated against the source model: a floor gate plus a task-level parity check, not a smoke test.
3. It holds on the deployment path it claims: the target backend, the target device, and a multi-turn conversation.
4. The record names the device, the backend, the runtime version and the toolchain versions. A number without those is not a result.

Each step can pass while the next one fails. A bundle that converts can die at engine creation. An engine that generates can be quantization garbage. A model that answers 8/8 single-turn can crash on message two. A file that passes on the GPU can refuse to load on the CPU. The gates below exist because every one of these has happened.

## 2. Before you convert: classify the architecture

Open `config.json` and look at `model_type`, `layer_types`, MoE fields and size before running anything. Architecture decides whether the model is convertible at all and which route it takes.

| Architecture | Status | Route |
|---|---|---|
| Dense decoder (Llama, Qwen, Phi, Mistral, Gemma, SmolLM, Falcon-dense, OLMo, Granite-dense, and their finetunes) | Works; the standard lane | section 4 |
| Reasoning model (`<think>` in the chat template) | Works; needs template and thought-channel care, and a larger output budget when you evaluate it | section 4, "Reasoning models" |
| Interleaved hybrid (SSM, linear-attention or short-conv layers between attention layers: LFM2, Granite-4.0-h, Qwen3.5 GatedDeltaNet class) | Runs on the CPU backend since litert-lm 0.15; needs a state-aware export and an executor-metadata section in the bundle | [recipe-selector.md §Hybrids](../skills/litert-conversion-workflow/references/recipe-selector.md) |
| Parallel hybrid (Mamba and attention inside every layer, Falcon-H1 class) | Blocked: one layer needs a KV cache and conv/SSM state at the same time | park |
| Mixture of experts | Blocked at the runtime: the custom `moe` kernel is GELU-only, and the generic lowering emits a sort op no released kernel set registers | park, or convert the dense sibling |
| MLA / latent KV cache (DeepSeek family, MiniCPM3) | Blocked in the released exporter (asymmetric K and V head dims) | park; dense distills of the same models convert fine |
| Vision-language model | Works when the decoder is convertible and the vision tower can be made static; the runtime contract is one image per prompt | [vlm-conversion.md](../skills/litert-conversion-workflow/references/vlm-conversion.md) |
| Pre-quantized checkpoint (MXFP4, FP8) | Not a wall: transformers dequantizes to bf16 at load | proceed as a float conversion |
| GPTQ int4 checkpoint | Never re-quantize it: a symmetric re-rounding breaks the calibration while smoke tests keep passing | start from the float checkpoint, or ingest it with the quantizer's dequantized-weight recovery ([accuracy-safe-quantization](../skills/accuracy-safe-quantization/SKILL.md)) |

The full table, with the exact error each wall dies with and the versions it was observed on, is [architecture-walls.md](../skills/litert-conversion-workflow/references/architecture-walls.md).

## 3. Set up the toolchain and record it

The MiniCPM5-2B bundles were built on this released stack, with no patched checkout:

```bash
pip install litert-torch==0.9.3 litert-converter==0.4.0 ai-edge-quantizer==0.9.0 \
    litert-lm-builder==0.16.1 transformers==5.14.1
pip install litert-lm        # the runtime and CLI; 0.16 or newer for the thought channel
```

Write the versions you used into your recipe README. litert-torch, litert-converter, litert-lm-builder, ai-edge-quantizer and transformers each decide what exports and what runs. A conversion note without them cannot be reproduced.

## 4. Export

### The dense export driver

One driver, three decisions: the chat template, the tokenizer, the quantization recipe. The load-bearing part:

```python
# convert_dense.py <hf_id> <out_dir>
import sys
from transformers import AutoTokenizer

MINIMAL_CHATML = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n"
    "{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

# Swap the vendor chat template for a minimal one on every tokenizer the
# exporter loads, so prefix/suffix extraction succeeds and the bundle
# embeds no Jinja at all (see "The template decision" below).
_orig = AutoTokenizer.from_pretrained.__func__
AutoTokenizer.from_pretrained = classmethod(
    lambda cls, *a, **k: (lambda t: (setattr(t, "chat_template", MINIMAL_CHATML), t)[1])(_orig(cls, *a, **k)))

from litert_torch.cli import main
sys.argv = ["litert-torch", "export_hf",
    "--model", sys.argv[1], "--output_dir", sys.argv[2],
    "--prefill_lengths", "1024,512,256,128,64,32,16,8,4,2,1",
    "--cache_length", "4096",
    "--use_jinja_template", "False",
    "--bundle_litert_lm", "True",
    "--quantization_recipe", "dynamic_wi8_afp32"]
sys.exit(main())
```

Import `litert_torch` before any `transformers.models.<arch>` module: the other order corrupts the converter's dialect loading and breaks every later conversion in the same process.

What the flags decide:

- **Prefill ladder.** Export the whole ladder from 1 to 1024. The engine chunks each prompt through the signatures it has; a sparse ladder forces padded chunks, which cost time-to-first-token on every model and corrupt state-carrying ones.
- **Cache length.** The export-time cache size is speed-neutral. The runtime's `--max-num-tokens` is not: attention runs over the full static cache, and a 4096 budget has cost 25–30 % of decode speed against 1024 on a 1B model. Put that tip on the model card.
- **Turn markers.** Use the model's own family: a ChatML model gets ChatML, a Llama-3 model gets its header tokens. Wrong-family markers leak literal token text into prompts.
- **Start token (BOS).** The engine prepends the bundle's `start_token`. Match the model's template: if the official template has no BOS, ship no start token. A wrong BOS is invisible on robust models and flips small ones (a 350M model went from 8/8 to 1/8). MiniCPM5-2B's template starts with `{{ bos_token }}`, which renders empty at runtime, so its bundle keeps `<s>` as the start token; a template that renders its own BOS literal needs the opposite.
- **`externalize_embedder=True`** for anything around 3B or larger headed to iPhone, and for large-vocabulary models. The embedding table becomes its own section, which keeps the main section under the roughly 2 GiB single-section budget of default-entitlement iOS apps.

### The template decision: the most common ship-killer

More conversions have died on the chat template than on any numerical issue. The runtime renders templates with a Rust minijinja build, with no Python. A vendor template that calls `.get()`, `.strip()`, `.startswith()` or `.split()` imports fine and crashes on the user's first message:

```
Failed to apply template: unknown method: map has no method named get
```

Some crash only on message two, when a completed assistant turn enters the history for the first time. A pip wheel and a source build of the same runtime version have rendered differently, so "it renders in my venv" is not evidence that it renders in a shipped app.

**Default: embed no Jinja at all.** Export with `use_jinja_template=False` and the minimal template swapped in, as the driver above does. The exporter then extracts plain prefix and suffix markers, and the bundle carries no template code. The driver swaps the template rather than trusting extraction of the vendor's, because extraction can fail on complex templates and fall back to embedding the raw Jinja.

**When to embed the vendor template verbatim** (`use_jinja_template=True`): when the model's behavior lives in template logic that plain markers cannot carry, such as a thinking switch (`enable_thinking`) or a tool-calling format. MiniCPM5-2B does this. It brings two obligations: read the template for Python method calls before you export, and run the multi-turn gate (section 7.4), the only gate that exercises the history branches.

**The multi-turn prefix contract.** At each message the engine renders the history without a generation prompt and requires the new render to extend the previous one as a string; it prefills only the new suffix. A template that rewrites history, typically a reasoning template that strips `<think>` blocks from past turns, dies at turn two:

```
new rendered template string does not start with the previous
```

The fix for a hybrid-thinking model shipped without thinking: the generation prompt appends `<|im_start|>assistant\n<think>\n\n</think>\n\n`, and history turns render with the same empty think block, so the render equals what was actually prefilled and generated.

### Reasoning models

- Use a thinking variant of the minimal template that emits `<|im_start|>assistant\n<think>\n` as the generation prompt. The pre-filled opener is quantization armor: think-discipline is among the first abilities int4 loses, and one int4 build deliberated 2,500 tokens of plain text without answering when the opener was absent.
- Declare the thought channel with the model's real markers: `channels { channel_name: "thought" start: "<think>" end: "</think>" }`. Placeholder strings never fire, and the thought text floods the answer. Recent exporters declare it automatically when the packed template contains `<think>`; confirm in the peek output (section 6) rather than assume.
- Watch `enable_thinking | default(...)` in the template. The engine injects `enable_thinking` into the render only when a thinking config is set, so the template's default silently decides the behavior of every plain conversation. Choose it to match how you gated the model.
- Evaluate with an output budget of at least 2048 tokens. At 512, an int4 that is fine looks degraded because `</think>` never closes.

### Tokenizer and stop tokens

- When the source ships a SentencePiece model or a convertible BPE, bundle an `SP_Tokenizer` and round-trip a prompt through the packed tokenizer. The HF-tokenizer path has garbled prompts end to end and has dropped every space on decode ("Theusersaid42").
- Special tokens added beyond the base vocabulary get dropped by SentencePiece conversion. A thinking model then generates `<think>` and the runtime crashes with `Token id N is out of range`. Append each added token as a `USER_DEFINED` piece at its exact id. This is a no-op for the Qwen family, whose specials sit inside the base vocabulary; check `added_tokens_decoder` against the base vocabulary size first.
- Declare every turn-end token, not only `config.json`'s `eos_token_id`: the union of `generation_config.eos_token_id` and the template's actual turn suffix. Declaring only eos makes the literal `<|im_end|>` text leak into every reply. Stop ids are per tokenizer, not per family; the same `<|im_end|>` was id 7 in one model and 124900 in its larger sibling.

## 5. Quantize

| Model or budget | Recipe | Why |
|---|---|---|
| Any decoder, the safe default | int8 dynamic (`dynamic_wi8_afp32`) | Often beats data-free int4 on quality; on the CPU backend it also prefills faster |
| The phone file, up to about 3B | int4 blockwise-32 with OCTAV clipping on linears, int8 embedding | The fastest GPU decode; block-32 is the quality granularity |
| Around 4B, or an iPhone section budget | int4 blockwise-128 | Lighter dequantization and a smaller section. Math and reasoning models want block-32 even at this size, and on Apple GPUs block-32 is also the faster kernel |
| Decoders under about 0.5B | fp16 | int4 and even dynamic int8 have corrupted task output at 0.3B where fp16 was bit-faithful |
| Hybrids with conv or scan layers | Linears and embedding only; conv and scan layers stay float | Whole-graph int8 has produced empty output. A/B per finetune |
| Ternary checkpoints | int4 blockwise with min-max, never OCTAV | The weights land exactly on the grid; clipping can move them |

Two rules with no exceptions:

- **Never channelwise int4 on a decoder.** It loads, answers short prompts coherently, then degenerates over length, and it has scored 0 % on a benchmark while passing the floor gate.
- **Weight-only recipes (explicit dequantize) are CPU-only bundles** on the released GPU delegates. The dequantized weight arrives at the delegate as a runtime tensor, and the delegate rejects it at engine creation on both a Mac (Metal) and a Galaxy S26 (OpenCL); the MiniCPM5-2B recipe measures both. Use the dynamic, integer-compute recipes: the delegate consumes the quantized weights directly.

The int4 recipe as the exporter's `--quantization_recipe` JSON (blockwise-32 with OCTAV on linears, int8 embedding):

```json
[
  {"regex": ".*", "operation": "FULLY_CONNECTED",
   "algorithm_key": "OCTAV",
   "op_config": {"weight_tensor_config": {"num_bits": 4,
     "granularity": "BLOCKWISE_32", "symmetric": true},
     "compute_precision": "INTEGER"}},
  {"regex": ".*", "operation": "EMBEDDING_LOOKUP",
   "algorithm_key": "min_max_uniform_quantize",
   "op_config": {"weight_tensor_config": {"num_bits": 8,
     "granularity": "CHANNELWISE", "symmetric": true},
     "compute_precision": "INTEGER"}}
]
```

Field names drift between ai-edge-quantizer versions; check the installed version's recipe presets before relying on it. [`build_minicpm5_2b.py`](minicpm/minicpm5_2b/converted/build_minicpm5_2b.py) registers this recipe programmatically.

**After every blockwise int4 export, scan for zero scales.** A dense checkpoint can carry all-zero weight rows (MiniCPM5-2B has 13 in decoder layer 0's MLP; a 2.6B hybrid carried about 746k all-zero blocks across 26 tensors). Blockwise quantization emits scale 0 for each of their blocks. The CPU backend refuses to load the tensor:

```
unsupported scale value (0.000000) in channel 15616 for INT4 tensor 372
Failed to allocate tensors
```

The GPU delegate accepts the same file silently, so a GPU-only gate ships a bundle that dies on every CPU. Set each zero scale to the tensor's smallest nonzero scale, in place: the quantized values in a zero block are 0, so the dequantized weights do not change. [`fix_zero_scales_inplace.py`](minicpm/minicpm5_2b/converted/fix_zero_scales_inplace.py) does this on the packed bundle. Blockwise scales live in separate fp16 scale tensors, not in the flatbuffer's `QuantizationParameters.scale`; patching the latter succeeds silently and changes nothing. Per-channel int8 is unaffected.

**CPU and GPU invert.** On the CPU backend int8 beats int4 on prefill and on quality; on the GPU, int4 prefills several times faster at similar decode speed. If both backends matter, ship both files and say on the card which is which.

## 6. Inspect the bundle before measuring anything

```bash
python -m litert_lm_builder.litertlm_peek_main --litertlm_file model.litertlm
```

Read the output against this list before running any gate:

- **Template.** `prompt_templates` only means safe. A `jinja_prompt_template` that contains `.get(`, `.strip(` or `.split(` is the first-message crasher.
- **Stop tokens.** Every turn-end id is present, and the ids come from this model's tokenizer. Recent exporters also write string stop tokens next to the ids; these are harmless.
- **Start token.** Present or absent according to the template's BOS convention (section 4).
- **Thought channel** on reasoning models, with the model's real markers.
- **Executor-metadata section** on hybrids. Without it the bundle loads and dies at the first generation.
- **Sections.** The embedding table is its own section if you asked for it; the largest section fits the target platform's budget.
- **Activation declaration.** Whether the bundle declares `prefer_activation_type` (section 8).

Then confirm the engine itself loads and answers one prompt on the CPU backend:

```bash
litert-lm run model.litertlm --backend cpu --cache no --prompt "What is the capital of Japan? Answer briefly."
```

## 7. Gate quality against the source model

Run the gates in this order; each catches what the previous one cannot. Every gate here has been passed by a broken model; only the combination is the verdict. Run each on the CPU backend first, which isolates the graph, then on the backend you ship.

### 7.1 The floor gate: eight questions

Eight fixed, unambiguously checkable questions through the engine's own conversation path, greedy, one fresh conversation per question:

| Question | Expected in the reply |
|---|---|
| What is 17 + 25? | 42 |
| What is the capital of Japan? | Tokyo |
| What is the opposite of "hot"? | cold |
| How many days are in a week? | seven or 7 |
| How do you say "thank you" in French? | merci |
| What is 8 times 7? | 56 |
| Which is larger: 0.9 or 0.11? | 0.9 |
| Complete the rhyme: "Roses are red, violets are ___" | blue |

Each question runs as `litert-lm run model.litertlm --prompt "<question> Answer briefly." --backend <cpu|gpu> --cache no --temperature 0 --seed 0`. Pass bar: at least 6/8 correct and zero degenerate answers (empty, or one word repeated for more than half the reply), on the CPU and on the backend you ship. The script, with the degeneration check and a machine-readable result file, is in [verification-gates.md §1](../skills/litert-conversion-workflow/references/verification-gates.md); [`verify_minicpm5_2b.py`](minicpm/minicpm5_2b/converted/verify_minicpm5_2b.py) is the same gate with thinking-aware scoring.

The gate is a floor, never a parity verdict. On record: 8/8 with a 14-point benchmark loss, and 6/8 with a 1 % benchmark. It catches degeneration, tokenizer garbage and template death; it cannot rank recipes. Calibrate the bar by running the same gate on an official published model of similar size.

Reasoning models: run each question separately, since a shared thinking budget false-fails later questions. Score only the text after the thought channel, because reasoning text routinely contains the expected string while checking alternatives. An unclosed thought is a degenerate answer, never a pass on the text that came before it.

### 7.2 Task parity

A benchmark the model family is actually used for (GSM8K-style for general and reasoning models), n of at least 100 (smaller n has produced wildly wrong rankings), with the identical prompt and answer extraction on both sides so that quantization is the only variable. The source model is the baseline; the ship bar is "within a few points". Two rules:

- **Run the reference in fp32.** On Apple-Silicon MPS, bf16 makes one-step arithmetic errors that fp32 recovers, and a quantized model that "beats" a broken baseline makes the bar meaningless. If the converted model beats its source, suspect the harness before celebrating.
- **The reference must come from a different implementation path** than your export. A conversion has scored correlation 1.0 against a reference whose rotary buffer had silently loaded as zeros; both sides shared the broken load path, so agreement proved nothing.

Reasoning models: output budget of at least 2048 tokens. Choice-output models (classifiers, rerankers): the task score is too blunt, so also measure label agreement and the correlation of the raw logit margin, and calibrate any threshold on the backend you deploy on ([verification-gates.md §2](../skills/litert-conversion-workflow/references/verification-gates.md)).

### 7.3 First-token length sweep

State-carrying models (hybrids) corrupt at specific templated prompt lengths while answering perfectly at others; two observed cases failed only at lengths 18–21 and 40, and only at 33–37. Sweep it: filler words first, a fixed instruction last ("Output only the word BANANA."), greedy, at most 8 output tokens, a fresh conversation per length, and for state-carrying models a fresh engine per length. Pass means the reply starts with the literal at every reachable length. The harness rules and the engine's Python API are in [verification-gates.md §3](../skills/litert-conversion-workflow/references/verification-gates.md). Dense models rarely need this gate; run it once on any bundle you have not shipped before.

### 7.4 Multi-turn

Three turns minimum through the conversation API: a fact in turn 1 recalled in turn 3, with arithmetic in between. Single-turn evaluation structurally cannot catch the two multi-turn killers: the template prefix contract (section 4) and state carry-over bugs. On thinking models, size the output budget so that every turn's thinking completes; a turn capped mid-thought leaves an unterminated assistant turn in the stream and derails everything after it, which reads exactly like state corruption.

## 8. Gate on the target backend and device

**Desktop GPU first, as a sieve.** `litert-lm benchmark model.litertlm --backend gpu -p 256 -d 256 --runs 3 --cache no` proves engine creation and gives real speed numbers; then repeat the floor gate on the GPU backend. A CPU pass is not a GPU pass: fp16 accumulation flips marginal answers. On failure, the log names the rejected op; route it through [architecture-walls.md §Backend walls](../skills/litert-conversion-workflow/references/architecture-walls.md).

**Gate the GPU with the bundle's activation dtype, and on a thinking model watch the thought close.** The GPU executor keeps activations in fp16 unless the bundle declares `prefer_activation_type = "fp32"` in its `model.toml`. Over a deep decoder that is enough to steer a reasoning chain away from its terminator: MiniCPM5-2B's int8 file ran one gate question past 2,000 thought tokens on the GPU and never emitted `</think>`, while the CPU answered in 452 tokens. The declaration is a repack with the weights untouched (`litert-lm unpack`, edit `model.toml`, `litert-lm pack`; [`set_activation_type.py`](minicpm/minicpm5_2b/converted/set_activation_type.py) automates it) and costs about 15 % of GPU decode speed. It is not a blanket fix: on the same model's int4 file the fp16 default passes and fp32 reproduces the CPU's non-terminating chains. Decide per file from a thinking-on A/B, and record what the bundle declares. `litert-lm pack` exits 0 without writing when the output file already exists, so delete the target first.

**Then the actual device.** A Mac GPU pass does not transfer: iOS Metal has its own shader compiler with its own bugs, and mobile GPUs reject ops that desktop accepts. Gate on the target device before any public claim, and record the device, the backend, the runtime version, the residency line (delegated nodes out of total) and the speeds. Verify the on-device file size or checksum against the local artifact before running anything: a truncated copy fails at engine creation with `Failed to map section` and `TF_LITE_PREFILL_DECODE not found`, which reads like a model defect. The [on-device-verification](../skills/on-device-verification/SKILL.md) skill carries the recording discipline.

**Benchmark hygiene.** Use `--cache no` for gating; the default disk cache writes delegate caches of up to twice the model size next to the bundle and warms later init numbers. The Python API writes caches beside the bundle too, so pass `cache_dir` and sweep it. Never quote a first-run number (shader compilation). `--max-num-tokens` is a total budget of prompt plus generation, so a short budget truncates and looks like an early-stop bug. Serialize runs on an idle machine; parallel load has produced phantom 2× regressions. Budget about twice the model size of disk and RAM for the first GPU load on any backend.

## 9. Publish

- The upload script refuses unless the machine-readable gate report says passed. Human discipline fails exactly once, on the model you most wanted to ship.
- After the push, verify that the remote checksum equals the file you gated. When repairing a published artifact, start from the published file, not from a local experiment, and re-gate after the repair.
- The card states: the minimum runtime version and why; the exact toolchain versions; the recipe per variant with file sizes; the gate results with device, backend and runtime named; which variant is the quality row and which the speed row per platform; and known limitations in actionable form ("prompts whose templated length lands on 33–37 tokens can end the reply early; adding or removing a word avoids it"). A hidden limitation becomes an issue report.

Lay the conversion out as a recipe, with export, verification and repair each separately re-runnable, so the next person can rebuild the published file:

```
<model>/
  build_<model>.py         export driver: flags, template choice, quantization recipe
  verify_<model>.py        gate scripts and their JSON results
  <post-export fixes>.py   each one a separate, re-runnable step
  README.md                versions, recipe, gate numbers (device, backend, runtime named), known limitations
```

## 10. When something breaks

| What you see | Cause | Where to look |
|---|---|---|
| `torch.export` raises during tracing | Data-dependent guards (MoE gating, dynamic rope, LongRoPE), a dtype in a constant path (`uint8` arange, float64 rope tables, int64 masks), or complex-valued rope | [architecture-walls.md §Export-time walls](../skills/litert-conversion-workflow/references/architecture-walls.md); all fixable in the driver |
| `ModuleNotFoundError: litert_converter.mlir.dialects.tfl` after a clean export earlier in the same process | A `transformers.models.<arch>` import ran before `litert_torch` | Fix the import order |
| Engine creation fails: `No KV cache inputs found` | A hybrid on a runtime older than 0.15, including an app that bundles one | Check the deployed runtime version |
| Loads, then dies at the first generation: `NOT_FOUND ... missing some output TensorBuffers` | A hybrid bundle without the executor-metadata section | [recipe-selector.md §Hybrids](../skills/litert-conversion-workflow/references/recipe-selector.md) |
| First message crashes: `unknown method: map has no method named get` | Vendor Jinja embedded in the bundle | Re-export per section 4 |
| Message two fails: `new rendered template string does not start with the previous` | The template rewrites history | Section 4, the prefix contract |
| Fluent output that ignores the prompt, or all spaces missing | Tokenizer packaging | Section 4, tokenizer |
| `Token id N is out of range. Vocab size is M` | Added special tokens dropped from the packed tokenizer | Section 4, tokenizer |
| A stop marker printed as literal text, or the model never stops | Stop-token metadata incomplete | Section 4, stop tokens |
| Load fails on CPU only: `unsupported scale value (0.000000) ... for INT4 tensor` | All-zero weight blocks under blockwise int4; at the engine level you only see `INTERNAL: Failed to invoke the compiled model` | Section 5, zero scales |
| int4 passes the floor gate but tanks the benchmark | The gate is a floor; or channelwise int4 | Section 7.2; section 5 |
| Answers fine at some prompt lengths, garbage or empty at others | Prefill-padding state corruption | Section 7.3 |
| A thinking model never closes `</think>` on the GPU but does on the CPU | fp16 activations on the GPU backend | Section 8, activation dtype |
| GPU engine creation fails naming an op (`not fully delegated`) | A backend wall: `GATHER_ND` from a gather or reorder, rank-5+ tensors, an unlowered softmax composite (fixed in litert-converter 0.4.0 and later) | [architecture-walls.md §Backend walls](../skills/litert-conversion-workflow/references/architecture-walls.md); classic-op rewrites in the [gpu-clean-conversion](../skills/gpu-clean-conversion/SKILL.md) skill |
| GPU engine creation fails with `Shape mismatch: {bhwc, {2048, 1, 1, 2048}} ...` | A weight-only (explicit dequantize) recipe | Section 5; use a dynamic recipe |
| `Failed to map section ... Cannot allocate memory` on iPhone | A single section above the iOS budget | `externalize_embedder=True`, then block-128 |
| Generation fails at the first message even though the model loaded, in your own app | `maxTokens` left unset defaults to the whole cache length | Pass an explicit bounded max-tokens |
| Quality flips on one backend only | Decompose by execution stack: torch eager, torch with quantized-dequantized weights, engine CPU, engine GPU. The layer where the flip appears is the cause | [verification-gates.md §Triage](../skills/litert-conversion-workflow/references/verification-gates.md) |
| The converted model beats its source | A mis-prompted or mis-rendered reference | Section 7.2 |

Never re-roll a failed gate and never call it noise. Packaging masquerades as quantization: one "quantization jitter" verdict was overturned by prepending the bundle's start token to the source model, which reproduced the failure verbatim. Walk the ladder before touching the recipe; metadata bugs flip small models hardest.

## 11. Worked example: MiniCPM5-2B

[`minicpm/minicpm5_2b/converted/`](minicpm/minicpm5_2b/converted/) converts [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B), a plain 42-layer `LlamaForCausalLM` with hybrid thinking, into the two bundles published at [litert-community/MiniCPM5-2B](https://huggingface.co/litert-community/MiniCPM5-2B). The architecture needs nothing from the exporter. What the recipe documents is how a bundle can pass on one backend and fail on the other, and the two post-export steps that close the gap. Each was found by gating on both backends, and each was measured before it went into the recipe.

| Symptom | Backend that shows it | Cause | Fix |
|---|---|---|---|
| The int4 bundle fails at load: `unsupported scale value (0.000000)` | CPU; the GPU loads the same file silently | 13 all-zero MLP rows in decoder layer 0 give zero block scales | `fix_zero_scales_inplace.py` sets those scales to an epsilon in place; 3,328 bytes of a 1.55 GB file change, the dequantized weights do not |
| The int8 bundle never closes `</think>` on a short question | GPU (fp16 activations by default); the CPU answers | fp16 rounding over 42 layers steers the reasoning chain | `set_activation_type.py --type fp32` declares fp32 activations; the question closes in 452 tokens |

Both bundles score 8/8 on the floor gate on both backends, and GSM8K parity against the bf16 source holds for int8 (91 % against 92 %) with int4 about five points lower. The recipe README carries the thinking-mode comparison, the Galaxy S26, iPhone 17 Pro and Mac M4 Max measurements, the weight-only comparison behind the rule in section 5, and a rebuild record showing the scripts reproduce the published bundles byte for byte in every weight and tokenizer section. The template decision there is the "embed verbatim" branch of section 4: the checkpoint's template carries the `enable_thinking` switch and the tool-calling format, which plain markers cannot express.

## 12. Other models: the `.tflite` lane

Vision, audio, diffusion and encoder models produce one or more `.tflite` graphs driven by a host loop through the CompiledModel API. The discipline is the same, in three skills that chain in lifecycle order:

1. [gpu-clean-conversion](../skills/gpu-clean-conversion/SKILL.md): convert plain first, verify through the CompiledModel API with the checker in [`utilities/litert_gpu_toolkit/`](../utilities/litert_gpu_toolkit/), map each rejected op to a rewrite from the toolkit, and finish with a numerical check on the device. Full GPU residency does not imply correct numbers.
2. [accuracy-safe-quantization](../skills/accuracy-safe-quantization/SKILL.md): the ladder from fp16 float-casting through dynamic int8 to int4 blockwise, verifying parity against the float source after every step.
3. [on-device-verification](../skills/on-device-verification/SKILL.md): reference dumps from the source model, device CPU as the control, a strict GPU compile (never CPU-or-GPU, which hides fallback), the residency line, and three gates.

[compiled-model-app-scaffolding](../skills/compiled-model-app-scaffolding/SKILL.md) then builds the Android app around the verified model. Worked examples: [SAM 3](sam3/sam3_image/converted/), [Qwen3-TTS](qwen/qwen3_tts/converted/) and [Bonsai Image 4B](bonsai/bonsai_image_4b/) below.

## 13. Recipes in this directory

| Recipe | What it is | Lane |
|---|---|---|
| [`minicpm/minicpm5_2b/`](minicpm/minicpm5_2b/) | MiniCPM5-2B to `.litertlm`: CPU-and-GPU bundles, two post-export fixes measured, 8-question and metadata gates | LLM |
| [`hunyuan/hy_mt2_1_8b/`](hunyuan/hy_mt2_1_8b/) | Hy-MT2-1.8B to `.litertlm`: the static rope baked before export and checked bitwise, an int8 bundle, 8-question and translation gates | LLM |
| [`nemotron/nemotron_3_nano_4b/`](nemotron/nemotron_3_nano_4b/) | Nemotron-3-Nano-4B (a Mamba2-attention hybrid) to `.litertlm`: a patched litert-torch, post-hoc int8 on the linears and the embedding, GPU with `--cache no` | LLM |
| [`bonsai/bonsai_image_4b/`](bonsai/bonsai_image_4b/) | Text-to-image diffusion: three `.tflite` graphs (text encoder, DiT, VAE decoder) with a Python host loop; ternary weights in the int4 block-32 container | `.tflite` |
| [`qwen/qwen3_tts/`](qwen/qwen3_tts/) | Qwen3-TTS: three `.tflite` graphs plus host tables, verified step by step against PyTorch; a Tensor API implementation alongside | `.tflite` |
| [`sam3/sam3_image/`](sam3/sam3_image/) | SAM 3 text-prompted detection and segmentation: three GPU-resident `.tflite` graphs, every step verified numerically | `.tflite` |
| [`wav2vec2/wav2vec2_kws/`](wav2vec2/wav2vec2_kws/) | wav2vec2 keyword spotting | `.tflite` |
| [`zipformer/zipformer_ctc/`](zipformer/zipformer_ctc/) | Zipformer CR-CTC speech recognition | `.tflite` |
| [`sam2/sam2_hiera_tiny_video/`](sam2/sam2_hiera_tiny_video/) | SAM 2.1 video tracking authored directly with the LiteRT Tensor API (C++, no converter) | Tensor API |
| [`llada/llada_8b/`](llada/llada_8b/) | LLaDA-8B diffusion-LM denoise step authored with the Tensor API | Tensor API |
| [`gemma/gemma3/`](gemma/gemma3/), [`gemma/gemma4/`](gemma/gemma4/) | Directories reserved for the Gemma recipes | LLM |

## 14. Going deeper

The agent skills carry the same procedure in a form a coding agent can execute, with the facts behind every rule above:

- [`skills/litert-conversion-workflow/`](../skills/litert-conversion-workflow/): the LLM and VLM lane. Its references are the lookup tables this page condenses: [architecture-walls.md](../skills/litert-conversion-workflow/references/architecture-walls.md), [recipe-selector.md](../skills/litert-conversion-workflow/references/recipe-selector.md), [template-tokenizer-traps.md](../skills/litert-conversion-workflow/references/template-tokenizer-traps.md), [verification-gates.md](../skills/litert-conversion-workflow/references/verification-gates.md), [vlm-conversion.md](../skills/litert-conversion-workflow/references/vlm-conversion.md).
- [`skills/gpu-clean-conversion/`](../skills/gpu-clean-conversion/), [`skills/accuracy-safe-quantization/`](../skills/accuracy-safe-quantization/), [`skills/on-device-verification/`](../skills/on-device-verification/), [`skills/compiled-model-app-scaffolding/`](../skills/compiled-model-app-scaffolding/): the `.tflite` lane and the app.
- [LiteRT-CLI](https://github.com/google-ai-edge/LiteRT-CLI): the `litert` command for download, convert, quantize, run and benchmark. Where a command exists, prefer it to an ad-hoc script; the judgment above stays the same.
