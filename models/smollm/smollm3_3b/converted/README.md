# SmolLM3-3B Conversion

Model recipes, instructions, scripts, and utilities for converting Hugging Face SmolLM3-3B to LiteRT-LM.

These scripts convert [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) (dense `SmolLM3ForCausalLM`, 3.08B parameters, Apache-2.0) into a `.litertlm` bundle for the [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) runtime, and verify the result with a quality gate. The published artifact is `SmolLM3-3B_q4_block32_ekv4096.litertlm` at [litert-community/SmolLM3-3B](https://huggingface.co/litert-community/SmolLM3-3B).

The model is a 36-layer dense transformer (hidden 2048, GQA 16:4, vocab 128256, tied embeddings) with SmolLM3's **NoPE** schedule — rotary embeddings disabled on every 4th layer (`no_rope_layer_interval=4`). That lowers to generic ops through the plain litert-torch HF export path; no re-authoring. What this conversion documents instead is a **template fact**: SmolLM3's reasoning-mode system prompt cannot ride the structured template the runtime applies, so the bundle is bare ChatML and the model's dual mode becomes the caller's choice.

## Build

| recipe | quantization | file | use |
|---|---|---|---|
| `int4` | blockwise-32 OCTAV int4 weights, int8 embedding (externalized) | 2.00 GB | phone GPU / desktop; GSM8K at parity with bf16 |

## Environment

```bash
pip install litert-torch==0.9.3 litert-converter==0.3.1 ai-edge-quantizer==0.8.0 \
    litert-lm-builder==0.16.0 transformers==5.14.1
pip install litert-lm   # verification CLI (>= 0.16.0)
```

## Run

```bash
python build_smollm3_3b.py --out out_int4                            # ~2.0 GB bundle
python verify_smollm3_3b.py out_int4/*.litertlm --backend cpu        # 8-question gate
python verify_smollm3_3b.py out_int4/*.litertlm --check-metadata     # bare ChatML, no start_token, sections < 2 GiB
python verify_smollm3_3b.py out_int4/*.litertlm --think-ab --backend gpu   # direct vs /think mode
```

## The template: why the bundle is bare ChatML

SmolLM3's official chat template inserts a long system block when the caller sends no system message — today's date, `Reasoning Mode: /think`, and the "Thought section / Solution section" instructions — inside an `{% if messages[0]['role'] != 'system' %}` branch. The runtime does not run that template. With `use_jinja_template=False` the exporter renders sample conversations through it and extracts **structured per-role prefixes and suffixes** (`<|im_start|>user\n` … `<|im_end|>\n`). Every sample conversation it renders begins with a system turn, so the conditional branch never fires and nothing conditional survives. Embedding the raw jinja instead is not an option: the runtime's minimal jinja engine cannot run `strftime_now` or the tool blocks, and the bundle dies on the first message.

The bundle therefore carries bare ChatML with a system-role prefix and no default system prompt. Two consequences, both measured on the published bundle (`--think-ab`, Mac GPU, greedy):

| prompt | think block | answer |
|---|---|---|
| plain user turn (the bundle's default) | empty — the model closes `<think></think>` at once | `42.` |
| same turn with SmolLM3's `/think` system prompt supplied as a system message | 120 words of reasoning | `42` |

By default the bundle behaves as a direct-answer instruct model; a caller who wants the reasoning mode passes SmolLM3's system prompt (the text is in `verify_smollm3_3b.py`) as a system message. What the bundle cannot do is apply a default on its own — a structured template has no place for "only when the caller sent nothing". To bake a default system turn in, fold it into the user prefix as the Qwen2.5-Coder recipe in this repository does, and accept that an explicit system message then renders alongside it.

Any model whose template guards its system prompt with a role check has the same shape. Check what the exporter extracted (`--check-metadata` prints the prefixes) before assuming the bundle carries what the template says.

## Quantization

GSM8K, n=100, greedy, 0-shot CoT, 1024-token budget, bare ChatML for both rows so the only variable is the quantization:

| build | GSM8K |
|---|---|
| bf16 (HF reference) | 81.0% |
| int4 blockwise-32 OCTAV, int8 embedding | 81.0% |

Parity, 0.0 pt at this n. Blockwise int4 plus OCTAV clipping is what holds it; channelwise min-max int4 is the recipe that collapses small models. The bundle's stop token is `<|im_end|>` (id 128012), from `generation_config`. SmolLM3 has no BOS token, so there is no `start_token` and no start-token trap.

## Prefill signatures and memory

The published bundle carries a **single 128-token prefill signature** (`--prefill 128`): the smallest engine-init memory, at the cost of prefilling long prompts in 128-token chunks and padding short ones to 128. Each extra signature is charged engine memory at init whether or not it is called. A 6-rung ladder (`--prefill 1024,256,64,16,4,1`, what the Qwen2.5-Coder recipe ships) buys TTFT on long prompts for a larger init footprint. `externalize_embedder=True` splits the tied 128256×2048 vocab table into its own bundle section (0.25 GiB), keeping the main weights section at 1.61 GiB — under the iOS ~2 GiB single-section mmap ceiling. No effect on weights or parity.

## Verification results

Quality gate on the published bundle (`verify_smollm3_3b.py`, litert-lm 0.16.0, Mac M4 Max): **CPU 8/8, GPU 8/8, non-degenerate**; `--check-metadata` 7/7.

Rebuilt from scratch with `build_smollm3_3b.py` on the pinned stack: 2,002,241,456 bytes against the published 2,002,257,840 (a 16 KB metadata difference — the published file was built on an earlier litert-torch), 7/7 on `--check-metadata`, 8/8 on the GPU gate.

Mac (M4 Max, `litert-lm benchmark`, `-p 256 -d 256 --runs 3`, max-num-tokens 4096, quiet machine, litert-lm 0.15.0):

| backend | prefill tok/s | decode tok/s | TTFT |
|---|---|---|---|
| GPU (Metal) | 1354 | 93.2 | 0.21 s |
| CPU | 141 | 24.1 | 2.14 s |

On device:

- **Pixel 8a** (Tensor G3, Mali-G715, 8 GB), GPU (OpenCL), `litert_lm_main` v0.16.0 driven directly: **full delegation** — 1476/1476 nodes in the 128-token prefill graph, 1308/1308 in decode, zero rejected ops — prefill 11.9 tok/s, decode 7.69 tok/s, TTFT 1.56 s, engine init 18.3 s, correct answer (the runtime's default factual prompt). The prefill figure is on a short prompt padded to the 128-token signature, so it reflects fixed per-turn cost rather than throughput.
- **iPhone 17 Pro** (Metal, iOS 27.0, one cold run through a LiteRT-LM Swift harness, 512-token budget): decode 22.5 tok/s, prefill 30.8 tok/s (short prompt), TTFT 0.63 s, load 7.7 s, footprint 1.24 GB.

A Mac's GPU-vs-CPU ratio does not transfer to a phone: on the M4 Max the Metal GPU decodes 3.9× the CPU, while on the Pixel 8a a 3B int4 bundle's GPU and CPU decode land within 1% of each other (7.07 vs 6.98 tok/s, measured on a granite-4.1-3b int4 bundle on the same phone), because a phone's CPU and GPU share the same LPDDR. On phone hardware the GPU buys prefill and TTFT, not tokens per second.

## Files

| File | What |
|---|---|
| `build_smollm3_3b.py` | HF checkpoint → `.litertlm`: bare-ChatML export, blockwise-32 OCTAV int4 + int8 externalized embedding, single 128 prefill signature, 4096 context. |
| `verify_smollm3_3b.py` | 8-question quality gate via the `litert-lm` CLI; `--check-metadata` asserts the template shape, stop token, no `start_token`, section sizes; `--think-ab` shows direct vs `/think` mode on the same question. |
