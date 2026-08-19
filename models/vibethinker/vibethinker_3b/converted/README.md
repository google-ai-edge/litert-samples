# VibeThinker-3B Conversion

Model recipes, instructions, scripts, and utilities for converting WeiboAI VibeThinker-3B to LiteRT-LM.

These scripts convert [WeiboAI/VibeThinker-3B](https://huggingface.co/WeiboAI/VibeThinker-3B) (dense `Qwen2ForCausalLM`, 3.09B parameters, MIT) into a `.litertlm` bundle for the [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) runtime, and verify the result with a quality gate and a math gate. The published artifact is `model.litertlm` (2.06 GB) at [litert-community/VibeThinker-3B](https://huggingface.co/litert-community/VibeThinker-3B).

VibeThinker is a **math/reasoning model**: it works a problem through an inline chain of thought, then states the answer (`\boxed{}` / `####`). Architecturally it is a 36-layer Qwen2 (hidden 2048, GQA 16:2, vocab 151936, tied embeddings), handled by the plain litert-torch HF export path with no re-authoring. What this conversion documents is a **stop-token gap** the vendor config leaves open under ChatML, and a **block-size finding**: the int4 block size that is a free speed knob on general-purpose models is not free on a model whose job is exact arithmetic.

## Build

| recipe | quantization | file | use |
|---|---|---|---|
| `int4` | blockwise-**32** OCTAV int4 weights, int8 embedding (externalized) | 2.06 GB | the only build — block-128 collapses this model (below) |

## Environment

```bash
pip install litert-torch==0.9.3 litert-converter==0.3.1 ai-edge-quantizer==0.8.0 \
    litert-lm-builder==0.16.0 transformers==5.14.1
pip install litert-lm   # verification CLI (>= 0.16.0)
```

## Run

```bash
python build_vibethinker_3b.py --out out_int4                                # ~2.06 GB bundle
python verify_vibethinker_3b.py out_int4/*.litertlm --backend cpu            # 8-question gate
python verify_vibethinker_3b.py out_int4/*.litertlm --check-metadata         # both stop ids, no start_token, sections
python verify_vibethinker_3b.py out_int4/*.litertlm --math --backend gpu     # 6 word problems, chain of thought
```

## Stop tokens: the vendor config declares one, ChatML needs two

VibeThinker's `generation_config.json` lists a single EOS, `<|endoftext|>` (151643). Under ChatML the model ends its turn with `<|im_end|>` (151645). The bundle builder derives the id-level stop set from `generation_config`, so a naive export gives the runtime no id stop for `<|im_end|>` and the model can finish its answer and keep going. `build_vibethinker_3b.py` adds 151645 to `eos_token_id` after the model loads; the published bundle carries both ids and `--check-metadata` asserts it. The tokenizer has no BOS, so there is no `start_token`.

The bundle's template is bare ChatML. Upstream's Qwen2 template also inserts `You are a helpful assistant.` as a default system turn; the bundle drops that line, and every number below was measured without it. To bake it in, fold the system turn into the user prefix as the Qwen2.5-Coder recipe in this repository does.

## Block size: 32, not 128

GSM8K, n=100, greedy, 0-shot CoT, **2048-token budget** (a reasoning model needs the room — at a short budget the chain of thought is cut off before the answer, and int4, whose chains run slightly longer, looks falsely degraded), identical prompt and extraction for every row:

| build | GSM8K |
|---|---|
| bf16 (HF reference) | 97.0% |
| int4 blockwise-**32** OCTAV, int8 embedding | **90.0%** (−7 pt) |
| int4 blockwise-128 OCTAV, int8 embedding | 64.0% (−33 pt) |

Block-128 has a quarter of the dequant scales and is the faster GPU build on general-purpose 4B reasoning models, where it measures at parity with block-32 or better. On this model it collapses. The difference is the task: exact arithmetic is precision-sensitive in a way open-ended reasoning is not, and the coarser grid loses it. Only block-32 is offered here; treat block size as a per-model measurement, not a default.

## Prefill signatures and memory

The published bundle carries a single 128-token prefill signature and a 4096-token KV cache. Keep the context at 4096 or above: the model spends 300–600 words per answer before it is done. `externalize_embedder=True` splits the tied 151936×2048 vocab table into its own section (0.29 GiB), keeping the main weights section at 1.62 GiB — under the iOS ~2 GiB single-section mmap ceiling.

## Verification results

Gates on the published bundle (`verify_vibethinker_3b.py`, litert-lm 0.16.0, Mac M4 Max), scored on the answer after the think block:

- **8-question gate: CPU 8/8, GPU 7/8**, non-degenerate (the GPU miss is the rhyme completion, answered "white"; every answer took 50–100 words including its reasoning).
- **Math gate: 6/6** on GPU — six GSM8K-shaped word problems, 294–582 words per answer, every final number correct.
- `--check-metadata` 7/7.

Rebuilt from scratch with these scripts on the pinned stack: 2,057,073,584 bytes against the published 2,057,106,352 (a 32 KB metadata difference — the published file was built on an earlier litert-torch), 7/7 on `--check-metadata` with both stop ids present, and the same 7/8 on the GPU gate with the same eight answers.

Mac (M4 Max, `litert-lm benchmark`, `-p 256 -d 256 --runs 3`, max-num-tokens 4096, quiet machine, litert-lm 0.15.0):

| backend | prefill tok/s | decode tok/s | TTFT |
|---|---|---|---|
| GPU (Metal) | 1386 | 94.1 | 0.20 s |
| CPU | 139 | 28.4 | 2.15 s |

On device:

- **Pixel 8a** (Tensor G3, Mali-G715, 8 GB), GPU (OpenCL), `litert_lm_main` v0.16.0 driven directly: **full delegation** — 1603/1603 nodes in the 128-token prefill graph, 1452/1452 in decode, zero rejected ops. On an arithmetic prompt ("What is 17 + 25? Answer with just the number.") it reasons briefly and answers `42`: prefill 17.3 tok/s, decode 8.11 tok/s, TTFT 1.51 s. CPU (XNNPACK): prefill 1.62 tok/s, decode 4.10 tok/s, TTFT 10.7 s. Engine init 20–50 s on this phone.
- **iPhone 17 Pro** (Metal): loads and generates correct answers (the 1.62 GiB main section is under the iOS limit); no on-device timing was taken.

This is a math-specialised model, and it shows on a phone: on the runtime's default factual prompt (the tallest building) it reasoned at length and settled on a wrong answer — identically in shape on GPU and CPU, with different hallucinated content, which is how you tell it is the model's domain rather than the backend. Ask it arithmetic.

## Files

| File | What |
|---|---|
| `build_vibethinker_3b.py` | HF checkpoint → `.litertlm`: bare-ChatML export with the `<|im_end|>` stop added, blockwise-32 OCTAV int4 + int8 externalized embedding, single 128 prefill signature, 4096 context. |
| `verify_vibethinker_3b.py` | 8-question quality gate via the `litert-lm` CLI (scored after the think block); `--check-metadata` asserts both stop ids, no `start_token`, template shape, section sizes; `--math` runs six word problems and checks the final number. |
