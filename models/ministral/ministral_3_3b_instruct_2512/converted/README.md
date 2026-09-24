# Ministral-3-3B-Instruct-2512 Conversion

Model recipes, instructions, scripts, and utilities for converting Mistral AI Ministral-3-3B-Instruct-2512 (text decoder) to LiteRT-LM.

These scripts convert the text decoder of [mistralai/Ministral-3-3B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512) (`Ministral3ForCausalLM`, 3.43B parameters, Apache-2.0) into a `.litertlm` bundle for the [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) runtime, and verify the result with a quality gate. The published artifact is `Ministral-3-3B-Instruct-2512_q4_block32_ekv4096.litertlm` (2.34 GB) at [litert-community/Ministral-3-3B-Instruct-2512](https://huggingface.co/litert-community/Ministral-3-3B-Instruct-2512).

The checkpoint is multimodal (`Mistral3ForConditionalGeneration`: a Pixtral vision tower, a projector and the Ministral3 text decoder). This recipe exports the **text decoder only** — a 26-layer dense transformer (hidden 3072, GQA 32:8, head dim 128, vocab 131072, tied embeddings, YaRN RoPE) that the plain litert-torch HF export path handles with no re-authoring once it is a standalone checkpoint. What this conversion documents is four things that each look like a model problem and are not: which source repo to load, a template that must be Mistral's own, a section-size ceiling on iOS, and a `start_token` that is correct.

## Build

| recipe | quantization | file | use |
|---|---|---|---|
| `int4` | blockwise-32 OCTAV int4 weights, int8 embedding (externalized) | 2.34 GB | phone GPU / iPhone / desktop |

## Environment

```bash
pip install litert-torch==0.9.3 litert-converter==0.3.1 ai-edge-quantizer==0.8.0 \
    litert-lm-builder==0.16.0 transformers==5.14.1
pip install litert-lm   # verification CLI (>= 0.16.0)
```

## Run

```bash
python extract_text_decoder.py --out ministral3_text                     # ~7.7 GB download, once
python build_ministral_3_3b.py --model ministral3_text --out out_int4    # ~2.34 GB bundle
python verify_ministral_3_3b.py out_int4/*.litertlm --backend cpu        # 8-question gate
python verify_ministral_3_3b.py out_int4/*.litertlm --check-metadata     # [INST] template, </s> stop, <s> start_token, sections < 2 GiB
python verify_ministral_3_3b.py --bos-ab --hf ministral3_text            # is the <s> start_token legitimate? (bf16, ~1 min)
```

## Source: the plain repo is FP8

`mistralai/Ministral-3-3B-Instruct-2512` ships its weights **FP8-quantized** (`config.quantization_config.quant_method = "fp8"`), which the CPU-side export path cannot load. Use the sibling **`mistralai/Ministral-3-3B-Instruct-2512-BF16`**. That repo also carries `consolidated.safetensors`, a second single-file copy of the same weights that the HF loader never reads; `extract_text_decoder.py` skips it (7.7 GB).

The extractor loads the multimodal wrapper, copies `model.language_model` and `lm_head` into a plain `Ministral3ForCausalLM`, refuses to continue if any key is missing or unexpected, and saves the decoder with the tokenizer. That standalone checkpoint is what `build_ministral_3_3b.py` exports.

## Template: Mistral's own, not ChatML

Ministral speaks `[INST] … [/INST]` with `</s>` as its end-of-turn token, and its tekken tokenizer has **no `<|im_end|>`**. Exported under a ChatML template — the default rail for most instruct models — the int4 bundle answers correctly and then never reaches a registered stop token: it runs on with `<|im_start|>` spam. Under its native template it stops cleanly. `build_ministral_3_3b.py` forces the plain Mistral template (upstream's full jinja carries tool-calling logic the runtime's minimal jinja engine cannot run, so the structured `[INST]` / `[SYSTEM_PROMPT]` prefixes are extracted instead), and `--check-metadata` asserts the shape. Match the template to what the tokenizer can emit.

## The start_token that is correct

The bundle carries `start_token { token_str: "<s>" }`, and the runtime prepends it on the first turn. On granite-4.1-3b the same field breaks the model: there BOS is the same token as EOS, and the model reads the prompt as an already-ended document. Here bos `<s>` (id 1) and eos `</s>` (id 2) are different tokens, and a leading `<s>` is the Mistral convention the model was trained on — so the field stays. `verify_ministral_3_3b.py --bos-ab` feeds the same rendered `[INST]` prompt to the bf16 decoder with and without the leading `<s>` and prints both answers. Eligibility for the trap is a tokenizer shape (bos == eos, or `add_bos_token: False`); this model has neither.

## Section size: the iOS ceiling

Without `externalize_embedder` this 3B's weights are one **~2.55 GiB** TFLite section, above the ~2 GiB single-section `mmap` ceiling on iOS: engine creation fails with *"Failed to map section: Cannot allocate memory"*, an error that reads like a memory problem rather than a layout one. Externalizing the tied 131072×3072 embedding into its own section (0.38 GiB) drops the main section to **1.80 GiB**, dedups the tied matrix (2.74 GB → 2.34 GB on disk), and the bundle loads on iPhone. Same weights, same parity; `--check-metadata` asserts every section is under 2 GiB.

## Quantization

GSM8K, n=100, greedy, 0-shot CoT, 512-token budget (a direct-answer model needs no more), identical prompt and extraction for both rows:

| build | GSM8K |
|---|---|
| bf16 (HF reference) | 89.0% |
| int4 blockwise-32 OCTAV, int8 embedding | 85.0% |

−4 pt, the ordinary healthy shape. A channelwise min-max int4 of the same model passed the 8-question sanity gate and was not at parity; blockwise-32 + OCTAV is what preserves it, and the sanity gate alone would not have shown the difference.

## Verification results

Quality gate on the published bundle (`verify_ministral_3_3b.py`, litert-lm 0.16.0, Mac M4 Max): **CPU 8/8, GPU 8/8, non-degenerate**, clean stop at `</s>`; `--check-metadata` 8/8.

Rebuilt from scratch with these scripts on the pinned stack: extraction maps every decoder tensor (missing 0 / unexpected 0), the bundle comes out at 2,340,966,384 bytes against the published 2,340,982,768 (a 16 KB metadata difference — the published file was built on an earlier litert-torch), 8/8 on `--check-metadata`, 8/8 on the GPU gate, and `--bos-ab` prints the two answering columns above.

Mac (M4 Max, `litert-lm benchmark`, `-p 256 -d 256 --runs 3`, max-num-tokens 4096, quiet machine, litert-lm 0.15.0):

| backend | prefill tok/s | decode tok/s | TTFT |
|---|---|---|---|
| GPU (Metal) | 1234 | 95.4 | 0.23 s |
| CPU | 123 | 22.3 | 2.39 s |

On device:

- **Pixel 8a** (Tensor G3, Mali-G715, 8 GB), GPU (OpenCL), `litert_lm_main` v0.16.0 driven directly: **full delegation** — 1187/1187 nodes in the 128-token prefill graph, 1087/1087 in decode, zero rejected ops — prefill 8.5 tok/s (short prompt padded to the 128-token signature), decode 6.80 tok/s, TTFT 1.68 s, engine init 20.9 s, correct answer. That is the runtime driven directly. The Google AI Edge Gallery app fixes its accelerator choice at import time from device RAM and offers only CPU for a file this size on an 8 GB phone; both are true at once — the app will not offer the GPU there, and the model runs on that phone's GPU when something else drives it.
- **iPhone 17 Pro** (Metal, iOS 27.0, three runs through a LiteRT-LM Swift harness, 512-token budget): prefill 39.7–42.3 tok/s (short prompt), decode 14.2–17.6 tok/s, TTFT 0.40–0.42 s, load 12–62 s, footprint 1.44–1.46 GB. Decode is a range because it varied with load time across the three runs; prefill and TTFT were steady.

## Files

| File | What |
|---|---|
| `extract_text_decoder.py` | BF16 multimodal checkpoint → standalone `Ministral3ForCausalLM` (drops the Pixtral tower and projector; loud failure on any unmapped key). |
| `build_ministral_3_3b.py` | text decoder → `.litertlm`: Mistral `[INST]` template, blockwise-32 OCTAV int4 + int8 externalized embedding, `<s>` start_token kept, single 128 prefill signature, 4096 context. |
| `verify_ministral_3_3b.py` | 8-question quality gate via the `litert-lm` CLI; `--check-metadata` asserts the `[INST]` template, `</s>` stop, `<s>` start_token, externalized embedder and section sizes; `--bos-ab` shows the start_token is legitimate on the bf16 decoder. |
