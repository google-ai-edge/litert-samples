# Qwen2.5-Coder-1.5B-Instruct Conversion

Model recipes, instructions, scripts, and utilities for converting Qwen2.5-Coder-1.5B-Instruct to LiteRT-LM.

These scripts convert [Qwen/Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) (dense `Qwen2ForCausalLM`, 1.54B parameters, Apache-2.0) into a `.litertlm` bundle for the [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) runtime, and verify the result with a quality gate and a code gate. The published artifact is `Qwen2.5-Coder-1.5B-Instruct_int4.litertlm` (1.12 GB) at [litert-community/Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/litert-community/Qwen2.5-Coder-1.5B-Instruct).

The model is a 28-layer dense transformer (hidden 1536, GQA 12:2, vocab 151936, tied embeddings) — standard Qwen2, handled by the plain litert-torch HF export path with no re-authoring. What this conversion documents is a **size trap** (a tied embedding stored seven times, silently), a **template fact** (a vendor default system prompt that a plain ChatML export drops), and why the 1–2B band in int4 is where a phone is quick rather than merely capable.

## Build

| recipe | quantization | file | use |
|---|---|---|---|
| `int4` | blockwise-32 OCTAV int4 weights, int8 embedding (externalized) | 1.12 GB | phone GPU/CPU, desktop |

## Environment

```bash
pip install litert-torch==0.9.3 litert-converter==0.3.1 ai-edge-quantizer==0.8.0 \
    litert-lm-builder==0.16.0 transformers==5.14.1
pip install litert-lm   # verification CLI (>= 0.16.0)
```

The published bundle was built on exactly this released stack — no patched checkout.

## Run

```bash
python build_qwen2_5_coder_1_5b.py --out out_int4                                    # 1.12 GB bundle
python verify_qwen2_5_coder_1_5b.py out_int4/*.litertlm --backend cpu                # 8-question gate
python verify_qwen2_5_coder_1_5b.py out_int4/*.litertlm --check-metadata             # template, stops, bytes/param
python verify_qwen2_5_coder_1_5b.py out_int4/*.litertlm --code-gate --backend gpu    # 6 functions, executed
```

## The size trap: one table, stored seven times

The first export of this checkpoint came out at **2.53 GB for 1.54 B parameters** — 1.64 bytes per parameter, for a recipe whose int4 linears should land near 0.5. Nothing had failed. Inspecting the tensors:

| tensor type | count | stored |
|---|---|---|
| INT4 | 197 | 771.8 MB — the linears, correct |
| **INT8** | **7** | **1,633.6 MB — the 151936×1536 vocab table, once per prefill signature** |

Qwen2.5-Coder ties its embedding and `lm_head`. A recipe that asks for int4 on `FULLY_CONNECTED` and int8 on `EMBEDDING_LOOKUP` describes that one tensor two ways, and the quantizer resolves the conflict by materializing the int8 copy inside every signature's graph — six prefill signatures plus decode, seven copies, 65% of the file. Decode on these runtimes is memory-bandwidth-bound; a file 2.26× too large is a bundle 2.26× slower at the thing that limits it.

**Fix: `externalize_embedder=True`** (the default in `build_qwen2_5_coder_1_5b.py`). The table moves into its own bundle section, referenced by every signature: **1,117,385,648 bytes, zero duplicated tensors, 0.72 bytes/param.** Same weights, same quality. `--check-metadata` asserts the bytes-per-parameter ratio so the duplicated build cannot pass unnoticed; `--inline-embedder` reproduces it for study.

The check generalizes: file size ÷ parameter count against the recipe's expected ratio (int4 ~0.5–0.75, int8 ~1.05–1.35). It costs nothing, and it is the only symptom this defect has.

## The template: the vendor's default system prompt

Upstream's chat template inserts `You are Qwen, created by Alibaba Cloud. You are a helpful assistant.` as a system turn whenever the caller sends no system message. The runtime applies structured per-role prefixes and has no place for a conditional default, so a plain ChatML export ships a model that never sees the system turn it was tuned with — quietly, on every default request. `build_qwen2_5_coder_1_5b.py` folds that system turn into the **user prefix**, which renders byte-identical to upstream for a single-turn request:

```
<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n
```

The trade: a caller who does send an explicit system message gets it as its own turn *and* the default inside the user prefix. A structured template cannot say "only if absent" — pick default-path fidelity (this recipe) or explicit-system fidelity (bare ChatML), knowing which you chose.

The stop set is `<|im_end|>` (151645) and `<|endoftext|>` (151643), both from `generation_config`. The tokenizer has no BOS, so no `start_token`.

## Verification results

Two gates on the published bundle (`verify_qwen2_5_coder_1_5b.py`, litert-lm 0.16.0, Mac M4 Max), because a general-knowledge check certifies nothing for a code model:

- **8-question gate: CPU 8/8, GPU 8/8**, non-degenerate.
- **Code gate: 6/6** on GPU — `fib`, `reverse_words`, `is_prime`, largest contiguous sublist sum, `count_vowels`, `flatten`; the generated code is executed against assertions, and a task passes only if it imports and every assertion holds. Kadane's problem (`max_sum`) is the useful signal: it needs reasoning rather than recall.
- `--check-metadata` 7/7 (default system turn in the user prefix, both stop ids, no `start_token`, externalized embedder, 0.72 bytes/param).

Rebuilt from scratch with `build_qwen2_5_coder_1_5b.py` on the pinned stack: 1,117,385,648 bytes, **byte-identical to the published bundle from the tokenizer section onward** (the 127 differing bytes are the header's uuid and creation timestamp), 7/7 metadata, 8/8 gate, 6/6 code gate.

Mac (M4 Max, `litert-lm benchmark`, `-p 256 -d 256 --cache no`, quiet machine, litert-lm 0.16.0):

| backend | prefill tok/s | decode tok/s | TTFT | init |
|---|---|---|---|---|
| GPU (Metal) | 3037 | 137.8 | 0.099 s | 2.38 s |
| CPU | 292 | 47.1 | 1.06 s | 2.87 s |

For scale, a 3B-class int4 bundle measured on the same machine and protocol on the same day (granite-4.1-3b, 2.19 GB) runs 1241 tok/s prefill / 86.3 decode / 0.233 s TTFT / 5.96 s init: 1.6× the decode, 2.4× the prefill and TTFT, 2.5× the startup, from halving the bytes.

On device:

- **iPhone 17 Pro** (Metal, LiteRT-LM Swift harness): loads at the bundle's default 4096 context, engine init 6.27 s, 7/8 on the composite eight-question prompt (the miss is the rhyme completion, answered "white"; granite-4.1-3b missed the same line on its CPU leg in that harness and got it right when the question was asked alone). Threshold is 6/8.
- **Pixel 8a** (Tensor G3, Mali-G715, 8 GB), `litert_lm_main` v0.16.0 driven directly, correct answers on both backends. GPU (OpenCL): **full delegation** — 1243/1243 nodes in every prefill signature, 1132/1132 in decode, zero rejected ops — prefill 83.7 tok/s, decode 12.0 tok/s, TTFT 0.54 s, engine init 15.1 s on the runtime's default factual prompt; on a code prompt (`is_prime`, correct function) prefill 116.6, decode 11.5, TTFT 0.54 s. CPU (XNNPACK): prefill 16.1, decode 12.6, TTFT 2.44 s, init 4.9 s.

Two things the phone rows say. Decode is 12.0 GPU vs 12.6 CPU: the phone's GPU and CPU share the same LPDDR, so on phone hardware the GPU buys prefill and TTFT (0.54 s vs 2.44 s), not tokens per second — the Mac's 2.9× GPU decode ratio does not carry over. And against the 3B int4 bundles measured on the same phone (7–8 tok/s decode), halving the bytes is worth about 1.6× on decode — the lever this band exists for.

## Files

| File | What |
|---|---|
| `build_qwen2_5_coder_1_5b.py` | HF checkpoint → `.litertlm`: coder template (default system turn in the user prefix), blockwise-32 OCTAV int4 + int8 externalized embedding, 6-signature prefill ladder, 4096 context. `--inline-embedder` reproduces the duplicated-table build. |
| `verify_qwen2_5_coder_1_5b.py` | 8-question quality gate via the `litert-lm` CLI; `--check-metadata` asserts the template shape, stop ids, no `start_token`, externalized embedder and bytes/param; `--code-gate` executes six generated functions against assertions. |
