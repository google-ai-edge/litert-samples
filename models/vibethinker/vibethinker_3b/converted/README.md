# VibeThinker-3B Conversion

Model recipes, instructions, scripts, and utilities for converting VibeThinker-3B to LiteRT-LM.

These scripts rebuild `model.litertlm` of [litert-community/VibeThinker-3B](https://huggingface.co/litert-community/VibeThinker-3B) from the [WeiboAI/VibeThinker-3B](https://huggingface.co/WeiboAI/VibeThinker-3B) checkpoint, with litert-torch's stock exporter and one `ModelExtension` plugin. The file was checked on a Mac with an M4 Max and on a Galaxy S26. To build it, run `python build_vibethinker_3b.py --out out_int4` in the environment below. On litert-torch 0.10.0, the latest release, the build stops before the export and says why.

VibeThinker-3B is a math and reasoning finetune in a stock `Qwen2ForCausalLM`: 36 layers, hidden 2,048, 16 query and 2 key-value heads, a 151,936-token vocabulary tied to the output layer, 3.09 B parameters, MIT. It works a problem through inside `<think>` … `</think>` and then answers.

## Build

| file | weights | prefill signature | KV cache | size |
|---|---|---|---|---|
| `model.litertlm` | int4 blockwise-32 (OCTAV) on every weight, the embedding int8 per channel in its own section (`recipe_int4.json`) | 128 tokens | 4,096 | 2.06 GB |

## Environment

```bash
pip install litert-torch-nightly==0.11.0.dev20261111 torch==2.14.0 transformers==5.20.2 \
    litert-lm-builder==0.19.1 litert-lm==0.19.1 litert-lm-api==0.19.1 \
    litert-converter==0.6.0.dev20261110 litert-quantizer-nightly==0.12.0.dev20261111 \
    ai-edge-litert-nightly==2.4.0.dev20261110 protobuf==7.36.2 numpy==2.5.4
```

Python 3.12. The extension needs litert-torch's main branch: a `.py` file in `extension_modules` came with commit 864a966 of 2026-10-28 and `ModelExtension.get_export_defaults` with 764beb0 of 2026-11-03, and no release has them yet. litert-lm 0.19.1 runs the checks.

## Run

```bash
python vibethinker_extension_test.py               # the extension's tests, CPU, no weights
python build_vibethinker_3b.py --out out_int4      # the file, a few minutes on an M4 Max
python verify_vibethinker_3b.py out_int4/model.litertlm --backend cpu
python verify_vibethinker_3b.py out_int4/model.litertlm --backend gpu
python verify_vibethinker_3b.py out_int4/model.litertlm --check-metadata
python verify_vibethinker_3b.py out_int4/model.litertlm --math --backend gpu
python verify_vibethinker_3b.py out_int4/model.litertlm --sections published.litertlm
```

The same export from the command line, with a local copy of the checkpoint in `$SRC`:

```bash
litert-torch export_hf --model=$SRC --output_dir=out_cli --extension_modules=vibethinker_extension.py
```

## The extension

`vibethinker_extension.py` is an out-of-tree `ModelExtension`. It holds every export argument and metadata field the published file needs, so litert-torch stays unchanged and nothing edits the file after the export. It registers for `model_type` `qwen2`, replaces litert-torch's built-in Qwen2 extension for the export it is passed to, patches no module and overrides two hooks:

| hook | what it sets | why |
|---|---|---|
| `get_export_defaults` | `quantization_recipe` = `recipe_int4.json` | the published file's weights |
| | `externalize_embedder=True` | one copy of the tied 151,936 × 2,048 table, in its own section, for every signature |
| | `prefill_lengths=[128]`, `cache_length=4096`, `llm_metadata_max_num_tokens_override=None` | the published signature and context; `max_num_tokens` is the cache length |
| | `use_jinja_template=False` | per-role templates, with the Jinja template from the hook |
| | `apply_gpu_composites=False`, `use_rope_composite=False`, `externalize_weights=False` | the published graph, without composite operators |
| `build_llm_metadata` | stops: the text `<\|im_end\|>\n`, `<\|endoftext\|>` (151643), `<\|im_end\|>` (151645) | the generation config declares only 151643, while the model ends a ChatML turn with 151645 |
| | the published Jinja template | ChatML with the vendor's default system turn, for content given as a string or as text parts |
| | the thought channel `<think>` … `</think>` | the runtime returns the thought apart from the answer |

Arguments passed to `export()` still win, and the defaults refuse another checkpoint's `config.json` and the `npu` target.

The Jinja template is the published file's, kept in the extension with its sha256 as `BUNDLE_TEMPLATE`. `build_llm_metadata` runs once the graphs are converted. It first renders four conversations, one and three turns with and without a system turn, through this template and the checkpoint's own. If any differ, or the checkpoint's turn-end ids or EOS ids change, the export stops there with `ValueError`, and `build_vibethinker_3b.py` leaves no file.

## Without the extension

`python build_vibethinker_3b.py --dry-run --no-extension` prints the stock plan: `dynamic_wi8_afp32` weights with the embedding in the graph, prefill signatures of 128 and 1,024 tokens over a 32,771-token cache, GPU composite operators, and LiteRT-LM's canonical Qwen2.5 metadata. At LiteRT-LM commit 61fd138 that metadata has no thought channel and the start token `<|endoftext|>`, which this tokenizer never adds, stops on the text `<|im_end|>` alone and has no default system turn.

Without the hook, the exporter's own metadata stops on the id 151643 alone and has no Jinja template or thought channel.

## After the export

The Hub's `main` took two more steps after `087512d2`, the revision this recipe rebuilds. On 2026-10-05, revision `61cbe45e`, hf-to-litertlm's [`tools/gpu_graph`](https://github.com/john-rocky/hf-to-litertlm/tree/main/tools/gpu_graph) moved the KV-cache update and the attention into the runtime's GPU composite operators and added a 16-token prefill signature. On 2026-10-30, revision `1fa937cc` took the copy from the repo's `web` branch, whose weights [`tools/externalize`](https://github.com/john-rocky/hf-to-litertlm/tree/main/tools/externalize) had put in `TFLiteWeights` sections, and its commit notes that Android GPU then needs LiteRT-LM 0.19.0 or later. Both steps run on a finished file, so neither is in this recipe.

## Verification results

Measured on 2026-11-12 on a Mac with an M4 Max in the environment above, and on a Galaxy S26. "This recipe" is a file built by `build_vibethinker_3b.py` from the checkpoint at revision `77bd2cce`. "Published" is `model.litertlm` at revision `087512d2`, the last one before the GPU graph rewrite, its sha256 checked against the Hub.

- **`vibethinker_extension_test.py`**: 18/18 on CPU.
- **Build**: 2,057,106,352 B, the published file's size, in a few minutes, with a peak of 40 GB of memory. litert-lm-builder 0.19.1 writes container version 1.9.0, the published file has 1.6.0.
- **Command line**: `litert-torch export_hf` with the extension gave every section byte-equal to the file of `build_vibethinker_3b.py`, the stop order included. The two files differ only in the header's uuid and creation time.
- **Checks** through the litert-lm 0.19.1 command-line tool, greedy, `--cache no`, 4 CPU threads, each reply scored on the answer after the thought:

| check | this recipe | published |
|---|---|---|
| 8 questions, CPU | 8/8 | 8/8 |
| 8 questions, GPU | 8/8 | 8/8 |
| `--check-metadata` | 9/9 | 9/9 |
| `--math`, GPU | 6/6 | 6/6 |

  The tool printed each thought apart from the answer, as `[thought] … [/thought]`. With the thought, no reply is byte-identical to the published file's. The answer after the thought is identical in five of eight on the CPU and four on the GPU, and worded differently in the rest. One GPU reply of this recipe's file, 8 × 7, reasons again after the thought and writes another `</think>` before "56".

- **Galaxy S26**, through LiteRT-LM v0.19.1's `litert_lm_advanced_main` for Android, greedy, one process per question: this recipe's file answered the 8 questions 8/8 on the CPU and 8/8 on the GPU. The answer after the thought is the same text as on the Mac in seven of eight on the CPU and four on the GPU. The published file was not run on the phone.

- **`--sections`** against the published file:

| section | this recipe | published | difference |
|---|---:|---:|---|
| `LlmMetadataProto` | 1,718 B | 1,701 B | a pad token |
| `ExecutorMetadataProto` | 4,894 B | none | new |
| `HF_Tokenizer_Zlib` | 2,142,143 B | 2,142,143 B | none, byte-equal |
| `TFLiteModel`, prefill and decode | 1,740,093,856 B | 1,740,096,624 B | the int4 weights, one reshape |
| `TFLiteModel`, embedder | 314,815,408 B | 314,815,408 B | none, byte-equal |

The stops, the Jinja template and the thought channel are the published file's, the stops in the same order. The published file's graphs were built with litert-torch 0.9.3 or earlier. Each other difference has a measured cause:

- The pad token is the tokenizer's `<|endoftext|>`, which litert-torch writes since commit 77e1fea of 2026-08-28, first released in 0.10.0.
- The executor section holds `max_history_size: 4096` and the 36 layers' key and value caches by name. litert-torch writes it since 17a3069 of 2026-07-31, first released in 0.9.4. An extension can replace the function that builds it, not drop the section.
- With the quantizer in the environment above, the int4 weights differ from the published file's, because litert-quantizer-nightly 0.12.0.dev20261105 changed how OCTAV picks its clipping range. The tensors checked, quantized again from the checkpoint with ai-edge-quantizer 0.8.0, give the published file's int4 values and scales.
- The prefill signature has one reshape fewer before the rotary angles: transformers 5.20.2 multiplies the positions by the inverse frequencies, where the earlier recipe's 5.14.1 takes a matrix product with an inner size of 1. Through the same converter, the two forms give the two files' operators and bit-equal angles.
- The section is 2,768 B smaller: the reshape and the rotary tensors' names. The weights take the same bytes.

<details>
<summary>The int4 difference, measured</summary>

- In the prefill and decode graphs all 253 int4 weights differ, in about 2.3 % of their values, never by more than one step, and in 99 % of their fp16 block scales. No value differs in a block whose scale is equal.
- OCTAV weighs the rounding noise of a 4-bit step at `4.0 ** (-bits) / exponent_divisor`, 1/768, in ai-edge-quantizer 0.8.0, and at `1.0 / (4.0 * exponent_divisor * half_range_steps**2)`, 1/588, from litert-quantizer-nightly 0.12.0.dev20261105 on, the first nightly with it. For symmetric int4 this is the one computational change in OCTAV's code between ai-edge-quantizer 0.8.0 and litert-quantizer-nightly 0.12.0.dev20261111.
- Three tensors were quantized again from the checkpoint: layer 0's `q_proj`, layer 17's `down_proj` and layer 35's `k_proj`. ai-edge-quantizer 0.8.0 gives the published file's int4 values and scales, litert-quantizer-nightly 0.12.0.dev20261111 gives this recipe's, and with the two noise weights swapped each gives the other file's.

</details>

The first published file, revision `65383ca9`, has the same graphs, tokenizer and stops, byte for byte, and no Jinja template or thought channel. Revision `1fa937cc`, now on `main`, has the published file's metadata and tokenizer, byte for byte, and the rewritten GPU graph with the weights in their own sections.

### Failure paths

- litert-torch 0.10.0, the latest release: `build_vibethinker_3b.py` stops before the export.

  ```
  litert-torch 0.10.0 cannot load vibethinker_extension.py: the recipe needs a .py file in extension_modules (litert-torch 864a966, 2026-10-28) and ModelExtension.get_export_defaults (764beb0, 2026-11-03), which no release carries yet. Install the version in the README.
  ```

- A checkpoint whose template has another default system prompt: the graphs are converted, then the export stops with this, and the script leaves no file.

  ```
  ValueError: vibethinker_extension: the chat template renders '<|im_start|>system\nYou are VibeThinker, a math assistant.<|im_end|>\n<|im_start|>user\nx<|im_end|>\n<|im_start|>assistant\n' for one user turn, without the default system turn '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'; the source template changed
  ```
- Without the extension: see [Without the extension](#without-the-extension).

### What is not measured

- Positions past 2,048 on a phone GPU. litert-torch turned its rope composite on by default because with plain operators, which this recipe keeps from the published file, ML Drift's fp16 GPU path cannot hold such a position exactly (0567035).
- Speed. The model card's numbers are the published file's.

## Files

| File | What |
|---|---|
| `vibethinker_extension.py` | The out-of-tree `ModelExtension`: `get_export_defaults` (the export arguments) and `build_llm_metadata` (the source check, the stops, the Jinja template, the thought channel). |
| `vibethinker_extension_test.py` | `ModelExtensionTestMixin` contract tests and 13 cases, among them a `torch.export` of a random 2-layer Qwen2, the plan with and without the extension, and source and template changes that stop the export. CPU, no weights. |
| `recipe_int4.json` | The quantization recipe. |
| `build_vibethinker_3b.py` | HF checkpoint → the `.litertlm` file, one stock `export()` call with the extension; `--dry-run` prints the plan. |
| `verify_vibethinker_3b.py` | The 8-question check through the `litert-lm` command-line tool, scored on the answer after the thought; `--check-metadata`; `--math`, six word problems; `--sections`, a section-by-section comparison with another file. |
