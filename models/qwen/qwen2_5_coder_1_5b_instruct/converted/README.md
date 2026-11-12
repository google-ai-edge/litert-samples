# Qwen2.5-Coder-1.5B-Instruct Conversion

Model recipes, instructions, scripts, and utilities for converting Qwen2.5-Coder-1.5B-Instruct to LiteRT-LM.

These scripts rebuild `Qwen2.5-Coder-1.5B-Instruct_int4.litertlm` of [litert-community/Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/litert-community/Qwen2.5-Coder-1.5B-Instruct) from the [Qwen/Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) checkpoint, with litert-torch's stock exporter and one `ModelExtension` plugin. The file was checked on a Mac with an M4 Max and on a Galaxy S26. To build it, run `python build_qwen2_5_coder_1_5b.py --out out_int4` in the environment below. On litert-torch 0.10.0, the latest release, the build stops before the export and says why.

The checkpoint is a stock `Qwen2ForCausalLM`: 28 layers, hidden 1,536, 12 query and 2 key-value heads, a 151,936-token vocabulary tied to the output layer, 1.54 B parameters, Apache-2.0. The exporter traces it as is, and the extension adds the published file's recipe and two facts of this model's template.

## Build

| file | weights | prefill signatures | KV cache | size |
|---|---|---|---|---|
| `Qwen2.5-Coder-1.5B-Instruct_int4.litertlm` | int4 blockwise-32 (OCTAV) on every weight, the embedding int8 per channel in its own section (`recipe_int4.json`) | 1,024, 256, 64, 16, 4 and 1 tokens | 4,096 | 1.13 GB |

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
python qwen2_5_coder_extension_test.py              # the extension's tests, CPU, no weights
python build_qwen2_5_coder_1_5b.py --out out_int4   # the file, a few minutes on an M4 Max
python verify_qwen2_5_coder_1_5b.py out_int4/Qwen2.5-Coder-1.5B-Instruct_int4.litertlm --backend cpu
python verify_qwen2_5_coder_1_5b.py out_int4/Qwen2.5-Coder-1.5B-Instruct_int4.litertlm --backend gpu
python verify_qwen2_5_coder_1_5b.py out_int4/Qwen2.5-Coder-1.5B-Instruct_int4.litertlm --check-metadata
python verify_qwen2_5_coder_1_5b.py out_int4/Qwen2.5-Coder-1.5B-Instruct_int4.litertlm --code-gate --backend gpu
python verify_qwen2_5_coder_1_5b.py out_int4/Qwen2.5-Coder-1.5B-Instruct_int4.litertlm --sections published.litertlm
```

The same export from the command line, with a local copy of the checkpoint in `$SRC`:

```bash
litert-torch export_hf --model=$SRC --output_dir=out_cli --extension_modules=qwen2_5_coder_extension.py
```

## The extension

`qwen2_5_coder_extension.py` is an out-of-tree `ModelExtension`. It holds every export argument and metadata field the published file needs, so litert-torch stays unchanged and nothing edits the file after the export. It registers for `model_type` `qwen2`, replaces litert-torch's built-in Qwen2 extension for the export it is passed to, patches no module and overrides two hooks:

| hook | what it sets | why |
|---|---|---|
| `get_export_defaults` | `quantization_recipe` = `recipe_int4.json` | the published file's weights |
| | `externalize_embedder=True` | one copy of the tied 151,936 × 1,536 table, in its own section, for every signature |
| | `prefill_lengths=[1024, 256, 64, 16, 4, 1]`, `cache_length=4096`, `llm_metadata_max_num_tokens_override=None` | the published signatures and context; `max_num_tokens` is the cache length |
| | `use_jinja_template=False` | per-role prompt templates, as published |
| | `apply_gpu_composites=False`, `use_rope_composite=False`, `externalize_weights=False` | the published graph, without composite operators |
| `build_llm_metadata` | the vendor's default system turn at the head of the user prefix | the source template adds it whenever the caller sends no system message |
| | the 16 stops in a fixed order | the exporter collects them in a Python set, whose order changes from one process to the next |

Arguments a caller passes to `export()` still win. The defaults refuse another checkpoint's `config.json` and the `npu` target.

`build_llm_metadata` runs once the graphs are converted. Before it writes anything, it renders the checkpoint's chat template for a user turn alone and after a system turn, compares the text with what the bundle's templates give, and checks the ids of `<|im_end|>` and `<|endoftext|>`. If the vendor changes the default system prompt or the turn markers, the export stops there with `ValueError`, and `build_qwen2_5_coder_1_5b.py` leaves no file.

## Without the extension

`python build_qwen2_5_coder_1_5b.py --dry-run --no-extension` prints the stock plan: `dynamic_wi8_afp32` weights with the embedding in the graph, prefill signatures of 128 and 1,024 tokens over a 32,771-token cache, GPU composite operators, and LiteRT-LM's canonical Qwen2.5 metadata. At LiteRT-LM commit 61fd138 that metadata has the start token `<|endoftext|>`, which this tokenizer never adds, stops on the text `<|im_end|>` alone and has no default system turn.

## After the export

On 2026-10-30 the Hub's `main` moved to revision `0cb76df9`, the copy from the repo's `web` branch, whose weights sit in `TFLiteWeights` sections next to the graphs, as hf-to-litertlm's [`tools/externalize`](https://github.com/john-rocky/hf-to-litertlm/tree/main/tools/externalize) makes them from a finished file; its commit notes that Android GPU then needs LiteRT-LM 0.19.0 or later. The GPU graph rewrite, [`tools/gpu_graph`](https://github.com/john-rocky/hf-to-litertlm/tree/main/tools/gpu_graph), was not applied to this model. Both tools run on a finished file, so neither is in this recipe.

## Verification results

Measured on 2026-11-12 on a Mac with an M4 Max in the environment above, and on a Galaxy S26. "This recipe" is a file built by `build_qwen2_5_coder_1_5b.py` from the checkpoint at revision `2e1fd397`. "Published" is the file of the same name at revision `6fbe02e7`, the last one with the weights inside the graphs, its sha256 checked against the Hub.

- **`qwen2_5_coder_extension_test.py`**: 18/18 on CPU.
- **Build**: 1,126,592,352 B in a few minutes, with a peak of 21 GB of memory. litert-lm-builder 0.19.1 writes container version 1.9.0, the published file has 1.6.0.
- **Command line**: `litert-torch export_hf` with the extension gave every section byte-equal to the file of `build_qwen2_5_coder_1_5b.py`, the stop order included. The two files differ only in the header's uuid and creation time.
- **Checks** through the litert-lm 0.19.1 command-line tool, greedy, `--cache no`, 4 CPU threads:

| check | this recipe | published |
|---|---|---|
| 8 questions, CPU | 8/8 | 8/8 |
| 8 questions, GPU | 8/8 | 8/8 |
| `--check-metadata` | 7/7 | 7/7 |
| `--code-gate`, GPU | 6/6 | 6/6 |

  Six of the eight replies are byte-identical to the published file's on each backend. The other two are worded differently, the same way on both backends, as in "Violets are blue." against "Violet are blue." for the rhyme. Five of the six generated functions are identical.

- **Galaxy S26**, through LiteRT-LM v0.19.1's `litert_lm_advanced_main` for Android, greedy, one process per question: this recipe's file answered the 8 questions 8/8 on the CPU and 8/8 on the GPU. Each reply is the same text as on the Mac, except the GPU's rhyme, "Violet are blue.". The published file was not run on the phone.

- **`--sections`** against the published file:

| section | this recipe | published | difference |
|---|---:|---:|---|
| `LlmMetadataProto` | 476 B | 459 B | a pad token, and the same 16 stops in another order |
| `ExecutorMetadataProto` | 3,790 B | none | new |
| `HF_Tokenizer_Zlib` | 2,142,126 B | 2,142,126 B | none, byte-equal |
| `TFLiteModel`, prefill and decode | 878,239,792 B | 878,182,176 B | the int4 weights, one reshape per prefill signature |
| `TFLiteModel`, embedder | 246,148,960 B | 237,024,176 B | 7 signatures against 2 |

The published file was built with litert-torch 0.9.3. Each difference has a measured cause:

- The pad token is the tokenizer's `<|endoftext|>`, which litert-torch writes since commit 77e1fea of 2026-08-28, first released in 0.10.0.
- The published file has the stops in the order of a Python set in the process that built it. The hook writes them in a fixed order.
- The executor section holds `max_history_size: 4096` and the 28 layers' key and value caches by name. litert-torch writes it since 17a3069 of 2026-07-31, first released in 0.9.4. An extension can replace the function that builds it, not drop the section.
- The embedder has a signature per prefill length since 18aa7ee of 2026-08-04, first released in 0.9.4. 0.9.3 exported the decode signature and the first prefill length only. Each signature carries the table's per-channel scales and zero points, 1,823,232 B, so five more signatures add 9,116,160 B. The table is byte-equal.
- With the quantizer in the environment above, the int4 weights differ from the published file's, because litert-quantizer-nightly 0.12.0.dev20261105 changed how OCTAV picks its clipping range. The tensors checked, quantized again from the checkpoint with ai-edge-quantizer 0.8.0, give the published file's int4 values and scales.
- Each prefill signature longer than one token has one reshape fewer before the rotary angles: transformers 5.20.2 multiplies the positions by the inverse frequencies, where the earlier recipe's 5.14.1 takes a matrix product with an inner size of 1. Through the same converter, the two forms give the two files' operators and bit-equal angles.
- The section is 57,616 B larger: its tensor names are 58,598 B longer, and the five reshapes are gone. The weights take the same bytes.

<details>
<summary>The int4 difference, measured</summary>

- In the prefill and decode graphs all 197 int4 weights differ, in about 2.3 % of their values, never by more than one step, and in 99 % of their fp16 block scales. No value differs in a block whose scale is equal.
- OCTAV weighs the rounding noise of a 4-bit step at `4.0 ** (-bits) / exponent_divisor`, 1/768, in ai-edge-quantizer 0.8.0, and at `1.0 / (4.0 * exponent_divisor * half_range_steps**2)`, 1/588, from litert-quantizer-nightly 0.12.0.dev20261105 on, the first nightly with it. For symmetric int4 this is the one computational change in OCTAV's code between ai-edge-quantizer 0.8.0 and litert-quantizer-nightly 0.12.0.dev20261111.
- Three tensors were quantized again from the checkpoint: layer 0's `q_proj`, layer 13's `down_proj` and layer 27's `k_proj`. ai-edge-quantizer 0.8.0 gives the published file's int4 values and scales, litert-quantizer-nightly 0.12.0.dev20261111 gives this recipe's, and with the two noise weights swapped each gives the other file's.

</details>

Revision `0cb76df9`, now on `main`, has the published file's metadata, tokenizer and graphs, with the weights moved into their own sections.

### Failure paths

- litert-torch 0.10.0, the latest release: `build_qwen2_5_coder_1_5b.py` stops before the export.

  ```
  litert-torch 0.10.0 cannot load qwen2_5_coder_extension.py: the recipe needs a .py file in extension_modules (litert-torch 864a966, 2026-10-28) and ModelExtension.get_export_defaults (764beb0, 2026-11-03), which no release carries yet. Install the version in the README.
  ```

- A checkpoint whose template has another default system prompt: the graphs are converted, then the export stops with this, and the script leaves no file.

  ```
  ValueError: qwen2_5_coder_extension: the chat template renders '<|im_start|>system\nYou are Qwen, a coding assistant.<|im_end|>\n<|im_start|>user\nx<|im_end|>\n<|im_start|>assistant\n' for [{'role': 'user', 'content': 'x'}], not '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nx<|im_end|>\n<|im_start|>assistant\n'; the source template changed, re-check the default system turn
  ```
- Without the extension, the stock exporter builds another file, see [Without the extension](#without-the-extension).

### What is not measured

- Positions past 2,048 on a phone GPU. litert-torch turned its rope composite on by default because with plain operators, which this recipe keeps from the published file, ML Drift's fp16 GPU path cannot hold such a position exactly (0567035).
- Speed. The model card's numbers are the published file's.

## Files

| File | What |
|---|---|
| `qwen2_5_coder_extension.py` | The out-of-tree `ModelExtension`: `get_export_defaults` (the export arguments) and `build_llm_metadata` (the template check, the default system turn, the stop order). |
| `qwen2_5_coder_extension_test.py` | `ModelExtensionTestMixin` contract tests and 13 cases, among them a `torch.export` of a random 2-layer Qwen2, the plan with and without the extension, and two template changes that stop the export. CPU, no weights. |
| `recipe_int4.json` | The quantization recipe. |
| `build_qwen2_5_coder_1_5b.py` | HF checkpoint → the `.litertlm` file, one stock `export()` call with the extension; `--dry-run` prints the plan. |
| `verify_qwen2_5_coder_1_5b.py` | The 8-question check through the `litert-lm` command-line tool; `--check-metadata`; `--code-gate`, six generated functions executed against assertions; `--sections`, a section-by-section comparison with another file. |
