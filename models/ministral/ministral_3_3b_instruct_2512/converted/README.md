# Ministral-3-3B-Instruct-2512 Conversion

Model recipes, instructions, scripts, and utilities for converting Mistral AI Ministral-3-3B-Instruct-2512 (text decoder) to LiteRT-LM.

This recipe rebuilds `Ministral-3-3B-Instruct-2512_q4_block32_ekv4096.litertlm`, the file of [litert-community/Ministral-3-3B-Instruct-2512](https://huggingface.co/litert-community/Ministral-3-3B-Instruct-2512), from [mistralai/Ministral-3-3B-Instruct-2512-BF16](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16) at revision `b6d637be`. The build is one call of litert-torch's stock exporter with `ministral3_extension.py`, an out-of-tree `ModelExtension` that reads the text decoder straight out of the multimodal checkpoint. The file was checked on a Mac with an M4 Max and on a Galaxy S26. To build it, run `python build_ministral_3_3b.py --out out_int4` in the environment below. On litert-torch 0.10.0, the latest release, the build stops before the export and says why.

## Build

| file | weights | prefill | KV cache | size |
|---|---|---|---|---|
| `Ministral-3-3B-Instruct-2512_q4_block32_ekv4096.litertlm` | int4 blockwise-32 with OCTAV clipping, the embedding int8 per channel in its own section | 128 tokens | 4,096 tokens | 2,340,999,152 B |

## Environment

```bash
pip install litert-torch-nightly==0.11.0.dev20261111 torch==2.14.0 transformers==5.20.2 \
    litert-lm-builder==0.19.1 litert-lm==0.19.1 litert-lm-api==0.19.1 \
    litert-converter==0.6.0.dev20261110 litert-quantizer-nightly==0.12.0.dev20261111 \
    ai-edge-litert-nightly==2.4.0.dev20261110 protobuf==7.36.2 numpy==2.5.4
```

The plugin needs litert-torch's out-of-tree extension files, which no release has yet. The export peaked at about 44 GiB of memory on a Mac.

## Run

```bash
python ministral3_extension_test.py                    # the plugin's tests: CPU, no weights
python build_ministral_3_3b.py --out out_int4          # downloads 7.7 GB, writes the file
F=out_int4/Ministral-3-3B-Instruct-2512_q4_block32_ekv4096.litertlm
python verify_ministral_3_3b.py $F --backend cpu       # 8 questions, greedy
python verify_ministral_3_3b.py $F --backend gpu
python verify_ministral_3_3b.py $F --check-metadata
python verify_ministral_3_3b.py --bos-ab               # the BF16 model with and without <s>
python verify_ministral_3_3b.py $F --sections other.litertlm   # section by section against another file
```

The download is the checkpoint's two Hugging Face shards and small files. It skips `consolidated.safetensors`, a second copy of the same weights in Mistral's own layout. The same export from the command line is `litert-torch export_hf --model=<checkpoint> --output_dir=<dir> --extension_modules=ministral3_extension.py`.

## The conversion is a `ModelExtension`

The checkpoint is `Mistral3ForConditionalGeneration`: a Pixtral vision encoder, a projector and a 26-layer Ministral3 text decoder, whose weights sit under `language_model.model.`. Three hooks load the decoder alone, and three set the published file's form:

| hook | what it sets |
|---|---|
| `load_config` | The checkpoint's text config, `ministral3`, in place of its `mistral3` config. |
| `get_hf_model_cls` | A constructor of transformers' `Ministral3ForCausalLM` from that config. It has no `from_pretrained`, so the exporter fills every weight from the checkpoint with its own loader. |
| `checkpoint_key_prefixes` | `language_model.model.`, the prefix of the decoder's weights. The vision encoder and projector weights are not loaded. |
| `get_export_defaults` | The published file's export arguments: `recipe_int4.json`, one 128-token prefill signature, a 4,096-token KV cache and limit, and the embedding table in its own section. It turns off GPU composite ops, weights outside the TFLite model and the rope composite, which the exporter's default CPU/GPU preset turns on. |
| `update_export_config` | The checkpoint's own `tokenizer.json` as the tokenizer section. It also checks the chat template before the conversion. |
| `build_llm_metadata` | The chat template, the `[INST]` and `[SYSTEM_PROMPT]` role markers, the start token `<s>`, the stop `</s>` (id 2) alone, and no pad token. |

Read this way, the decoder's 237 tensors are bit-identical to those of a standalone `Ministral3ForCausalLM` copied out of the multimodal model, which the earlier version of this recipe wrote to disk first. The two configs differ only in `_name_or_path` and `architectures`. The exporter writes an ExecutorMetadata section that declares the KV cache of all 26 layers.

## The revision and the tokenizer

The build reads revision `b6d637be`, the checkpoint's `main` on 2026-11-12. Its weight files are the same as at `ecc3ba8b`, the `main` of 2026-06-23 when the published file was first uploaded. The two revisions differ in the tokenizer and the chat template.

The tokenizer section is the checkpoint's `tokenizer.json`, byte for byte. At `b6d637be` it differs from the published file's tokenizer in ids 34 and 35 only, which Mistral renamed from `[THINK]` and `[/THINK]` to `<SPECIAL_34>` and `<SPECIAL_35>` on 2026-06-30. Neither token is in the chat template. The text `[THINK]` in a prompt is now four ordinary tokens instead of id 34.

## The chat template

The template is the published file's, byte for byte. It writes Ministral's default system prompt when the caller sends no system message, and it reads message content as a string or as LiteRT-LM's list of content parts. The published file's first upload, on 2026-06-23, had no chat template. It got the default system prompt on 2026-08-28 and the content-parts form on 2026-09-21.

The plugin builds this template at export time from the checkpoint's template. It renders the checkpoint's template for one user message and takes the default system turn in front of it. Two checks stop the export right after the model loads, before the conversion: the assembled template must hash to the published one, and it must render as the checkpoint's template does for a one-turn, a three-turn and a system-message conversation, after the `<s>` that the checkpoint's template writes first.

The tests change the default turn and a role marker, and both stop. So does the real checkpoint at `ecc3ba8b`, where the default system prompt's sentences on audio and video read differently.

## From this file to the Hub's `main`

This recipe writes the file of revision [`613e437f`](https://huggingface.co/litert-community/Ministral-3-3B-Instruct-2512/tree/613e437fea973ca9e04a4063953ee514f1d4075e) of 2026-09-21. The file on `main` came from it by two rewrites with the tools of [hf-to-litertlm](https://github.com/john-rocky/hf-to-litertlm). On 2026-10-05, in [`85b1913e`](https://huggingface.co/litert-community/Ministral-3-3B-Instruct-2512/commit/85b1913e39cedd484e0c174fcf1638ea6daf227e), [`tools/gpu_graph`](https://github.com/john-rocky/hf-to-litertlm/tree/main/tools/gpu_graph) rewrote the KV-cache update and attention into the runtime's GPU composite ops and added a 16-token prefill signature. On 2026-10-30, in [`cb1a823a`](https://huggingface.co/litert-community/Ministral-3-3B-Instruct-2512/commit/cb1a823a3689405cc524bfe3ae051e58529422fd), [`tools/externalize`](https://github.com/john-rocky/hf-to-litertlm/tree/main/tools/externalize) moved the weights into TFLiteWeights sections. Neither changed the metadata: the LlmMetadata section on `main` is byte-equal to this revision's.

## Quantization

`recipe_int4.json` is the published file's quantization: int4 blockwise-32 weights with OCTAV clipping, and int8 per channel on the embedding table. With the quantizer in the environment above, the int4 weights differ from the published file's, because litert-quantizer-nightly 0.12.0.dev20261105 changed how OCTAV picks its clipping range. The tensors checked, quantized again from the checkpoint with ai-edge-quantizer 0.8.0, give the published file's int4 values and scales.

## Verification results

The 8 questions, greedy, on this file and on the published file of revision `613e437f`, in the same session:

| run | this file | `613e437f` |
|---|---|---|
| Mac CPU, 4 threads | 8/8 | 8/8 |
| Mac GPU | 8/8 | 8/8 |

On both backends the two files give the same reply to every question, byte for byte, although their int4 values differ.

`--check-metadata` passes 11 of 11 on both files. One of its checks follows the Hub's later fixes: the chat template must be the published one, of 2026-09-21. `--bos-ab` gives the BF16 model the same rendered prompt with and without the leading `<s>`, and both forms answer its three questions. So the file keeps `<s>` as its start token, as the earlier version of this recipe did.

- **Galaxy S26**, through LiteRT-LM v0.19.1's `litert_lm_advanced_main` for Android, greedy, one process per question: this recipe's file answered the 8 questions 8/8 on the CPU and 8/8 on the GPU. Each reply is the same text as on the Mac. The published file was not run on the phone.

Section by section against `613e437f`, the LlmMetadata is byte-equal, and the tokenizer differs only in ids 34 and 35, as above. The ExecutorMetadata section is only in this file, and the bundle header is version 1.9.0 against 1.6.0. Both TFLite models have the same signatures, subgraphs, composite ops and int4 tensors. Apart from the int4 weights and their block scales, every weight buffer is byte-equal. The LlmMetadata of the first upload, `a540524f`, lacks the template, and that of `main` equals this file's.

The prefill/decode model has one RESHAPE fewer, as in the SmolLM3 recipe's comparison. Ministral3's rotary embedding changed in transformers 5.19.0 the same way as SmolLM3's, from a matrix product to a multiplication, and the published file has the earlier form's two RESHAPEs of the token positions and one 12-byte buffer more.

<details>
<summary>The int4 values, traced</summary>

28 weights of four layers were quantized again from the checkpoint with the recipe's settings. ai-edge-quantizer 0.7.0 and 0.8.0 give the published file's int4 values and scales, 28 of 28, and this environment's nightly gives this file's, 28 of 28. The same nightly with only `_guess_clipping_with_octav` taken from 0.8.0 gives the published file's again. That function weighs the quantization noise at 1/768 for int4 in 0.8.0 and at 1/588 in the nightly. The nightlies changed it on 0.12.0.dev20261105, and the releases ai-edge-quantizer 0.9.0 to 0.10.1 and litert-quantizer 0.11.0 keep 1/768. In those 28 weights, about 2.4 % of the int4 values differ between the two files.

</details>

### Failure paths

- litert-torch 0.10.0: `build_ministral_3_3b.py` exits before the export with `litert-torch 0.10.0: this litert-torch cannot load ministral3_extension.py: …`, and writes nothing.
- The checkpoint at `ecc3ba8b`, or a template changed in a test: the export stops in `update_export_config`, before the conversion, with `the checkpoint template changed` or `renders differently`.
- Without the plugin (`--dry-run --no-extension`): the exporter refuses the checkpoint, `ValueError: … has architectures ['Mistral3ForConditionalGeneration'] (model_type 'mistral3'), which no litert-torch ModelExtension handles and which are not an AutoModelForCausalLM.` With the decoder loaded as above but the stock metadata, the file would have 15 stops, the EOS id and `</s>` as text, alone and with 13 characters in front, and `<pad>` as the pad token.

### What is not measured

No task benchmark: the 8 questions are a smoke test. No speed. No iOS. No image input: this file is the text decoder only.

## Files

| File | What |
|---|---|
| `ministral3_extension.py` | The out-of-tree `ModelExtension`: the decoder loaded from the multimodal checkpoint, export defaults, the tokenizer section and the metadata, with the template built from the checkpoint and checked. |
| `ministral3_extension_test.py` | `ModelExtensionTestMixin` contract tests and 11 cases of its own: a tiny prefill and decode `torch.export`, the exporter's loader on a tiny checkpoint with the real layout, the export defaults, the tokenizer section, the template and the three ways it stops, the metadata from litert-torch's own builder, and the metadata without the plugin. CPU, about 17 MB of downloads. |
| `recipe_int4.json` | The quantization recipe. |
| `build_ministral_3_3b.py` | The checkpoint at `b6d637be` to the `.litertlm` file in one `export()` call; `--dry-run` prints the export arguments, `--no-extension` leaves the plugin out. |
| `verify_ministral_3_3b.py` | 8 questions through the `litert-lm` CLI, `--check-metadata`, `--bos-ab` and `--sections`. |
