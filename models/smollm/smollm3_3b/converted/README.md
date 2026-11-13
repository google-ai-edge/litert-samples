# SmolLM3-3B Conversion

Model recipes, instructions, scripts, and utilities for converting Hugging Face SmolLM3-3B to LiteRT-LM.

This recipe rebuilds `SmolLM3-3B_q4_block32_ekv4096.litertlm`, the int4 file of [litert-community/SmolLM3-3B](https://huggingface.co/litert-community/SmolLM3-3B), from [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) at revision `a07cc9a0`. The build is one call of litert-torch's stock exporter with `smollm3_extension.py`, an out-of-tree `ModelExtension`. The file was checked on a Mac with an M4 Max and on a Galaxy S26. To build it, run `python build_smollm3_3b.py --out out_int4` in the environment below. On litert-torch 0.10.0, the latest release, the build stops before the export and says why.

## Build

| file | weights | prefill | KV cache | size |
|---|---|---|---|---|
| `SmolLM3-3B_q4_block32_ekv4096.litertlm` | int4 blockwise-32 with OCTAV clipping, the embedding int8 per channel in its own section | 128 tokens | 4,096 tokens | 2,002,274,224 B |

The repo's other file, `SmolLM3-3B.litertlm` of about 3.1 GB, is the int8 form, and this recipe does not build it.

## Environment

```bash
pip install litert-torch-nightly==0.11.0.dev20261111 torch==2.14.0 transformers==5.20.2 \
    litert-lm-builder==0.19.1 litert-lm==0.19.1 litert-lm-api==0.19.1 \
    litert-converter==0.6.0.dev20261110 litert-quantizer-nightly==0.12.0.dev20261111 \
    ai-edge-litert-nightly==2.4.0.dev20261110 protobuf==7.36.2 numpy==2.5.4
```

The plugin needs litert-torch's out-of-tree extension files, which no release has yet. The export peaked at about 37 GiB of memory on a Mac.

## Run

```bash
python smollm3_extension_test.py                       # the plugin's tests: CPU, no weights
python build_smollm3_3b.py --out out_int4              # downloads 6.2 GB, writes the file
F=out_int4/SmolLM3-3B_q4_block32_ekv4096.litertlm
python verify_smollm3_3b.py $F --backend cpu           # 8 questions, greedy
python verify_smollm3_3b.py $F --backend gpu
python verify_smollm3_3b.py $F --check-metadata
python verify_smollm3_3b.py $F --think-ab              # reasoning with and without a system message
python verify_smollm3_3b.py $F --sections other.litertlm   # section by section against another file
```

The same export from the command line is `litert-torch export_hf --model=<checkpoint> --output_dir=<dir> --extension_modules=smollm3_extension.py`. Its `model.litertlm` is byte-equal to the Python call's file, apart from the creation time and uuid that the bundle header records.

## The conversion is a `ModelExtension`

| hook | what it sets |
|---|---|
| `get_export_defaults` | The published file's export arguments: `recipe_int4.json`, one 128-token prefill signature, a 4,096-token KV cache and limit, and the embedding table in its own section. It turns off three things the exporter's default CPU/GPU preset turns on: GPU composite ops, weights outside the TFLite model and the rope composite. |
| `update_export_config` | The checkpoint's own `tokenizer.json` as the tokenizer section, instead of the tokenizer that transformers saves again. It also checks the chat template before the conversion. |
| `build_llm_metadata` | The chat template, bare ChatML role markers, the stops `<\|im_end\|>` (id 128012) and `"<\|im_end\|>\n"`, no pad token, and the thought channel `<think>\n` … `\n</think>`. |

What the exporter already does for this model, unchanged:

- it loads `SmolLM3ForCausalLM` from transformers in float32, with the output layer tied to the embedding;
- the layers without rotary embeddings, every fourth layer, trace through the same attention path as the others, and in the written file their query weights carry the attention scale, 1/√128;
- it writes an ExecutorMetadata section that declares the KV cache of all 36 layers;
- the stock packager writes the bundle.

## The chat template

The template is the published file's, byte for byte. It writes SmolLM3's default system turn when the caller sends no system message: the metadata header with today's date, `Reasoning Mode: /think` and the default instructions. It reads message content as a string or as LiteRT-LM's list of content parts. A system message from the caller is written as bare ChatML, where the checkpoint's template puts it under the metadata header.

The plugin builds this template at export time from the checkpoint's template. It renders the checkpoint's template for one user message, takes the default system turn in front of it, and keeps the date as a live `strftime_now` call. Two checks stop the export right after the model loads, before the conversion: the assembled template must hash to the published one, and it must render as the checkpoint's template does for a one-turn and a three-turn conversation. The tests change the default turn and a role marker, and both stop.

The earlier version of this recipe replaced the checkpoint's template with bare ChatML at run time. The published file's first upload, on 2026-06-23, had bare ChatML role markers and no chat template. It got the default system turn on 2026-08-28, the thought channel on 2026-08-31 and the content-parts form on 2026-09-21. The plugin writes all three.

## From this file to the Hub's `main`

This recipe writes the file of revision [`c29e7655`](https://huggingface.co/litert-community/SmolLM3-3B/tree/c29e7655125fe754c483dfdf5e31a10acf0a51fb) of 2026-09-21. The file on `main` came from it by two rewrites with the tools of [hf-to-litertlm](https://github.com/john-rocky/hf-to-litertlm). On 2026-10-05, in [`af261561`](https://huggingface.co/litert-community/SmolLM3-3B/commit/af261561ec8d30a53034bbd19aa399fc87e69dca), [`tools/gpu_graph`](https://github.com/john-rocky/hf-to-litertlm/tree/main/tools/gpu_graph) rewrote the KV-cache update and attention into the runtime's GPU composite ops and added a 16-token prefill signature. On 2026-10-30, in [`a8567034`](https://huggingface.co/litert-community/SmolLM3-3B/commit/a85670349a17630b34771dc18e70f8d98bda1f68), [`tools/externalize`](https://github.com/john-rocky/hf-to-litertlm/tree/main/tools/externalize) moved the weights into TFLiteWeights sections. Neither changed the metadata: the LlmMetadata section on `main` is byte-equal to this revision's.

## Quantization

`recipe_int4.json` is the published file's quantization: int4 blockwise-32 weights with OCTAV clipping, and int8 per channel on the embedding table. With the quantizer in the environment above, the int4 weights differ from the published file's, because litert-quantizer-nightly 0.12.0.dev20261105 changed how OCTAV picks its clipping range. The tensors checked, quantized again from the checkpoint with ai-edge-quantizer 0.8.0, give the published file's int4 values and scales.

## Verification results

The 8 questions, greedy, on this file and on the published file of revision `c29e7655`. The Mac rows come from one session, and the Galaxy S26 rows from LiteRT-LM v0.19.1's `litert_lm_advanced_main` for Android, one process per question:

| run | this file | `c29e7655` |
|---|---|---|
| Mac CPU, 4 threads | 7/8: the rhyme's thought runs to the 4,096-token limit | 6/8: "hot" and the rhyme run to the limit |
| Mac CPU, 4 threads, "hot" again in a new process | correct | correct |
| Mac CPU, 1 thread, 1,024-token cap | 6/8: "hot" and the rhyme run to the cap | 7/8: the rhyme runs to the cap |
| Mac GPU | 8/8 | 6/8: the capital runs to the limit, the rhyme ends "violets are red" |
| Mac GPU, the capital and the rhyme again | the same replies | the same replies |
| Galaxy S26 CPU | 6/8: "hot" and the rhyme run to the 1,024-token cap | 7/8: the rhyme runs to the cap |
| Galaxy S26 GPU | 7/8: the rhyme runs to the cap | 7/8: the rhyme runs to the cap |

No reply of this file is byte-equal to the published file's, which the different int4 values explain. With greedy decoding a reply can change in a new process. On the Mac's CPU, "hot" went either way for both files.

`--check-metadata` passes 11 of 11 on both files. Two of its checks follow the Hub's later fixes: the chat template must be the published one, of 2026-09-21, and the thought channel must be `<think>\n` … `\n</think>`, of 2026-08-31. `--think-ab` gives both files a thought and the answer 42, with the default system turn and with SmolLM3's `/think` prompt as a system message.

Section by section against `c29e7655`, the LlmMetadata and the tokenizer are byte-equal. The ExecutorMetadata section is only in this file, and the bundle header is version 1.9.0 against 1.6.0. Both TFLite models have the same signatures, subgraphs, composite ops and int4 tensors. Apart from the int4 weights and their block scales, every weight buffer is byte-equal. The LlmMetadata of the first upload, `a2f0c541`, lacks the template and the thought channel, and that of `main` equals this file's.

The prefill/decode model has one RESHAPE fewer. The published file reshapes the rotary embedding's token positions twice and has one 12-byte buffer more. That comes from transformers: since 5.19.0, the rotary embedding multiplies the positions by the frequencies, where earlier versions took a matrix product. A tiny SmolLM3 exported with only that function changed back to the 5.14.1 form gets the published file's two RESHAPEs and its extra buffer.

<details>
<summary>The int4 values, traced</summary>

Compared tensor by tensor, no int4 weight of this file is byte-equal to the published file's: about 2.4 % of the int4 values differ, by one step at most, and the block scales are about 1.1 % smaller. 33 weights of five layers were quantized again from the checkpoint with the recipe's settings. ai-edge-quantizer 0.7.0 and 0.8.0 give the published file's int4 values and scales, 33 of 33, and this environment's nightly gives this file's, 33 of 33. The same nightly with only `_guess_clipping_with_octav` taken from 0.8.0 gives the published file's again. That function weighs the quantization noise at 1/768 for int4 in 0.8.0 and at 1/588 in the nightly. The nightlies changed it on 0.12.0.dev20261105, and the releases ai-edge-quantizer 0.9.0 to 0.10.1 and litert-quantizer 0.11.0 keep 1/768.

</details>

### Failure paths

- litert-torch 0.10.0: `build_smollm3_3b.py` exits before the export with `litert-torch 0.10.0: this litert-torch cannot load smollm3_extension.py: …`, and writes nothing.
- A changed checkpoint template: the export stops in `update_export_config`, before the conversion, with `the checkpoint template changed` or `renders differently`. The tests show both.
- Without the plugin (`--dry-run --no-extension`): the exporter takes its CPU/GPU preset, with GPU composite ops, external weights, a 32,771-token cache, prefill signatures of 128 and 1,024 tokens, the int8 recipe `dynamic_wi8_afp32` and the embedding table inside the model. Its metadata has 15 stops, `<|im_end|>` as the pad token, the thought channel without its newlines and the checkpoint's template with the exporter's own edits.

### What is not measured

No task benchmark: the 8 questions are a smoke test. No speed. No iOS.

## Files

| File | What |
|---|---|
| `smollm3_extension.py` | The out-of-tree `ModelExtension`: export defaults, the tokenizer section and the metadata, with the template built from the checkpoint and checked. |
| `smollm3_extension_test.py` | `ModelExtensionTestMixin` contract tests and 10 cases of its own: a tiny prefill and decode `torch.export`, the export defaults, the tokenizer section, the template and the three ways it stops, the metadata from litert-torch's own builder, and the metadata without the plugin. CPU, about 17 MB of downloads. |
| `recipe_int4.json` | The quantization recipe. |
| `build_smollm3_3b.py` | The checkpoint at `a07cc9a0` to the `.litertlm` file in one `export()` call; `--dry-run` prints the export arguments, `--no-extension` leaves the plugin out. |
| `verify_smollm3_3b.py` | 8 questions through the `litert-lm` CLI, `--check-metadata`, `--think-ab` and `--sections`. |
