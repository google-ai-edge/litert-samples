# LFM3-1.7B Conversion

Model recipes, instructions, scripts, and utilities for converting LFM3-1.7B to LiteRT-LM.

These scripts convert [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) (`Lfm3ForCausalLM` in the checkpoint's own remote code, LFM Open License v1.0, Liquid AI's text model for local agents) at revision `40ebe0aa` into the four `.litertlm` files of [litert-community/LFM3-1.7B-LiteRT](https://huggingface.co/litert-community/LFM3-1.7B-LiteRT) for the [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) runtime, and verify them with a quality gate, a metadata check, the runtime's own rendering of the chat template against transformers', and a section-by-section comparison with the published files.

The model is a 24-layer hybrid: 13 gated short convolutions of width 7, 5 sliding-window attention layers (window 256) and 6 global attention layers, q / k norms and a sigmoid output gate in every attention layer, a rotary embedding on half of each head in the global layers (θ 10⁶) and on the whole head in the sliding layers (θ 10⁴), sandwich norms, hidden 2048, and a 128,000-token vocabulary shared with the output (1.73 B parameters). litert-torch has no LFM3 module, and the conversion needs none: `lfm3_extension.py` is an out-of-tree litert-torch `ModelExtension` that the stock exporter loads (`--extension_modules`), with no change to litert-torch and no edit of a bundle after the export. The stock loader reads the checkpoint with `trust_remote_code` alone, and every other stage is the exporter's own: the prefill and decode exportables, the conversion, the quantization, the tokenizer, the ExecutorMetadata that declares the convolution and attention states, and the packaging.

## Build

| name | file | weights | KV cache | size |
|---|---|---|---|---|
| `q1b_2048` | `LFM3-1.7B_mixed_int4_ekv2048.litertlm` | int4 blockwise-32 on the MLP gate and up projections, int8 per channel on every other weight (`recipe_q1b.json`) | 2,048 | 1.44 GB |
| `q1b_4096` | `LFM3-1.7B_mixed_int4_ekv4096.litertlm` | the same | 4,096 | 1.44 GB |
| `int8_2048` | `LFM3-1.7B_int8_ekv2048.litertlm` | int8 per channel on every weight (the stock `dynamic_wi8_afp32`) | 2,048 | 1.80 GB |
| `int8_4096` | `LFM3-1.7B_int8_ekv4096.litertlm` | the same | 4,096 | 1.80 GB |

All four have prefill signatures of 1,024, 256, 64, 16, 4 and 1 tokens, the checkpoint's `tokenizer.json` byte for byte, a thought channel (`<think>` … `</think>`), and the sampler top-k 50 at temperature 0.1 (TOP_P with p 1). Which file to take for which device is on the [model card](https://huggingface.co/litert-community/LFM3-1.7B-LiteRT#files).

## Environment

```bash
pip install --pre "litert-torch @ git+https://github.com/google-ai-edge/litert-torch@a456eebebcf2b1cfe56cfc6cbab51f3a73850f9f" \
    torch==2.14.0 transformers==5.20.2 litert-lm-builder==0.19.0 litert-lm==0.19.0 \
    litert-converter==0.6.0.dev20261027 litert-quantizer-nightly==0.12.0.dev20261029 \
    ai-edge-litert-nightly==2.4.0.dev20261027
```

litert-torch at that commit (main, 2026-10-29) has the out-of-tree `ModelExtension` path (`extension_modules` takes a `.py` file, commit 864a966 of 2026-10-28); no release carries it yet. One environment runs the build and every check: at this commit the exporter packages with litert-lm-builder 0.19.0, and litert-lm 0.19.0 is the verification CLI and Python package. transformers must be 5.15.0 or later: the checkpoint's remote code imports `use_kernel_func_from_hub_with_fallback` from `transformers.integrations`, which 5.12.1 to 5.14.1 do not have (the checkpoint's `config.json` names 5.12.1). `build_lfm3_1_7b.py` stops before the export, with that reason, on an older transformers or on a litert-torch without the extension path.

## Run

```bash
python lfm3_extension_test.py                                                   # the extension's tests, CPU, no weights
python build_lfm3_1_7b.py --out out_lfm3                                        # the four files, 4 to 5 min each on an M4 Max
python verify_lfm3_1_7b.py out_lfm3/LFM3-1.7B_mixed_int4_ekv2048.litertlm --backend cpu   # 8-question gate
python verify_lfm3_1_7b.py out_lfm3/LFM3-1.7B_mixed_int4_ekv2048.litertlm --backend gpu
python verify_lfm3_1_7b.py out_lfm3/LFM3-1.7B_mixed_int4_ekv2048.litertlm --check-metadata  # metadata, tokenizer, states, int4 tensors
python verify_lfm3_1_7b.py out_lfm3/LFM3-1.7B_mixed_int4_ekv2048.litertlm --render          # runtime template rendering vs transformers
python verify_lfm3_1_7b.py out_lfm3/LFM3-1.7B_mixed_int4_ekv2048.litertlm --sections LFM3-1.7B_mixed_int4_ekv2048.litertlm  # vs a published file
python build_lfm3_1_7b.py --out out_stock --files q1b_2048 --no-extension       # the stock export without the extension, for study
```

## The conversion is a `ModelExtension`

`lfm3_extension.py` subclasses litert-torch's `ModelExtension` for `model_type` `lfm3` and overrides three hooks:

| hook | what it does |
|---|---|
| `patch_model` | Runs the short convolutions as litert-torch's LFM2 short convolution (`model_ext/lfm2/short_conv.py`) at width 7 and keeps padded prefill slots out of the convolution state ([below](#the-short-convolution)). |
| `update_export_config` | Turns `use_rope_composite` off, so the export config says what the graph has. litert-torch turns the flag on for every export without a split cache (0567035, 2026-10-27), but only its built-in model patches act on it; this extension does not route LFM3's rotary embeddings through `odml.rope`, so their angles trace as plain operators, as in the published files ([below](#what-is-not-measured)). |
| `build_llm_metadata` | Makes the EOS id 124900 (`<\|im_end\|>`) the only stop, and changes one line of the embedded chat template ([below](#the-chat-template)). |

What the exporter already does for this model, unchanged:

- the loader builds `Lfm3ForCausalLM` from the checkpoint's remote code: all 286 checkpoint tensors load bit for bit in float32, the output layer tied to the embedding;
- litert-torch's core patches swap the remote code's RMSNorm for its composite `odml.rms_norm`;
- the attention (the output gate, the q / k norms, the rotary embeddings) traces through litert-torch's attention interface, with the sliding window in the mask the exportable builds;
- the ExecutorMetadata builder declares a convolution state for each conv layer and global K / V caches for the 11 attention layers, from `config.layer_types`.

The export stops on `use_qkv_norm_rope_composite`, `fuse_qkv` and `split_cache`, which the extension has no LFM3 form for (not wired or tested here). `use_short_conv_composite`, `fuse_gate_up` and `use_swiglu_composite` pass through to LFM2's modules but were not exported or tested here.

`build_lfm3_1_7b.py` passes the file to `export(extension_modules=...)` once per file; from the command line the same export is `litert-torch export_hf --extension_modules=lfm3_extension.py` with the flags in the file's docstring.

`lfm3_extension_test.py` runs litert-torch's `ModelExtensionTestMixin` contract tests on the extension, and eleven cases of its own on a random 6-layer LFM3 built from the checkpoint's remote code (conv, conv, global, conv, sliding, conv; window 8) and on the checkpoint's tokenizer and template:

- a `torch.export` of prefill and decode, and the patch restoring every module and forward on exit;
- the exporter's module against transformers in float32 on the four paths LiteRT-LM drives, for token and for embeddings input: one prefill call, token-by-token decode, 16-token prefill calls with the last one padded, and that prefill followed by decode (41 tokens);
- the refused flags, and `use_rope_composite` turned off;
- the hook inside litert-torch's own `LlmMetadata` builder, the stops without the hook, and the stops taken from the generation config;
- three template cases: the change is one line and applies once, an unknown template stops, and the result renders as the source template.

## The short convolution

LFM3's gated short convolution has the same B, C, x split as LFM2's, with a kernel of 7 instead of 3 (and a `seq_idx` input for packed documents, which only training uses). The hook swaps each one for litert-torch's LFM2 module, which keeps a `[1, hidden, 6]` state per layer between calls and computes the prefill tap by tap.

A prefill call shorter than its signature is padded: the LiteRT-LM executor writes position 0 into every unused slot of the call and, for token input, the pad token (v0.18.1 `runtime/executor/llm_litert_compiled_model_executor.cc`). The guard keeps those slots out of the convolution state, which the next call reads. In transformers the decoder layer hands its convolution the model's padding mask (`causal_mask_mapping['conv']`), but the exportable builds masks for the attention layers only. So the hook's model forward reads the mask dict with `.get` per layer type, and its layer forward gives the LFM2 module a `valid_mask` from the positions: a slot is real if it is the first of the call or its position grows. The exportable's own `valid_mask` for token input is `tokens != pad_id`; the hook uses the position rule for both inputs, so a real pad id in a prompt is not dropped and an embeddings input, whose padded slots hold no pad token, takes the same path. The test runs padded prefill calls (with random noise in the padded slots of the embeddings input); on all eight paths the module matches transformers within 2.03e-6 in every logit, with the same top token at all 41 positions.

## The chat template

LiteRT-LM keeps a conversation's earlier prompt as a prefix of the next one, so a turn must render the same after a later user message. The source template does not: it keeps the thought of an assistant turn while that turn follows the last user message, and drops it once the next user message arrives. The hook changes that one line, `keep_thinking = preserve_thinking or loop.index0 > ns.last_user_index`, to `keep_thinking = preserve_thinking`, so every earlier thought is dropped wherever the turn sits. A conversation that ends with a user message renders as the source template renders it; in a tool loop, the assistant turn after the last user message renders `<think></think>` and its content instead of its thought. If the line is not in the template exactly once, the hook stops the export: the source template changed.

The rest of the template needs nothing from the hook: litert-torch's packager removes the two `{%- generation -%}` tags and writes the 26 `.get("…")` lookups as `["…"]`, and LiteRT-LM turns the Python string methods the template uses (`.split(…)[-1]`, `.endswith(…)`, `.items()` and the like) into minijinja filters when it loads it (v0.18.1 `runtime/components/prompt_template.cc`, `EditTemplateForMinijinja`). `--render` measures the result on litert-lm 0.19.0: the runtime's own rendering (`Conversation.render_message_to_string`) equals transformers' rendering of the source template for one turn, three turns and a tool round trip, thinking off and on, with two differences: the start token, which the runtime renders empty from `bos_token` and adds once from the metadata, and the dropped thought in the tool round trip.

## Stops and sampler

Without the hook the exporter writes 15 stops: the EOS id 124900, and the template's end-of-turn string `<|im_end|>\n` as text, alone and with each of ` .,?!:;"')]}*` in front (litert-torch adds these for SentencePiece vocabularies). The hook keeps the EOS id alone, as the published files do. A study build that kept the 15 stops and the rest of the extension answered the 8 questions with the same replies as this recipe's file on litert-lm 0.19.0's GPU (8 of 8) and CPU (7 of 8; the eighth is the rhyme reply that changes between runs, [below](#verification-results)); `--check-metadata` fails it on the stop check.

The exporter writes a sampler only when a sampler flag is set: loading the model, it turns the generation config's `do_sample` off. `sampler_top_p=1.0` makes the file declare TOP_P with the generation config's k 50 and temperature 0.1 and p 1, which LiteRT-LM 0.18.1 accepts on both backends: its CPU sampler builds only TOP_P, and a TOP_K file, the other choice, stops there with `UNIMPLEMENTED: Sampler type: 1`. The source card's repetition penalty of 1.1 is not in the file; an app passes it per call.

## Quantization

`recipe_q1b.json` is a litert-quantizer recipe: int8 per channel on every weight, then int4 blockwise-32 on the fully connected layers of the MLP's `w1` and `w3` (the gate and up projections of the 24 layers). The int8 files use the stock recipe `dynamic_wi8_afp32`; the published int8 files were quantized with a JSON copy of it, entry for entry. Why the mixed int4 file stops at the gate and up projections is on the model card: int4 on more of the weights matched the source model's top token at fewer test positions, and int4 on all three MLP projections missed one of the 8 check questions. XNNPACK, LiteRT-LM's CPU path, refuses a block scale that is zero or subnormal (the scales are fp16, so anything under 2^-14); `--check-metadata` checks every scale (the smallest in this model is 7.21e-4).

## Verification results

Measured on files built with this recipe in a fresh environment set up as above, Mac M4 Max, 2026-10-30. The published files are the four of [litert-community/LFM3-1.7B-LiteRT](https://huggingface.co/litert-community/LFM3-1.7B-LiteRT) at revision `9bbda264` (sha256 checked against the Hub), measured with the same scripts.

- **`lfm3_extension_test.py`: 16/16** on CPU. The eight runtime paths match transformers within 2.03e-6 in every logit, with the same top token at all 41 positions.
- **Build**: the four files in 264, 259, 270 and 308 s, with litert-lm-builder 0.19.0, which writes container version 1.9.0 (the published files: 1.7.0); litert-lm 0.18.1 runs this recipe's mixed int4 2,048 file too (one question, Mac CPU). A second build gave every section of all four files byte for byte, and so did a build of the mixed int4 2,048 file in an environment made with `python -m venv` and the pip line above.
- **`--check-metadata`**: 16/16 on each mixed int4 file, 15/15 on each int8 file (no int4 tensors, so no scale check).
- **`--render`**: 6/6 on litert-lm 0.19.0 (one turn, three turns and a tool round trip, thinking off and on); the runtime's rendering of this recipe's file and of the published file are the same text in all six.
- **8-question gate** (litert-lm 0.19.0 CLI, greedy, thinking off, `--cache no`, 4 CPU threads):

| file | CPU, this recipe | CPU, published | GPU, this recipe | GPU, published |
|---|---|---|---|---|
| `LFM3-1.7B_mixed_int4_ekv2048` | 8/8 | 8/8 | 8/8 | 8/8 |
| `LFM3-1.7B_mixed_int4_ekv4096` | 8/8 | 8/8 | 8/8 | 8/8 |
| `LFM3-1.7B_int8_ekv2048` | 7/8 | 7/8 | 8/8 | 8/8 |
| `LFM3-1.7B_int8_ekv4096` | 7/8 | 7/8 | 8/8 | 8/8 |

  The int8 files miss the rhyme on the CPU ("violets are **violet**"), as the model card records for the published ones. No answer is degenerate. The replies are byte-identical to the published files' in 62 of the 64 runs. The two others (the mixed int4 2,048 file: the rhyme on the CPU, the French question on the GPU) are long replies with the same verdict. In two more runs of each, both files' replies changed between identical runs: the published file gave three different CPU replies in three runs and two GPU replies in three, this recipe's file two in three on the CPU and one in three on the GPU.

- **`--sections`** against the published file of the same name:

| section | mixed int4 2,048 / 4,096 | int8 2,048 / 4,096 |
|---|---|---|
| tokenizer (`HF_Tokenizer_Zlib`, 2,574,276 B) | byte-equal | byte-equal |
| `ExecutorMetadataProto` (2,299 B) | byte-equal | byte-equal |
| `LlmMetadataProto` | 12,045 B against 12,043 B: every field equal but the template, which has two more line breaks where litert-torch removed the `generation` tags (the published template had the lines removed; both render the same text) | the same |
| `TFLiteModel` | 1,440,574,688 / 1,440,582,880 B against 1,438,983,104 / 1,438,991,296 B: all 526 weight buffers byte-equal, the same 7 signatures, 816 subgraphs and 12,980 operators of the same kinds, 48 int4 tensors with the same smallest scale | 1,800,469,600 / 1,800,477,792 B against 1,798,911,440 / 1,798,919,632 B: all 478 weight buffers byte-equal, the same signatures and operators |

  The TFLite sections differ from the published ones in their tensor names only. The published graphs were traced through the conversion lane's own exportable class (`lfm3_ext.exportables.Lfm3Decode/…`), these through litert-torch's stock one (`…exportable_module.LiteRTExportableModuleForDecoderOnlyLMGenerate/…`), and every name also carries the remote-code module path, which transformers derives from the checkpoint directory's name. So the section size depends on that name: the sizes above come from a copy in a directory named `LFM3-1.7B`; from the Hub snapshot, the default of `build_lfm3_1_7b.py`, the mixed int4 2,048 section is 1,442,120,000 B, with the same 526 weight buffers and operators. The names of the mixed int4 2,048 graph take 10,116,642 B in the first case, 11,657,013 B in the second and 8,521,024 B in the published file.

### Galaxy S26

Galaxy S26 (SM-S942Q, Android 16), LiteRT-LM v0.18.1's `litert_lm_advanced_main` (the binary the model card's numbers came from), this recipe's `LFM3-1.7B_mixed_int4_ekv2048.litertlm` and the published one in one session on 2026-10-30. Every benchmark leg ran cold (no compiled-model cache) with `--benchmark_prefill_tokens=1024 --benchmark_decode_tokens=256 --max_num_tokens=2048`:

| file | backend | prefill 1,024 (tok/s) | decode 256 (tok/s) | time to first token (s) | peak memory (MiB) |
|---|---|---:|---:|---:|---:|
| this recipe | GPU | 1,321.07 | 31.69 | 0.81 | 1,000.8 |
| published | GPU | 1,320.15 | 31.38 | 0.81 | 899.4 |
| this recipe | CPU, 4 threads | 342.62 | 33.91 | 3.02 | 2,611.5 |
| published | CPU, 4 threads | 340.48 | 32.41 | 3.04 | 2,485.9 |

The model card's numbers for the published file (the mean of two legs, 2026-10-29) are 1,308.5 / 32.42 on the GPU and 316.1 / 35.53 on the CPU, whose two legs read 274.73 and 357.55 tokens/s of prefill. Both GPU legs ran whole on the GPU (7 of 7 signatures, one partition each). During the CPU legs the CPU clock was below its maximum in 10 of 13 and 12 of 15 one-second samples. Peak memory is the runtime's own "Peak system ram usage" line, one leg each; for the published file the card gives 1,002–1,012 MiB on the GPU and 2,211–2,212 MiB on the CPU.

The 8 questions on the phone (thinking off, the source template rendered on the host, each file's own sampler): both files 8 of 8 on the GPU and on the CPU, with byte-identical replies.

### Failure paths

- transformers 5.14.1: `build_lfm3_1_7b.py` stops before the export: `transformers 5.14.1 cannot load LFM3's remote code: it imports use_kernel_func_from_hub_with_fallback from transformers.integrations, which transformers has since 5.15.0.`
- litert-torch-nightly 0.11.0.dev20261028 (built before 864a966): it stops before the export: `this litert-torch has no out-of-tree extension path (a .py file in extension_modules, litert-torch commit 864a966 of 2026-10-28; no release carries it yet).`
- The source template with one character changed in its keep_thinking line (`>` to `>=`): the export stops in the hook, `ValueError: lfm3_extension: the keep_thinking line occurs 0 times in the chat template (expected once); the source template changed, re-check how it renders earlier assistant turns`, and leaves no file.
- Without the extension (`--no-extension`): the export stops while tracing, `KeyError: 'conv'` in the source `Lfm3Model.forward`, which reads the exportable's mask dict for every layer type; the dict has entries for the attention layers only.

### What is not measured

- Positions past 2,048 on a phone GPU. Without `odml.rope` the rotary angles trace as plain operators, and litert-torch turned the composite on by default because ML Drift's default fp16 GPU path rounds positions past 2,048 in that form (0567035). The published files have no rope composite either. The phone runs here end at position 1,280 (prefill 1,024 + decode 256); the model card's long run (GSM8K with thinking on, up to 3,584 tokens, the 4,096-token files) was on a Mac GPU.
- The other phone of the model card (Pixel 8a), and the 4,096-token and int8 files on a phone.

## Files

| File | What |
|---|---|
| `lfm3_extension.py` | The out-of-tree `ModelExtension`: `patch_model` (LFM2's short convolution at width 7, the position guard for padded prefill slots), `update_export_config` (no rope composite), `build_llm_metadata` (the EOS-only stop, the one-line template change). |
| `lfm3_extension_test.py` | `ModelExtensionTestMixin` contract tests and eleven cases: `torch.export`, the patch's exit, the exporter's module against transformers on the four runtime paths for token and embeddings input, the refused flags, the rope flag, the hook inside litert-torch's metadata builder and the stops, three template cases; CPU, no weights (downloads the checkpoint's code, config, tokenizer and template, about 18 MB). |
| `recipe_q1b.json` | The mixed int4 quantization recipe. |
| `build_lfm3_1_7b.py` | HF checkpoint → the four `.litertlm` files, one stock `export()` call each with `lfm3_extension.py`; stops early on a transformers or a litert-torch that cannot run it. `--no-extension` exports without the extension, for study. |
| `verify_lfm3_1_7b.py` | 8-question quality gate via the `litert-lm` CLI; `--check-metadata`; `--render` (the runtime's rendering of the template against transformers'); `--sections` (section-by-section comparison with another bundle). |
