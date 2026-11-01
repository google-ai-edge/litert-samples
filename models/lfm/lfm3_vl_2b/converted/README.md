# LFM3-VL-2B Conversion

Model recipes, instructions, scripts, and utilities for converting LFM3-VL-2B to LiteRT-LM.

These scripts convert [LiquidAI/LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B) (`Lfm3VlForConditionalGeneration` in the checkpoint's own remote code, LFM Open License v1.0, Liquid AI's vision-language model) at revision `f83d2b0a` into `LFM3-VL-2B_int4.litertlm` of [litert-community/LFM3-VL-2B-LiteRT](https://huggingface.co/litert-community/LFM3-VL-2B-LiteRT) for the [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) runtime, and verify it with a quality gate on pictures and text, a metadata check, the runtime's own rendering of the chat template against transformers', and a section-by-section comparison with the published file.

The model has 1.92 B parameters: an LFM3 text decoder of 1.73 B (the architecture of [LFM3-1.7B](../../lfm3_1_7b/), with its own weights: 24 layers, 13 gated short convolutions, 5 sliding-window and 6 global attention layers, hidden 2048, a 128,000-token vocabulary shared with the output), a 164 M vision encoder (14 layers: 8 gated depthwise 2D convolutions and 6 bidirectional attention layers, hidden 1024, patches of 16 × 16) and a 25 M projector that merges 2 × 2 patches into one token. litert-torch has no LFM3-VL module, and the conversion needs none: two out-of-tree litert-torch `ModelExtension` files, which the stock exporter loads (`--extension_modules`), with no change to litert-torch and no edit of the bundle after the export. One of them is the LFM3-1.7B recipe's `lfm3_extension.py`, unchanged, for the text decoder; `lfm3_vl_extension.py` here adds the vision tower and what the image path needs in the bundle's metadata.

## Build

| file | weights | KV cache | size |
|---|---|---|---|
| `LFM3-VL-2B_int4.litertlm` | text decoder: int4 blockwise-32 on every fully connected weight, int8 per channel on the output projection (`recipe_int4.json`); token embedding table: int8 per channel; vision encoder: float32; projector: float16 (`recipe_vision.json`) | 2,048 | 2.08 GB |

Prefill signatures of 1,024, 256, 64, 16, 4 and 1 tokens, the checkpoint's `tokenizer.json` byte for byte, a thought channel (`<think>` … `</think>`), model type `fast_vlm` with a 512 × 512 image (one image = 256 tokens), and no sampler, as the published file. Which file of the model repository to take for which device is on the [model card](https://huggingface.co/litert-community/LFM3-VL-2B-LiteRT); this recipe makes the int4 file with a 2,048-token context.

## Environment

```bash
pip install --pre "litert-torch @ git+https://github.com/google-ai-edge/litert-torch@49af12281d2de0217371dfe54f8fbc189b276265" \
    torch==2.14.0 torchvision==0.29.2 transformers==5.20.2 litert-lm-builder==0.19.0 litert-lm==0.19.0 \
    litert-converter==0.6.0.dev20261027 litert-quantizer-nightly==0.12.0.dev20261029 \
    ai-edge-litert-nightly==2.4.0.dev20261027
```

litert-torch at that commit (main, 2026-10-30) has the out-of-tree `ModelExtension` path (`extension_modules` takes a `.py` file, commit 864a966 of 2026-10-28); no release carries it yet. The LFM3-1.7B recipe pins an earlier commit of the same branch (a456eeb); its extension and its tests run unchanged at this one. torchvision is for the checkpoint's image processor, which litert-torch loads for this task. transformers must be 5.15.0 or later (the checkpoint's remote code imports `use_kernel_func_from_hub_with_fallback` from `transformers.integrations`). `build_lfm3_vl_2b.py` stops before the export, with its reason, on an older transformers, on a litert-torch without the extension path, or without torchvision.

## Run

```bash
python lfm3_vl_extension_test.py                                            # the extension's tests, CPU, no weights
python build_lfm3_vl_2b.py --out out_lfm3_vl                                # the int4 file
python verify_lfm3_vl_2b.py out_lfm3_vl/LFM3-VL-2B_int4.litertlm --backend cpu   # 8 picture and 8 text questions
python verify_lfm3_vl_2b.py out_lfm3_vl/LFM3-VL-2B_int4.litertlm --backend gpu
python verify_lfm3_vl_2b.py out_lfm3_vl/LFM3-VL-2B_int4.litertlm --check-metadata  # metadata, tokenizer, states, weights
python verify_lfm3_vl_2b.py out_lfm3_vl/LFM3-VL-2B_int4.litertlm --render          # runtime template rendering vs transformers
python verify_lfm3_vl_2b.py out_lfm3_vl/LFM3-VL-2B_int4.litertlm --sections LFM3-VL-2B_int4.litertlm  # vs the published file from the model repository
python build_lfm3_vl_2b.py --out out_stock --no-extension                   # without the extensions: where the stock export stops
```

`build_lfm3_vl_2b.py` downloads the checkpoint at the pinned revision (code, configuration, weights, tokenizer and template, 3.85 GB), or takes a local copy of it with `--model`. `verify_lfm3_vl_2b.py` takes the pictures and the checkpoint's template and tokenizer from the Hub at pinned revisions unless `--photos-dir` and `--source` name local copies.

## Two `ModelExtension` files

The exporter resolves an extension for the checkpoint's `model_type` (`lfm3_vl`) and for its `text_config.model_type` (`lfm3`), and composes the two (`resolve_model_extension` in litert-torch's `model_ext/extension.py`). `build_lfm3_vl_2b.py` passes both files to one `export(extension_modules=[...])` call. From the command line, `litert-torch export_hf --extension_modules=../../lfm3_1_7b/converted/lfm3_extension.py,lfm3_vl_extension.py` with the flags in `lfm3_vl_extension.py`'s docstring runs the same export without the two wraps of [What stays in the script](#what-stays-in-the-script): it asks on stdin whether to run the image processor's remote code, and its file keeps the 3,517 int4 block scales under 2^-14 that XNNPACK refuses ([Quantization](#quantization)).

| file | hook | what it does |
|---|---|---|
| `lfm3_vl_extension.py` | `get_vision_exportables` | The vision encoder and the projector as two graphs for a 512 × 512 picture ([below](#the-vision-tower)). |
| | `get_llm_model_type_proto` | Model type `fast_vlm` with a 512 × 512 image tensor (the decoder's positions are 1-D, as in the source). |
| | `build_llm_metadata` | Image parts of the chat template render `<image_soft_token>` ([below](#the-chat-template)); no sampler unless a sampler flag is set ([below](#sampler)). |
| `../../lfm3_1_7b/converted/lfm3_extension.py` (unchanged) | `patch_model` | The text decoder's short convolutions as litert-torch's LFM2 short convolution at width 7, with the position guard for padded prefill slots. |
| | `update_export_config` | Turns `use_rope_composite` off. |
| | `build_llm_metadata` | The EOS id 124900 (`<\|im_end\|>`) as the only stop, and the one-line change of the template's `keep_thinking` line. |

The composite runs `build_llm_metadata` of `lfm3_vl_extension.py` first, then that of `lfm3_extension.py`. The LFM3-1.7B recipe's README explains the decoder's hooks; the two decoders' code is the same (`modeling_lfm3.py` of the two checkpoints differs in two docstring repository names), and their text configurations differ only in `max_position_embeddings`.

What the exporter already does for this model, unchanged:

- the loader builds `Lfm3VlForConditionalGeneration` from the checkpoint's remote code: all 424 checkpoint tensors load bit for bit in float32, the output layer tied to the embedding;
- for this task with a vision encoder it exports the token embedding table as its own graph (`externalize_embedder`, set in `exportable_module_config.py`), and the decoder from the full model through its stock prefill and decode exportables;
- the ExecutorMetadata builder declares a convolution state for each of the 13 conv layers and global K / V caches for the 11 attention layers;
- the packager writes the four graphs, the tokenizer, the metadata and the template into one file.

`lfm3_vl_extension_test.py` runs litert-torch's `ModelExtensionTestMixin` contract tests on the extension and nine cases of its own on a random LFM3-VL built from the checkpoint's remote code (4 vision layers: conv, attention, conv, attention; 2 text layers) and on the checkpoint's tokenizer, image processor and template:

- the encoder and the projector against transformers' own tower, fed by the checkpoint's image processor, in float32 on two 512 × 512 pictures with flat regions: the largest difference is 1.94e-7 and 1.79e-7 against outputs up to 0.30 (the test's bar is 1e-5);
- the input step: the graph recovers every 8-bit value from the runtime's `u8 / 255`;
- a `torch.export` of both graphs, and the stock resolver composing the two extensions for the checkpoint's model types;
- both hooks inside litert-torch's own `LlmMetadata` builder (stops, start token, model type, no sampler, the thought channel, the template), and a sampler flag that keeps its sampler;
- the template edit and its guard, and the edited template rendering as the source with `<|image|>` as `<image_soft_token>` (one and three turns, thinking off and on);
- a vision configuration the rewrite does not cover stops.

## The vision tower

LFM3-VL's vision tower reads a picture at its own size (NaViT style: patches of 16 × 16 grouped in 2 × 2 merge blocks, 2-D positions per picture). LiteRT-LM hands a vision encoder one picture stretched to a fixed size, so `lfm3_vl_extension.py` writes the tower again in plain PyTorch for a 512 × 512 picture, with the checkpoint's weights and the same arithmetic:

| graph | input | output | weights |
|---|---|---|---|
| vision encoder | `images` [1, 512, 512, 3] in [0, 1] | `features` [1, 1024, 1024] | float32 |
| projector | `soft_tokens` [1, 1024, 1024] | `mm_embedding` [1, 256, 2048] | float16 |

- Input step: LiteRT-LM hands the encoder `float(u8) * (1 / 255)`. The graph takes the exact 8-bit value back, `floor(x * 255 + 0.5)`, and normalizes it as the checkpoint's processor does, `(u8 - 127.5) / 127.5`.
- Patches: the patchifier reads a temporal pair of identical frames, so it equals a 16 × 16 stride-16 `Conv2d` whose kernel sums the pair.
- Order: inside one picture the attention has no mask, each token's rotary position is its own (h, w), and the MLPs and norms work per token, so the tower runs in raster order with constant cos / sin tables; the depthwise 3 × 3 convolution runs on the raster grid (the source converts to raster for it). The projector's 2 × 2 merge is four strided slices concatenated on channels, which gives the source's merge-block token order.
- Attention: grouped-query attention without `repeat_kv` (query heads 2j and 2j + 1 against key / value head j), and the 1 / 8 scale folded into the query norm's weight (a power of two, exact).

The encoder stays float32: the model card records that float16 or int8 encoder weights, or the GPU's float16 precision, lowered the worst per-token cosine against the source's image features from 0.9999978 to 0.20–0.70. The vision sections declare no activation type (the stock packager writes none for them; the published file declares float32), and LiteRT-LM runs an undeclared vision encoder at float32 on the GPU: `SetCommonGpuOptions` falls back to `ActivationDataType::FLOAT32` when the settings carry no activation type (`runtime/executor/litert_compiled_model_executor_utils.cc`, v0.18.1 and v0.19.0; `runtime/engine/engine_settings.cc` notes "Vision executor defaults to F32").

## The chat template

The source template writes an image part as `"<|vision_start|><|image|><|vision_end|>"` in two places, and the checkpoint's processor expands `<|image|>` to one token per image embedding. LiteRT-LM splices the projector's 256 embeddings at `<image_soft_token>`, so the hook writes `<image_soft_token>` there; video parts stay as the source has them (the file carries no video path). If the image part is not in the template exactly twice, the hook stops the export: the source template changed. The decoder's extension then changes its one `keep_thinking` line (the LFM3-1.7B recipe's README, "The chat template").

`--render` measures the result on litert-lm 0.19.0: the runtime's own rendering (`Conversation.render_message_to_string`) equals transformers' rendering of the source template for one turn with a picture and three turns with two pictures, thinking off and on, with two differences: the start token, which the runtime renders empty from `bos_token` and adds once from the metadata, and `<|image|>`, which the bundle renders as `<image_soft_token>`.

## Sampler

For text generation litert-torch's loader turns the checkpoint's `do_sample` off, so the exporter writes a sampler only when a sampler flag is set. For this task it leaves it on, and the packager would write the generation config's sampler: TOP_K with k 50 and temperature 0.1 (the config has no top_p). The hook removes it unless a sampler flag is set, so the file carries no sampler, as the published one, and an app sets its own; the source card recommends temperature 0.1, top_k 50 and repetition_penalty 1.1.

## Quantization

`recipe_int4.json` is the litert-quantizer recipe of the published file: int4 blockwise-32 (symmetric, from each block's largest value) on every fully connected weight of the decoder, int8 per channel on the embedding table and on the output projection (the op that writes the `decode` signature's logits). `recipe_vision.json` casts the projector's two fully connected weights to float16 and selects nothing in the encoder: litert-torch applies one vision recipe to both graphs, and the regex `VisionAdapter|vision_adapter` matches the projector's operator names only (the second weight's output is the signature's output, `vision_adapter_mm_embedding_output`, so a regex on the module name alone misses it).

XNNPACK, LiteRT-LM's CPU path, refuses a fully connected weight with a block scale that is zero or subnormal (the scales are float16, so anything under 2^-14), and litert-quantizer floors a min-max scale at 1e-9 only. In this model 3,517 block scales of 4 weights come out under 2^-14. `build_lfm3_vl_2b.py` raises them to 2^-14 before the weights are quantized, as the published file was made; `--check-metadata` checks every scale.

## What stays in the script

Two things no extension hook reaches, each scoped to the one `export()` call in `build_lfm3_vl_2b.py`:

- litert-torch loads the image processor with `AutoImageProcessor.from_pretrained` and no `trust_remote_code`, and this checkpoint's image processor is remote code (`auto_map` in `processor_config.json`), so transformers asks on stdin and, with no terminal, stops. The vision graphs here do not use the processor; the script passes `trust_remote_code=True` to that call. The `patch_context` hook wraps only the model's loading.
- The int4 scale floor above (litert-quantizer has no option for it).

## Verification results

Measured on the file built with this recipe, Mac M4 Max, 2026-11-02. The published file is `LFM3-VL-2B_int4.litertlm` of [litert-community/LFM3-VL-2B-LiteRT](https://huggingface.co/litert-community/LFM3-VL-2B-LiteRT) at revision `58085927` (sha256 checked against the Hub), measured with the same scripts.

- **`lfm3_vl_extension_test.py`: 14/14** on CPU; the LFM3-1.7B recipe's `lfm3_extension_test.py` 16/16 in the same environment.
- **Build**: 322 s, 25.1 GB peak resident memory; 2,078,404,640 B. litert-lm-builder 0.19.0 writes container version 1.9.0 (the published file: 1.7.0). A build in an environment made with `python3.12 -m venv` and the pip line above, with the checkpoint from the Hub (no `--model`), passed the same tests, metadata check and default gate; its sections are byte-equal to this build's except for the text decoder's tensor names (`--sections` below).
- **`--check-metadata`: 19/19.** The published file passes 17: it has no `pad_token`, and its template differs from the expected one in whitespace at the two places described under `--sections`.
- **`--render`: 4/4** on litert-lm 0.19.0 (one turn with a picture and three turns with two pictures, thinking off and on). The runtime renders this recipe's file and the published file to the same text in all four.
- **Picture and text gate** (litert-lm 0.19.0 CLI, greedy, thinking off, `--cache no`, 4 CPU threads):

| questions | CPU, this recipe | CPU, published | GPU, this recipe | GPU, published |
|---|---|---|---|---|
| the 8 picture questions of the default gate (4 pictures) | 8/8 | 8/8 | 8/8 | 8/8 |
| the 18 picture questions with one right answer of the model card's test set (12 pictures, `--photos-json`) | 18/18 | 18/18 | 18/18 | 18/18 |
| the 8 text questions | 8/8 | 8/8 | 8/8 | 8/8 |

  On the 18 picture questions and the 8 text questions, this recipe's file and the published file gave byte-identical replies in all 52 comparisons (26 questions on each backend). Of the default gate's 32 comparisons, 31 were byte-identical: on the CPU the published file answered the days-in-a-week question "7 days.", where its other run and both runs of this recipe's file gave "There are $\boxed{7}$ days in a week." (the same verdict). No answer is degenerate. The 8 pictures of the model card's set that the default gate does not include (drawings of shapes, a gate number, a table, left / right and top / bottom placement, a large shape, and two NASA photographs) are not in the model repository.

- **`--sections`** against the published file:

| section | this recipe | published | |
|---|---:|---:|---|
| tokenizer (`HF_Tokenizer_Zlib`) | 2,574,262 B | 2,574,262 B | byte-equal |
| `ExecutorMetadataProto` | 2,299 B | 2,299 B | byte-equal |
| token embedder | 263,682,208 B | 263,682,208 B | byte-equal |
| text decoder | 1,106,995,312 B | 1,100,474,784 B | all 626 weight buffers byte-equal; the same 7 signatures, 816 subgraphs and 12,959 operators of the same kinds; tensor names differ |
| vision encoder | 654,693,840 B | 654,412,748 B | all 166 weight buffers byte-equal; 694 operators of the same kinds; the signature and tensor names differ |
| projector | 50,376,736 B | 50,371,856 B | all 20 weight buffers byte-equal; 28 operators of the same kinds; the signature and tensor names differ |
| `LlmMetadataProto` | 12,053 B | 12,038 B | below |

  What the comparison finds different in the three TFLite sections is their tensor and signature names; the weight buffers and the operator kinds and counts are the same (the table). The published decoder was traced through the exportable class of the published conversion, this one through litert-torch's stock one, and 17,458 of its 20,677 tensor names also carry the remote-code module path, which transformers derives from the checkpoint directory's name. So the section size depends on that name: the sizes above come from a copy in a directory named `LFM3-VL-2B`; from the Hub snapshot, the default of `build_lfm3_vl_2b.py`, the text decoder section is 1,108,594,208 B, with the same 626 weight buffers and operators. The published vision graphs have one signature `serving_default` with inputs and outputs `args_0` / `output_0`; this recipe's are `vision_encoder` (`images` → `features`) and `vision_adapter` (`soft_tokens` → `mm_embedding`). LiteRT-LM runs a one-signature vision graph by index (`runtime/executor/vision_litert_compiled_model_executor.cc`, v0.18.1 and v0.19.0), so the names do not reach the runtime.

  `LlmMetadata`: the same stop, context, model type, channel and no sampler. The differences are the stock packager's: the start token as the string `<|startoftext|>` (the published file: its id 124894), a `pad_token` (`<|pad|>`), no empty `prompt_templates` block, and the template, which differs only in whitespace at the two places where litert-torch removed the `{%- generation -%}` / `{%- endgeneration -%}` tags (+2 B; both templates render the same text, `--render`). The vision sections declare no activation type (above, [The vision tower](#the-vision-tower)).

### Galaxy S26

Galaxy S26 (SM-S942Q, Android 16), LiteRT-LM v0.18.1's `litert_lm_advanced_main` (the binary the model card's numbers came from), this recipe's file and the published one in one session on 2026-11-02, each file's sha256 read back on the phone after the push. The benchmark runs were cold (no compiled-model cache), one process each, with `--benchmark_prefill_tokens=1024 --benchmark_decode_tokens=256 --max_num_tokens=2048` and no picture:

| file | backend | prefill 1,024 (tok/s) | decode 256 (tok/s) | time to first token (s) | peak memory (MB) |
|---|---|---:|---:|---:|---:|
| this recipe | GPU | 1,330.65 | 37.91 | 0.80 | 1,189.18 |
| published | GPU | 1,324.77 | 37.46 | 0.80 | 1,170.43 |
| this recipe | CPU, 4 threads | 218.02 | 38.86 | 4.72 | 2,224.51 |
| published | CPU, 4 threads | 232.44 | 35.14 | 4.43 | 2,304.39 |

This recipe's benchmark runs used the file as built, the published file's runs its think-off copy (below). The model card's GPU numbers for the published file (the mean of two runs, 2026-10-31) are 1,328 / 37.88 tok/s with 0.79–0.80 s to the first token; the card has no CPU row for this file. During the CPU runs the CPU clock was below its maximum in 22 of 30 and 25 of 29 half-second samples. Peak memory is the runtime's own "Peak system ram usage" line.

The 8 picture questions of the default gate and the 8 text questions, one process per question, with thinking off: this binary has no thinking switch, so each file ran as a copy whose template line `enable_thinking | default(true)` reads `enable_thinking |default(false)` (12 bytes differ; every other section byte-equal). The picture runs give the picture to the vision encoder on the same backend:

| file | backend | picture questions | text questions | vision encoder (ms) | then first token (s) | sum (s) | peak memory (MB) |
|---|---|---|---|---:|---:|---:|---|
| this recipe | GPU | 8/8 | 8/8 | 279.8 | 0.475 | 0.751 | 1,498–1,583 |
| published | GPU | 8/8 | not run | 287.6 | 0.51 | 0.797 | 1,433–1,567 |
| this recipe | CPU, 4 threads | 8/8 | 8/8 | 789.4 | 4.185 | 4.963 | 2,498–2,932 |

The times are medians over the 8 picture questions (the vision encoder's "vision_executor" mark, the runtime's time to first token, and their sum per question); peak memory is the range over the picture runs. For the published file on this phone the model card gives 291 ms of vision encoder and 0.855 s in all (the medians of its 8 picture questions, 2026-10-31), and 1,475–1,616 MB in its question runs. Both files ran the same way on the GPU: the decoder's 7 signatures whole on the GPU (one partition each), the vision encoder's 694 operators in one partition, the projector on XNNPACK (22 of 28 operators). Every reply on the phone is byte-equal to the Mac reply of the same file and backend above, but one: on the GPU, the 0.9-or-0.11 question got the CPU's wording ("0.9 is larger than 0.11 because 0.9 = 0.90, and 0.90 > 0.11. $\boxed{0.9}$"), where the Mac GPU wrote "0.9 is larger than 0.11. To compare: …", with the same verdict.

### Failure paths

- transformers 5.14.1: `build_lfm3_vl_2b.py` stops before the export: `transformers 5.14.1 cannot load LFM3's remote code: it imports use_kernel_func_from_hub_with_fallback from transformers.integrations, which transformers has since 5.15.0.`
- litert-torch-nightly 0.11.0.dev20261028 (built before 864a966): it stops before the export: `this litert-torch has no out-of-tree extension path (a .py file in extension_modules, litert-torch commit 864a966 of 2026-10-28; no release carries it yet).`
- The source template with one of its two image parts changed (a line break added after it): the export stops in the hook (after the graphs are exported, when the metadata is built), `ValueError: lfm3_vl_extension: the image part occurs 1 times in the chat template (expected 2); the source template changed, re-check how it renders images`, and leaves no file.
- Without the extensions (`--no-extension`): the export stops at the vision graphs, `ValueError: Unsupported model type: lfm3_vl` (litert-torch's `model_ext/exportables.py`, `get_vision_exportables`), and leaves no file.

### What is not measured

- Answers with thinking on. The gates ask with thinking off; `--render` checks the thinking-on text, not its answers.
- More than one picture in a question, or pictures in later turns, on a runtime. `--render` checks the text of three turns with two pictures; every gate question has one picture in a single turn.
- The other phones of the model card (Pixel 8a, Pixel 11 Pro), the NPU, and the repository's 4,096-token and float16 files, which this recipe does not build.

## Files

| File | What |
|---|---|
| `lfm3_vl_extension.py` | The out-of-tree `ModelExtension` for `lfm3_vl`: `get_vision_exportables` (the encoder and the projector for 512 × 512), `get_llm_model_type_proto` (`fast_vlm`), `build_llm_metadata` (image parts as `<image_soft_token>`, no sampler without a sampler flag). |
| `lfm3_vl_extension_test.py` | `ModelExtensionTestMixin` contract tests and nine cases: the tower against transformers, the input step, `torch.export`, the composition with the decoder's extension, both hooks in litert-torch's metadata builder, the sampler flag, the template edit and its rendering, an unsupported configuration; CPU, no weights (downloads the checkpoint's code, configuration, processor, tokenizer and template, about 18 MB). |
| `recipe_int4.json` | The decoder's quantization recipe. |
| `recipe_vision.json` | The projector's float16 recipe. |
| `build_lfm3_vl_2b.py` | HF checkpoint → `LFM3-VL-2B_int4.litertlm`, one stock `export()` call with the two extension files; stops early on a transformers, litert-torch or torchvision that cannot run it. `--no-extension` exports without the extensions, for study. |
| `verify_lfm3_vl_2b.py` | 8 picture and 8 text questions via the `litert-lm` CLI (`--photos-json` for other rows); `--check-metadata`; `--render` (the runtime's rendering of the template against transformers'); `--sections` (section-by-section comparison with another bundle). |
