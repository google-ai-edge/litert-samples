# LFM3-Audio-2B Conversion

Model recipes, instructions, scripts, and utilities for converting LFM3-Audio-2B to LiteRT-LM.

These scripts convert [LiquidAI/LFM3-Audio-2B](https://huggingface.co/LiquidAI/LFM3-Audio-2B) (`Lfm3AudioForConditionalGeneration` in the checkpoint's own remote code, LFM Open License v1.0, Liquid AI's audio model: text or speech in, text, speech or both out) at revision `9d49701c` into `lfm3-audio-2b_int4_b128_c2048_h30.litertlm` of [litert-community/LFM3-Audio-2B-LiteRT](https://huggingface.co/litert-community/LFM3-Audio-2B-LiteRT) for the [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) runtime, and verify it with a quality gate on text and speech input, a metadata check, the runtime's own rendering of the chat template against transformers', and a section-by-section comparison with the published file. That file is the text path: text or speech in, text out. Speech output runs outside LiteRT-LM, with four more graphs and a host loop: [`speech/README.md`](speech/README.md).

The checkpoint has 2.00 B parameters: an LFM3 text decoder of 1.73 B (the architecture of [LFM3-1.7B](../../lfm3_1_7b/), with its own weights: 24 layers, 13 gated short convolutions, 5 sliding-window and 6 global attention layers, hidden 2048, a 128,000-token vocabulary shared with the output), a 110 M audio encoder (a 128-bin log-mel front end, a convolutional downsampler that turns 8 mel frames into one vector, and a 17-layer LFM2 encoder: 10 short convolutions and 7 attention layers, hidden 768), a 6 M projector to the decoder's width, and 158 M of speech-output layers (the depth decoder and its embeddings), which the `.litertlm` file does not carry. litert-torch has no LFM3-Audio module, and the conversion needs none: two out-of-tree litert-torch `ModelExtension` files, which the stock exporter loads (`--extension_modules`), with no change to litert-torch and no edit of the bundle after the export. One of them is the LFM3-1.7B recipe's `lfm3_extension.py`, unchanged, for the text decoder; `lfm3_audio_extension.py` here adds the audio section, the decoder's second output for speech, and what the audio path needs in the bundle's metadata.

## Build

| file | weights | KV cache | size |
|---|---|---|---|
| `lfm3-audio-2b_int4_b128_c2048_h30.litertlm` | text decoder: int4 blockwise-128 on every fully connected weight, int8 per channel on the output projection (`recipe_int4.json`); token embedding table: int8 per channel; audio section: float16 fully connected weights, the rest float32 (`recipe_audio.json`) | 2,048 | 1.53 GB |

Prefill signatures of 1,024, 256, 64, 16, 4 and 1 tokens, each with a second output `hidden` (the decoder's last hidden state, which a speech loop reads), the checkpoint's `tokenizer.json` byte for byte, a thought channel (`<think>` … `</think>`), model type `generic_model` with the audio fields (raw 16 kHz mono PCM, 30 s per call of the audio section), and no sampler, as the published file. The file names keep the model repository's names; the repository's speech graphs are rebuilt by `speech/export_speech_graphs.py`.

## Environment

```bash
pip install --pre "litert-torch @ git+https://github.com/google-ai-edge/litert-torch@49af12281d2de0217371dfe54f8fbc189b276265" \
    torch==2.14.0 transformers==5.20.2 litert-lm-builder==0.19.0 litert-lm==0.19.0 \
    litert-converter==0.6.0.dev20261027 litert-quantizer-nightly==0.12.0.dev20261029 \
    ai-edge-litert-nightly==2.4.0.dev20261027
```

litert-torch at that commit (main, 2026-10-30) has the out-of-tree `ModelExtension` path (`extension_modules` takes a `.py` file, commit 864a966 of 2026-10-28) and the audio hook (`get_audio_exportables`); no release carries them yet. The LFM3-1.7B recipe pins an earlier commit of the same branch (a456eeb); its extension and its tests run unchanged at this one. transformers must be 5.15.0 or later (the checkpoint's remote code imports `use_kernel_func_from_hub_with_fallback` from `transformers.integrations`). `build_lfm3_audio_2b.py` stops before the export, with its reason, on an older transformers or on a litert-torch without the extension path or the audio hook.

## Run

```bash
python lfm3_audio_extension_test.py                                          # the extension's tests, CPU, no weights
python build_lfm3_audio_2b.py --out out_lfm3_audio                           # the h30 file
F=out_lfm3_audio/lfm3-audio-2b_int4_b128_c2048_h30.litertlm
python verify_lfm3_audio_2b.py $F --backend cpu                              # the 8 text questions
python verify_lfm3_audio_2b.py $F --backend gpu --asr-json clips.json --voicemail-json voicemails.json   # with speech input
python verify_lfm3_audio_2b.py $F --check-metadata                           # metadata, tokenizer, states, the three graphs
python verify_lfm3_audio_2b.py $F --render --wav clip.wav                    # runtime template rendering vs transformers
python verify_lfm3_audio_2b.py $F --sections lfm3-audio-2b_int4_b128_c2048_h30.litertlm   # vs the published file
python build_lfm3_audio_2b.py --out out_stock --no-extension                 # without the extensions: where the stock export stops
```

`build_lfm3_audio_2b.py` downloads the checkpoint at the pinned revision (code, configuration, feature extractor, weights, tokenizer and template), or takes a local copy of it with `--model`. `verify_lfm3_audio_2b.py` takes the checkpoint's template and tokenizer from the Hub at the pinned revision unless `--source` names a local copy. The gate runs through the `litert-lm` Python package, one engine per question, greedy. `--asr-json` takes clips to transcribe (`{"clips": [{"name", "path", "reference"}]}`, paths relative to the JSON's folder; the prompt is the system turn `Perform ASR in English.` and the clip) and scores the word error rate with the FLEURS rule (upper case, `-` as a space, every character but A–Z, 0–9 and the apostrophe as a space); `--voicemail-json` takes recorded messages (`{"items": [{"id", "caller", "callback"}]}`, audio `<id>.wav`) and asks `Summarize this voicemail in one or two short sentences: who called, what they want, and the number to call back.`; an answer counts when it names the caller's first and last name and has the callback number's digits in a row (digit words read as digits).

## Two `ModelExtension` files

The exporter resolves an extension for the checkpoint's `model_type` (`lfm3_audio`) and for its `text_config.model_type` (`lfm3`), and composes the two (`resolve_model_extension` in litert-torch's `model_ext/extension.py`). `build_lfm3_audio_2b.py` passes both files to one `export(extension_modules=[...])` call with task `multimodal_lm`. From the command line, `litert-torch export_hf --extension_modules=../../lfm3_1_7b/converted/lfm3_extension.py,lfm3_audio_extension.py` with the same flags runs the same export without the two wraps of [What stays in the script](#what-stays-in-the-script): it asks on stdin whether to run the feature extractor's remote code, and its file keeps the 720 int4 block scales under 2^-14 that XNNPACK refuses ([Quantization](#quantization)).

| file | hook | what it does |
|---|---|---|
| `lfm3_audio_extension.py` | `get_audio_exportables` | The audio section as one graph: raw PCM in, the decoder-width vectors and their count out ([below](#the-audio-section)). No adapter or end-of-audio graph. |
| | `get_prefill_decode_exportables` | The stock embeddings-input prefill and decode with a second output `hidden`: the decoder's final norm output, `[1, T, 2048]` from prefill and `[1, 1, 2048]` from decode. LiteRT-LM reads `logits`; a speech loop feeds `hidden` to the depth decoder, so the speech path and the text path read the same decoder graph. |
| | `get_llm_model_type_proto` | Model type `generic_model` with the audio fields: `<\|audio\|>` as the delimiter, raw PCM at 16 kHz, one channel, 160-sample frames, no mel spectrogram in the runtime. |
| | `build_llm_metadata` | Stops the export if the chat template no longer writes an audio part as `<\|audio\|>` exactly once ([below](#the-chat-template)). |
| `../../lfm3_1_7b/converted/lfm3_extension.py` (unchanged) | `patch_model` | The text decoder's short convolutions as litert-torch's LFM2 short convolution at width 7, with the position guard for padded prefill slots. |
| | `update_export_config` | Turns `use_rope_composite` off. |
| | `build_llm_metadata` | The EOS ids from the generation config as the only stops: 124900 (`<\|im_end\|>`) and 125014 (`<\|text_end\|>`); the one-line change of the template's `keep_thinking` line. |

The composite runs `build_llm_metadata` of `lfm3_audio_extension.py` first, then that of `lfm3_extension.py`. The LFM3-1.7B recipe's README explains the decoder's hooks.

What the exporter already does for this model, unchanged:

- the loader builds `Lfm3AudioForConditionalGeneration` from the checkpoint's remote code (its `auto_map`): every tensor of the class loads bit for bit from the checkpoint (527 tensors, bfloat16 read as float32), the output layer tied to the embedding; the class ignores 12 more checkpoint tensors on load (`depth_embeddings.*.to_logits`, speech output);
- for this task it exports the token embedding table as its own graph (`externalize_embedder`) and the decoder through its stock prefill and decode exportables (here with `hidden`);
- it exports the audio section through its own audio step (`export_audio_encoder_models_for_llm`: one conversion, the audio quantization recipe) and packages it as the bundle's audio encoder;
- the ExecutorMetadata builder declares a convolution state for each of the 13 conv layers and global K / V caches for the 11 attention layers;
- the packager writes the three graphs, the tokenizer, the metadata and the template into one file.

`lfm3_audio_extension_test.py` runs litert-torch's `ModelExtensionTestMixin` contract tests on the extension and ten cases of its own on a random LFM3-Audio built from the checkpoint's remote code (an encoder of 4 layers: conv, attention, conv, attention; a decoder of 4: conv, sliding, conv, global) and on the checkpoint's feature extractor, tokenizer and template:

- the audio section against the checkpoint's feature extractor and transformers' encoder and projector, in float32, on three clips of 1.0, 2.37 and 3.9 s in a 400-row window: the same number of audio tokens (13, 30 and 49), the largest difference 5.96e-8; the section's log-mel against the feature extractor's, 236 frames, the largest difference 1.75e-5; the extension's check of the feature extractor's settings;
- the decoder's prefill and decode against transformers under the LFM3-1.7B extension's patch: the largest difference 4.99e-6 in `hidden` and 9.98e-7 in the logits, the same top token;
- a `torch.export` of the three graphs, and the stock resolver composing the two extensions for the checkpoint's model types;
- the hooks inside litert-torch's own `LlmMetadata` builder (stops, start token, model type with the audio fields, no sampler, the thought channel, the template), and a text-only export without the audio fields;
- the template guard, and the bundle's template rendering as the source template with an audio part.

## The audio section

With `skip_mel_spectrogram_extraction` set, LiteRT-LM's generic audio path hands the audio section the clip itself, as rows of 160 PCM samples in windows of the section's row count, with no length input, and takes the number of audio tokens in a window from the section's `mask`: the position of its last non-zero entry plus one (`GetValidCount` in `runtime/executor/audio_litert_compiled_model_executor.cc`, v0.19.0). `lfm3_audio_extension.py` writes the checkpoint's audio encoder again in plain PyTorch for that contract, with the checkpoint's weights and the same arithmetic:

| graph | input | outputs | weights |
|---|---|---|---|
| audio section (`encode`) | `audio` [1, 3000, 160]: 30 s of 16 kHz PCM | `features` [1, 375, 2048], `mask` [1, 375] (uint8) | fully connected float16, the rest float32 |

- Log-mel in the graph: the feature extractor's 400-sample periodic Hann window and hop of 160 as a fixed DFT basis (201 one-sided bins), 128 slaney mel bins, log10 floored at -12, then (x + 4) / 4. Each frame is scaled by its own peak before the DFT and log10 of the peak squared is added back (the same value; it keeps a quiet frame's power out of float16's subnormal range on a GPU).
- Clip length: a frame counts when a non-zero sample lies in its span past the previous frame's, which gives the feature extractor's frame count for the clip up to its last non-zero sample. The validity is a 0 / 1 prefix vector per frame, carried through the downsampler's three stride-2 steps; `mask` marks the first ceil(frames / 8) vectors. A clip that ends in exact zeros is read as shorter.
- Encoder: the checkpoint's LFM2 encoder over the 375 vectors, every attention layer under its causal window of 32 vectors, grouped-query heads folded into the matrix rows (no rank-5 tensor), the window mask as a runtime rank-4 tensor of -30000 (float16 holds it).

`--sections` below compares this section with the published file's.

## The chat template

The source template writes an audio part as `"<|audio|>"` once, and the metadata names `<|audio|>` as the delimiter where the runtime puts the audio section's vectors. The template needs no change for it; the hook checks that the audio part is in the template exactly once and stops the export otherwise: the source template changed. The decoder's extension then changes its one `keep_thinking` line (the LFM3-1.7B recipe's README, "The chat template"). The source template has no thinking switch; the model sometimes opens a thought on its own, which LiteRT-LM returns in the thought channel.

`--render` measures the result on litert-lm 0.19.0: the runtime's own rendering (`Conversation.render_message_to_string`) equals transformers' rendering of the source template for a transcription request (the system turn and a clip), a question about a clip, and three text turns with a thought, with one difference: the start token, which the runtime renders empty from `bos_token` and adds once from the metadata.

## Sampler

The checkpoint's generation config is greedy (`do_sample` false, top_k 1, temperature 0), and litert-torch writes a sampler only when `do_sample` is on or a sampler flag is set, so the file carries none, as the published one, and an app sets its own. The source card's audio sampling settings (`audio_temperature`, `audio_top_k`, the repetition penalty) are for speech output, which `speech/` runs.

## Quantization

`recipe_int4.json` is the litert-quantizer recipe of the published decoder: int4 blockwise-128 (symmetric, from each block's largest value) on every fully connected weight, int8 per channel on the embedding table and on the output projection (the op that writes the `decode` signature's logits). `recipe_audio.json` casts every fully connected weight of the audio section to float16 (105 weights); the convolutions, norms and the DFT and mel tables stay float32.

XNNPACK, LiteRT-LM's CPU path, refuses a fully connected weight with a block scale that is zero or subnormal (the scales are float16, so anything under 2^-14), and litert-quantizer floors a min-max scale at 1e-9 only. In this model 720 block scales of 3 weights come out under 2^-14. `build_lfm3_audio_2b.py` raises them to 2^-14 before the weights are quantized, as the published file was made; `--check-metadata` checks every scale.

## What stays in the script

Two things no extension hook reaches, each scoped to the one `export()` call in `build_lfm3_audio_2b.py`:

- litert-torch loads the feature extractor with `AutoFeatureExtractor.from_pretrained` and no `trust_remote_code`, and this checkpoint's feature extractor is remote code (`auto_map` in `preprocessor_config.json`), so transformers asks on stdin and, with no terminal, stops. The export needs it (without a feature extractor there is no audio step); the audio section does not use it at run time, and the extension checks its settings (sampling rate, window, hop, mel bins, floor and normalization) against the values the graph computes with. The script passes `trust_remote_code=True` to that call. The `patch_context` hook wraps only the model's loading.
- The int4 scale floor above (litert-quantizer has no option for it).

The script also passes `export_vision_encoder=False`: task `multimodal_lm` turns the vision export on by default, and this checkpoint has no image processor.

## Verification results

Measured on the file built with this recipe, Mac M4 Max, 2026-11-02. The published file is `lfm3-audio-2b_int4_b128_c2048_h30.litertlm` of [litert-community/LFM3-Audio-2B-LiteRT](https://huggingface.co/litert-community/LFM3-Audio-2B-LiteRT) at revision `8e14ffaa` (sha256 checked against the Hub), measured with the same scripts.

- **`lfm3_audio_extension_test.py`: 15/15** on CPU in 1.3 s; the LFM3-1.7B recipe's `lfm3_extension_test.py` 16/16 in the same environment.
- **Build**: 318 s, 24.7 GB peak resident memory; 1,532,687,200 B. litert-lm-builder 0.19.0 writes container version 1.9.0 (the published file: 1.7.0).
- **A fresh environment**: a new Python 3.12 venv with the `pip install` line above and nothing else: the tests 15/15; the build from the Hub at the pinned revision (no `--model`) 330 s, 1,533,473,632 B; `--check-metadata` 19/19; the 8 text questions 8/8 on CPU and on GPU. Five of its six sections are byte-equal to the first build's. In the decoder all 633 weight buffers are byte-equal and `tfl_struct` (below) finds it structurally equal; its tensor names carry `transformers_modules._9d49701c…` (the Hub snapshot's directory, named by the revision) where the first build, from a local copy, has `transformers_modules.LFM3_hyphen_Audio_hyphen_2B`.
- **`--check-metadata`: 19/19.** The published file passes 18: its template differs from the expected one in whitespace at the two places described under `--sections`.
- **`--render`: 3/3** on litert-lm 0.19.0 for this recipe's file and for the published file, and the runtime renders the two files to the same text in all three.
- **Gate** (litert-lm 0.19.0 Python API, one engine per question, greedy, 4 CPU threads; the audio section on the same backend as the decoder). The clips are the first 10 of FLEURS en_us test and 10 of LibriSpeech dev-clean (404 words, 1.6–29.4 s each); the voicemails are six synthetic English messages (19–23 s, Kokoro-82M voices, fictional names and 555-01xx numbers). Two runs of each file on each backend:

| questions | CPU, this recipe | CPU, published | GPU, this recipe | GPU, published |
|---|---|---|---|---|
| the 8 text questions | 8/8, 8/8 | 8/8, 8/8 | 8/8, 8/8 | 8/8, 8/8 |
| transcription, 20 clips: word errors in 404 words | 28, 29 | 27, 28 | 24, 24 | 25, 25 |
| the 6 voicemails: caller and callback number | 6/6, 6/6 | 6/6, 6/6 | 5/6, 5/6 | 5/6, 5/6 |

  The text replies are byte-identical across the two files and the runs on each backend. On the GPU each file gave the same replies in both runs, and the two files' replies are byte-identical but one transcript: `en_us_1665`, where this recipe's file wrote "borders dispute" (the reference's words) and the published file "border dispute". On the CPU the transcripts of each file changed between its two identical runs (the published file 3 of 20 clips, this recipe's 2 of 20), and the two files' transcripts differ in 2 and 3 of 20 clips in the paired runs; `libri_1272_128104_0004` came out as the same two transcripts (8 and 9 errors) from both files. The voicemail question the GPU misses (`v2`) has the same reply from both files, without the number. No answer is degenerate.

- **`--sections`** against the published file:

| section | this recipe | published | |
|---|---:|---:|---|
| tokenizer (`HF_Tokenizer_Zlib`) | 2,574,275 B | 2,574,275 B | byte-equal |
| `ExecutorMetadataProto` | 2,299 B | 2,299 B | byte-equal |
| token embedder | 263,682,208 B | 263,682,208 B | byte-equal |
| text decoder | 1,032,652,688 B | 1,031,710,368 B | all 633 weight buffers byte-equal; the same 7 signatures, 840 subgraphs and 13,315 operators of the same kinds; tensor names differ |
| audio section | 233,698,144 B | 233,193,840 B | all 283 weight buffers byte-equal; the same signature `encode` and 1,118 operators of the same kinds; tensor names differ |
| `LlmMetadataProto` | 9,346 B | 9,331 B | below |

  hf-to-litertlm's `tools/tfl_compare/tfl_struct.py`, which compares each subgraph's operators by their options and inputs and ignores indices and names, finds the decoder and the audio section structurally equal to the published ones; as controls it finds the audio section different from one with float32 weights and the decoder different from an int4 blockwise-32 decoder with a 4,096-token cache (both from the published conversion). The tensor names carry the Python module of the code that traced each graph: here `_litert_torch_ext_file_lfm3_audio_extension_<hash>`, the name litert-torch gives an extension file it loads (the hash is of the file's absolute path, `load_extension_file` in `model_ext/extension.py`), and `transformers_modules.<checkpoint directory name>` for the checkpoint's remote code; in the published file, the conversion's own modules. The name strings are 943,466 B and 504,612 B longer than the published ones in the decoder and the audio section, against section size differences of 942,320 B and 504,304 B. A build from another path gives other names and another sha256.

  `LlmMetadata`: the same stops, context, model type with the same audio fields, channel and no sampler. The differences are the stock packager's: the start token as the string `<|startoftext|>` (the published file: its id 124894), a `pad_token` (`<|pad|>`), no empty `prompt_templates` block, and the template, which differs only in whitespace at the two places where litert-torch removed the `{%- generation -%}` / `{%- endgeneration -%}` tags (+2 B; both templates render the same text, `--render`).

### Galaxy S26

Galaxy S26 (SM-S942Q, Android 16), LiteRT-LM v0.18.1's `litert_lm_advanced_main` (the binary the model card's numbers came from), this recipe's file on 2026-11-02, its sha256 read back on the phone after the push. The benchmark runs were cold (no compiled-model cache), one process each, with `--benchmark_prefill_tokens=1024 --benchmark_decode_tokens=256 --max_num_tokens=2048`:

| backend | prefill 1,024 (tok/s) | decode 256 (tok/s) | time to first token (s) | peak memory (MiB) |
|---|---:|---:|---:|---:|
| GPU, run 1 | 1,274.31 | 39.14 | 0.83 | 983.5 |
| GPU, run 2 | 1,292.53 | 41.34 | 0.82 | 979.6 |
| CPU, 4 threads | 290.37 | 38.81 | 3.55 | 2,306.8 |

The model card's GPU numbers for the published file (the mean of two runs) are 1,295.4 / 40.61 tokens/s with 0.81–0.82 s to the first token; the mean of the two runs here, 1,283.4 / 40.24, is 0.9 % below both, and each run is within 3.7 % of the card. The card has no CPU row. Both GPU runs ran the decoder whole on the GPU (7 of 7 signatures, one partition each) and held 2.10 GB of GPU memory besides (the card: 951–975 MiB plus 2.08 GB). During the CPU run the CPU clock was below its maximum in 20 of 23 half-second samples. Peak memory is the process's VmHWM.

The 8 text questions and the 6 voicemails ran one process per question, each voicemail through the audio section on the decoder's backend (`--audio_backend`), with at most 512 output tokens:

| backend | text questions | voicemails | audio section (ms) | then first token (s) | sum (s) | peak memory (MiB) |
|---|---|---|---:|---:|---:|---|
| GPU | 8/8 | 5/6 | 94.6 | 0.445 | 0.541 | 956–1,023 |
| CPU, 4 threads | 8/8 | 4/6 | 265.2 | 2.345 | 2.609 | 2,134–2,386 |

The times are medians over the 6 voicemails (the audio section's `audio_executor` mark, one 30 s window per clip; the runtime's time to first token; their sum per voicemail); peak memory is the range over the voicemail runs. For the published file on this phone the model card gives 94.4 ms per 30 s window on the GPU (the mean of four runs). The audio section ran whole on the GPU (1,118 of 1,118 operators). The 8 text answers have the same text as the Mac's on each backend. The voicemail the GPU misses, `v2`, has the Mac GPU's reply, without the number. On the CPU `v2` has the same reply, and the answer to `dev3` stops at the 512-token limit after a thought that names the caller and the number; on the Mac CPU both files answered all 6 in all four runs. The published file was not run on the phone.

### Failure paths

- transformers 5.14.1: `build_lfm3_audio_2b.py` stops before the export: `transformers 5.14.1 cannot load LFM3's remote code: it imports use_kernel_func_from_hub_with_fallback from transformers.integrations, which transformers has since 5.15.0.`
- litert-torch-nightly 0.11.0.dev20261028 (built before 864a966): it stops before the export: `this litert-torch has no out-of-tree extension path (a .py file in extension_modules, litert-torch commit 864a966 of 2026-10-28; no release carries it yet).`
- The source template with its audio part changed (a line break added after `<|audio|>`): the export stops in the hook (after the graphs are exported, when the metadata is built), `ValueError: lfm3_audio_extension: the audio part occurs 0 times in the chat template (expected once); the source template changed, re-check how it renders audio`, and leaves no file.
- Without the extensions (`--no-extension`): the export stops at the audio graph, `ValueError: Unsupported model type: lfm3_audio` (litert-torch's `model_ext/exportables.py`, `get_audio_exportables`), and leaves no file.

### What is not measured

- Clips longer than one window of the audio section (30 s): every clip here is 29.4 s or shorter.
- Languages other than English, and answers with thinking on (the template has no thinking switch; `--render` checks the text of a turn with a thought, not answers).
- The model card's other phone (Pixel 8a) and the NPU, which this recipe builds no file for.

## Files

| File | What |
|---|---|
| `lfm3_audio_extension.py` | The out-of-tree `ModelExtension` for `lfm3_audio`: `get_audio_exportables` (the audio section), `get_prefill_decode_exportables` (the decoder with `hidden`), `get_llm_model_type_proto` (`generic_model` with the audio fields), `build_llm_metadata` (the audio part guard). |
| `lfm3_audio_extension_test.py` | `ModelExtensionTestMixin` contract tests and ten cases: the audio section and its log-mel against the checkpoint's feature extractor and transformers, the decoder's `hidden` and logits against transformers, `torch.export`, the composition with the decoder's extension, the hooks in litert-torch's metadata builder, a text-only export, the template guard and the rendering; CPU, no weights (downloads the checkpoint's code, configuration, feature extractor, tokenizer and template). |
| `recipe_int4.json` | The decoder's quantization recipe. |
| `recipe_audio.json` | The audio section's float16 recipe. |
| `build_lfm3_audio_2b.py` | HF checkpoint → `lfm3-audio-2b_int4_b128_c2048_h30.litertlm`, one stock `export()` call with the two extension files; stops early on a transformers or litert-torch that cannot run it. `--no-extension` exports without the extensions, for study. |
| `verify_lfm3_audio_2b.py` | 8 text questions, transcription (`--asr-json`) and voicemail questions (`--voicemail-json`) through the `litert-lm` Python package; `--check-metadata`; `--render` (the runtime's rendering of the template against transformers'); `--sections` (section-by-section comparison with another bundle). |
| `speech/` | Speech output: the depth decoder, frame embedding and codec graphs and the host loop ([`speech/README.md`](speech/README.md)). |
