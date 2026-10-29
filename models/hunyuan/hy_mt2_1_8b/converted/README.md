# Hy-MT2-1.8B Conversion

Model recipes, instructions, scripts, and utilities for converting Hy-MT2-1.8B to LiteRT-LM.

These scripts convert [tencent/Hy-MT2-1.8B](https://huggingface.co/tencent/Hy-MT2-1.8B) (dense `HunYuanDenseV1ForCausalLM`, Apache-2.0, Tencent's 33-language translation model) into a `.litertlm` bundle for the [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) runtime, and verify the result with a quality gate, a translation A/B against the HF reference, and two one-variable proofs. The published artifact is `Hy-MT2-1.8B_int8.litertlm` (1.83 GB) at [litert-community/Hy-MT2-1.8B](https://huggingface.co/litert-community/Hy-MT2-1.8B); it comes from the previous version of this recipe, updated twice since ([below](#the-published-bundle)).

The model is a 32-layer dense transformer (hidden 2048, GQA 16:4 with QK-norm, head_dim 128, intermediate 6144, vocab 120,818, tied embeddings — 2.04 B tensor parameters, 1.79 B unique). The architecture needs nothing from the exporter. What the checkpoint needs is a metadata fix, and it is a **plugin**: `hy_mt2_extension.py` is an out-of-tree litert-torch `ModelExtension` that the stock exporter loads, with no change to litert-torch and no edit of the bundle after the export. The fix rests on two **runtime facts**: the LiteRT-LM engine prepends the metadata `start_token` to every prompt, so a template that renders its own BOS literal must not also declare one (proved inside the runtime, not inferred from outputs), and LiteRT-LM 0.18 and later hand the template each message's content as a list of parts, which the vendor template prints as JSON text. A **config fact** (a `dynamic` rope that is static in practice) needed a fix too, until transformers 5.20.1.

## Build

| recipe | quantization | file | use |
|---|---|---|---|
| `int8` | export-time dynamic int8 on linears + embedding (`dynamic_wi8_afp32`) | 1.82 GB | desktop CPU/GPU; phone not measured |

## Environment

```bash
pip install --pre "litert-torch @ git+https://github.com/google-ai-edge/litert-torch@30d0be00501037b57b825434d74d64959f40d668" \
    torch==2.14.0 transformers==5.21.0 litert-lm-builder==0.18.1 \
    litert-converter==0.6.0.dev20261027 litert-quantizer-nightly==0.12.0.dev20261029 \
    ai-edge-litert-nightly==2.4.0.dev20261027
pipx install litert-lm==0.19.0   # verification CLI, in its own environment
```

litert-torch at that commit (main, 2026-10-28) has the out-of-tree `ModelExtension` path (`extension_modules` takes a `.py` file); no release carries it yet. Two pins keep it working today. `litert-lm-builder==0.18.1`: 0.19.0 no longer has the `TfLiteModelType` enum this exporter still uses when it packages the bundle, so the export fails at its last step. The verification CLI in its own environment: `litert-lm` 0.19.0 requires `litert-lm-builder==0.19.0`. Run `verify_hy_mt2_1_8b.py` from the first environment (`--translate` and `--rope-ab` need torch and transformers) with the `litert-lm` command on `PATH`.

## Run

```bash
python hy_mt2_extension_test.py                                                     # the extension's tests, CPU, no weights
python build_hy_mt2_1_8b.py --out out_int8                                          # 1.82 GB bundle, ~2 min export
python verify_hy_mt2_1_8b.py out_int8/*.litertlm --backend cpu                      # 8-question gate
python verify_hy_mt2_1_8b.py out_int8/*.litertlm --check-metadata                   # start_token, stop, template, bytes/param, K/V buffers
python verify_hy_mt2_1_8b.py out_int8/*.litertlm --translate                        # the model's task, vs HF bf16
python verify_hy_mt2_1_8b.py out_int8/*.litertlm --rope-ab                          # vendor rope = the published bundle's baked rope
python build_hy_mt2_1_8b.py --out out_int8_bos --keep-start-token                   # the default (double-BOS) export, for study
python verify_hy_mt2_1_8b.py out_int8/*.litertlm --bos-discriminator out_int8_bos/*.litertlm
```

## The fix is a `ModelExtension`, not a patch

`hy_mt2_extension.py` subclasses litert-torch's `ModelExtension` for `model_type` `hunyuan_v1_dense` and overrides one hook, `build_llm_metadata`, which the exporter calls while it writes the bundle's `LlmMetadata`. The hook does three things: it reads the embedded template's three content sites through a `format_content` macro (below); when the chat template renders the BOS even with `bos_token` bound to an empty string, it clears `start_token` (two sections down); and it keeps the tokenizer's eos, id 120020 (`<｜hy_place▁holder▁no▁2｜>`), among the stop ids. `build_hy_mt2_1_8b.py` passes the file to `export(extension_modules=...)`; from the command line the same export is `litert-torch export_hf --extension_modules=hy_mt2_extension.py` with the flags in the file's docstring.

`hy_mt2_extension_test.py` runs litert-torch's `ModelExtensionTestMixin` contract tests on the extension, a tiny-model parity and `torch.export` check, and six metadata and template cases: the double BOS is dropped, the eos stop is kept once, a template that leaves the BOS to the runtime (`{{ bos_token }}`) keeps its `start_token`, the rewritten template renders a system + multi-turn conversation the same for string content and for content parts, an unexpected template stops the export, and the hook rewrites the embedded template.

## Content parts: LiteRT-LM 0.18 and later

Since 0.18, LiteRT-LM hands the chat template each message's content as a list of parts (`[{"type": "text", "text": ...}]`) instead of a string. The vendor template prints `message['content']` as is, so a bundle that embeds it verbatim renders every user turn as JSON text. Measured on litert-lm 0.19.0 with `Conversation.render_message_to_string`: `<｜hy_begin▁of▁sentence｜><｜hy_User｜>[{"type": "text", "text": "Translate the following …"}]<｜hy_Assistant｜>`. The model still answers, which is why a gate alone does not catch it. `hy_mt2_extension.py` reads the three places where the template takes a message's content (the system message and the user and assistant turns) through a `format_content` macro that accepts both forms; the template it writes is byte-equal to the one the published bundle has carried since 2026-09-21, and on 0.19.0 it renders the three translation probes exactly as HF's `apply_chat_template` does.

## The rope: static from transformers 5.20.1

`config.json` declares `rope_scaling: {type: dynamic, alpha: 1000.0, ...}` with `rope_theta: 10000` and `max_position_embeddings: 262144`. On transformers 5.14.1 a plain export died inside `torch.export`:

```
torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode:
Could not guard on data-dependent expression Eq(u0, 1)
  File transformers/modeling_rope_utils.py, in dynamic_frequency_update
    if seq_len > max_seq_len_cached:  # growth
```

The branch that failed is dead for this model: the alpha form is resolved **statically at init** — `base = rope_theta · alpha^(dim/(dim−2))`, computed once — and only the generic `@dynamic_rope_update` growth guard was left in the forward. The published bundle was therefore exported from a copy of the config with the resolved base baked into `rope_theta` (**11,158,839.92507748**, `0x1.548a6fd9a3c16p+23`) and `rope_scaling` dropped. From transformers 5.20.1 the HunYuan rotary embedding maps a `dynamic` type with a fixed `alpha` to its own static `ntk_alpha` type, the guard is gone, and the export traces the vendor config as is (5.19.0 still has the dynamic branch; `build_hy_mt2_1_8b.py` stops with a message on such a version). `--rope-ab` measures that the two are the same computation on the pinned transformers:

| quantity | vendor config | baked config |
|---|---|---|
| `rope_type` | `ntk_alpha` (transformers 5.21.0) | `default` |
| `inv_freq` | — | **bitwise-equal** (max abs diff 0) |
| teacher-forced logits, 17-token chat prompt, bf16 | — | **bitwise-equal** (max abs diff 0) |

## The engine prepends `start_token` — proved inside the runtime

Hy-MT2's template opens with the BOS written as a **literal** — `<｜hy_begin▁of▁sentence｜>` is the first thing it renders — and its tokenizer adds no BOS at encode time. The exporter still writes the BOS as `LlmMetadata.start_token` (its check sees the BOS at the start of the rendered template), so the question is what the engine does with it. Comparing outputs against HF could not settle that: int8 greedy divergence happened to match the "HF + extra BOS" arm on 2 of 3 probes even after the field was removed. The discriminator that decides runs entirely inside the runtime, same weights (the two bundles' model sections are byte-identical), greedy, CPU, all three streams through the raw `--no-template` path:

| stream | how | result |
|---|---|---|
| X — `[start_token] + rest` | default bundle (`--keep-start-token`), prompt = rendered text minus its leading BOS string | |
| Y — `[literal BOS] + rest` | this bundle (no `start_token`), the whole rendered text, BOS literal included | **X ≡ Y byte-for-byte on 3 of 3 probes** |
| X′ — `rest` only | this bundle, the same BOS-stripped text | X ≠ X′ on 1 of 3 (English) |

So the engine does prepend the start token, it encodes to the same id as the template's literal, and the default export feeds the model `[BOS][BOS]…`. Two runtime details keep this comparison clean on litert-lm 0.19.0: every CPU run uses the same thread count (`--cpu-thread-count 4`; at a near-tie, greedy output can change with it — the French probe reads *spectaculaire* with 4 threads and *significative* with 1), and Y is sent raw rather than through the chat template, as the previous version of this check did, so that the comparison stays inside one runtime path. This bundle drops the field; the on-device stream equals the training stream. Honest numbers from the published bundle's build (litert-lm 0.16.0, 2026-08-30): the double-BOS bundle happened to score **8/8** on the sanity gate and the stream-faithful one **6/8** (misses "Cool" for the opposite of hot and "pink" for the rhyme — noise-level for a translation-tuned 1.8B, no degeneration). The faithful file ships anyway; matching the training stream is the property that generalizes, a lucky gate is not.

What generalizes, from auditing every published bundle afterwards: the engine renders the template with minijinja, which leaves `bos_token` **unbound** — `{{ bos_token }}` renders empty on device, and only a BOS written as a literal survives. So a template that says `{{ bos_token }}` (Falcon-H1, LFM2.5, gemma) needs the `start_token` to supply its single BOS, and dropping it would ship no BOS at all; a template with a literal BOS (Hy-MT2, Zamba2, SmolVLM2) gets one from the template and a second from the field. Render the template twice — `bos_token` bound and empty — and the two shapes separate; that is the test `hy_mt2_extension.py` applies. Do not diagnose this by comparing generations against HF; quantization noise can mimic the wrong arm.

## Verification results

Measured on a bundle built with this recipe in a fresh environment set up as above, `verify_hy_mt2_1_8b.py`, litert-lm 0.19.0, Mac M4 Max, 2026-10-29:

- **8-question gate: CPU 6/8, GPU 6/8**, non-degenerate, the same two misses on both backends as the published bundle ("Cool", "pink"). Threshold is 6/8.
- **`--check-metadata` 8/8**: no `start_token`; stop id 120020; the embedded Jinja is the repo's `chat_template.jinja` read through `format_content` (1,037 bytes, byte-equal to the published bundle's template); `max_num_tokens` 4096; 1,815,909,088 B / 1,791,080,448 unique params = 1.01 bytes/param; one 1.69 GiB model section; the ExecutorMetadata section the exporter now writes declares the 32 + 32 global K/V caches at 4096 and nothing else.
- **Translation A/B** (`--translate`, the source card's default prompt, greedy, HF bf16 vs the bundle): byte-identical on **2 of 3** probes (Japanese, French); the English probe is a fluent alternate ("the weather is good" → "the weather is nice"). No degeneration.
- **`--rope-ab`**: `inv_freq` and teacher-forced logits bitwise-equal (table above).
- **`--bos-discriminator`**: X ≡ Y on 3 of 3 and X ≠ X′ on 1 of 3: the engine prepends `start_token` (table above).
- transformers 5.19.0: `build_hy_mt2_1_8b.py` stops before the export with the rope message.

### The published bundle

[litert-community/Hy-MT2-1.8B](https://huggingface.co/litert-community/Hy-MT2-1.8B) was first uploaded on 2026-08-27 from the previous version of this recipe — `litert-torch==0.9.3 litert-converter==0.3.1 ai-edge-quantizer==0.8.0 litert-lm-builder==0.15.0 transformers==5.14.1`, the rope bake, and a pack/unpack pass that dropped `start_token` after the export — and gated with litert-lm 0.16.0 on 2026-08-30: gate CPU 6/8, GPU 6/8 ("Cool", "pink" on both); `--check-metadata` 7/7 (1,815,622,960 B); translation byte-identical on 1 of 3 probes; a rebuild on that stack was byte-identical in every section. It has been updated twice since: on 2026-09-21 its chat template started reading the content-parts form (the template this recipe now writes), and on 2026-10-05 its prefill/decode graph was rewritten into the runtime's GPU composite ops with a 16-token prefill signature added (weights, tokenizer and metadata unchanged; 1,827,026,224 B). Its card carries the current file's measured numbers. This recipe's file differs in bytes from both (a newer converter and quantizer) and passes the checks above.

Mac (M4 Max, `litert-lm benchmark`, `-p 256 -d 256 --runs 3 --cache no`, litert-lm 0.16.0, quiet host at load ~3.3; two protocol runs per cell, the GPU cell after a ≥300 s rest), on the 2026-08-27 file:

| backend | prefill tok/s | decode tok/s | TTFT |
|---|---|---|---|
| GPU (Metal) | 2008 / 2003 | 105.8 / 105.4 | 0.137 s |
| CPU | 211.0 / 210.0 | 34.0 / 32.4 | 1.24 / 1.25 s |

Both backends were gated on real generations before benchmarking (a broken backend still prints benchmark numbers). GPU repeats within 0.4%, CPU decode within 5%. Unlike the Nemotron-3-Nano-4B hybrid in this repository, GPU works with the runtime's default compiled-graph cache on this dense bundle.

Not measured on a phone. The bundle carries the exporter's stock single 128-token prefill signature plus decode (`--prefill` changes that; the ladder was not measured).

## Files

| File | What |
|---|---|
| `hy_mt2_extension.py` | The out-of-tree `ModelExtension`: `build_llm_metadata` reads the template's content sites through `format_content`, drops the metadata `start_token` when the template writes the BOS itself, and keeps the eos stop id 120020. |
| `hy_mt2_extension_test.py` | `ModelExtensionTestMixin` contract tests, tiny-model parity and `torch.export`, six metadata and template cases; CPU, no weights (downloads the tokenizer). |
| `build_hy_mt2_1_8b.py` | HF checkpoint → `.litertlm`: the stock int8 export (embedded vendor Jinja, 4096 context) with `hy_mt2_extension.py`; stops early on a transformers whose rope still has the dynamic branch. `--keep-start-token` exports without the extension (the default double-BOS bundle) for study. |
| `verify_hy_mt2_1_8b.py` | 8-question quality gate via the `litert-lm` CLI; `--check-metadata`; `--translate` (three translation probes vs HF bf16 greedy); `--rope-ab` (the vendor rope vs the published bundle's baked rope); `--bos-discriminator` (the in-runtime X / Y′ / X′ comparison). |
