# Hy-MT2-1.8B Conversion

Model recipes, instructions, scripts, and utilities for converting Hy-MT2-1.8B to LiteRT-LM.

These scripts convert [tencent/Hy-MT2-1.8B](https://huggingface.co/tencent/Hy-MT2-1.8B) (dense `HunYuanDenseV1ForCausalLM`, Apache-2.0, Tencent's 33-language translation model) into a `.litertlm` bundle for the [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) runtime, and verify the result with a quality gate, a translation A/B against the HF reference, and two one-variable proofs. The published artifact is `Hy-MT2-1.8B_int8.litertlm` (1.82 GB) at [litert-community/Hy-MT2-1.8B](https://huggingface.co/litert-community/Hy-MT2-1.8B).

The model is a 32-layer dense transformer (hidden 2048, GQA 16:4 with QK-norm, head_dim 128, intermediate 6144, vocab 120,818, tied embeddings — 2.04 B tensor parameters, 1.79 B unique). The architecture needs nothing from the exporter. What this conversion documents is a **config fact** (a `dynamic` rope that is static in practice, and how to remove the one branch that kills `torch.export` without changing a bit) and a **runtime fact** (the LiteRT-LM engine prepends the metadata `start_token` to every prompt, so a template that renders its own BOS literal must not also declare one — proved inside the runtime, not inferred from outputs).

## Build

| recipe | quantization | file | use |
|---|---|---|---|
| `int8` | export-time dynamic int8 on linears + embedding (`dynamic_wi8_afp32`) | 1.82 GB | desktop CPU/GPU; phone not measured |

## Environment

```bash
pip install litert-torch==0.9.3 litert-converter==0.3.1 ai-edge-quantizer==0.8.0 \
    litert-lm-builder==0.15.0 transformers==5.14.1
pip install litert-lm   # verification CLI (>= 0.16.0 used here)
```

The published bundle was built on exactly this released stack — no patched checkout. `litert-lm-builder >= 0.15` is what the post-export metadata edit needs (pack/unpack API).

## Run

```bash
python build_hy_mt2_1_8b.py --out out_int8                                          # 1.82 GB bundle, ~2 min export
python verify_hy_mt2_1_8b.py out_int8/*.litertlm --backend cpu                      # 8-question gate
python verify_hy_mt2_1_8b.py out_int8/*.litertlm --check-metadata                   # start_token, stop, template, bytes/param
python verify_hy_mt2_1_8b.py out_int8/*.litertlm --translate                        # the model's task, vs HF bf16
python verify_hy_mt2_1_8b.py out_int8/*.litertlm --rope-ab                          # the rope bake is bitwise-exact
python build_hy_mt2_1_8b.py --out out_int8_bos --keep-start-token                   # the default (double-BOS) export, for study
python verify_hy_mt2_1_8b.py out_int8/*.litertlm --bos-discriminator out_int8_bos/*.litertlm
```

## Why the stock export fails, and why the fix is exact

`config.json` declares `rope_scaling: {type: dynamic, alpha: 1000.0, ...}` with `rope_theta: 10000` and `max_position_embeddings: 262144`. A plain `export(model, output_dir)` dies inside `torch.export`:

```
torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode:
Could not guard on data-dependent expression Eq(u0, 1)
  File transformers/modeling_rope_utils.py, in dynamic_frequency_update
    if seq_len > max_seq_len_cached:  # growth
```

The branch that fails is dead for this model. transformers 5.14's hunyuan implementation resolves the alpha form **statically at init** — an init-time special case its code comments label "DynamicNTKAlphaRotary" computes `base = rope_theta · alpha^(dim/(dim−2))` once and copies the resulting `inv_freq` — and never rescales below `max_position_embeddings`. Only the generic `@dynamic_rope_update` growth guard is left in the forward, and its data-dependent comparison is what the tracer cannot guard.

`build_hy_mt2_1_8b.py` therefore bakes the resolved base into `rope_theta` (**11,158,839.92507748**, `0x1.548a6fd9a3c16p+23`) and drops `rope_scaling`, in a local hub-format copy of the snapshot whose other files are symlinks. The export then goes through the stock path untouched. `--rope-ab` measures that this is not an approximation:

| quantity | vendor config | baked config |
|---|---|---|
| `rope_type` | `dynamic` | `default` |
| `inv_freq` | — | **bitwise-equal** (int32 view; max abs diff 0.0) |
| `attention_scaling` | 1.0 | 1.0 |
| teacher-forced logits, 17-token chat prompt, bf16 | — | **bitwise-equal** (max abs diff 0.0) |

Faithful for every position up to 262,144; past that the HF model would enter the non-alpha dynamic recomputation, which is irrelevant to a bundle with a 4,096-token context.

## The engine prepends `start_token` — proved inside the runtime

Hy-MT2's template opens with the BOS written as a **literal** — `<｜hy_begin▁of▁sentence｜>` is the first thing it renders — and its tokenizer adds no BOS at encode time. The bundle builder sets `LlmMetadata.start_token` from `tokenizer.bos_token` regardless, so the question is what the engine does with it. Comparing outputs against HF could not settle that: int8 greedy divergence happened to match the "HF + extra BOS" arm on 2 of 3 probes even after the field was removed. The discriminator that decides runs entirely inside the runtime, same weights, greedy, CPU:

| stream | how | result |
|---|---|---|
| X — `[start_token] + rest` | default bundle, `--no-template`, prompt = rendered text minus its leading BOS string | |
| Y′ — `[template BOS] + rest` | this bundle (no `start_token`), normal templating | **X ≡ Y′ byte-for-byte on 3 of 3 probes** |
| X′ — `rest` only | this bundle, `--no-template`, same BOS-stripped text | X ≠ X′ on 2 of 3 |

So the engine does prepend the start token, it encodes to the same id as the template's literal, and the default export fed the model `[BOS][BOS]…`. This bundle drops the field; the on-device stream equals the training stream. Honest numbers: the double-BOS bundle happened to score **8/8** on the sanity gate and the stream-faithful one scores **6/8** (misses "Cool" for the opposite of hot and "pink" for the rhyme — noise-level for a translation-tuned 1.8B, no degeneration). The faithful file ships anyway; matching the training stream is the property that generalizes, a lucky gate is not.

What generalizes, from auditing every published bundle afterwards: the engine renders the template with minijinja, which leaves `bos_token` **unbound** — `{{ bos_token }}` renders empty on device, and only a BOS written as a literal survives. So a template that says `{{ bos_token }}` (Falcon-H1, LFM2.5, gemma) needs the `start_token` to supply its single BOS, and dropping it would ship no BOS at all; a template with a literal BOS (Hy-MT2, Zamba2, SmolVLM2) gets one from the template and a second from the field. Render the template twice — `bos_token` bound and empty — and the two shapes separate. Do not diagnose this by comparing generations against HF; quantization noise can mimic the wrong arm.

## Verification results

Measured on the published bundle with `verify_hy_mt2_1_8b.py`, litert-lm 0.16.0, Mac M4 Max, re-run 2026-08-30:

- **8-question gate: CPU 6/8, GPU 6/8**, non-degenerate, the same two misses on both backends ("Cool", "pink") — a property of this translation-tuned checkpoint, not of a backend. Arithmetic, factual and translation items all correct. Threshold is 6/8.
- **`--check-metadata` 7/7**: no `start_token`; stop id 120020 (`<｜hy_place▁holder▁no▁2｜>`); the embedded Jinja is byte-equal to the repo's `chat_template.jinja` (654/654 bytes — the repo carries no `tokenizer_config.json` copy to disagree with); `max_num_tokens` 4096; 1,815,622,960 B / 1,791,080,448 unique params = 1.01 bytes/param; one 1.69 GiB model section.
- **Translation A/B** (`--translate`, the source card's default prompt, greedy, HF bf16 on CPU vs the bundle): byte-identical on **1 of 3** probes (Japanese); the French and English probes are fluent alternates of the usual int8-vs-bf16 kind (*spectaculaire* → *significative*; "the weather is good" → "the weather is nice"). No degeneration.
- **`--rope-ab`**: `inv_freq` and teacher-forced logits bitwise-equal (table above).

Rebuilt from scratch with `build_hy_mt2_1_8b.py` on the pinned stack (2026-08-30): 1,815,622,960 bytes, **byte-identical to the published bundle in every section** — LlmMetadata (673 B), tokenizer (2,006,574 B) and the model (1,813,574,960 B); the 39 differing bytes are the file header's uuid and creation timestamp. The rebuild passes the same metadata checks and gate with the same answers.

Mac (M4 Max, `litert-lm benchmark`, `-p 256 -d 256 --runs 3 --cache no`, litert-lm 0.16.0, quiet host at load ~3.3; two protocol runs per cell, the GPU cell after a ≥300 s rest):

| backend | prefill tok/s | decode tok/s | TTFT |
|---|---|---|---|
| GPU (Metal) | 2008 / 2003 | 105.8 / 105.4 | 0.137 s |
| CPU | 211.0 / 210.0 | 34.0 / 32.4 | 1.24 / 1.25 s |

Both backends were gated on real generations before benchmarking (a broken backend still prints benchmark numbers). GPU repeats within 0.4%, CPU decode within 5%. Unlike the Nemotron-3-Nano-4B hybrid in this repository, GPU works with the runtime's default compiled-graph cache on this dense bundle.

Not measured on a phone. The bundle carries the exporter's stock single 128-token prefill signature plus decode (`--prefill` changes that; the ladder was not measured).

## Files

| File | What |
|---|---|
| `build_hy_mt2_1_8b.py` | HF checkpoint → `.litertlm`: bakes the static rope (`rope_theta` 11,158,839.925, `rope_scaling` dropped), stock int8 export (embedded vendor Jinja, 4096 context), drops the metadata `start_token`. `--keep-start-token` reproduces the default double-BOS export for study. |
| `verify_hy_mt2_1_8b.py` | 8-question quality gate via the `litert-lm` CLI; `--check-metadata`; `--translate` (three translation probes vs HF bf16 greedy); `--rope-ab` (bitwise proof of the bake); `--bos-discriminator` (the in-runtime X / Y′ / X′ comparison). |
