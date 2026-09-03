# Nemotron-3-Nano-4B Conversion

Model recipes, instructions, scripts, and utilities for converting Nemotron-3-Nano-4B to LiteRT-LM.

These scripts convert [nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16) (`NemotronHForCausalLM`, 3.97 B parameters, [NVIDIA Nemotron Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/)) into a `.litertlm` bundle for the [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) runtime, and verify the result with a think-aware quality gate, a metadata check, and two one-variable reproducers. The published artifact is `Nemotron-3-Nano-4B_int8.litertlm` (4.13 GB) at [litert-community/Nemotron-3-Nano-4B](https://huggingface.co/litert-community/Nemotron-3-Nano-4B). **Requires litert-lm ≥ 0.15**, which is where the runtime started binding hybrid state through the `ExecutorMetadata` section.

The model is a three-kind hybrid: **21 Mamba2 selective-scan layers + 17 plain MLP layers + 4 grouped-query attention layers** (42 in all, `hybrid_override_pattern` `M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-`), hidden 3136, 40 query / 8 KV heads, mamba 96 heads × 80 dim (state 128, conv kernel 4, 8 groups), vocab 131,072, untied embeddings. It is a reasoning model: ChatML turns, and every reply opens with a `<think>` block. Only the 4 attention layers keep KV; the mamba layers carry constant-size conv + SSM state and the MLP layers none — **50 state buffers** in total, so memory stays nearly flat with context.

No released litert-torch (0.9.3, 0.9.4) carries a Mamba2 cache layer — 0.9.4's generic path maps the mamba layers to a linear-attention cache and dies before tracing (measured on a granite-4.0-h checkpoint, the same Mamba2 cache gap; no released wheel was run on this checkpoint). This recipe therefore pins a litert-torch checkout and applies **`nemotron_h_litert_torch.patch`** (1,878 added lines: the NemotronH export extension, the granite-4-h extension it ports its scan from, and a hybrid cache with a cache-less layer type). What this conversion documents is the **family recipe** (six steps, each with its measured reason), the **`--cache no` GPU trap** (isolated as a one-variable comparison), and three facts from evaluating the checkpoint — `auto_map` is not proof of remote code, the ≥3B prefill ladder, and a spurious `start_token` the model happens to tolerate.

## Build

| recipe | quantization | file | use |
|---|---|---|---|
| `int8` | post-hoc dynamic int8 on linears + embedding (convolutions and the selective scan stay float); fp32 activations declared | 4.13 GB | desktop CPU, desktop GPU with `--cache no`; phone not measured |

## Environment

```bash
pip install litert-converter==0.3.1 ai-edge-quantizer==0.8.0 ai-edge-litert==2.1.6 \
    litert-lm-builder==0.15.0 transformers==5.14.1 torch==2.12.1
pip install litert-lm   # verification CLI (>= 0.15; 0.16.0 used here)
python build_nemotron_3_nano_4b.py --setup   # clones litert-torch @ 115a1360 (2026-06-19) into ./litert-torch-nemotron and applies the patch
```

The build script puts the patched checkout first on `sys.path` and refuses to run if `litert_torch` imports from anywhere else. A released `litert-torch` wheel may be installed alongside; it is not used. The published bundle was built with this checkout + patch on the pinned dependencies above, packaged with litert-lm-builder 0.15/0.16 and gated with litert-lm 0.16.0.

## Run

```bash
python build_nemotron_3_nano_4b.py --out out_int8                                   # ~9 min on an M4 Max; a 15.97 GB float intermediate plus converter temporaries
python verify_nemotron_3_nano_4b.py out_int8/*.litertlm --backend cpu               # 8-question gate, think-aware
python verify_nemotron_3_nano_4b.py out_int8/*.litertlm --backend gpu               # same, GPU (the script always passes --cache no)
python verify_nemotron_3_nano_4b.py out_int8/*.litertlm --check-metadata            # start_token, stops, template, 50 state buffers, fp32
python verify_nemotron_3_nano_4b.py out_int8/*.litertlm --cache-ab                  # GPU with the default cache vs --cache no
python verify_nemotron_3_nano_4b.py out_int8/*.litertlm --bos-ab                    # HF bf16 greedy with / without the leading <s>
```

## The family recipe, step by step

`build_nemotron_3_nano_4b.py` exports once in float and then edits the bundle through the litert-lm-builder pack/unpack API — one unpack, one pack; the weights are quantized once and never touched again.

1. **Float export through the patched exporter** — bundle, vendor Jinja embedded verbatim, 4096-token KV budget for the 4 attention layers, reduced 7-signature prefill ladder (1024/256/64/16/4/1 + decode). The patch's scan is a *port*, not reuse: NemotronH's `torch_forward` is an older spelling of the same SSD idiom as granite-4-h, with a min-only `dt` clamp (pads must be forced to `dt = 0` *after* the clamp or a partially filled prefill chunk decays the state) and two width-0 `d_mlp` slices in the in_proj split. The lesson that cost the most time: `NemotronHBlock` picks its mixer class from a module-level `MIXER_TYPES` dict bound at import, so swapping the class attribute alone exports the **unrewritten reference scan** — with every parity gate green, because reference math is correct math. A parity gate can never prove which scan form got traced; only a padded-prefill sweep can. The patch swaps the dict entry and asserts loudly that its mixers were instantiated.
2. **Post-hoc int8 on the linears and the embedding only.** `FULLY_CONNECTED` and `EMBEDDING_LOOKUP` (channelwise) go to dynamic int8; the causal convolutions and the selective scan stay float. Export-time int8 quantizes the convolutions too; the one export-time-int8 comparison on a hybrid of another family scored lower on the gate (cause not isolated between the recipe and the template shape), and keeping the state path float is the rule every published bundle of the granite-4-h, Qwen3.5 and Nemotron-H families ships with. 3.97 B parameters → 4,126,697,184 B, 1.04 bytes/param.
3. **An `ExecutorMetadata` section.** litert-lm ≥ 0.15 binds per-layer state buffers through it; without one a state-carrying bundle fails at inference with `NOT_FOUND: The given map is missing some output TensorBuffers`. The pinned exporter writes none, so the script reads the decode signature's `kv_cache_*` inputs and declares them: `kv_cache_mc_N` / `kv_cache_mr_N` (mamba conv and recurrent state, 21 + 21) as `TYPE_LINEAR_ATTENTION`, opaque to the executor; `kv_cache_k_N` / `kv_cache_v_N` (attention, 4 + 4) with their sequence axis and the 4096 maximum. 50 buffers, cross-checked against the layer pattern.
4. **`<|im_end|>` (id 11) added to the stop set.** The vendor `generation_config` declares only `eos_token_id: 2`, so the export carries id 2 plus the template-derived string `"<|im_end|>\n"`. The tokenizer's own `eos_token` is `<|im_end|>` = 11; the recipe adds it as an id-level stop. (The case that made this guard exist was measured on another family whose bundle never stopped; here the string form was already present, and the id is belt-and-braces.)
5. **The metadata `start_token` dropped.** The bundler sets `start_token` from `tokenizer.bos_token` (`<s>`) without consulting `add_bos_token: False`, and the template opens with `<|im_start|>` — the model was never trained on a leading `<s>`. `--bos-ab` measures the consequence on the bf16 model: greedy decoding of the same rendered prompt with and without the `<s>` is **byte-identical on 2 of 3 probes**; the third diverges at token 28, inside the thought, and reaches the same final answer. The gate scores 7/8 with the field and 7/8 without it (same single miss). So at 4B this is a stream-fidelity fix, not a rescue — the 350M granite sibling is where the same mismatch bites. The published sibling Nemotron-H-4B bundle carries the field; no defect claim is made for it.
6. **`prefer_activation_type = fp32` declared on the model section.** The GPU executor runs activations in fp16 by default; on these hybrids the scan's intermediates leave that range and the engine's sampler reports invalid results and emits token 0 (measured on sibling hybrids, which is why this is the family rule — the fp16 arm was not measured on this checkpoint; `--fp16-activations` builds it for study). The GPU gate below is the fp32 bundle.

## GPU needs `--cache no`

Same runner, same file, one flag flipped (litert-lm 0.16.0, M4 Max):

- `litert-lm run --backend gpu` (default compiled-graph cache): fails with WebGPU `Invalid BindGroup` validation errors; an 8-question sweep through the Mac verifier, which has no cache flag, returned token soup — `<unk>[INST]<SPECIAL_22>…` on every question, 0/8.
- `litert-lm run --backend gpu --cache no`: answers, **8/8**.

The cache path is where it goes wrong; the root cause is not isolated further here, and the two symptoms (a validation error vs. token soup) are not asserted to be the same bug. The sibling Nemotron-H-4B card documents the same invocation, so treat it as a family-level caveat, not a property of this checkpoint. Two operational consequences: `verify_nemotron_3_nano_4b.py` passes `--cache no` on every run (on CPU the cache is a weight-sized file written next to the model), and a green CPU gate says nothing about GPU for this family — gate the backend you ship. `--cache-ab` reproduces the comparison in two runs.

## Three facts from evaluating the checkpoint

- **`auto_map` is not proof of remote code.** The repo declares `auto_map` (a back-compat shim for older transformers), and a converter that refuses on `auto_map` alone refuses this model for no reason. transformers 5.14.1 registers `nemotron_h` natively, so without `trust_remote_code` the library implementation loads and the repo's Python is never imported (`AutoConfig.from_pretrained` returns transformers' own `NemotronHConfig`; the export ran with `trust_remote_code: False`). Refuse only when `auto_map` is present *and* the model_type is absent from `CONFIG_MAPPING`.
- **≥3B exports use the reduced 7-signature ladder.** Every exported signature costs engine RAM whether or not it is called: a 12 GB iPhone jetsam-kills a 248k-vocab hybrid 4B during engine creation with the full 11-signature ladder, and the export's converter passes scale with the merged module (a full-ladder 4B export was killed three times on a 128 GB host). Both measured on a Qwen3.5-4B-class hybrid — a different model family, not this checkpoint or its Nemotron-H sibling; no correctness cost — the runtime plans coarser prefill chunks.
- **The template to compare against is the one `AutoTokenizer` resolves.** The embedded Jinja is byte-equal to the repo's `chat_template.jinja` (10,504 / 10,504 bytes). The same repo's `tokenizer_config.json` carries a *different* 10,497-byte copy; compare against that one and you report a false mismatch.

## Verification results

Measured on the published bundle with `verify_nemotron_3_nano_4b.py`, litert-lm 0.16.0, Mac M4 Max, re-run 2026-08-30 (thought budget: none imposed — the CLI runs to the stop token; thoughts on the gate are 17–48 words):

- **8-question gate: CPU 7/8, GPU (`--cache no`) 8/8**, non-degenerate. The single CPU miss is the rhyme completion — it answers "purple" where the gate wants "blue"; the GPU run answers "blue". Every arithmetic, factual and translation item is correct on both backends. Threshold is 6/8.
- **`--check-metadata` 9/9**: no `start_token`; stop ids {2, 11}; embedded Jinja byte-equal to the repo's `chat_template.jinja`; `max_num_tokens` 4096; `ExecutorMetadata` present with 42 linear-attention + 8 K/V buffers, attention caches sized to 4096; `prefer_activation_type` fp32; 1.04 bytes/param. The weights are one 3.84 GiB section (this recipe path does not externalize the embedder).
- **`--cache-ab`**: the default-cache arm exits 0 with an **empty** reply and `Validation error: Buffer size (54896640) wrapping…` followed by `[Invalid BindGroup] is invalid due to a previous error` on stderr, and leaves a 93 MB program cache and a 3,972 MB weight cache next to the model (the script removes them); the `--cache no` arm answers "42.".
- **`--bos-ab`** (HF bf16, MPS, `use_cache=False`, 96 new tokens): identical on 2 of 3, third diverges at token 28 with the same final answer (こんにちは).

Rebuilt from scratch with `build_nemotron_3_nano_4b.py` on the pinned checkout (2026-08-30, 9 min end to end on an M4 Max, float intermediate 15.97 GB): 4,126,697,184 bytes, **byte-identical to the published bundle in every section** — LlmMetadata (10,543 B, including the stop-token edit and the dropped start_token), ExecutorMetadata (3,365 B, all 50 buffers), tokenizer (2,806,488 B) and the model (4,123,829,984 B); the 193 differing bytes are the file header's uuid, creation timestamp and an `Authors` key. The rebuild passes the same 9/9 metadata checks and gates CPU 7/8 / GPU (`--cache no`) 8/8 with the same answers.

Mac (M4 Max, `litert-lm benchmark`, `-p 256 -d 256 --runs 3 --cache no`, litert-lm 0.16.0; two independent runs per cell):

| backend | prefill tok/s | decode tok/s | TTFT | init |
|---|---|---|---|---|
| GPU (Metal, `--cache no`) | 803 / 792 | 83.3 / 82.4 | 0.33 / 0.34 s | 7.2 / 7.4 s |
| CPU | 99.6 / 113.3 | 22.7 / 22.5 | 2.64 / 2.30 s | 25.7 / 24.9 s |

GPU repeats within 1.4%; CPU prefill spreads 13%, partly because the host was **not idle** during these runs (a concurrent export) — read the CPU column as an order of magnitude. `--cache no` is part of the protocol, not tidiness: with the cache the benchmark reports a much faster CPU prefill (395 tok/s was measured that way) because it is not doing the same work.

**Not measured on a phone.** A 4B of this family did not fit an 8 GB Android phone when the sibling Nemotron-H-4B was measured (engine creation aborts on both backends), so expect to need a higher-RAM device; that is an expectation carried over from a different bundle, not a measurement of this one.

## Files

| File | What |
|---|---|
| `build_nemotron_3_nano_4b.py` | `--setup` pins and patches litert-torch; then HF checkpoint → float export → post-hoc int8 (linears + embedding) → `ExecutorMetadata` (50 buffers) → stop id 11 → `start_token` dropped → fp32 activations → `.litertlm`. `--keep-start-token`, `--fp16-activations`, `--keep-float` for study. |
| `nemotron_h_litert_torch.patch` | Against litert-torch `115a13607c730c81018bb9789138a3e5e5119e3d`: `model_ext/nemotron_h` (folded rank≤4 SSD scan port with the pad guard and the `MIXER_TYPES` swap), `model_ext/granitemoehybrid` (the scan it derives from), `core/cache.py` (hybrid cache; cache-less layer type for the MLP blocks). Apache-2.0 headers. |
| `verify_nemotron_3_nano_4b.py` | Think-aware 8-question gate via the `litert-lm` CLI (always `--cache no`); `--check-metadata`; `--cache-ab` (the GPU trap, one variable); `--bos-ab` (HF bf16 greedy with / without the leading `<s>`). |
