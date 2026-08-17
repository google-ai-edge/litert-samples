# Recipe selector — architecture family → export route

Classify from `config.json` first (`architecture-walls.md` gates what is
worth attempting at all). Everything here assumes `pip install
litert-torch litert-lm litert-lm-builder ai-edge-quantizer` and records
the exact versions next to the result.

## Dense decoders (the standard lane)

Llama / Qwen / Phi / Mistral / Gemma / SmolLM / Falcon-dense / OLMo /
Granite-dense and their finetunes. One export driver, three decisions:
template, tokenizer, quant.

```python
# convert_dense.py <hf_id> <out_dir>  — the load-bearing parts
import sys
from transformers import AutoTokenizer

MINIMAL_CHATML = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n"
    "{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

# Swap the vendor chat template for a minimal one on every tokenizer the
# exporter loads, so prefix/suffix extraction succeeds and the bundle
# embeds NO Jinja (see template-tokenizer-traps.md for why this matters).
_orig = AutoTokenizer.from_pretrained.__func__
AutoTokenizer.from_pretrained = classmethod(
    lambda cls, *a, **k: (lambda t: (setattr(t, "chat_template", MINIMAL_CHATML), t)[1])(_orig(cls, *a, **k)))

from litert_torch.cli import main
sys.argv = ["litert-torch", "export_hf",
    "--model", sys.argv[1], "--output_dir", sys.argv[2],
    "--prefill_lengths", "1024,512,256,128,64,32,16,8,4,2,1",
    "--cache_length", "4096",
    "--use_jinja_template", "False",
    "--bundle_litert_lm", "True",
    "--quantization_recipe", "dynamic_wi8_afp32"]
sys.exit(main())
```

(`--model`/`--output_dir` are positional in some CLI versions; the flag
syntax above works on both.)

- **Prefill ladder** `1..1024`: the engine picks tight chunks per prompt;
  a sparse ladder (only large signatures) forces padded chunks — a
  quality hazard for state-carrying models and a TTFT cost for all.
- **Cache length**: export-time cache size is speed-neutral, but the
  *runtime* `--max-num-tokens` is not (4096 vs 1024 cost ~25–30% decode
  on a ~1B model — attention runs over the full static cache). Put the
  tip on the model card.
- **Use the model's own conventions**: match the real turn markers (a
  ChatML model gets ChatML; a Llama-3 model gets its header tokens —
  wrong-family markers leak literal token text into prompts), and match
  the BOS convention (§Start token, below).
- **`externalize_embedder=True`** for anything ≥ ~3B headed to iPhone
  (the >2 GiB section wall), and for large-vocab models — it also dedups
  tied embeddings.
- **Looped transformers** (`num_loops` shared-weight variants): the KV
  cache needs one slot per (loop, layer) — 2×22 layers = 44 slots —
  registered as a custom cache; weights stay stored once.

**Reasoning models** (`<think>` in the template): use a thinking variant
of the minimal template (emit `<|im_start|>assistant\n<think>\n` as the
generation prompt), declare the thought channel in metadata
(`channels { channel_name: "thought" start: "<think>" end: "</think>" }`),
and keep the *empty* think block in history renders (the multi-turn
prefix contract — `template-tokenizer-traps.md`). Evaluate at
max-tokens ≥ 2048, or quantization damage is indistinguishable from
truncated thinking.

The think-opener in the generation prompt is quantization armor, not just
etiquette: think-discipline is among the first abilities int4 degrades.
With a bare `assistant\n` prompt one int4 build deliberated ~2,500 tokens
of plain text on turn 2 without ever answering, while the same weights
with the pre-filled opener answered tersely — and the pre-filled opener
also routes the thought channel cleanly (with self-emitted markers, the
closing marker can leak into the text channel).

## Hybrids (short-conv / Mamba2 / gated-delta linear attention)

Runtime ≥ 0.15 executes these on CPU via the executor-metadata section
(`architecture-walls.md`). Two extra obligations on top of the dense lane:

**1. The bundle must carry the executor-metadata section.** Released
litert-torch ≤ 0.9.3 does not write it — a fresh hybrid export loads on
0.15 and dies at first generation (`executor.cc:708`). Retrofit it
(weights untouched): read the `decode` signature's `kv_cache_*` input
names + shapes with `ai_edge_litert.Interpreter`, classify by exporter
naming convention, emit the pbtext, and repack with `litert-lm unpack` /
`litert-lm pack`:

| input name | state type |
|---|---|
| `kv_cache_k_N` / `kv_cache_v_N` | `TYPE_GLOBAL_KEY_CACHE` / `TYPE_GLOBAL_VALUE_CACHE`, `sequence_axis` = the max-size dim, `maximum_sequence_length` = that dim |
| `kv_cache_c_N` (conv), `kv_cache_s_N` (ssm), `mc_`/`mr_` (mamba), `lc_`/`lr_` (gated-delta) | `TYPE_LINEAR_ATTENTION` (opaque pass-through) |

⚠ `litert-lm pack` **silently exits 0 without writing when the output
file already exists** — delete the target first, always.

**2. If litert-torch does not support the architecture**, this becomes
exporter work, and three properties must hold before any quality gate is
meaningful — each has silently failed in practice:

- **Decode-state continuity**: the decode graph must consume *and
  produce* the running state. Check with per-position parity (below):
  position-0 correlation 1.0 with positions ≥ 1 dead = the decode graph
  has no state continuation, and a prefill-only check will never see it.
- **Prefill pad guards**: the engine's chunk planner runs **partially
  filled** chunks, and batch-1 models often skip their own pad masking —
  pads then poison conv/SSM state at specific prompt lengths. Guards
  that work: zero the pad embeddings (`tokens != pad_id`), force the SSM
  step to identity on pads (dt → −∞ pre-softplus), select the conv
  window from the last *valid* position (one-hot matmul, not gather).
- **Chunked-prefill continuation**: multi-chunk prefill must compose —
  trace the state-continuation branch for prefill signatures too.

How those properties are obtained in practice (each failed silently once):

- **Decode-state continuity comes from tracing in decode mode**: a flag on
  the exportable module makes the cache layer report
  `has_previous_state` truthy so the modeling traces its seq==1
  precomputed branch, plus roll-and-return semantics for the conv-state
  update. ⭐ The flag must ride the **pytree flatten/unflatten context**
  — tracing rebuilds cache layers, and a flag carried anywhere else is
  dropped, producing an export **byte-identical** to the non-decode one
  (you will diff two files and see nothing).
- ⭐ **The `prefill_1` signature takes the seq==1 branch**, where a plain
  `copy_` conv-state update silently **broadcasts** `[B,dim,1]` over the
  K-wide buffer and destroys the state. Use one unified
  `cat(old, x)[..., -K:]` expression across all signatures.
- **The prefill ladder does not fix pad poisoning.** The engine's chunk
  planner is "cautious greedy": it runs the remainder chunk **partially
  filled** (e.g. 256 real tokens through a 512 signature), so only
  prompt lengths that exactly hit a signature are pad-free — the pad
  guards are mandatory, the ladder only narrows exposure. Magnitude when
  missed: a short-conv model shipped with pad-poisoned prefill produced
  one junk first-token per reply and silently cost ~22 GSM8K points —
  no crash, nothing visibly wrong.
- **Re-check the graph after adding guards**: a mask-based guard
  (`mask.sum()`) can itself introduce **int64** ops that kill the GPU
  delegate. After any masking patch: (a) no GATHER_ND, (b) no int64.
- **transformers minors move the state contract** (layer-type strings
  renamed, states became one-element containers, private mask hooks
  removed) — re-verify any layer-type registration and mask patch per
  transformers version, and verify patched-vs-stock eager logits are
  **exactly** equal before trusting anything downstream.
- **Decode-state contracts differ per family even within "hybrids"**:
  one family mutates its conv state in-place inside the modeling
  (functionalization carries it out), another expects the cache layer to
  roll-and-return. The checklist above transfers between families; the
  code does not — re-derive the contract from the modeling source each
  time.

**Quantization house rule for conv-bearing hybrids**: quantize linears +
embedding only (int8 or int4-blockwise); conv/scan layers stay float —
whole-graph int8 has produced empty output, and export-time conv-int8 has
cost a finetune 9 GSM8K points while being harmless on its sibling.
**A/B it per finetune**, never assume across a family.

## Start token (BOS) — small models flip on this

The engine prepends `start_token` from the bundle metadata. Match the
model's own template convention: if the official chat template has no
BOS, ship no start_token. A wrong BOS is invisible on robust models and
flips greedy trajectories on small ones (a 350M answered 8/8 without BOS
and 1/8 with it — engine faithful both times, packaging wrong). Repair is
metadata-only: unpack → delete the `start_token` block → pack.

## Quantization decision facts (LLM lane)

The general ladder lives in the `accuracy-safe-quantization` skill; these
are the LLM-specific facts that override intuition:

- **int8 dynamic (`dynamic_wi8_afp32`) is the safe default** and often
  *beats* data-free int4 on quality (one 1B scored int8 63 vs int4 48
  GSM8K from identical weights).
- **int4 must be blockwise.** Channelwise int4 collapses decoders (0%
  GSM8K with a passing smoke gate, degeneration over length). Block-32
  for quality ≤ ~3B; block-128 for ~4B (fits the iOS section budget,
  lighter dequant). Two overrides: **math/reasoning models want block-32
  even at 3–4B** (one 4B reasoner: b32 −8 vs b128 −15 GSM8K points), and
  on **Apple GPU block-32 is also the faster kernel** (see the backend
  walls). Both failure directions exist — one 3B collapses at block-128
  (90→64%) while one 4B corrupts at block-32 on a phone GPU — so when a
  model underperforms at one granularity, try the other before giving
  up.
- **OCTAV (data-free optimal clipping) over min-max** for int4 weights;
  embeddings stay int8 in every int4 recipe. **Exception: ternary
  checkpoints use min-max, never OCTAV** — min-max lands the {-1,0,+1}
  grid exactly; OCTAV's clipping can move it.
- **Ternary checkpoints are a free lunch**: {-1,0,+1}-scaled weights land
  exactly on the int4 blockwise grid (with min-max, above) — zero
  rounding decisions, verify the roundtrip rather than budgeting for
  loss.
- **The int4-fragile family is "deep-narrow"**, not "small": deep-narrow
  decoders with activation multipliers have lost 14–28 points data-free
  at every granularity while shallow-wide peers of the same size ship at
  parity. When a family scans fragile-diffuse, prefer a shallow-wide
  alternative over recipe iteration.
- **Sub-0.5B decoders: fp16, not int**. At ~0.3B, int4 and even dynamic
  int8 have corrupted task output where fp16 float-casting was
  bit-faithful — below half a billion parameters the headroom just
  isn't there.
- **A calibrated artifact's score may be unreachable data-free.** If a
  published int4 of the same model scores higher than your data-free
  int4 rebuild, suspect a calibrated pass, not your pipeline — int8
  typically reaches the same level. Never ingest a calibrated (GPTQ)
  checkpoint by re-quantizing it: symmetric full-range grids re-round and
  break the calibration compensation while smoke gates keep passing.
- **Weight-only (explicit dequantize) vs dynamic**: weight-only buys
  quality when activation quantization is the floor — but it
  materializes fp32 weights at prepare time (RAM = fp32, not the file
  size), and its int4 form crashes the Apple GPU delegate. Prefer
  dynamic; use weight-only deliberately.
- **CPU and GPU invert**: on CPU, int8 beats int4-blockwise on prefill
  *and* quality; on GPU, int4 prefills ~4× faster at equal decode. If
  both backends matter, ship both variants and say which is which.

Example recipe JSON for the exporter's `--quantization_recipe` (blockwise
int4 + OCTAV, embedding int8 — the LLM quality recipe):

```json
[
  {"regex": ".*", "operation": "FULLY_CONNECTED",
   "algorithm_key": "OCTAV",
   "op_config": {"weight_tensor_config": {"num_bits": 4,
     "granularity": "BLOCKWISE_32", "symmetric": true},
     "compute_precision": "INTEGER"}},
  {"regex": ".*", "operation": "EMBEDDING_LOOKUP",
   "algorithm_key": "min_max_uniform_quantize",
   "op_config": {"weight_tensor_config": {"num_bits": 8,
     "granularity": "CHANNELWISE", "symmetric": true},
     "compute_precision": "INTEGER"}}
]
```

(Field names drift across ai-edge-quantizer versions — validate against
the installed version's `recipe.py` presets before relying on it.)

## Adjacent lanes (same discipline, different bundles)

- **VLMs**: a full recipe family of this skill — see
  `vlm-conversion.md` (runtime contract, tower static-rewrite toolkit,
  candidate selection, vision gates).
- **Encoders** (embedding/classification): no KV cache — direct
  multi-signature trace, not `export_hf`; gate = bit-exact parity at
  valid positions + pad-content invariance (padded-vs-unpadded abs diff
  is reduction-order float noise; don't gate on it).
- **Decoder-based classifiers** (safety/guard models, rerankers, judges):
  these ride the dense lane unchanged — the export is ordinary — but they
  are **not chat models**, and the gate stack has to be rebuilt around
  that. The reply is one token chosen from a fixed pair, and the shipped
  quantity is usually the softmax over those two logits.
  - Read it out with `Session.run_text_scoring`, minding both traps in
    `verification-gates.md` §0; the generate path gives the thresholded
    verdict for free (one prefill) while the continuous score costs two.
  - The 8-question floor gate saturates — use borderline items
    (`verification-gates.md` §1) — and task parity becomes label
    agreement + margin correlation (§2).
  - If the model's system prompt is fixed by its own card rather than
    user-tunable, **bake it into the user prefix** rather than shipping a
    system slot (`template-tokenizer-traps.md` §Fixed system prompts).
  - Budget the *whole* decision: `cache_length` matters because the
    document is the prompt, while the usual decode-length tax does not
    apply at all — one token out.
- **Multi-graph TTS / audio**: LM talker + per-stage graphs + a host
  loop; the real quality gate is a task-level round-trip (ASR on the
  audio), not per-token match.
