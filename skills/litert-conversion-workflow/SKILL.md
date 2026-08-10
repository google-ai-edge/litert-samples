---
name: litert-conversion-workflow
description: Convert a Hugging Face LLM or vision-language model checkpoint into a .litertlm bundle that runs on the LiteRT-LM runtime with verified quality - classify the architecture against known runtime walls, pick the recipe family (dense, reasoning, hybrid SSM, VLM), export, quantize, gate the result against the source model, and publish. Use when converting a new LLM or VLM to LiteRT-LM, when a converted bundle crashes on the first message or dies at engine creation, when a quantized model answers worse than its source, or when deciding whether a model is convertible at all.
---

# LiteRT-LM conversion workflow

A conversion is done when three things hold, in this order:

1. the bundle loads and generates through the LiteRT-LM engine (not just
   the raw interpreter),
2. **output quality is gated against the source model** — a floor gate plus
   a task-level parity check, not a smoke test,
3. it holds up on the deployment path it claims: the target backend, the
   target device, and multi-turn conversation.

Each step can pass while the next one fails. A bundle that converts can
die at engine creation; an engine that generates can be quantization
garbage; a model that answers 8/8 single-turn can crash on message two.
The gates exist because every one of these has happened.

Scope: models producing a `.litertlm` bundle consumed by the LiteRT-LM
engine (`pip install litert-lm`) — text LLMs and vision-language models
(a VLM bundle is an LLM bundle plus two vision graphs;
`references/vlm-conversion.md` covers the delta). Standalone/classic
`.tflite` models go through the `gpu-clean-conversion` →
`accuracy-safe-quantization` → `on-device-verification` lane; this skill
is the LM sibling and reuses their discipline where it applies.

## Step 0: classify before you convert

Look at `config.json` (`model_type`, `layer_types`, MoE fields, size)
**before** running anything. Architecture decides everything downstream,
and some structures die at a known point no recipe can route around.
`references/architecture-walls.md` is the lookup table: structure → where
it dies (export / load / engine / backend) → the error signature you'll
see. Check it first; it turns a day of debugging into a table lookup.

Then pick the lane in `references/recipe-selector.md`: plain dense
decoders ride a standard export; hybrids (SSM / linear-attention /
short-conv) need state-aware export plus executor metadata; reasoning
models need template care; MoE and MLA are currently walls.

## Loop

**1. Export with a minimal, extraction-safe template.** The single most
common ship-killer is not math — it is the chat template. Export with
`use_jinja_template=False` and a minimal ChatML-style template swapped in,
so the bundle carries plain prefix/suffix markers and **no Jinja at all**.
Vendor templates routinely call Python methods (`.get()`, `.startswith()`,
`.strip()`) that the runtime's minijinja renderer does not implement — such
a bundle imports fine and dies on the first message. Details, the
multi-turn prefix contract, and the tokenizer traps that pair with this:
`references/template-tokenizer-traps.md`.

**2. Check the bundle before measuring anything.**

```bash
python -m litert_lm_builder.litertlm_peek_main --litertlm_file model.litertlm
```

`prompt_templates`-only = safe. A `jinja_prompt_template` carrying
Python-method calls = the first-message crasher. Also confirm the stop
tokens and (for hybrids) the executor-metadata section are present.

**3. Quantize on the LLM lane.** int8 dynamic is the safe default;
int4 must be **blockwise** (block-32 quality, block-128 for ~4B / iPhone
section limits) — channelwise int4 collapses decoders while still passing
smoke tests. Conv/scan layers of hybrids stay float. The decision facts
and the recipe mechanics: `references/recipe-selector.md` §Quantization,
plus the `accuracy-safe-quantization` skill for the general ladder.

**4. Gate quality — floor, parity, then structure.** Run the gate stack
in `references/verification-gates.md`:

- **8-question floor gate** on the engine (CPU, then the target backend).
  Catches collapse, never proves parity.
- **Task parity** (e.g. GSM8K n≥100) against the source model, same
  prompt and extraction both sides. Reasoning models need
  max-tokens ≥ 2048 or int4 falsely looks degraded.
- **First-token length sweep** — chat-templated prompt lengths are
  chunked by the engine's prefill planner, and state-carrying models
  corrupt at *specific lengths* while answering perfectly at others.
- **Multi-turn** — single-turn evals structurally cannot catch
  template-contract violations that kill message two.

**5. Gate on the target device.** Desktop GPU pass ≠ mobile GPU pass
(different delegates, different compilers). Record device, backend,
runtime version, and speeds next to the quality numbers — the
`on-device-verification` skill's rules apply unchanged.

**6. Publish behind the gate.** The upload step must mechanically refuse
unless the quality report passed — a collapsed quant that reaches a public
repo costs more than every hour the gates cost. Card conventions and the
publish checklist: `references/verification-gates.md` §Publish.

## Symptom router

| What you see | Where to look |
|---|---|
| Export raises in `torch.export` / tracing | `architecture-walls.md` — data-dependent guards (MoE gating, dynamic rope), dtype-in-constant-path traps |
| Export OK, engine creation fails: `No KV cache inputs found` | Hybrid on a pre-0.15 runtime — runtime too old for the architecture |
| Engine OK, generation dies: `NOT_FOUND ... missing some output TensorBuffers` | Hybrid bundle missing the executor-metadata section (`recipe-selector.md` §Hybrids) |
| First message crashes: `unknown method: map has no method named get` | Vendor Jinja embedded in the bundle — re-export per Loop step 1 |
| Output is fluent garbage / prompt seems ignored | Tokenizer packaging (`template-tokenizer-traps.md` §Tokenizer) |
| Runtime crash: `Token id N is out of range` | Added special tokens dropped from the packaged tokenizer — same file, §Added tokens |
| Stop token printed as literal text / model never stops | Stop-token metadata incomplete — §Stop tokens |
| Answers fine at some prompt lengths, garbage/empty at others | Prefill-padding state corruption — run the length sweep, `verification-gates.md` |
| Message two fails or quality drops mid-conversation | Template prefix contract violated — `template-tokenizer-traps.md` §Multi-turn |
| int4 passes the 8-question gate but tanks the benchmark | The floor-gate trap — gate is a floor, parity is the verdict (`verification-gates.md`) |
| Every build scores 8/8 and the gate never discriminates | Floor items are saturated — rebuild the gate from borderline cases (`verification-gates.md` §1) |
| The converted model *beats* its source | Suspect the harness, not the recipe: a mis-prompted or mis-rendered reference (`verification-gates.md` §2, `vlm-conversion.md` §Gates) |
| Scored numbers look plausible but wrong; generation is fine | The scoring-API traps — session reuse and `apply_prompt_template=True` (`verification-gates.md` §0) |
| Quality flips only on one backend | 4-layer triage: torch → torch-control → engine-CPU → engine-GPU (`verification-gates.md` §Triage) |
| GPU rejects the graph (`not fully delegated`, named op) | `architecture-walls.md` §Backend walls; classic-op rewrites live in `gpu-clean-conversion` |
| Vision tower aborts export (`grid_thw`, `cu_seqlens` guards) | Dynamic-resolution tower — static rewrite (`vlm-conversion.md`) |
| VLM bundle: "Failed to create conversation" | Structured prompt_templates missing from metadata (`vlm-conversion.md`) |
| VLM engine creation fails on device naming GATHER_ND | Patch reorder in the vision tower — raster-order rewrite (`vlm-conversion.md`) |
| VLM answers ignore the image / describe the wrong thing | Preprocessing contract (mean/std, NCHW) or embedder injection — gates in `vlm-conversion.md` |

## Watch for

- **Pin and record the toolchain.** litert-torch / litert-converter /
  litert-lm-builder / ai-edge-quantizer / transformers versions decide
  what exports and what runs; several walls in the references are
  version-bounded facts. A conversion note without versions is not
  reproducible.
- **The runtime moves under you.** A wall table entry is a dated fact,
  not a law — re-test walls on each runtime/converter release, in both
  directions: walls fall (blocked architectures start working) and
  regressions appear (bundles that ran stop running). Both have happened
  in the same release.
- **Never diagnose quality without a control.** Run the same eval on the
  source model with the same prompt and extraction; a broken harness
  undermeasures everyone and reads as a conversion bug.
- **The engine adds behavior the graph does not have** — start-token
  prepending, prefill chunking, cross-conversation caching. When engine
  output differs from a raw-graph replay, reproduce the engine's exact
  token stream before blaming the graph.
- **One variable at a time at ship time.** Repair published artifacts
  from the published files, not from local experiments; verify by
  checksum that what you gated is what you shipped.

## Output layout

Ship each conversion as a recipe, the same split the model-recipe skills
use — export, verification, and repair each separately re-runnable:

```
<model>/
  convert_<model>.py       export driver: flags, template swap, quant recipe
  templates/<model>.jinja  the minimal template the bundle embeds
  verify/                  gate scripts + their JSON results
  README.md                versions, recipe, gate numbers (device + backend
                           + runtime named), known limitations — honest ones
```

State known limitations on the card in actionable form ("prompts whose
templated length lands on 33–37 tokens can end the reply early; adding or
removing a word avoids it" — a real example). An honest limitation note
survives contact with users; a hidden one becomes an issue report.
