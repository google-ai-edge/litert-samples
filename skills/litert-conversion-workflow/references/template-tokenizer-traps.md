# Template & tokenizer traps — the non-math ship-killers

More conversions have died here than on any numerical issue. Everything
below was hit on a real model; each entry gives the observable signature.

## The renderer is minijinja, not Jinja2 — and not Python

The runtime renders chat templates with a Rust minijinja build (no Python
compatibility layer). Python-style *methods* on dicts/strings are
unknown at render time; support varies by runtime build, so the only safe
template uses attribute access, `|` filters, and plain tests. Observed
crash signatures (all thrown on the user's *first or second message*,
never at import):

```
Failed to apply template: unknown method: map has no method named get
Failed to apply template: unknown method: string has no method named strip
Failed to apply template: unknown method: string has no method named startswith
Failed to apply template: unknown method: string has no method named split
too many arguments            (tojson(ensure_ascii=False) vs the built-in filter)
```

A template can also crash **only on message two**: assistant-history
branches (reasoning re-render logic) execute for the first time when a
completed assistant turn enters the history. Single-turn tests cannot
catch this — the multi-turn gate exists for it. The repair pattern for a
crashing history branch in a non-thinking ship: replace the whole
assistant-history branch with a plain verbatim emit of the turn.

Three sharper facts about the render environment:

- **The `| split` *filter* form works where the `.split()` *method*
  crashes** — minijinja has the filter, jinja2 does not; method-to-filter
  rewrites are the safe direction.
- ⭐ **The pip wheel and an OSS source build of the same runtime version
  render differently** — a template that renders on the wheel has thrown
  `unknown method` on a source build. "It renders in my venv" is not
  evidence it renders in a shipped app; only the minimal-template route
  is build-independent.
- Recent exporters ship an **experimental jinja→minijinja transpiler**
  flag. It rewrites some string methods to filters but not `.get()` /
  `.startswith()` — and **on any exception it returns the template
  unchanged**, so enabling it can look like a fix and silently not be
  one. Don't substitute it for the minimal template.

Also real jinja semantics, not a runtime bug: `{% set x = x ~ ... %}`
inside a `for` loop does not escape the loop — templates that accumulate
tool-call arguments that way silently emit empty arguments on re-render.

Exporter packaging behaviors worth knowing (recent litert-torch): a
`thought` channel is auto-declared when the packed template contains
`<think>`; stop tokens get punctuation-prefix expansion (a SentencePiece
greedy-merge guard); `sampler_top_k/top_p/temperature` export args bake a
sampler config (`top_k=1` = greedy); `suppress_tokens` is pulled from
`generation_config`. Check the peek output rather than assuming any of
these fired.

## The standing defense: embed no Jinja at all

Export with `use_jinja_template=False` and a **minimal** template swapped
onto the tokenizer (`recipe-selector.md` has the driver). The exporter
then extracts plain prefix/suffix turn markers and the bundle carries no
template code. Two caveats:

- Extraction can fail on complex vendor templates (namespaces, tool
  branches) and **fall back to embedding the raw Jinja** — exactly the
  trap. That is why the driver swaps in the minimal template rather than
  trusting extraction of the vendor one.
- Triage any bundle (yours or a third party's) in one line:

```bash
python -m litert_lm_builder.litertlm_peek_main --litertlm_file model.litertlm
```

`prompt_templates`-only = safe. A `jinja_prompt_template` containing
`.get(`/`.strip(`/`.split(` = the first-message crasher.

## Fixed system prompts belong in the user prefix

When a model's system prompt is part of its **inference contract** — the
card states it verbatim and the model was trained against it — rather than
something the caller tunes, ship it inside the user prefix instead of as a
system slot:

```
user.prefix = "<sys-open>" + THE_FIXED_SYSTEM_PROMPT + "<sys-close>" + "<turn-open>"
user.suffix = "<turn-close>"
```

The reason is API shape, not aesthetics: only the conversation path can
carry a `Role.SYSTEM` message, while `Session.run_prefill` (the scoring
entry point — `verification-gates.md` §0) takes plain strings. A separate
system slot therefore makes the model's own reference prompt *unreachable
from the API you actually classify with*, and every caller who forgets it
silently gets a different model.

Extraction handles this cleanly: a template that simply ignores the system
role takes the exporter's "no system prompt" path, and the fixed text
lands in the user prefix where it belongs. Verify with peek, then check
that one rendered turn is byte-identical to the vendor template's render.

Caveat to state on the card: a caller who *does* pass a system message
will have it dropped. That is the right trade when the prompt is fixed by
the model, and the wrong one when it is a real knob.

## Multi-turn: the prefix contract

At each message the engine renders the history *without* a generation
prompt and requires the new render to **string-extend** the previous one
— it prefills only the suffix. Hard failure:

```
new rendered template string does not start with the previous
```

Templates that rewrite history violate it — the classic case is a
reasoning template that strips `<think>` blocks from past assistant
turns: turn 2 dies. Fix pattern for hybrid-thinking models shipped
non-thinking: the generation prompt appends
`<|im_start|>assistant\n<think>\n\n</think>\n\n`, and history assistant
turns render **with the same empty think block**, so
render(history) ≡ what was actually prefilled + generated. Subtler
hazard even when the check passes: if the render diverges from the live
token stream, the engine resolves the difference by rewinding — safe for
slot-addressed KV, corrupting for running-state models. History renders
must be append-only extensions of the real stream.

## Tokenizer packaging

- **The HF-tokenizer bundle path can mis-tokenize.** Two observed forms:
  prompts garbled end-to-end (fluent output ignoring the prompt), and
  SentencePiece-style `▁` markers unhandled on decode so **all spaces
  vanish** ("Theusersaid42"). When the source ships a real SP model or a
  convertible BPE, bundle an `SP_Tokenizer`; verify by round-tripping a
  prompt through the packed tokenizer.
- **BPE→SP conversion trap**: converters that read `tokenizer.vocab_file`
  choke when it points at `vocab.json` ("Wire format was corrupt" — it
  expects an SP ModelProto). Clear `vocab_file` unless it ends in
  `.model`/`.spiece` so the conversion builds from vocab+merges.
- **Added special tokens get dropped** by SP conversion (they live in
  `added_tokens_decoder`, not the base vocab). A thinking model then
  *generates* `<think>` and the runtime crashes:

  ```
  NOT_FOUND: Token id 166103 is out of range. Vocab size is 166100
  ```

  Fix: append every added token as a `USER_DEFINED` piece **at its exact
  id**, pad with `UNUSED` pieces up to the model's embedding
  `vocab_size`. Test by encoding `<think>` and asserting the id. This
  bites models whose specials sit **beyond** the base vocab (typical for
  Llama/Mistral-based finetunes that append ChatML or `<think>` tokens)
  — also as a missing stop: `<|im_end|>` absent from the SP model means
  the stop never fires. It is a **no-op for the Qwen family**, whose
  specials live inside the base vocab — check
  `added_tokens_decoder` ids against the base vocab size before planning
  the patch.

## Stop tokens

- The bundle must declare **every** turn-end token, not just
  `config.json`'s `eos_token_id`. ChatML finetunes commonly stop on
  `<|im_end|>` while eos is `<|endoftext|>` — declaring only eos makes
  the literal `<|im_end|>` text leak into every reply. Current
  litert-torch exporters usually get this right by reading
  `generation_config.eos_token_id` (often a list) — so **verify via
  peek first, and intervene only if the union is missing**. The union =
  `generation_config.eos_token_id` ∪ the template's actual turn suffix.
- Metadata stop tokens can be token **ids** or literal **strings**;
  string stops catch a model that spells a stop marker in text form that
  never tokenizes to the stop id.
- Reusing a sibling model's metadata as the recipe carries the sibling's
  **token ids**. Stop ids are per-tokenizer, not per-family: the same
  `<|im_end|>` was id 7 in a 1.2B and 124900 in its 2.6B sibling.
  Re-derive every id from the target model's tokenizer before packing —
  peek only proves ids are present, not that they are the right ones.

## Thinking-model metadata

- Declare the thought channel with the model's real markers:
  `channels { channel_name: "thought" start: "<think>" end: "</think>" }`
  — placeholder marker strings never fire and the thought text floods
  the main channel.
- Thinking-budget enforcement is inert unless the metadata carries
  `thinking_end_token_ids` (the runtime logs a warning and ignores the
  budget).
- Beware `enable_thinking | default(...)` in templates: the engine
  injects `enable_thinking` into the render context only when a thinking
  config is set, so the template's default silently decides behavior for
  every plain conversation. Choose the default to match how you gated
  the model.

## Start token

BOS convention lives in `recipe-selector.md` §Start token — wrong BOS is
a metadata bug that looks exactly like a quality problem, and small
models are the ones it flips. Check what the exporter actually shipped
before planning a repair: no `start_token` block in the peek output (and
a None BOS from the engine API) means the bundle already matches a
no-BOS template.
