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
catch this — the multi-turn gate exists for it.

Also real jinja semantics, not a runtime bug: `{% set x = x ~ ... %}`
inside a `for` loop does not escape the loop — templates that accumulate
tool-call arguments that way silently emit empty arguments on re-render.

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
  bites *any* model whose specials sit beyond the base vocab — also as a
  missing stop: `<|im_end|>` absent from the SP model means the stop
  never fires and generation runs forever.

## Stop tokens

- Declare **every** turn-end token, not just `config.json`'s
  `eos_token_id`. ChatML finetunes commonly stop on `<|im_end|>` while
  eos is `<|endoftext|>` — declaring only eos makes the literal
  `<|im_end|>` text leak into every reply. Check
  `generation_config.eos_token_id` (often a list) and the template's
  actual turn suffix; declare the union.
- Metadata stop tokens can be token **ids** or literal **strings**;
  string stops catch a model that spells a stop marker in text form that
  never tokenizes to the stop id.

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
models are the ones it flips.
