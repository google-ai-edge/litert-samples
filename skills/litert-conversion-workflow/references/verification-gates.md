# Verification gates — floor, parity, structure, device, publish

Run them in this order; each catches what the previous one cannot. The
recurring lesson: **every gate here has been passed by a broken model** —
except the combination.

## 0. The gate harness: the engine's Python API

The `litert-lm` CLI is single-prompt and has no output-token cap — two of
the gates below (the length sweep and multi-turn) need the Python API
that ships in the same package:

```python
import litert_lm
engine = litert_lm.Engine("model.litertlm")          # backend selectable
conv = engine.create_conversation()                   # fresh state
r = conv.send_message("hello", max_output_tokens=8)   # greedy via sampler config
text = r["content"][0]["text"]                        # NOT r["text"] — note the shape
ids = engine.tokenize(conv.render_message_to_string("hello"))  # templated length
```

Two rules before reading any harness result:

- **Prove the harness on a known-good case first.** A wrong response key
  (`r["text"]`) returns "" at every length — byte-identical to the
  corruption signature the sweep hunts. The "never diagnose without a
  control" rule applies to your harness, not just the model.
- Run structure gates on **CPU first** (isolates the graph), then repeat
  on the backend you ship.

### Scoring: `Session`, not `Conversation` — and two silent traps

Some models are read out by *scoring* candidate continuations rather than
generating (classifiers, rerankers, anything whose answer is a choice
between fixed strings). `Session.run_text_scoring` returns per-token
**log-probabilities**, so `sigmoid(z_a − z_b)` is the softmax over two
candidates. Both traps below return plausible numbers rather than errors,
and the generate path is unaffected by either — a bundle can generate
perfectly while every scored number is wrong.

- **`run_text_scoring` advances the session.** Scoring a second candidate
  after the same prefill returns a value that is neither candidate's true
  score (measured on a known-good bundle: a candidate scoring −16.71 on a
  fresh session read −12.93 when scored right after another). Give every
  candidate **its own session and its own prefill**. It also refuses more
  than one target per call (`INVALID_ARGUMENT: Target text size should
  be 1`), so sharing the *prefill* is the tempting shortcut — and it is
  the broken one.
- **`create_session(apply_prompt_template=True)` prefills the user
  *prefix* only** — no user suffix, no start token — so scoring happens
  mid-prompt. Isolated by reproducing the value from hand-built strings:
  the flag's output was bit-identical to `prefix + body` with the suffix
  and BOS missing. On one bundle that read a margin of 1.060 where the
  correct stream reads 8.087, and it inverted a clearly-correct item.

```python
prompt = "<s>" + user_prefix + body + user_suffix      # render it yourself
z = []
for cand in ("yes", "no"):
    s = engine.create_session(apply_prompt_template=False)   # NOT True
    s.run_prefill([prompt]); z.append(s.run_text_scoring([cand]).token_scores[0][0])
    s.close()                                                # fresh session per candidate
```

Verify the render once: `engine.tokenize(pre_rendered)` should equal the
source tokenizer's ids exactly, and the bundle's own
`Conversation.render_message_to_string(body)` should equal that string
minus the start token the engine prepends.

## 1. The 8-question floor gate

Eight fixed, unambiguously checkable questions through the engine's own
conversation path, greedy, one question per conversation:

```python
# gate8q.py <model.litertlm> <cpu|gpu>   (pip install litert-lm)
import json, re, subprocess, sys
QUESTIONS = [
    ("17+25",   "What is 17 + 25?",                       r"\b42\b"),
    ("capital", "What is the capital of Japan?",           r"tokyo"),
    ("opposite",'What is the opposite of "hot"?',          r"\bcold\b"),
    ("week",    "How many days are in a week?",            r"\bseven\b|\b7\b"),
    ("merci",   'How do you say "thank you" in French?',   r"merci"),
    ("8x7",     "What is 8 times 7?",                      r"\b56\b"),
    ("compare", "Which is larger: 0.9 or 0.11?",           r"0\.9"),
    ("rhyme",   'Complete the rhyme: "Roses are red, violets are ___"', r"\bblue\b"),
]
def degenerate(t):
    w = re.findall(r"\w+", t.lower())
    if len(w) >= 12 and max(w.count(x) for x in set(w)) / len(w) > 0.5: return True
    return not t.strip()
ok = 0
for label, q, pat in QUESTIONS:
    p = subprocess.run(["litert-lm", "run", sys.argv[1],
        "--prompt", q + " Answer briefly.", "--backend", sys.argv[2],
        "--cache", "no", "--temperature", "0", "--seed", "0"],
        capture_output=True, text=True, timeout=600)
    t = p.stdout.strip()
    good = bool(re.search(pat, t, re.I)) and not degenerate(t)
    ok += good
    print("[ok]" if good else "[NG]", label, "|", t[:70].replace("\n", " "))
print(f"{ok}/8")
json.dump({"correct": ok, "of": 8, "passed": ok >= 6}, open(f"gate8q_{sys.argv[2]}.json", "w"))
```

(Adjust the `litert-lm` path to your venv's `bin/` if it is not on PATH.)
Pass bar: ≥ 6/8 correct **and** zero degenerate answers, on CPU **and**
on the backend you ship. Run each question separately — a reasoning model
burns a shared budget thinking and false-fails later questions. Keep the
JSON — it is the machine-readable report the publish gate (§6) consumes.

**The gate is a floor, never a parity verdict.** On record: 8/8 with a
−14-point benchmark loss; 6/8 with a 1% benchmark (total collapse). It
catches degeneration, tokenizer garbage, template death — it cannot rank
recipes. Calibrate the bar by running the same gate on an official
published model of similar size.

⭐ **Build the floor gate from items the model finds *hard*.** The
questions above work because a general LLM's easy answers are still
reachable when it degrades. A model whose easy cases are saturated gets no
signal from them: on a binary classifier the unambiguous items carried
±9 to ±14 logit margins, so **every** build scored 8/8 — including one
whose scoring readout was silently wrong (§0). Before trusting a floor
gate, check the margin distribution of its items; if nothing sits near the
decision boundary, the gate cannot fail and is not a gate. Pick borderline
items instead, and let the parity stage (§2) carry the verdict.

Two more shapes that do not transfer: `degenerate()` is inert on
single-token outputs, and a gate whose items are all one class ("all
answers should be *no*") is passed by a model that has stopped reading the
input entirely — include both classes.

One known false positive in `degenerate()`: legitimately repetitive text
(a quoted verse restated) trips the top-word ratio. If a flagged answer
is otherwise correct, require a co-occurring diversity collapse (shrinking
vocabulary over the reply) before counting it degenerate — repetition
alone is not collapse.

## 2. Task parity vs the source model

A benchmark the model family is actually used for (GSM8K-style for
general/reasoning models), **n ≥ 100** (smaller n produced wildly wrong
rankings), **identical prompt + extraction on both sides** so quantization
is the only variable. Source model, eager, is the baseline; ship bar is
"within a few points". **Run the reference in fp32 on Apple-Silicon MPS**
— bf16-on-MPS makes one-step arithmetic errors that fp32 recovers, and a
quantized engine "beating" a broken baseline by several points makes the
ship bar meaningless. The rule generalizes: **if the quantized model
beats the baseline, suspect the baseline** (precision, backend, harness)
before celebrating. Reasoning models: max-tokens ≥ 2048 — at 512 an int4
that is actually fine looks degraded because `<think>` never closes. A
broken eval undermeasures *everyone* — if the baseline scores far below
the model's reported numbers, fix the harness before reading any
comparison.

For deeper isolation, teacher-forced logit parity with **controls**:
torch-fp32 (reference), torch-bf16 (precision floor), and a known-good
4-bit runtime of the same model if one exists — "our int4 tracks the
control 4-bit" separates conversion bugs from int4 physics.

**Choice-output models (classifiers, rerankers): three quantities, in
increasing sensitivity.** A task score alone is too blunt when the answer
is one of two fixed strings — a handful of borderline items moves it more
than the recipe does.

| quantity | what it tells you |
|---|---|
| task F1 / accuracy | the number the model is used for |
| label agreement vs the reference | how often it makes the same call |
| **correlation of the raw logit margin** | moves long before any label flips — the actual instrument |

Run the precision-floor control (torch bf16 vs fp32) through the same
three, because it sets the scale: on one 3B classifier that floor was
r = 1.0000, mean |Δmargin| 0.063, and **2 label flips out of 200** — so a
bundle showing 5 flips is at the noise level, not degraded. Print the
regression of engine margin on reference margin too; "compressed" and
"expanded" are both possible and eight floor items will not tell you which
(they suggested compression where 200 items showed a slope of 1.16).

⭐ **On these models the backend moves the score more than the bit width
does.** Same 3B, same 200 items, mean |Δmargin| against the fp32
reference: int8 CPU 1.414 · int4-b32 CPU 1.168 · int8 GPU 0.642 ·
int4-b32 GPU 0.735 — and the same int8 weights across the two backends
differ by 1.203, larger than int4-vs-int8 at a fixed backend. Thresholded
verdicts are unaffected (agreement ≥ 97.5% everywhere), but any consumer
who calibrates a threshold away from the default must calibrate it **on
the backend they deploy on**, and the card should say so.

⭐ **A parity check against a broken reference is a tautology and reads
as a PASS.** Twice on record: a conversion scored correlation 1.0
against a reference whose rotary buffer had silently loaded as zeros —
both sides shared the same broken load path, so agreement proved
nothing. The referee must come from a **different implementation path**
(a different transformers pin, the vendor's own harness, a native port)
— a second copy of your own loader is not a control.

**Hybrids: per-position parity is the decode-state diagnostic.** Drive
the raw `decode` signature 8 steps, feeding states forward, and compare
per-position logits against the eager model. Position-0 correlation 1.0
with positions ≥ 1 dead = the decode graph has no state continuation —
a prefill-only check can never see this. Aggregate correlation hides it;
always print per-position.

## 3. First-token length sweep (structure gate)

The engine chunks prompts through the prefill-signature ladder, and
state-carrying models corrupt at **specific templated prompt lengths**
while being perfect at others (observed: BAD exactly at lengths
{18–21, 40}; another model only at {33–37}). Sweep it:

- Prompt shape: **filler first, instruction last** — e.g.
  `"word word … word Output only the word BANANA."` — so the instruction
  stays adjacent to the generation prompt at every length. Measure the
  templated length with
  `len(engine.tokenize(conv.render_message_to_string(prompt)))` (+1 if
  the bundle declares a start token — check peek) and grow the filler
  until each target length is hit.
- **Validate compliance before sweeping**: run the instruction at 3–4
  spot lengths first. Models differ in which phrasing they obey greedily
  — a refusal ("I don't understand…") at *every* length is the model
  disliking your prompt, and it is byte-identical to a total-corruption
  reading. If the spot check refuses, change the phrasing, not the
  verdict.
- Greedy, ≤ 8 output tokens, fresh conversation per length; PASS = reply
  starts with the literal at **every reachable** length. (The shortest
  lengths can be unreachable — template overhead plus the shortest
  compliant instruction has a floor, and token merges can skip a value.
  Record them as unreachable, not failed.) Empty or junk at a length =
  prefill state corruption at that chunk plan.

**For state-carrying models run the sweep hermetically** — a fresh engine
per length. Conversations sharing one engine can interact through prefix
caching, which contaminates the measurement (clean graphs have shown 36/40
BAD on a shared engine and 40/40 clean hermetic — the shared-engine result
was real engine-interaction behavior, not a graph bug). Gate the graph
hermetically, then run **one** shared-engine multi-conversation sequence
as a separate probe so you know both behaviors before shipping.

## 4. Multi-turn gate

Three turns minimum through the conversation API (§0 — one
`create_conversation()`, sequential `send_message` calls): a fact in
turn 1 recalled in turn 3, plus arithmetic mid-way. Single-turn evals
structurally cannot catch the two multi-turn killers: the template
prefix contract (`template-tokenizer-traps.md` §Multi-turn — turn 2 dies
or silently rewinds) and running-state carryover bugs.

For thinking models, two more rules. Size the output budget so thinking
turns **complete** — a turn capped mid-think leaves an unterminated
assistant turn in the stream and derails every later turn, which reads
exactly like state corruption. And expect the live stream to retain past
turns' reasoning (renders append; history is not re-rendered stripped):
a model trained on think-stripped history can degrade over turns. The
enforced think-opener (`recipe-selector.md` §Reasoning) is the working
mitigation; note the context-growth caveat on the card.

## 5. Backend and device gates

- **Desktop GPU sieve** (fast): `litert-lm benchmark <model> --backend
  gpu -p 256 -d 256` — engine creation + real speed numbers, plus the 8Q
  gate on the GPU backend (CPU pass ≠ GPU pass; fp16 accumulation flips
  marginal answers). On failure, grep the log for the named op and route
  through `architecture-walls.md`.
- **Device verdict**: the desktop GPU sieve does not transfer — iOS Metal
  has its own compiler bugs, mobile GPUs reject ops desktop accepts.
  Gate on the actual target device before any public claim; the
  `on-device-verification` skill's recording discipline (device, runtime
  version, residency, speeds — or it didn't happen) applies unchanged.
- **Getting the file onto the device is part of the gate.** Three failure
  modes, each of which first read as a *model* defect:
  - the transfer tool reports failure on stdout and still needs its **exit
    code** checked — piping it through `tail` to trim the noise discards
    the status, and a truncated copy then fails at engine creation with
    `Failed to map section: Length (…) and offset (…) are too large for
    file size (…)` → `TF_LITE_PREFILL_DECODE not found in the model`.
    Read that as "the file on the device is not the file you built".
  - **out-of-space surfaces as a transport error**: the first attempt died
    as `NSPOSIXErrorDomain error 54` (socket closed) over a *wired*
    connection; only the retries gave the honest `openat(2) POSIX error
    code 28` (ENOSPC).
  - `devicectl` has **no file-delete subcommand** — reclaiming space in an
    app container means copying a 0-byte file over the target, which
    leaves a placeholder rather than unlinking.
  Verify the on-device size (or checksum) against the local artifact
  before running anything.
- Benchmark hygiene: serialize runs on an idle machine (parallel load has
  produced 2× phantom regressions that survived into analysis);
  `--max-num-tokens` is a **total** budget — prompt + generation — so a
  short budget truncates and masquerades as an early-stop bug; never
  quote a first-run number (shader compilation). The default
  `--cache disk` silently writes delegate cache files up to ~2× model
  size next to the bundle **and** warms later init-time numbers — use
  `--cache no` for gating, and clean the caches up either way. CPU
  (XNNPACK) caches accumulate beside models too — a gating campaign has
  quietly consumed ~8 GB of disk; they are regenerable, delete freely.
  **The Python API has no `--cache no`**: `litert_lm.Engine(...)` writes
  beside the bundle by default, so pass `cache_dir` and sweep it. Measured
  from gating one 3B in two variants: `*.xnnpack_cache` 3.44 GiB on CPU,
  and on **Mac** GPU `*_mldrift_weight_cache.bin` + `*_mldrift_program_cache.bin`
  at 3.43 GiB + 26 MiB (int8) and 1.71 GiB + 53 MiB (int4) — 5.2 GiB from
  the GPU rows alone. The ML Drift weight cache is filed under *Android*
  GPU in `architecture-walls.md`, but it appears on Mac too, so budget
  "≈ 2× model size on first load" for any GPU backend, not just Android.

## 6. Publish, behind a mechanical gate

- The upload script **refuses** unless the machine-readable gate report
  says passed — human discipline fails exactly once, on the model you
  most wanted to ship.
- After push, verify **remote checksum == the file you gated** (LFS
  sha256). When repairing a published artifact, start from the
  *published* file, not a local experiment, and re-gate after repair.
- The card states: minimum runtime version and why; exact toolchain
  versions; recipe per variant with sizes; gate results with device +
  backend + runtime named; which variant is the quality row vs the speed
  row per platform; and **honest, actionable known limitations** ("
  prompts whose templated length lands on 33–37 tokens can end the reply
  early; adding or removing a word avoids it"). A hidden limitation
  becomes an issue report.

## Triage: when a gate fails, decompose by execution stack

Never re-roll a failed gate and never call it noise. Run the failing
question through four layers; the layer where the flip appears is the
cause:

| layer | isolates |
|---|---|
| 1. torch bf16 eager | the model itself (margin too thin?) |
| 2. torch + quantized-dequantized weights | weight damage from the recipe |
| 3. engine, CPU backend | runtime kernels + activation quantization |
| 4. engine, GPU backend | fp16 accumulation / delegate numerics |

On record: a 1.7B failed "17+25" at layer 3 (int-compute cost one flip)
and "8×7" at layer 4 (fp16 cost two more) with weights, tokenizer, and
template each exonerated by direct test — the honest verdict was "model
margin too thin to ship", which no amount of recipe iteration would have
found. Engine-vs-graph divergence has the same rule: reproduce the
engine's exact token stream (including its start token) on the raw graph
before blaming either side.

Two refinements from cases where the first diagnosis was wrong:

- **Packaging masquerades as quantization.** A "quantization jitter"
  verdict was overturned by this ladder: device failure → reproduces on
  desktop CLI (not the device) → source model answers everything (not
  the model) → long-context recall intact (not state) → **prepending the
  bundle's start token to the source model reproduced the failure
  verbatim** (the packaging). Walk the ladder before touching the
  recipe; metadata bugs flip small models hardest.
- **Ship an fp16 variant as the control for int-quant anomalies.** A
  length-band failure that vanishes on the fp16 variant of the same
  bundle is quantization-noise × chunk-plan interaction — document it as
  a known limitation; a failure that persists at fp16 is the graph or
  the engine, and no recipe will fix it.
