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
- Benchmark hygiene: serialize runs on an idle machine (parallel load has
  produced 2× phantom regressions that survived into analysis);
  `--max-num-tokens` is a **total** budget — prompt + generation — so a
  short budget truncates and masquerades as an early-stop bug; never
  quote a first-run number (shader compilation). The default
  `--cache disk` silently writes delegate cache files up to ~2× model
  size next to the bundle **and** warms later init-time numbers — use
  `--cache no` for gating, and clean the caches up either way.

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
