---
name: routing-accuracy
description: Gate a natural-language classifier that picks an action before anything runs - pair every probe with a paraphrase that avoids the label's vocabulary, score only over labels actually offered, let "decline" compete as a real class, calibrate a decision margin, and catch the three failure modes accuracy alone hides: negated requests executing the opposite action, compound requests silently half-dropped, and antonym pairs that an embedding model cannot separate. Use when routing free text to actions, when a small model is replacing a generator at the routing step, when a tool-calling agent needs to decide whether to act or ask, or when a router scores well on prompts you wrote and badly on ones you did not.
---

# Routing accuracy

A routing layer is done when five things hold:

1. every probe has a paraphrased twin that avoids its label's own vocabulary, and both are reported separately,
2. the score covers only probes whose expected label was actually offered,
3. `decline` is a candidate in the same comparison, not a threshold applied afterwards,
4. the margin between winner and runner-up is recorded per probe, and the decline threshold is calibrated on it,
5. negated, compound and antonym probes are in the set, and their results are reported separately from the headline number.

A router that has only literal probes will look finished while being a keyword matcher. That is the failure this skill exists to prevent.

## Why gate this at all

Measured on one budget Android phone (SM-A145F, Exynos 3830, 3.5 GB), routing four actions:

| | 270M function-calling `.litertlm` | 22M sentence embedder |
|---|---|---|
| routing accuracy | **22%** | **100%** |
| refusal rate | 72% | 0% |
| median latency per probe | 3120 ms | 594 ms |
| peak RSS | **983 MB** | 285 MB |

The generator was not failing at generation. It was failing at routing: it matches prompt tokens against label vocabulary, so `"do i have meetings today"` shares almost nothing with "Lists the events on the user's calendar for today" and it refuses. A generator is often the wrong model for the routing step, and the swap is worth measuring rather than assuming in either direction.

Peak RSS there is `VmHWM` from `/proc/self/status`. **Do not use a before/after heap delta** — the collector runs between samples and one such harness read `-6`.

## Loop

**1. Write the probe set with labels committed before running.** Include, per action: a literal phrasing, a paraphrase that avoids the label's vocabulary, and — for the whole set — one negated, one compound, one out-of-domain, and one antonym pair if any exists. Tag each probe `literal` / `paraphrase` / `negation` / `compound` / `out_of_domain` / `antonym`.

**2. Score by bucket, not in total.** A 67% overall that is 100% literal and 8% paraphrase is a keyword matcher wearing a disguise. Report the table; the headline number is the least informative row.

**3. Offer a decline class from the start.** Cosine similarity, a softmax over classes, and an argmax all have no "none of these". Left to themselves they invent an answer for every input. Give the decline class its own prototype bank — real out-of-domain phrasings, not a threshold — and let it compete in the same comparison.

**4. Record the margin on every probe.** Winner minus best alternative. This is the instrument; accuracy is the summary. It is what lets a caller choose act-or-ask, and it is the only way to set the decline threshold from data.

**5. Calibrate the threshold on the margin distribution, then re-check accuracy.** Pick the threshold that would have declined every observed failure without declining a majority of successes, then verify accuracy has not collapsed. If no such threshold exists, the router needs work, not a threshold.

## Reading the margin

Margins are only interpretable relative to a confusion. On the run above:

| probe | margin |
|---|---|
| "what is on my calendar today" | 0.80 |
| "do i have meetings today" | 0.73 |
| "turn on the flashlight" | 0.062 |
| "turn off the flashlight" | 0.037 |
| "no more light" | 0.002 |

Two different populations, ~20× apart. A single global threshold tuned on the easy cluster will decline nothing from the hard one. **Report the two distributions separately**, and if the hard cluster cannot be separated by any threshold, say so — that is a model finding, not a tuning problem.

A correct answer won on a margin of 0.002 is a coin flip that landed well. Treat min-margin across a bucket as the health number, not the mean.

## Three failures accuracy cannot see

| Failure | What it looks like | Why a score misses it |
|---|---|---|
| **Negation inverts** | "do not switch the torch on" routes to the *off* action | It is a correct-looking action on an inverted intent. On a measured set, 4 of 5 negated probes executed the inverse; accuracy would only look "wrong" if the label set contained a `no_action` for it, which nobody adds |
| **Compound silently half-dropped** | "turn on the light and check my calendar" routes to one of the two | Single-label argmax has no representation for it, and whichever it picks can be scored correct |
| **Antonym pair unroutable** | `open` and `close` swap, or tie | Mean cosine between opposing phrasings was **0.806**; "switch the torch on" vs "...off" sat at **0.949**. The information is not in the embedding |

Handle the first with a lexical prohibition check *ahead of* the model — negation is lexical, and a 5-line guard took that bucket from 1/5 to 5/5 where no threshold would have helped. Detect conjunctions for the second and split or ask. For the third, resolve polarity lexically and use the embedding only to pick the action class; more prototypes do not move it.

## Beware a swap that trades refusals for wrongness

Constrained decoding, grammar-constrained output and any "never abstain" pressure all raise the score while removing the refusals that were the honest outcome. Measured on a function-calling bundle, pinning the emitted name to a tool enum moved routing 28% → 38% and refusals **72% → 0%**: 14 of 18 replies were the first enum entry, literal-probe accuracy *fell* 75% → 33%, and the remainder was noise.

A grammar guarantees a valid call, not a correct one. Whenever a change removes refusals, re-check whether it also removed them *from the probes that should have been answered*.

## Held-out means held out

Writing the probe bank after seeing which probes fail converts the probe set into a training set, and the resulting number measures nothing. On record: a router at 100% on 18 self-written probes scored **67% (30/45)** on a separate set written afterwards with zero string overlap. The self-written number was not wrong, it was not a generalization estimate, and the gap was almost entirely indirect phrasing (43%) and out-of-domain input (40%).

Report both numbers and label which is which. Fixing a failure you observed means the next evaluation set must be fresh; a third set after that is not paranoia, it is the only way the number means anything.

## Output layout

```
<agent>/routing/
  probes.py             probe set with bucket tags, labels committed before the run
  route.py              the router under test, returning per-probe margins
  score.py              per-bucket table, margin distributions, held-out vs tuned
  results/*.json        per-probe rows
```

Keep the probe set separate from the router so the same set can be replayed against a replacement. Print the per-probe margin in the result JSON, not only the aggregate.

## What to report

Per bucket: n, accuracy, median and **min** margin. Then, separately: false acts (expected decline, acted anyway), over-fires (expected action, declined anyway), and the antonym result if any pair exists. Then name device, runtime and accelerator for every latency number, and state which number is tuned and which is held out. A routing claim without those five is an anecdote.

## Related

- [on-device-verification](on-device-verification/SKILL.md) — the residency line and the device-recording discipline this borrows
- [verification-gates.md](litert-conversion-workflow/references/verification-gates.md) §2 — label agreement and logit-margin correlation for choice-*output* models; that covers a model that outputs a class directly, this skill covers text in, action out
- [`samples/litert/intent_router/`](https://github.com/google-ai-edge/litert-samples/tree/main/samples/litert/intent_router) — a working implementation of the prototype-expansion, decline-class and margin-printing pattern