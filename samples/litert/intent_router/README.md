# Intent routing with a sentence embedder

Routes a free-text request to one of several actions — or declines — using a
sentence embedder through the LiteRT CompiledModel API. This is the layer *before*
a language model: the decision of which action applies, made without generating a
token.

It is the companion to [`semantic_similarity`](../semantic_similarity/), which
computes cosine similarity between two sentences. That sample answers "how similar
are these?"; this one answers "which of these, and should I answer at all?".

## Why not just use the language model

Measured on a Samsung SM-A145F (Exynos 3830, 3.5 GB RAM, Android 15), routing a
paired prompt set against a 270M function-calling `.litertlm` bundle:

| | 270M `.litertlm` (CPU) | 22M sentence embedder (CPU) |
|---|---|---|
| Routing accuracy, 4 actions | **22%** | **100%** |
| Refusal rate | 72% | 0% |
| Median latency per prompt | 3120 ms | **594 ms** |
| Peak RSS | **983 MB** | **285 MB** |

Peak RSS was `VmHWM` from `/proc/self/status`, not a heap delta — a before/after
`Runtime.totalMemory()` delta is not a memory measurement, because the collector
can run between the two samples. On a 3.5 GB device, 983 MB for a 289 MB bundle is
28% of RAM.

The generator was not broken, and this is not an argument against language models
here. It was failing at *routing*: it matches prompt tokens against action
descriptions, so `"do i have meetings today"` shares almost nothing with "Lists the
events on the user's calendar for today" and it refuses. Spending 3 seconds and a
gigabyte on the routing decision is the wrong trade when a 22M-parameter embedder
does it in half a second.

## Build

```bash
bazel build -c opt //samples/litert/intent_router
```

## Run

```bash
./bazel-bin/samples/litert/intent_router/intent_router \
  --tokenizer=/path/to/tokenizer.model \
  --embedder=/path/to/embedder.tflite \
  --sequence_length=128
```

`--embedder` must be a single-signature sentence embedder taking token ids and
producing one pooled vector. The build and push path for a phone is the same shape
as [`semantic_similarity`'s deploy script](../semantic_similarity/build_from_source/deploy_and_run_android.sh).

Route your own text instead of the built-in evaluation set:

```bash
... --utterances="turn on the flashlight;am i busy today;what is the weather"
```

## The three things worth copying

**1. Prototypes, not centroids.** Each action carries several prototypes and is
scored by max cosine over them. One embedded description cannot represent that
users phrase a request many ways; on a paired prompt set this was the largest
single accuracy lever.

**2. `no_action` is a candidate, not a threshold on the side.** A plain argmax
over actions always invents an answer for out-of-domain input — cosine similarity
has no notion of "none of these". Letting the decline class compete in the same
comparison is what makes `"what is the weather"` able to lose to nothing.

**3. The margin is the output that matters.** Each line prints the gap between the
winner and the best alternative. An agent's real decision is act-or-ask, and that
is what the margin is for. Calibrate `--abstain_margin` on real traces: the
default of 0.02 was measured on one device with one action set and is a starting
point, not a constant.

## What this cannot do: antonyms

Included in the action set on purpose, because the limitation should be
reproducible rather than theoretical. Cosine similarity between opposing
phrasings, same encoder:

```
0.949  'switch the torch on'  /  'switch the torch off'
0.911  'flashlight on please' /  'flashlight off please'
0.859  'turn on the light'    /  'turn off the light'
```

A one-token difference lands at 0.949. Sentence embeddings encode topic, not
polarity, so `open_flashlight` and `close_flashlight` are nearly the same point in
embedding space, and adding prototypes does not separate them.

If your action set contains an on/off pair, or play/pause, or show/hide, resolve
the polarity with a lexical check on the polarity token and use the embedding only
to select the action class. A small abstain margin will not fix this: it will
decline both members of the pair, which is safer but not useful.

## Two more failure modes to check for

**Negated requests invert.** A router asked "do not switch the torch on" will
plausibly route to `close_flashlight` — the opposite of the instruction. On a
measured set, 4 of 5 negated utterances executed the inverse. Handle prohibition
with a lexical check *ahead of* the model; no accuracy number captures this one,
because the failure is not "wrong action" but "opposite action".

**Compound requests have no single answer.** "Turn on the light and check my
calendar" cannot be routed to one action. Detect the conjunction and split or ask;
do not let the argmax silently drop half the request.

## Scoring more honestly than the built-in set does

The built-in probes are a smoke test, and the binary says so: a high score means
the pairing is too easy. Before trusting a router, add

- a paraphrase of each intent that avoids the action's own vocabulary, since
  literal phrasings route nearly perfectly and hide a keyword matcher;
- out-of-domain utterances, to check the decline path;
- negated utterances, to check for inversion;
- an antonym pair, to confirm the limitation above.

Score only over probes whose expected action was actually offered, so a narrow
action set is not credited for prompts it had no way to answer.

## Cold start scales with the prototype count

Prototype embedding happens once at startup. A single embedding pass took ~600 ms
on the device referenced above, so a large prototype bank dominates cold start
and startup latency grows linearly in it. Embed the actions needed on the critical
path first and defer the rest, or keep the bank small.

## References

- [`semantic_similarity`](../semantic_similarity/) — pairwise cosine similarity, the primitive used here
- [`on-device-verification`](../../../skills/on-device-verification/SKILL.md) — the recording discipline for device numbers
- Field report with the full measurement record: [litert-samples#382](https://github.com/google-ai-edge/litert-samples/issues/382)