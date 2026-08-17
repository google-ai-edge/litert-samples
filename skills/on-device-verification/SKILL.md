---
name: on-device-verification
description: Prove a converted or quantized LiteRT model on the actual device via the CompiledModel API - confirm GPU residency, compare device output against the source model, and diagnose device-only failures such as silent CPU fallback, whole-graph compile ceilings, and fp16 range breaks. Use after conversion or quantization, when device output is wrong or NaN, when a clean graph fails to compile only on device, or when GPU and CPU outputs are suspiciously identical.
---

# On-device verification

A device result is done when three things hold:

1. the model compiles and runs on the accelerator you claim it runs on,
2. the output matches the source model numerically **and** on a task-level
   gate,
3. the record names the device, the runtime version, and the residency
   line. A number without those is not reproducible and not a result.

Host-side checks (the CompiledModel checker used in `gpu-clean-conversion`)
exercise the host GPU. The device has its own shader compiler, its own
precision behavior, and its own memory ceiling — every failure mode in the
table below was hit by a model that had already passed on the host.

## Loop

**1. Dump references from the source model, once.** Fixed inputs — one
real sample plus fixed-seed random — saved as `.npy` next to the recipe
(`dump_*_ref.py`). These are ground truth for every later step; regenerate
them only when the source model changes.

**2. Run the same inputs on the device: CPU first, then GPU.** One
argument switches the accelerator:

```python
from ai_edge_litert.compiled_model import CompiledModel
from ai_edge_litert.hardware_accelerator import HardwareAccelerator

model = CompiledModel.from_file(
    "model.tflite", hardware_accel=HardwareAccelerator.GPU)  # or .CPU
```

The device-CPU run is the control. If it already diverges from the source
dump, the problem is the conversion, not the GPU — go back to
`gpu-clean-conversion`. A full worked example of the A/B lives in this
repo at `samples/litert/speech_recognition/convert/verify_tflite.py`.

Ask for the strict accelerator. Compiling with
`HardwareAccelerator.CPU | HardwareAccelerator.GPU` permits partial
delegation and hides fallback; use the combined mode only to discover
*which* ops fell back after a strict GPU compile fails.

**3. Read the delegate log before reading any numbers.**

```
Replacing N out of M node(s) with delegate ... X partitions
```

Record `N/M` and the partition count. `N < M` or `X > 1` means part of the
graph runs on the CPU — decide whether that is acceptable before quoting
any accuracy or latency number.

**4. Gate the output three ways.**

- **Numeric**: correlation and max abs diff against the source dump.
  Expect genuine GPU fp16 to drift in the last digits — exact equality is
  a symptom, not a pass (see below).
- **Task**: argmax match, IoU, token-for-token greedy decode — whatever
  the model is for.
- **Artifact**, for generative models: the decoded text / audio / image.
  An artifact gate catches argmax ties that numeric tolerance misses.

**5. Record it**: device, runtime version, `N/M` + partitions,
correlation, max abs diff, task result — one row per device in the recipe
README.

## The silent CPU fallback

The most common false positive: everything runs and the numbers match
perfectly. If the GPU output is **bit-identical to the device CPU fp32
output, it almost certainly did not run on the GPU.** Perfect equality is
the tell, not the goal. Cross-check the residency line; only when `N/M`
is full *and* the outputs drift in the last digits are you looking at a
real GPU run.

## Device-only failures

| What you see | What it is, what to do |
|---|---|
| GPU compile fails on device for a graph that is op-clean and passed the host check | A whole-graph compile ceiling, not a bad op — a fused graph can fail where each half compiles. Split at a natural block boundary (conv frontend / transformer encoder), verify split == monolith bit-exact on the host, ship the halves |
| The file refuses to load at all | The >2 GB flatbuffer limit. Split, or quantize below it |
| Full residency, wrong numbers | Bisect by intermediates: re-export with block-boundary tensors as extra outputs and find the first one that diverges. The cause is usually a reduction (mean, variance, Σx²) in fp16, not the op you suspect. If materializing a tap *fixes* the numbers, that localizes the bug to a fusion boundary — record it as a finding, not a nuisance |
| Wrong-but-plausible output, and every hypothesis costs a slow re-export | Micro-probe instead: export ~1 KB graphs — the suspect subexpression and each candidate fix — and run them through the device harness you already have. Minutes per hypothesis instead of a re-export per hypothesis |
| NaN or garbage only on the GPU, in a model with large-magnitude residuals or deep modulation paths | An fp16 range break. Confirm the attribution by forcing fp32 on the delegate where the API exposes it (~2× memory, slower); then fix it properly with the fp16-safe rewrites in `gpu-clean-conversion`, or keep the offending block on CPU and run the rest on GPU |
| The process dies while loading or running a large float model | A memory ceiling, not a model bug. Weights + activations + delegate buffers must fit in available memory. Do not run a multi-GiB fp32 build in-process on a phone "just to check" — quantize or split first, then verify the smaller thing |
| A deep transformer shipped as an fp16 graph is bit-exact on desktop and noise on the device CPU | Android ARM XNNPACK computes **native fp16**; desktop XNNPACK upcasts to fp32. Deep residual streams compound the difference to collapse. Ship fp32 graphs for CPU inference on device; fp16 is a GPU-side format |
| Attention quality collapses only for a small-head submodule | head_dim is the fp16-fragile axis, not token count: the same graph at head_dim 64 held corr 0.998 where head_dim 16 fell to 0.86. Pad heads to ≥32 or keep the small-head module on CPU |
| A `[1,N,C]` token tensor that is a graph output **and** feeds other consumers comes back corrupted | 3-D fan-out corruption — the later branch is clobbered, it cascades, and it reads exactly like an fp16 wall downstream (4-D NCHW maps with the same fan-out are fine). Keep token tensors as sole leaf outputs (or keep them 4-D) and push per-token heads to the host — exact, since per-token ops commute with the gather |
| A recurrent/streaming graph gives correct output on call 1 and drifts on repeated calls | Fused-LSTM-style **variable tensors persist across `invoke()`** on a reused interpreter — and a fresh-interpreter-per-call verify script structurally cannot see it. Call `reset_all_variables()` before every invoke (cost ≈ 0). Related: the CompiledModel loader rejects variable tensors outright, so such graphs are Interpreter-only |

## Watch for

- **Two references, two verdicts.** The source-framework dump is the
  truth; the device CPU run is the control that isolates the GPU. A
  GPU-vs-CPU comparison alone can pass while both are wrong.
- **The fp32-forcing knob is for attribution, not shipping.** It tells
  you precision is the cause; the fix is a rewrite or hybrid placement.
- **First inference includes shader compilation.** Correctness on the
  first run is fine; never quote first-run latency, and never let a
  latency number travel without its residency line.
- **One device proves correctness, not portability.** GPU compilers
  differ per vendor — a compile ceiling on one chip may not exist on
  another. "Runs on Android GPUs" means a device matrix (your own
  devices, or a farm service such as AI Edge Portal), recorded as one
  row each.
- **One runtime proves it for that runtime.** A delegate rejection or
  miscompute is a fact about the runtime version you measured: ops have
  been *dropped* between minor versions, a miscompute's victim output
  has *moved* between versions, and mixing accelerator and core
  libraries across versions silently falls back to CPU. When a wall
  appears after an upgrade, bisect the runtime pin on the real graph —
  micro-probes have repeatedly failed to reproduce walls that only fire
  in full-graph context, so a negative micro-probe is not a refutation.
- **Localize miscomputes with single-output graphs, never fan-out
  taps.** A multi-output tapped probe is itself exposed to
  output-aliasing bugs and has produced a confidently wrong culprit;
  in single-output form every op was exact and the *assembly* was the
  bug.
- **Sweep the delegate options to classify a miscompute.** Run the same
  graph across precision, buffer-storage, and backend options: a
  bit-identical wrong result across all of them places the bug in the
  shared graph-compilation layer and rules out precision/storage in one
  pass. And know what the precision flag can do: forcing fp32 rescues
  overflow→NaN cases only — it does **not** fix precision compounding
  (the delegate still reduces in fp16), so "fp32 didn't help" does not
  exonerate fp16.
- **Time the enqueue and the readback as separate counters.** `run()`
  is asynchronous; timing it alone has reported a 4× GPU win that did
  not exist. A large readback time is usually the deferred compute, not
  the transfer. Corollary economics: per-call overhead makes small
  per-step graphs (KV-cache decoders re-uploading state every token) a
  net GPU loss — estimate `calls × per-call overhead` against the CPU
  time before re-exporting for GPU; the crossover sits around
  hundreds of nodes per call.
- **The desktop build is the CPU reference, not a GPU sieve** — desktop
  Python runtimes exercise CPU/XNNPACK only, which is exactly what makes
  them the right numerical reference. For the device loop, a minimal
  push-run-pull binary (tflite in, output tensor out) iterates in
  seconds without an app rebuild.

## Output layout

Verification is part of the model recipe, not a side script:

```
models/<family>/<model>/converted/
  dump_*_ref.py            source-model reference dumps (.npy)
  verify_*.py              parity vs those references, accelerator as a flag
  README.md                per-device table: device | accelerator |
                           N/M nodes, partitions | corr | max abs | task gate
```

Keep the dump and the verify separately runnable: references are dumped
once on the host, verification re-runs on every device and after every
model change.
