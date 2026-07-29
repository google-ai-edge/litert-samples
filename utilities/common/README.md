# common/ — canonical shared Kotlin utilities

**Status: proposal.** The files here are extracted from patterns that recur
across the sample apps, but only `RealtimeCameraPipeline.kt` has been in
production use (in a separate model zoo). The other four have not been vendored
into any sample yet, here or elsewhere — the first sample that adopts one is its
compile and device verification. Nothing in this directory is wired into a build
today, so merging it changes no existing sample.

## The problem

The same helpers are re-written per sample. Counted across the sample apps in
this repository plus a ~40-app zoo built on the same APIs:

- CameraX realtime capture pipeline (~250 lines): 22 near-identical copies
- Bitmap → normalized float tensor (mean/std, NCHW/NHWC, letterbox): ~40 copies
- 16 kHz `AudioRecord` capture loop: ~10 copies; mel spectrogram: 4 copies that
  have already drifted apart
- softmax / argmax / NMS / IoU: ~10 copies each
- In this repository, `ModelDownloader.kt` exists 5 times and all five have
  diverged

CompiledModel GPU lifecycle boilerplate is the worst case, because its sharp
edges are not written down anywhere: named-signature `run`, reusing an output
buffer as the next step's input for stateful models, `TensorBuffer` being
`AutoCloseable`. Every sample re-discovers them.

## The proposal: vendored copies, not a shared module

Each sample stays an independent Gradle project. Shared utilities are
distributed as **vendored copies** — the canonical source lives here, every
sample keeps its own physical copy with a `// Vendored from ...` provenance
line, and `utilities/tools/sync_common.py` keeps the copies byte-identical
(only the `package` line differs; the canonical carries the
`package __MODULE_PACKAGE__` placeholder).

```bash
python utilities/tools/sync_common.py --check   # drift check, exit 1 on divergence
python utilities/tools/sync_common.py --apply   # push the canonical out to all copies
```

Copies rather than a Gradle module or a published artifact, because:

- every sample stays standalone — a reader can copy one directory and build it;
- these are teaching material, so the helper code should stay readable inside
  the sample rather than hidden behind a dependency;
- there is zero build coupling between samples today, and this keeps it that way.

If the team later wants a real `com.google.ai.edge:…-utils` artifact, this
directory is the source it would be built from, so graduating is mechanical.

## Rules if this is adopted

1. **Fix bugs in the canonical first**, then `--apply`. Never patch one
   sample's copy in place — that is how the four diverged `MelSpectrogram.kt`
   variants happened.
2. **Model-specific parameters go in constructor arguments**, not edits to the
   copy (normalization mean/std, mel bin count, and so on).
3. Task-specific visualization (overlay views) is deliberately not shared. That
   is the part of a sample readers most want to see whole.

## Files

| File | What it provides | Adopted |
|---|---|---|
| `kotlin/RealtimeCameraPipeline.kt` | CameraX two-thread capture loop, RGBA `ImageProxy` → pooled `Bitmap` including rotation, FPS counter | in ~22 zoo modules; none here yet |
| `kotlin/CompiledModelRunner.kt` | CompiledModel GPU lifecycle — create from assets or file, buffers, run, close-all including `TensorBuffer`s. KDoc records the undocumented behaviour above | not yet |
| `kotlin/ImageTensor.kt` | Bitmap → NCHW/NHWC float tensor; parameterized mean/std, RGB/BGR, 0–1 vs 0–255; stretch and letterbox with a coordinate `Mapping` back to the source | not yet |
| `kotlin/AudioCapture.kt` | 16 kHz mono `AudioRecord` daemon loop delivering normalized float chunks | not yet |
| `kotlin/MathOps.kt` | sigmoid, in-place softmax, argmax, IoU, greedy NMS over flat xyxy arrays | not yet |

Planned next, not included: `MelSpectrogram.kt`, unifying the four diverged
copies behind explicit mel parameters. That one needs per-sample device
re-verification before any switch, so it should not be extracted on paper.

## Known gap

`sync_common.py` was written against a flat zoo layout and its file search has
been re-pointed at this repository's variable-depth sample paths
(`**/src/main/java/**`, `**/src/main/kotlin/**`); the comparison logic is
unchanged. `--check` runs here and correctly picks up all five canonical files,
but with no vendored copies yet it reports `0 copies` for each and exits 0 — so
the drift comparison itself has not been exercised in this repository. The first
sample that adopts a utility is also this tool's first real run.
