---
name: compiled-model-app-scaffolding
description: Build a new Android app (Kotlin, Compose) around a verified LiteRT model using the CompiledModel API - the app architecture, the inference-layer lifecycle rules, model delivery, and the UI traps that masquerade as model bugs. Use when turning a converted and device-verified model into a demo or product app, when an app's inference layer leaks memory or blocks the UI, or when a model that verified clean looks wrong inside an app.
---

# CompiledModel app scaffolding

An app around a verified model is done when three things hold, in this order:

1. the app reproduces the model recipe's verification numbers on the
   accelerator the recipe verified — **before any UI exists**,
2. inference is confined and leak-free: one dispatcher owns the model,
   every buffer is closed, benchmarks include the readback,
3. the inference code is liftable — another app could take the helper
   file unchanged.

Scope: this scaffolds a **new** app from a model recipe. Migrating an
existing TFLite Interpreter app to CompiledModel is a different task with
its own skill. LM models are consumed through the LiteRT-LM Engine rather
than raw CompiledModel — `samples/litert/text_to_speech_lm` is the
reference for that lane; everything below is the non-LM CompiledModel app.

## Step 0: prove parity before building UI

The first milestone has no UI: a bare harness that loads the `.tflite`,
runs the recipe's verification input on the target accelerator, and
reproduces the numbers recorded in the recipe README (correlation,
residency). If they do not reproduce, stop — that is an
`on-device-verification` problem, and no amount of app code fixes it.
Only then build the ViewModel and the screen.

## The shape

One sample = one standalone Gradle project, four layers:

```
app/src/main/java/<pkg>/
  <Task>Helper.kt        inference infra: owns the CompiledModel and its
                         buffers; pre/post-processing; NO Android UI types
  MainViewModel.kt       state machine: drives the helper on a confined
                         dispatcher, exposes UiState
  UiState.kt             one immutable data class
  MainActivity.kt        ComponentActivity + setContent, nothing else
  view/                  Screen.kt, Theme.kt, Color.kt — Compose only
app/src/main/res/values/ strings.xml etc. — no UI strings in Kotlin
```

The worked example of the full shape is
`samples/litert/image_segmentation/kotlin_cpu_gpu/android`. The layer
boundary that matters most is the helper's: pre/post-processing is part of
the model contract (it must match what the recipe exported against), so it
lives with the model, not in the screen. Reusable inference code is worth
more than UI polish — a reader will lift `<Task>Helper.kt` and delete the
rest.

## Inference-layer rules

1. **One confined dispatcher owns the model.**
   `Dispatchers.IO.limitedParallelism(1, "ModelDispatcher")` in the
   helper; create, run, and close the model only inside
   `withContext(singleThreadDispatcher)`. Not a bare executor in the
   Activity, and never the main thread.
2. **Buffers are created once, reused every frame, and closed.**
   `TensorBuffer` is `AutoCloseable`; forgetting the buffers leaks native
   memory even when the model itself is closed. The vendored
   `CompiledModelRunner` (below) gets the whole lifecycle right.
3. **`run()` enqueues; the readback waits.** `run()` may return before
   the GPU finishes — the output read is the synchronization point.
   Benchmark run + readback together, never `run()` alone.
4. **Warm up once at init.** The first GPU inference includes shader
   compilation. Run one inference on dummy input right after create, so
   the first user action is not the compile, and no first-run number
   ever gets quoted as latency.
5. **Ask for the strict accelerator the recipe verified.**
   `Accelerator.GPU` fails compilation on an unsupported op instead of
   silently falling back — that is a feature. Surface the failure as a
   visible error and route it back to the model recipe; do not paper over
   it with a CPU fallback that makes a 10× slowdown look like a working
   app.
6. **Stateful and multi-graph models: move references, not data.** Feed
   step N's output buffers as step N+1's inputs (buffer ping-pong)
   instead of copying state through the host. When one app creates many
   `CompiledModel`s (pipelines, chunked models), create one `Environment`
   and pass it to every `CompiledModel.create` call, and close per-run
   buffers — per-create GPU contexts leak, and a create-per-step loop
   will eventually take the process down (observed at ~20 creates).
   Treat the GPU serialization/program-cache options as untested per
   device — enabling program-cache serialization has aborted a process
   on first compile, and the compiler-cache environment option targets
   NPU JIT, not GPU shader caching.

## Vendor the helpers, don't rewrite them

`utilities/common/kotlin/` holds the canonical copies of the code every
app needs and every app gets subtly wrong when written from scratch:

| File | What it provides |
|---|---|
| `CompiledModelRunner.kt` | the lifecycle in rules 1–3, with the sharp edges documented |
| `ImageTensor.kt` | Bitmap → float tensor; mean/std, NCHW/NHWC, RGB/BGR, letterbox with coordinate mapping back |
| `AudioCapture.kt` | 16 kHz mono `AudioRecord` loop delivering float chunks |
| `RealtimeCameraPipeline.kt` | CameraX capture → pooled Bitmap incl. rotation |
| `MathOps.kt` | softmax, argmax, IoU, NMS |

Vendor a copy with its provenance line, keep model-specific values in
constructor arguments, and fix bugs in the canonical first —
`utilities/tools/sync_common.py --check` is the drift gate. The
preprocessing arguments **are** the model contract: a wrong mean/std or
RGB/BGR looks exactly like a broken model, and it is the first thing to
diff against the recipe's export script when app output is subtly wrong.

## Model delivery

Weights are never committed.

- **Bundleable models**: fetch at build with a `download_model.gradle`
  (Hugging Face URL → `assets/`), and set
  `androidResources { noCompress += "tflite" }` so the asset stays
  mmappable. Worked example:
  `samples/litert/image_segmentation/kotlin_cpu_gpu`.
- **Models too big to bundle**: stage into the app's private `filesDir`
  with an `install_to_device.sh` (`adb push` to `/data/local/tmp`, then
  `run-as <pkg> cp`), and load with the from-file path. Worked example:
  `samples/litert/text_to_speech/kotlin_cpu_gpu/android`.

Link the model recipe (`models/<family>/<model>/`) from the app README;
conversion and verification scripts belong to the recipe, not the app.

## UI traps that masquerade as model bugs

- **The tiny-output trap.** `Image(contentScale = Fit)` with
  `fillMaxWidth()` inside a `verticalScroll` renders the bitmap at native
  pixel size: with no height constraint the row is exactly as tall as the
  bitmap and Fit never upscales, so a 256-px model output looks broken on
  a 1080-px screen. Drop the scroll and give each image `weight(1f)` in
  the Column.
- **Audio I/O belongs to the ViewModel.** Mic capture and `AudioTrack`
  playback run on the confined dispatcher in the ViewModel; the screen
  only requests permission (`rememberLauncherForActivityResult`) and
  renders state. For file input use `OpenDocument` with audio MIME types
  — the photo picker cannot see audio.
- **A "slow model" is often the UI thread.** If the app stutters, check
  that every helper call site goes through the ViewModel's dispatcher
  before profiling the model — a verified model that benchmarked fine in
  step 0 did not get slower by being in an app.

## Output layout

```
samples/litert/<task>/kotlin_cpu_gpu/android/
  app/src/main/java/...   the four layers above
  app/download_model.gradle   or install_to_device.sh at the app root
  README.md               what it demos, which model recipe it consumes,
                          device + accelerator it was verified on
```

State the device the app was verified on, the same way the model recipe
does. An app README that names no device inherits none of the recipe's
verification.
