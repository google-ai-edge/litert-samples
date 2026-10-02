# SAM 2 video: one C++ Tensor API pipeline, native and in the browser

The whole SAM 2.1 Hiera-Tiny video pipeline is written in C++ with the LiteRT
Tensor API and wired together with LiteRT's `ModelChain`
(`tensor/runners/model_chain.{h,cc}`). That covers frame preprocessing, the image
encoder, the per-object prompt and track steps with the memory bank, and the
mask composite shown on screen. The same C++ runs natively (CPU or Metal) and in
the browser as WebAssembly, where every stage executes on WebGPU.

```
                 ┌──────────── encoder chain ────────────┐
RGBA frame ──▶ preprocess ──pixels──▶ encode ──▶ pix_raw, feat_s1, feat_s0 (2-frame cache)
  [1,H,W,4]      (Tensor API)          (SAM 2)                 │
                                                               ▼
                 ┌──────────────── step chain (one run per frame) ─────────────────┐
                 │ prompt{k}   (click frame: k = 1..8 clicks, +/-)          ─┐      │
                 │ track{2|7}  × each object: memory slots mem_0.. ptr_0..  ─┼─low_mask─▶ composite ──▶ display RGB
                 │   (slots bound to buffers earlier steps wrote: no copies) ─┘      │   (fill / outline /
                 └─────────────────────────────────────────────────────────────────┘    spotlight / cutout)
```

## Layout

| Path | What |
|---|---|
| `cc/chain_graphs.{h,cc}` | Tensor API graphs. **SAM 2 model:** `encode`, `prompt1..8`, `track2`/`track7`. The track steps take every memory slot and object pointer as a separate input and concatenate the bank in-graph. **Frame model** (per video size): `preprocess` (crop → anti-alias average pool → bilinear → ImageNet norm) and `composite` (bilinear upsample → signed distance → fill + outline, or union matte for spotlight / green cutout). |
| `cc/host_plan.{h,cc}` | Plain C++ host logic: click encoding, pointer temporal encodings, and which stored frames fill which memory slot. |
| `cc/signature_stage.{h,cc}` | A `ModelStage` that runs one signature of a **shared** compiled model. The stock `CompiledModelStage` owns its model, which would compile the 156 MB model once per stage. |
| `cc/sam2_pipeline.{h,cc}` | `Sam2Pipeline`: the ModelChains, the memory bank as buffer bindings, result retention. |
| `cc/sam2_chain_main.cc` | Native end-to-end driver: authors the model from weights, runs a clip, dumps results. |
| `cc/host_plan_test.cc` | gtest: encodings vs Hugging Face goldens, memory-plan semantics (7 tests). |
| `wasm/litert_js_runtime.cc` | A LiteRT runtime for the browser. It fills the `LiteRtRuntimeCApiStruct` table the LiteRT C++ API calls. Model introspection parses the `.tflite` in C++; compile, run and buffer I/O go to **LiteRT.js on WebGPU**. Unused entries fail loudly by name. |
| `wasm/litert_js_bridge.js` | The JS half: `loadAndCompile`, `run`, WebGPU buffers, readback (async via JSPI). Input `core.Tensor` wrappers are cached per buffer and shape, and all per-object score/mask readbacks of a step are batched into one suspension (`lrtjs_read_many`). |
| `wasm/sam2_wasm.cc` | The embind API (`Sam2Chain`). |
| `wasm/CMakeLists.txt`, `build.sh` | emscripten build from the **same** LiteRT snapshot, absl and flatbuffers as the native Bazel build. |
| `app/` | The web demo (same UI as the earlier LiteRT.js demo); `src/chain/runtime.ts` loads and drives the wasm, `src/display.ts` blits the composite. |
| `tools/verify_chain.py` | Verifier for native and browser dumps. |
| `tools/native_e2e.sh`, `verify_all.sh` | Native end-to-end matrix (CPU and Metal); one-command verification of all layers. |
| `tools/build_models.sh`, `export_weights.py` | Setup: weights and models. |
| `docs/` | Implementation report for LiteRT reviewers (HTML, PDF, Markdown). |

## Why the web half is a "hybrid"

Open-source LiteRT can't compile its WebGPU runtime to wasm. The `ml_drift` GPU
accelerator source is private, only native WebGPU accelerator binaries are
published, and the Tensor API's own WebGPU backend and runner
(`tensor/backends/webgpu`, `tensor/runners/webgpu`) are internal-only. So all
the C++ (Tensor API graph authoring and serialization, `ModelChain`, the
pipeline, the LiteRT C++ API) is compiled to wasm unchanged. The one thing
swapped is the runtime underneath the LiteRT C API: here it executes on WebGPU
through LiteRT.js.

- `model_chain.cc` and `litert/cc` compile unmodified.
- One upstream file is patched at build time: `tflite_flatbuffer_conversion.cc`.
  Its eager `Run()` helper embeds a TFLite interpreter, so it is stubbed;
  `ModelFactory` serialization is untouched.
- Tensors are WebGPU buffers end to end. Frames go into the pipeline by GPU copy
  (`copyExternalImageToTexture` → `copyTextureToBuffer`), and the composite
  goes to the canvas with a blit. There is no CPU pixel work.
- With `?build=browser`, the wasm module also **authors the SAM 2 model in the
  page** from the safetensors weights (0.2 s), instead of downloading a
  `.tflite` built natively by the same code.

## Graph optimizations

All of these are build-time rewrites in the Tensor API graph code: the SAM 2
builders in litert-samples (`models/sam2/sam2_hiera_tiny_video/tensor_api/`,
merged upstream in PRs #356–#358) and `cc/chain_graphs.cc`. Apart from the
step-graph threshold (equal to `x > 0` except within ~1e-4 of 0), none of them
change the math. Every rewrite is checked against HF with
`tools/verify_chain.py`.

- **Attention scale folded into weights.** $1/\sqrt{d_k}$ is baked into the Q rows
  of the Hiera `qkv` projections, into the decoder `q_proj`, and into the
  memory-attention RoPE query tables. This removes every score-matrix `Mul`.
- **RoPE without `Neg`.** The rotate-half sign is baked into the `sin` table, so
  rotation is `x·cos + swap_halves(x)·sin`.
- **Leaner Hiera windows.** Window partition stays 4-D (`[nH, nW·ws, ws, C]`),
  which removes the reshapes around `FullyConnected` and `MaxPool2D`. When the
  map needs padding, `qkv` runs before padding without its bias. The bias is
  added after padding, so padded tokens still carry `qkv = bias` as in Hiera,
  where the block input is zero-padded before `qkv`. `proj` runs after
  un-padding. Single-head blocks skip the head transposes.
- **Step graphs are 100 % GPU-delegable.** Thresholds and gates (`Greater`,
  `Cast(bool)`, `ReduceMax`) are replaced by `1 - Relu(1 - s·Relu(x))`. Mask
  selection uses rank-4 `BatchMatMul`. So `prompt*` and `track*` contain no
  BOOL tensors and never fall back to CPU mid-signature.
- **Mode-specialized steps.** Click steps skip the unused mask-0 hypernetwork
  and only emit the binarized mask. Track steps project only candidate 0's
  object pointer and only emit the sigmoid mask. Upsampling uses one
  `ResizeBilinear` instead of two constant interpolation matrices.
- **Fewer ops around the memory bank.** Slots and pointers are concatenated
  before a single reshape. The composite hoists per-frame terms out of the
  per-object loop.

| Signature (384 and 512) | Before | After |
|---|---|---|
| `encode` | 594 ops | 521 ops |
| `prompt1` | 438 ops | 420 ops |
| `track2` / `track7` | 804 / 809 ops | 733 / 733 ops |
| `Greater` / `Cast(bool)` / `ReduceMax` in step signatures | yes | none (0 BOOL tensors) |
| `sam2_chain_384.tflite` / `_512` | 166 / 173 MB | 156 / 162 MB |

`tools/inspect_model.py model.tflite` prints these op histograms.

## Verification (football sample, 3 objects; M4 Pro, `tools/verify_all.sh 384`)

Test prompts: object 0 has 2 clicks at frame 0, object 1 has 1 click at frame 0,
and object 2 has 3 clicks (including a negative) and joins at frame 6.

| Run | Masks vs HF `Sam2VideoModel` on the same preprocessed frames (mean IoU per object) | Preprocess graph vs numpy | Composite vs numpy |
|---|---|---|---|
| Native CPU fp32, 7-frame, overlay | 1.0000 / 1.0000 / 1.0000 (0 px differ) | 7e-5 | 1e-5 |
| Native Metal fp16, 2-frame, cutout | 0.996 / 0.999 / 0.997 | 4e-3 | 3e-4 |
| **Browser WebGPU fp16 (wasm), 7-frame, overlay** | 0.998 / 1.000 / 0.994 | 3e-3 | 6e-4 |
| **Browser, SAM 2 model authored in the page (0.2 s), 2-frame, cutout** | 0.999 / 1.000 / 0.994 | 3e-3 | 2e-4 |
| **Browser WebGPU fp32 (wasm), 7-frame, overlay** | 1.0000 / 1.0000 / 1.0000 (0 px differ) | 7e-5 | 1e-5 |

384 px, 10 frames each. At 1024 px (native CPU fp32, 7-frame, 24 frames) mean
IoU is 0.9996 / 1.0000 / 0.9999, at most 7 pixels different on any frame.

Every native and browser run also passes a per-frame criterion: IoU ≥ 0.95
(fp32) or ≥ 0.85 (fp16), or at most 3 pixels different.

The few imperfect frames come from **SAM 2's own hard decisions**, not the
pipeline:

- the best of 3 candidate masks by predicted IoU, where two candidates tied
  within 0.0009;
- for 2+ clicks, token 0 is used unless its stability is below 0.98. The
  whole-player prompt used first had stability 0.9817, and WebGPU fp16 landed on
  the other side of the threshold. That prompt is noted as a knife-edge; the
  test uses a prompt with a 0.011 margin.

The UI check (`npm run e2e:ui`) covers:

- the sample opens by default;
- positive and negative clicks and Reset;
- a 192-frame, two-object track with no identity switches (occlusion-aware);
- every effect;
- no page scroll at 1440, 1280 and 390 px widths;
- the camera.

## Live camera (M4 Pro, 48 GB, Chrome, WebGPU fp16)

1280×720 camera at 30 fps (33.3 ms per frame), one object, 2-frame memory.
Camera → screen = Delivery + Wait + Pipeline (see Notes).

| Input | Display | Rate | Camera → screen | Delivery | Wait | Pipeline |
|---|---|---|---|---|---|---|
| 384 | Smooth | video 30 fps, masks 30.1 fps (lag 37 ms) | 3 ms* | 3 ms | 0 ms | 27 ms |
| 384 | Aligned | 30.4 fps | 31 ms | 8 ms | 0 ms | 23 ms |
| 512 | Smooth | video 26 fps, masks 26.0 fps (lag 68 ms) | 36 ms* | 5 ms | 24 ms | 38 ms |
| 512 | Aligned | 29.1 fps | 62 ms | 5 ms | 23 ms | 34 ms |

\* Smooth: camera frame to the display composite being submitted; the masks
trail by the lag shown.

Where a frame goes (`?profile=gpu`: GPU time per stage, with a GPU sync after
each, so the total is a little longer than the pipelined step; medians of 60
frames, ms):

| Input | upload | preprocess | encode | track2 | composite | C++ / JSPI | readback | total |
|---|---|---|---|---|---|---|---|---|
| 384 | 1.0 | 0.7 | 12.2 | 7.7 | 3.5 | 0.5 | 0.6 | 26.5 |
| 512 | 0.7 | 0.6 | 20.0 | 12.0 | 3.7 | 0.3 | 0.6 | 37.7 |
| 1024 | 0.6 | 1.1 | 91.0 | 81.7 | 3.7 | 0.3 | 0.5 | 179.1 |

CPU dispatch is small (512, `?profile=cpu`: encode 0.2 ms, track2 0.5 ms, composite
0.1 ms), so the live rate is bound by GPU time. At 384 the step fits in the
camera interval and Wait is 0. At 512 it is just over 33 ms, so each frame
waits about one camera interval. Keeping a second frame in flight doesn't
help: Wait moves into Pipeline. Composite, upload and preprocess scale with
the camera size (`?cam=WxH`).

## One-command verification

```bash
tools/verify_all.sh 384      # ~2.5 min; also: 512
```

| Layer | Step | Checks |
|---|---|---|
| **A. Tensor API pipeline (native C++)** | A1 unit tests + build | host plan vs HF goldens (gtest) |
| | A2 CPU fp32, 7-frame, overlay | preprocess vs numpy · masks vs HF · composite vs numpy |
| | A3 Metal fp16, 2-frame, cutout | same |
| **B. wasm target** | B1 emscripten build | same LiteRT / Tensor API / ModelChain sources, links clean |
| | B2 WebGPU fp16, 7-frame, overlay | same checks, in Chrome |
| | B3 model authored in the page, 2-frame, cutout | same checks, SAM 2 model built by the wasm Tensor API |
| | B4 WebGPU **fp32**, 7-frame | same checks: exactness of the WebGPU path apart from fp16 |
| **C. Web demo on WebGPU** | C1 UI | clicks ± / Reset, 192-frame 2-object tracking, effects, layout, camera smooth + aligned |
| | C2 WebGPU health | hardware adapter (not a fallback) · every signature of both models on WebGPU · zero WebGPU errors, no device loss · GPU buffers bounded across tracking, re-tracking, playback, Reset and camera · steady per-frame time |

Every pipeline run (A2–B3) uses 10 frames and 5 objects (all slots). All runs compare
against **one cached Hugging Face reference** per (size, memory, clip,
prompts), computed from the numpy reference of the preprocess graph, so native
and browser results face the same ground truth and HF runs only once.

Last run (384, M4 Pro): every step except the wasm rebuild (B1) passes in about
2 min; B2–B4 ran on the checked-in wasm, which is built from the same sources.

WebGPU results: Apple GPU (Metal-3) through Chrome's WebGPU, both models fully
accelerated. At **fp32 on WebGPU the masks match HF exactly** (0 pixels differ
on every frame, all objects), so the fp16 differences below are rounding, not
the WebGPU path. Zero WebGPU errors over a full session. Two objects over 192
frames: steady per-frame time, the same in the last quarter as the first. GPU
buffers: flat across repeated tracking (1,284), freed by Reset (to 778), flat in
camera mode (143).

## Setup from a fresh clone

Requirements: macOS or Linux, Bazel 7 (bazelisk), Node 20+, Python 3.11+ with
`torch transformers safetensors numpy pillow ai-edge-litert`, `ffmpeg`, CMake
(Ninja optional; `wasm/build.sh` falls back to Makefiles), and Chrome 137+
(WebGPU and JSPI). Metal runs need macOS.

The native code builds in this repository's Bazel workspace, which provides the
LiteRT sources and the SAM 2 Tensor API network builders
(`models/sam2/sam2_hiera_tiny_video/tensor_api`); `cc/` is the Bazel package
`//samples/web_demos/src/sam2/cc`. Run the commands below from this directory
(`samples/web_demos/src/sam2`). Defaults, overridable with environment variables:

| Variable | Default | What |
|---|---|---|
| `LITERT_SAMPLES` | the repository root | the litert-samples workspace |
| `ARTIFACTS` | `./artifacts` | weights, test clips, run outputs |
| `PYTHON` | `./.venv/bin/python` | Python with the packages above |

This sample is self-contained: it has its own `app/package.json` and Vite
config and is not part of the `web_demos` site build (`dist/`), because its
models are built locally (below).

```bash
# 1. Weights (from facebook/sam2.1-hiera-tiny) and the SAM 2 models, authored
#    with the Tensor API -> app/public/models/   (add 1024 for the 1024 px model)
tools/build_models.sh 384 512

# 2. emscripten, then the wasm build -> app/public/wasm/sam2_chain.{mjs,wasm}
git clone https://github.com/emscripten-core/emsdk.git third_party/emsdk
third_party/emsdk/emsdk install latest && third_party/emsdk/emsdk activate latest
wasm/build.sh

# 3. The app
cd app && npm install && npm run dev      # http://localhost:5175  (?size=512|1024, ?nmm=7, ?build=browser, ?sample=flowers)
```

The page keeps downloaded models in the browser's Cache API. Later loads only
send a HEAD request and reuse the cached copy while the server's ETag,
Last-Modified and size are unchanged, so a rebuilt model is fetched again.
`?cache=0` always downloads; `__clearModelCache()` in the console empties the
cache.

Not checked in (see `.gitignore`): the model and weight files (156–198 MB each,
over GitHub's 100 MB limit; rebuild with step 2 or publish them with Git LFS or
as a release asset), emsdk, `node_modules`, the LiteRT.js runtime copy
(`npm install` restores it) and test outputs. The prebuilt wasm
(`app/public/wasm`, 1.7 MB) and the 75 KB host tables are checked in.

## Tests

```bash
tools/verify_all.sh 384            # everything below, ~2.5 min
tools/native_e2e.sh 384            # native CPU / Metal matrix on 24 frames
cd app
npm run e2e                        # wasm on WebGPU vs HF  (-- --nmm=2 --effect=cutout --build=browser)
npm run e2e:ui                     # the demo UI, incl. camera (fake camera)
npm run e2e:webgpu                 # WebGPU health: adapter, acceleration, errors, GPU memory, speed
npm run e2e:gemma                  # Ask Gemma against a local server (tools/gemma_server.sh)
```

## Notes

Camera: **smooth** (default) composites every camera frame with the newest
masks through the display chain, so the video runs at camera rate. **Aligned**
shows each processed frame with its own masks, at the pipeline's rate. The model
size can't be switched while the camera runs.

Prompts: positive / negative clicks and **boxes** (the Box tool: drag around the
object), in both file and camera mode. As in SAM 2 video, a box is encoded as two
corner points (labels 2 and 3) placed before any clicks, so it runs through the
same `prompt{k}` signatures; clicks can refine a box, and a new box replaces the
old one. Up to 8 points per object, a box counting as 2. Box encoding is checked
against HF's prompt encoder (C++ unit test) and end to end (test object 2 is a
box + negative click).

### Ask Gemma: text → boxes → masks

Type or say what to select ("the ball", "all players") in the Objects panel and
press **Find**. **Gemma 4 (E4B or E2B)** looks at the frame on screen
and returns bounding boxes. Each Find replaces the selection: all objects are
reset and each box (up to 5) becomes a fresh object with a SAM 2 box prompt,
which you can refine with clicks and track as usual. If Gemma finds nothing,
your objects are kept. Works on video files and the camera.

Gemma runs either in the page (see "In the browser" below) or on the same
machine in `tools/gemma_server.py`, a small OpenAI-compatible server on the
LiteRT-LM Python API (LiteRT-LM's web Gemma 4 builds are text-only today; the
full `.litertlm` models include the vision encoder). On the server, language
model **and** vision encoder run on the GPU (ML Drift's WebGPU delegate through
Dawn, on Metal; all 1,477 vision-encoder ops delegated). SAM 2 runs on WebGPU
in Chrome.

```bash
tools/gemma_server.sh          # installs LiteRT-LM, imports Gemma 4 E4B + E2B, serves on :9379
```

**In the browser, no server:** choose **E4B web** or **E2B web**
in the model menu (the default when WebGPU is available). Gemma 4 then runs in
the page on WebGPU through the MediaPipe LLM Inference task, vision encoder
included (`app/src/gemma_web.ts`). The models are the hand-written GPU
`.litertlm` builds (E4B 3.2 GB, E2B 2.2 GB), downloaded on the first Find and
kept in the browser's Cache API. The published `@mediapipe/tasks-genai` loads
only the text decoder of these files, so `app/public/genai/` has a MediaPipe
GenAI build that also loads the vision encoder and adapter sections; its
33.7 MB `.wasm` comes from the same public bucket as the models.

**Voice:** the 🎤 button in the text box uses Chrome's built-in speech
recognition (Web Speech API; `app/src/speech.ts`). Recognition runs on the
device when Chrome has, or can install, the language pack (`processLocally`,
Chrome 139+). Otherwise it uses Chrome's server recognition, and the hint says
so. The phrase fills the box and runs it.

**Tool calling (browser models):** with a browser model the request is a plan,
not only a query. Gemma 4 gets a short list of the app's own actions as tools
and answers with calls, one per line
(`<tool_call>{"name": "set_effect", "arguments": {"effect": "cutout"}}</tool_call>`).
Each call runs as soon as it is complete in the stream, in order, and shows up
as a chip under the box (`find_objects(what: "the players") ✓ 2.1 s`). So
*"find all the players and cut them out"* is `find_objects` then `set_effect`,
with the first mask on screen while Gemma is still writing the second call.

| Tool | Does |
|---|---|
| `find_objects(what, max?)` | The Find above: boxes from the frame on screen, each a SAM 2 box prompt, streamed |
| `set_effect(effect, outline?)` | overlay / spotlight / cutout |
| `remove_objects(labels? \| keep? \| all?)` | by Gemma's labels ("keep only the ball") or everything |
| `playback(action)` | track / play / pause / restart / stop |
| `use_camera(on)` | live webcam on or off |
| `set_quality(size?, memory?)` | 384 / 512 / 1024 px model, 2- or 7-frame memory |
| `describe_scene()` | what is tracked and the settings; Gemma answers in words |
| `measure()` | measured ms per frame, fps, camera → screen; Gemma answers in words |

`app/src/toolcalls.ts` holds the declarations, the prompt and a tolerant
streaming parser; `app/src/agent.ts` runs a turn on any engine (the planning
turn is text-only, so it costs no image tokens; `find_objects` reuses the
vision call); `app/src/tools.ts` validates arguments and maps them to the
app's actions. Escape cancels a turn. `?agent=0` restores plain Find, which
the LiteRT-LM server models always use. The same declarations fit
LiteRT-LM.js's `AutoToolChat` (native tool calling with constrained decoding)
once its web Gemma 4 builds take images. Unit tests: `npm test`
(`test/agent.test.ts`).

The page sends the frame on screen (any frame, or the camera) as a JPEG with
Gemma's native detection prompt (`Detect … Output a json list … "box_2d" …
"label"`) and parses `box_2d = [ymin, xmin, ymax, xmax]` normalized to 0–1000
(`app/src/gemma.ts`; `?llm=http://host:port` points it elsewhere). The reply is
**streamed**: each box becomes an object and gets its SAM 2 mask as soon as
Gemma has written it, while the rest are still coming.

Where the time goes (E4B): ~0.9 s for the image (vision encoder + ~310
prompt tokens; the frame is always resized to 1056×576, so a smaller JPEG does
not help) and then decoding at ~30 tokens/s, ~30 tokens per box. The server
avoids two costs of `litert-lm serve` (which also works, with
`--config tools/litert_lm_config.json`): serve sets up constrained decoding on
every request (~0.8 s, and slower decoding) and puts the vision encoder on the
CPU by default. It keeps one model in memory at a time: two Gemma engines next
to a browser push a 16 GB machine into swap, and decoding slows 2–3×.
Measured in the demo on the sample (M3 Air, 16 GB):

| | E4B (default) | E2B |
|---|---|---|
| "the soccer ball" → SAM 2 mask | whole ball (0.92% of the frame) | box offset right, part of the ball |
| "all players" | 4 found | 3 found |
| First object on screen | ~3.1 s | ~1.5 s |
| Each further object | +1.1 s | +0.5 s |
| Before (`litert-lm serve`, all boxes at the end) | 3.7 s ball / 6.8 s all players | 2.4 s / 4 s |

`tools/gemma_boxes.py` asks the server from Python and scores boxes against a
reference. Tests: `npm test` (reply parsing), `npm run e2e:gemma` (the UI
against the real server: ball, all players + tracking, camera, no-server hint).

The live panel splits camera → screen into **Delivery** (capture to the page's
video frame callback), **Wait** (until the pipeline takes the frame) and
**Pipeline** (upload to masks on screen). Wait is near 0 only while the pipeline
step fits in the camera's frame interval. The page asks for 60 fps; the header
shows what the camera granted. Diagnostics:

- `?cam=WxH` — requested camera size (default `1280x720`). Upload, preprocess
  and composite scale with it.
- `?profile=cpu` — per-signature time for LiteRT.js `run()` to return, plus
  stage wall times (median of 60 frames, in the status line and
  `__profile()`). `?profile=gpu` also waits for the GPU after every signature,
  so each row is that stage's GPU time (the total gets longer).

1024 px is available too (`tools/build_models.sh 1024`). Natively on CPU fp32
it matches HF to mean IoU 0.9996–1.0000 (at most 7 pixels on any frame). On the
M4 Pro a live 1024 frame takes about 180 ms with one object (see Live camera).
Its track step grows fastest: memory attention runs over 64×64 tokens.

`tools/inspect_model.py model.tflite` lists a model's signatures, op histogram
and weight sharing. The full implementation report, written for LiteRT
reviewers, is in `docs/`.
