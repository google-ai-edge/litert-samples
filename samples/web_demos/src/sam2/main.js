// Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

/**
 * Click to segment with SAM 2.1 (Hiera-Tiny), fully client-side.
 *
 * Per photo: square resize to 1024×1024 (bilinear, no letterbox) →
 * ImageNet-normalized NCHW float32 → image encoder → decoder-ready features
 * image_embeddings [1,256,64,64], feat_s1 [1,64,128,128], feat_s0 [1,32,256,256].
 * Per click: host prompt encoding (one positive point + the padding token,
 * sparse_prompt [1,2,256]) → mask decoder → pred_masks [1,3,256,256] logits
 * + iou_scores [1,3] → the mask with the highest predicted IoU, bilinearly
 * upsampled to the canvas and thresholded at 0.
 *
 * Both graphs run on WebGPU when available and fall back to WASM per graph
 * if WebGPU fails to compile or warm up. Model files stream from
 * litert-community on Hugging Face (~97 MB, Cache API after the first visit).
 *
 * Every click logs one SAM2_STATS JSON line (tools/check.mjs reads it).
 * Debug URL params: ?img=<url>|example (encode that photo at boot) ·
 *   &point=x,y (then click there; x, y = fractions of the photo's width and
 *   height, 0..1) · ?backend=wasm|webgpu (both graphs) · ?precision=fp32
 *   (WebGPU compute precision; fp16 by default) · ?models=<base url> (all
 *   three files from another location) · ?debug=1. With ?img= or ?debug=1
 *   the last click's arrays stay in window.__lastResult. ?debug=1 also logs
 *   the tensor names (SAM2_IO) and enables what reproduces the README's
 *   numbers: &bench=N (time N encoder and N decoder runs, logs SAM2_BENCH),
 *   &enc= / &dec= (one graph's backend), &threads=0 (single-thread WASM).
 */
import {
  Tensor,
  getWebGpuDevice,
  isWebGPUSupported,
  loadAndCompile,
  loadLiteRt,
} from '@litertjs/core';
import {
  DECODER_INPUT_ORDER,
  SHAPES,
  SIZE,
  argmax,
  bindByShape,
  countPositive,
  encodePoint,
  maskContains,
  parsePromptConstants,
  resizeBilinear,
  toNchw,
  upsampleMask,
} from './sam2.js';

const params = new URLSearchParams(location.search);
// The automation hooks run only when asked for, so a normal visit keeps
// nothing it does not show.
const DEBUG = params.get('debug') === '1';
const KEEP_RESULT = DEBUG || params.has('img');

// Weights stream from the two model cards' repos on Hugging Face and are
// cached with the Cache API after the first visit. ?models=<base url> loads
// the same three file names from one other location (a local copy, a mirror).
const FILES = {
  encoder: 'sam2_tiny_image_encoder_v2_fp16.tflite', // the card's decoder-ready variant
  decoder: 'sam2_tiny_mask_decoder_v2_fp16.tflite', // the card's recommended build (rank-4 attention)
  prompt: 'prompt_encode_const.bin', // posmat + point embeddings, 3,072 bytes
};
const HF = 'https://huggingface.co/litert-community/';
const MODEL_BASE = params.get('models')?.replace(/\/?$/, '/');
const MODEL_URLS = MODEL_BASE
  ? Object.fromEntries(Object.entries(FILES).map(([k, f]) => [k, MODEL_BASE + f]))
  : {
    encoder: `${HF}SAM2.1-Hiera-Tiny-Image-Encoder/resolve/main/${FILES.encoder}`,
    decoder: `${HF}SAM2.1-Hiera-Tiny-Mask-Decoder/resolve/main/${FILES.decoder}`,
    prompt: `${HF}SAM2.1-Hiera-Tiny-Mask-Decoder/resolve/main/${FILES.prompt}`,
  };
// File sizes on the Hub, for the progress bar until a response has given its
// Content-Length (a mirror behind ?models= may not send one).
const EXPECTED_BYTES = { encoder: 80_278_528, decoder: 16_968_160, prompt: 3_072 };
const CACHE_NAME = 'sam2-demo-v1';
// The LiteRT.js WASM runtime is served from litert-wasm/ at the site root
// (vite.config.js copies it there from node_modules). Resolve against this
// module's own URL: in dev it is <root>/sam2/main.js, in the build
// <root>/assets/<hash>.js — one level below the runtime dir either way.
const WASM_DIR = new URL(/* @vite-ignore */ '../litert-wasm/', import.meta.url).href;
const EXAMPLE_URL = new URL('./example.jpg', import.meta.url).href;
// Default click on the bundled example (fractions of width, height): the cat.
const EXAMPLE_POINT = [0.6, 0.5];
// Photos larger than this (long side) are shrunk by the browser first; the
// model only sees 1024×1024 anyway.
const MAX_SOURCE_SIDE = 4096;
const GRAPHS = ['encoder', 'decoder'];
const FEATURE_KEYS = ['imageEmbeddings', 'featS1', 'featS0'];
// WebGPU compute precision. fp16 runs the encoder 27% faster than fp32 on an
// M4 Max (60.8 vs 83.9 ms, headless Chromium), and its masks match a CPU run
// of the same files at IoU 0.9997–1.0 on three clicks of the example;
// ?precision=fp32 switches back for comparison.
const PRECISION = params.get('precision') === 'fp32' ? 'fp32' : 'fp16';

const statusEl = document.getElementById('status');
const latencyEl = document.getElementById('latency');
const encMsEl = document.getElementById('enc-ms');
const decMsEl = document.getElementById('dec-ms');
const envEl = document.getElementById('env');
const hintEl = document.getElementById('hint');
const backendButtons = [...document.querySelectorAll('#backend-switch button')];
const fileEl = document.getElementById('file');
const camBtn = document.getElementById('cam');
const shutterBtn = document.getElementById('shutter');
const exampleBtn = document.getElementById('example');
const stage = document.getElementById('stage');
const placeholder = document.getElementById('placeholder');
const view = document.getElementById('view');
const videoEl = document.getElementById('video');
const dropOverlay = document.getElementById('drop-overlay');

let wasmOpts = null; // which loadLiteRt attempt succeeded
const bytes = {}; // downloaded files by key
let consts = null; // prompt encoder constants
let models = null; // { encoder, decoder } compiled
let buildInfo = null; // compile / warm-up times of the current models
let downloadInfo = null;
let wanted = null; // requested accelerator per graph
const webgpuFailed = new Set(); // graphs that failed to compile or warm up on WebGPU
let gpuAvailable = false; // a WebGPU device exists (set once the runtime is loaded)
let photo = null; // { id, bitmap, width, height, rgb (KEEP_RESULT only), nchw, source }
let features = null; // decoder-ready encoder outputs for `photo`
let encodeError = null; // why `photo` has no features (its encode failed)
let encoderMs = null;
let prepMs = null;
let lastPoint = null; // [fx, fy] of the last click
let lastMask = null; // { logits, iou, best }
let pendingPhoto = null; // a photo given before the models were ready
let fatal = null; // status line of a failure that left no models (shown again on new input)
let photoCount = 0;
const READY = 'Ready — choose a photo, use the camera, drop or paste an image, or try the example.';

function status(text, pct = null) {
  statusEl.textContent = text;
  if (pct !== null) {
    const bar = document.createElement('span');
    bar.className = 'bar';
    const fill = document.createElement('i');
    fill.style.width = `${Math.round(pct * 100)}%`;
    bar.appendChild(fill);
    statusEl.appendChild(bar);
  }
}

/** Runtime failures are not always Error objects (a failed <script> load
 * rejects with an Event; WebKit sometimes throws bare strings). */
function errText(err) {
  if (err instanceof Error) return err.message;
  if (typeof err === 'string') return err;
  if (err && typeof err.type === 'string') return `${err.type} event`;
  return String(err);
}

const round = (v, digits = 1) => +v.toFixed(digits);

// --- model loading --------------------------------------------------------

/** One file, from the Cache API when it holds it, else from the network
 * (then stored). Where the Cache API is missing or fails (storage blocked,
 * quota), the page still runs and downloads on every visit. */
async function fetchCached(url, onProgress) {
  let cache = null;
  try {
    cache = 'caches' in window ? await caches.open(CACHE_NAME) : null;
    const hit = cache && (await cache.match(url));
    if (hit) {
      const data = new Uint8Array(await hit.arrayBuffer());
      onProgress(data.length, data.length, true);
      return { data, cached: true };
    }
  } catch (err) {
    console.warn(`[sam2] Cache API unavailable (${errText(err)}); downloading without it.`);
    cache = null;
  }
  let response;
  try {
    response = await fetch(url);
  } catch (err) {
    // The browser reports a blocked or unreachable host as a bare
    // "Failed to fetch" — name the URL so the failure is diagnosable.
    const offline = navigator.onLine === false ? ', browser is offline' : '';
    throw new Error(`could not fetch ${url} (${errText(err)}${offline})`);
  }
  if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`);
  const size = Number(response.headers.get('Content-Length')) || 0;
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(received, size, false);
  }
  const data = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    data.set(chunk, offset);
    offset += chunk.length;
  }
  if (cache) {
    // In the background: a failed store (quota, storage blocked) only means
    // the next visit downloads the file again.
    cache.put(url, new Response(data)).catch((err) => {
      console.warn(`[sam2] could not cache ${url} (${errText(err)}); the next visit downloads it again.`);
    });
  }
  return { data, cached: false };
}

async function downloadModels() {
  const received = {};
  const sizes = {}; // Content-Length per file (its byte count on a cache hit); 0 = not sent
  let fromCache = true;
  const start = performance.now();
  const results = await Promise.all(Object.entries(MODEL_URLS).map(async ([key, url]) => {
    const result = await fetchCached(url, (n, size, cached) => {
      // Once one file has failed, the others may still be streaming: their
      // progress must not cover the failure line.
      if (fatal) return;
      received[key] = n;
      sizes[key] = size;
      fromCache &&= cached;
      if (fromCache) {
        status('Loading the models from cache…');
        return;
      }
      const mb = (count) => (count / 1e6).toFixed(0);
      const done = Object.values(received).reduce((a, b) => a + b, 0);
      const total = Object.keys(MODEL_URLS).reduce((sum, k) => sum + (sizes[k] || EXPECTED_BYTES[k]), 0);
      status(`Downloading models (one-time)… ${mb(done)} / ${mb(total)} MB`, Math.min(done / total, 1));
    });
    bytes[key] = result.data;
    return result;
  }));
  downloadInfo = {
    bytes: results.reduce((n, r) => n + r.data.length, 0),
    ms: round(performance.now() - start, 0),
    cached: results.every((r) => r.cached),
  };
}

function compileOptions(acc) {
  if (acc === 'webgpu') return { accelerator: 'webgpu', gpuOptions: { precision: PRECISION } };
  return { accelerator: 'wasm' };
}

/** What a compiled graph actually runs on: LiteRT.js can silently recompile
 * for WASM (or split the graph) when WebGPU does not take every op. */
function placement(model) {
  const acc = model.options.accelerator;
  return acc !== 'wasm' && !model.isFullyAccelerated ? `${acc}+wasm` : acc;
}

const wasmLabel = () => (wasmOpts?.threads ? 'wasm threads' : 'wasm 1-thread');

function showEnv() {
  const graphs = GRAPHS.map((g) => {
    const failed = wanted[g] === 'webgpu' && webgpuFailed.has(g) ? ' (WebGPU failed)' : '';
    return `${g} ${placement(models[g])}${failed}`;
  });
  const warm = (buildInfo.warmupMs.encoder + buildInfo.warmupMs.decoder) / 1000;
  envEl.textContent = `SAM 2.1 ${PRECISION} · ${graphs.join(' · ')} · ${wasmLabel()} · warm-up ${warm.toFixed(1)} s`;
  envEl.style.display = 'block';
}

function releaseFeatures(set) {
  if (!set) return;
  for (const key of FEATURE_KEYS) set[key]?.delete();
}

/** Photo tensor → decoder-ready features, placed where the decoder reads
 * them (once per photo, not on every click). */
async function encode(set, nchw) {
  const input = Tensor.fromTypedArray(nchw, SHAPES.image);
  const start = performance.now();
  let outputs;
  try {
    outputs = await set.encoder.run([input]);
  } finally {
    input.delete();
  }
  const feats = {};
  try {
    const out = bindByShape(set.encoder.getOutputDetails(), FEATURE_KEYS);
    for (const key of FEATURE_KEYS) feats[key] = outputs[out[key].index];
    const target = set.decoder.options.accelerator === 'webgpu' ? 'webgpu' : 'wasm';
    for (const key of FEATURE_KEYS) {
      if (feats[key].accelerator !== target) feats[key] = await feats[key].moveTo(target);
    }
    // WebGPU work is asynchronous: wait for the queue so the time covers the
    // encoder itself, not just its submission.
    if (FEATURE_KEYS.some((key) => feats[key].accelerator === 'webgpu')) {
      await getWebGpuDevice().queue.onSubmittedWorkDone();
    }
  } catch (err) {
    // feats alias the outputs until moveTo() replaces one (and deletes the
    // original): free each tensor that is still alive, once.
    for (const t of new Set([...outputs, ...Object.values(feats)])) if (!t.deleted) t.delete();
    throw err;
  }
  return { features: feats, ms: performance.now() - start };
}

/** One click → 3 mask logits + their predicted IoUs (read back to the CPU). */
async function decode(set, feats, sparse) {
  const prompt = Tensor.fromTypedArray(sparse, SHAPES.sparsePrompt);
  const inputs = bindByShape(set.decoder.getInputDetails(), DECODER_INPUT_ORDER, DECODER_INPUT_ORDER);
  const ordered = [];
  for (const key of DECODER_INPUT_ORDER) {
    ordered[inputs[key].index] = key === 'sparsePrompt' ? prompt : feats[key];
  }
  const start = performance.now();
  let outputs = null;
  try {
    outputs = await set.decoder.run(ordered);
    const out = bindByShape(set.decoder.getOutputDetails(), ['predMasks', 'iouScores']);
    const logits = await outputs[out.predMasks.index].data();
    const iou = await outputs[out.iouScores.index].data();
    return { logits, iou, ms: performance.now() - start };
  } finally {
    prompt.delete();
    if (outputs) for (const output of outputs) output.delete();
  }
}

// Which boot stage is in flight, so a failure names the culprit
// (runtime / download / compile encoder webgpu / warm-up decoder wasm / …).
let bootStage = 'runtime';

/** Compile both graphs for `want` and burn one throwaway run of each: the
 * first run after compile carries shader / kernel warm-up and must never
 * land on a user photo or in the latency display. A graph that fails on
 * WebGPU is retried on WASM (the other graph keeps its accelerator). */
async function buildGraphs(want) {
  for (;;) {
    const plan = {};
    for (const g of GRAPHS) plan[g] = want[g] === 'webgpu' && webgpuFailed.has(g) ? 'wasm' : want[g];
    const built = {};
    const info = { compileMs: {}, warmupMs: {} };
    let current = null;
    try {
      for (const g of GRAPHS) {
        current = g;
        bootStage = `compile ${g} ${plan[g]}`;
        status(`Compiling the ${g} for ${plan[g] === 'webgpu' ? 'WebGPU' : 'WASM'}…`);
        const start = performance.now();
        built[g] = await loadAndCompile(bytes[g], compileOptions(plan[g]));
        info.compileMs[g] = round(performance.now() - start, 0);
        if (plan[g] === 'webgpu' && built[g].options.accelerator !== 'webgpu') {
          // In a browser without JSPI, LiteRT.js compiles a graph that WebGPU
          // cannot fully take for WASM instead, without an error.
          webgpuFailed.add(g);
          console.warn(`[sam2] the ${g} was compiled for WASM: WebGPU could not take all of it`);
          status(`WebGPU could not take all of the ${g} — it runs on WASM…`);
        }
      }
      status('Warming up (one throwaway run)…');
      current = 'encoder';
      bootStage = `warm-up encoder ${plan.encoder}`;
      let start = performance.now();
      let warmFeatures;
      if (plan.encoder === 'webgpu') {
        warmFeatures = (await encode(built, new Float32Array(3 * SIZE * SIZE))).features;
        info.warmupMs.encoder = round(performance.now() - start, 0);
      } else {
        // No warm-up run on WASM: a CPU graph has no shaders to compile, and
        // the encoder takes seconds there. Zero features warm the decoder.
        warmFeatures = Object.fromEntries(FEATURE_KEYS.map((key) => [key,
          Tensor.fromTypedArray(new Float32Array(SHAPES[key].reduce((a, b) => a * b)), SHAPES[key])]));
        info.warmupMs.encoder = 0;
      }
      try {
        current = 'decoder';
        bootStage = `warm-up decoder ${plan.decoder}`;
        start = performance.now();
        await decode(built, warmFeatures, encodePoint(consts, SIZE / 2, SIZE / 2));
        info.warmupMs.decoder = round(performance.now() - start, 0);
      } finally {
        releaseFeatures(warmFeatures);
      }
      return { built, info };
    } catch (err) {
      for (const model of Object.values(built)) model.delete();
      if (current && plan[current] === 'webgpu') {
        // WebGPU exists on paper in more browsers than it works in (mobile
        // WebKit in particular) — fall back to WASM instead of dying.
        webgpuFailed.add(current);
        console.warn(`[sam2] the ${current} failed on WebGPU, retrying it on WASM:`, err);
        status(`WebGPU failed for the ${current} (${errText(err)}) — retrying on WASM…`);
        continue;
      }
      throw err;
    }
  }
}

async function install(want) {
  wanted = want;
  const { built, info } = await buildGraphs(want);
  models = built;
  buildInfo = info;
  showEnv();
  syncBackendButtons();
  if (DEBUG) {
    // The tensor names LiteRT.js reports (binding goes by shape; see sam2.js).
    const names = (details) => details.map((d) => `${d.name}[${[...d.shape].join(',')}]`);
    const io = (model) => ({ inputs: names(model.getInputDetails()), outputs: names(model.getOutputDetails()) });
    console.log('SAM2_IO ' + JSON.stringify({ encoder: io(models.encoder), decoder: io(models.decoder) }));
  }
}

/** Rebuild both graphs for `want`. If that fails, the previous backends are
 * built again, so a failed switch leaves a working page (then rethrows). */
async function setBackends(want) {
  const previous = models ? wanted : null;
  // Free the old graphs first: two copies of the 80 MB encoder can exhaust
  // the WASM heap on small devices.
  releaseFeatures(features);
  features = null;
  if (models) for (const g of GRAPHS) models[g].delete();
  models = null;
  try {
    await install(want);
  } catch (err) {
    if (previous) {
      console.error('[sam2] building the new backends failed, restoring the previous ones:', err);
      await install(previous); // if this throws too, no models are left
    }
    throw err;
  }
}

/** The switch position that matches what runs (a graph may have fallen
 * back to WASM). */
function activeBackend() {
  const onGpu = models
    ? GRAPHS.some((g) => models[g].options.accelerator === 'webgpu')
    : !wanted || GRAPHS.some((g) => wanted[g] === 'webgpu');
  return onGpu ? 'webgpu' : 'wasm';
}

function syncBackendButtons(busy = false) {
  const gpuUsable = gpuAvailable && GRAPHS.some((g) => !webgpuFailed.has(g));
  const active = activeBackend();
  for (const b of backendButtons) {
    b.classList.toggle('active', b.dataset.backend === active);
    b.disabled = busy || !models || (b.dataset.backend === 'webgpu' && !gpuUsable);
    if (b.dataset.backend === 'webgpu' && !gpuUsable) {
      b.title = gpuAvailable ? 'WebGPU failed on this device' : 'WebGPU is not available in this browser';
    }
  }
}

// --- photos and clicks ----------------------------------------------------

// Every model call goes through one queue, so a click never races an encode
// or a backend switch. Clicks coalesce: only the latest pending one runs.
let chain = Promise.resolve();
function enqueue(task) {
  const run = chain.then(task);
  chain = run.catch(() => {});
  return run;
}

const scratch = document.createElement('canvas');
function readPixels(bitmap) {
  scratch.width = bitmap.width;
  scratch.height = bitmap.height;
  const ctx = scratch.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  const data = ctx.getImageData(0, 0, bitmap.width, bitmap.height).data;
  // Free the canvas pixels now, not at garbage collection (iOS caps the
  // canvas memory of a page).
  scratch.width = 0;
  scratch.height = 0;
  return data;
}

async function preparePhoto(bitmap, source) {
  let bmp = bitmap;
  const side = Math.max(bmp.width, bmp.height);
  if (side > MAX_SOURCE_SIDE) {
    const scale = MAX_SOURCE_SIDE / side;
    bmp = await createImageBitmap(bitmap, {
      resizeWidth: Math.round(bitmap.width * scale),
      resizeHeight: Math.round(bitmap.height * scale),
      resizeQuality: 'high',
    });
    bitmap.close(); // the full-size decode is not needed any more
  }
  const start = performance.now();
  // Same steps as the Python usage on the model card: square resize
  // (Pillow-exact bilinear), /255, ImageNet mean/std, NCHW.
  const rgb = resizeBilinear(readPixels(bmp), bmp.width, bmp.height, SIZE, SIZE);
  const nchw = toNchw(rgb);
  prepMs = performance.now() - start;
  return {
    id: String(++photoCount), bitmap: bmp, width: bmp.width, height: bmp.height, nchw, source,
    rgb: KEEP_RESULT ? rgb : null, // the encoder's input, for window.__lastResult
  };
}

/** Resolves after the browser has painted pending DOM changes (capped, as
 * requestAnimationFrame does not fire in a hidden tab). */
function nextPaint() {
  return new Promise((resolve) => {
    setTimeout(resolve, 100);
    requestAnimationFrame(() => setTimeout(resolve, 0));
  });
}

async function encodePhoto() {
  const onGpu = models.encoder.options.accelerator === 'webgpu';
  // The numbers and the hint below belong to the previous encode.
  latencyEl.style.display = 'none';
  hintEl.style.display = 'none';
  status(onGpu ? 'Encoding the photo…' : 'Encoding the photo on WASM — this takes a while…');
  // The WASM encoder holds the main thread for seconds: show the new photo
  // and this message before it starts.
  if (!onGpu) await nextPaint();
  releaseFeatures(features);
  features = null;
  encodeError = null;
  try {
    const result = await encode(models, photo.nchw);
    features = result.features;
    encoderMs = result.ms;
  } catch (err) {
    encodeError = errText(err);
    throw err;
  }
  encMsEl.textContent = `${encoderMs.toFixed(0)} ms`;
  decMsEl.textContent = '–';
  latencyEl.style.display = 'block';
  hintEl.style.display = 'block';
  status('Click anything in the photo.');
}

/** Encode the photo on screen (a new photo, or new graphs after a backend
 * switch) and repeat the click it should show; failures end in the status. */
async function encodeAndSegment(point) {
  try {
    await encodePhoto();
    if (point) await segmentAt(point);
  } catch (err) {
    status(`Failed: ${errText(err)}`);
    console.error('[sam2] photo failed:', err);
  }
}

async function loadPhoto(bitmap, source, point = null) {
  if (!models) {
    if (fatal) {
      // The models are not coming: keep the reason on screen.
      bitmap.close();
      status(fatal);
      return;
    }
    pendingPhoto?.bitmap.close(); // a newer photo replaces the waiting one
    pendingPhoto = { bitmap, source, point };
    status('Still loading the model — your photo runs as soon as it is ready.');
    return;
  }
  let next;
  try {
    next = await preparePhoto(bitmap, source);
  } catch (err) {
    bitmap.close();
    status(`Failed: ${errText(err)}`);
    console.error('[sam2] photo failed:', err);
    return;
  }
  photo?.bitmap.close(); // the previous photo leaves the screen
  photo = next;
  lastPoint = null;
  lastMask = null;
  showStage('photo');
  await encodeAndSegment(point);
}

async function segmentAt(point) {
  if (!features) {
    // A click on a photo whose encode failed: say why nothing happens.
    if (fatal) status(fatal);
    else if (encodeError) status(`No mask: this photo could not be encoded (${encodeError}). Choose another photo or switch the backend.`);
    return;
  }
  const [fx, fy] = point;
  const x = fx * SIZE;
  const y = fy * SIZE;
  status('Segmenting…');
  try {
    const { logits, iou, ms } = await decode(models, features, encodePoint(consts, x, y));
    const best = argmax(iou);
    lastPoint = point;
    lastMask = { logits, iou, best };
    render();
    decMsEl.textContent = `${ms.toFixed(0)} ms`;
    status(`Done (predicted IoU ${iou[best].toFixed(2)}) — click anywhere else for a new mask.`);
    const backends = Object.fromEntries(GRAPHS.map((g) => [g, placement(models[g])]));
    // machine-readable stats for automation (tools/check.mjs reads this line)
    const stats = {
      backends,
      wasmThreads: !!wasmOpts?.threads,
      numThreads: models.encoder.options.cpuOptions?.numThreads ?? null,
      precision: PRECISION,
      image: { width: photo.width, height: photo.height, source: photo.source },
      point: { fx, fy, x, y },
      prepMs: round(prepMs),
      encoderMs: round(encoderMs),
      decoderMs: round(ms),
      iouScores: [...iou].map((v) => round(v, 4)),
      best,
      maskPixels: countPositive(logits, best), // logits > 0 in the 256×256 best mask
      hit: maskContains(logits, best, x, y), // the clicked point is inside that mask
      compileMs: buildInfo.compileMs,
      warmupMs: buildInfo.warmupMs,
      download: downloadInfo,
    };
    if (KEEP_RESULT) stats.maskPixelsAll = [0, 1, 2].map((i) => countPositive(logits, i));
    console.log('SAM2_STATS ' + JSON.stringify(stats));
    if (KEEP_RESULT) {
      window.__lastResult = {
        logits, // Float32Array 3×256×256, pred_masks
        iouScores: iou, // Float32Array 3
        best,
        point: { fx, fy, x, y },
        backends,
        input: photo.rgb, // the 1024×1024 RGB bytes the encoder saw
        encoderMs,
        decoderMs: ms,
      };
    }
  } catch (err) {
    status(`Failed: ${errText(err)}`);
    console.error('[sam2] segmentation failed:', err);
  }
}

let pendingPoint = null;
let pointQueued = false;
function requestSegment(point) {
  pendingPoint = point;
  if (pointQueued) return;
  pointQueued = true;
  enqueue(async () => {
    // Let clicks that queued up while the main thread was busy (the WASM
    // graphs run on it) arrive first, so a burst of them decodes only the last.
    await new Promise((resolve) => setTimeout(resolve, 0));
    pointQueued = false;
    const next = pendingPoint;
    pendingPoint = null;
    if (next) await segmentAt(next);
  });
}

// --- drawing --------------------------------------------------------------

const viewCtx = view.getContext('2d');
const baseCanvas = document.createElement('canvas'); // the photo at view size
const maskCanvas = document.createElement('canvas');

function showStage(what) {
  placeholder.style.display = what === 'placeholder' ? 'block' : 'none';
  view.style.display = what === 'photo' ? 'block' : 'none';
  videoEl.style.display = what === 'video' ? 'block' : 'none';
  stage.classList.toggle('has-photo', what !== 'placeholder');
  if (what === 'photo') render();
}

function render() {
  if (!photo || view.style.display === 'none') return;
  const box = stage.getBoundingClientRect();
  const scale = Math.min(box.width / photo.width, box.height / photo.height);
  const cssW = Math.max(1, Math.floor(photo.width * scale));
  const cssH = Math.max(1, Math.floor(photo.height * scale));
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.round(cssW * dpr);
  const h = Math.round(cssH * dpr);
  view.style.width = `${cssW}px`;
  view.style.height = `${cssH}px`;
  if (view.width !== w || view.height !== h || baseCanvas.dataset.photo !== photo.id) {
    view.width = w;
    view.height = h;
    baseCanvas.width = w;
    baseCanvas.height = h;
    const baseCtx = baseCanvas.getContext('2d');
    baseCtx.imageSmoothingQuality = 'high';
    baseCtx.drawImage(photo.bitmap, 0, 0, w, h);
    baseCanvas.dataset.photo = photo.id;
  }
  viewCtx.drawImage(baseCanvas, 0, 0);
  if (lastMask) {
    // Tint the mask and outline it; the frame of the photo is not an edge.
    const mask = upsampleMask(lastMask.logits, lastMask.best, w, h);
    const image = new ImageData(w, h);
    const px = image.data;
    const edge = Math.max(1, Math.round(dpr));
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (!mask[i]) continue;
        let border = false;
        for (let d = 1; d <= edge && !border; d++) {
          border = (x >= d && !mask[i - d]) || (x < w - d && !mask[i + d]) ||
            (y >= d && !mask[i - d * w]) || (y < h - d && !mask[i + d * w]);
        }
        const o = i * 4;
        if (border) {
          px[o] = 255; px[o + 1] = 255; px[o + 2] = 255; px[o + 3] = 235;
        } else {
          px[o] = 124; px[o + 1] = 196; px[o + 2] = 255; px[o + 3] = 125;
        }
      }
    }
    maskCanvas.width = w;
    maskCanvas.height = h;
    maskCanvas.getContext('2d').putImageData(image, 0, 0);
    viewCtx.drawImage(maskCanvas, 0, 0);
  }
  if (lastPoint) {
    const [fx, fy] = lastPoint;
    viewCtx.beginPath();
    viewCtx.arc(fx * w, fy * h, 6 * dpr, 0, Math.PI * 2);
    viewCtx.fillStyle = '#7cc4ff';
    viewCtx.fill();
    viewCtx.lineWidth = 2 * dpr;
    viewCtx.strokeStyle = '#ffffff';
    viewCtx.stroke();
  }
}

let renderQueued = false;
new ResizeObserver(() => {
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => {
    renderQueued = false;
    render();
  });
}).observe(stage);

view.addEventListener('click', (event) => {
  const rect = view.getBoundingClientRect();
  const fx = (event.clientX - rect.left) / rect.width;
  const fy = (event.clientY - rect.top) / rect.height;
  if (fx < 0 || fy < 0 || fx >= 1 || fy >= 1) return;
  requestSegment([fx, fy]);
});

// --- inputs: file, drop, paste, camera, example ---------------------------

function usePhoto(bitmap, source, point = null) {
  // A photo takes the stage: a running camera stops, one still opening is
  // dropped when its stream arrives.
  cancelCameraStart();
  stopCamera();
  return enqueue(() => loadPhoto(bitmap, source, point));
}

async function fetchBitmap(url) {
  let response;
  try {
    response = await fetch(url);
  } catch (err) {
    throw new Error(`could not fetch ${url} (${errText(err)})`);
  }
  if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`);
  return createImageBitmap(await response.blob());
}

/** A file from the picker, a drop or a paste. A file the browser cannot
 * decode as an image gets a message (and leaves the camera running). */
async function useFile(file, source) {
  let bitmap;
  try {
    bitmap = await createImageBitmap(file);
  } catch (err) {
    status(`Could not read that file: ${errText(err)}`);
    return;
  }
  await usePhoto(bitmap, source);
}

fileEl.addEventListener('change', () => {
  const file = fileEl.files?.[0];
  fileEl.value = '';
  if (file) useFile(file, 'file');
});

window.addEventListener('dragover', (event) => {
  event.preventDefault();
  dropOverlay.style.display = 'flex';
});
window.addEventListener('dragleave', (event) => {
  if (event.relatedTarget === null) dropOverlay.style.display = 'none';
});
window.addEventListener('drop', (event) => {
  event.preventDefault();
  dropOverlay.style.display = 'none';
  const file = event.dataTransfer?.files?.[0];
  if (file) useFile(file, 'drop');
});

window.addEventListener('paste', (event) => {
  const item = [...(event.clipboardData?.items ?? [])].find((entry) =>
    entry.type.startsWith('image/'),
  );
  const file = item?.getAsFile();
  if (file) useFile(file, 'paste');
});

exampleBtn.addEventListener('click', async () => {
  try {
    await usePhoto(await fetchBitmap(EXAMPLE_URL), 'example', EXAMPLE_POINT);
  } catch (err) {
    status(`Example: ${errText(err)}`);
  }
});

let stream = null;
let camOpening = null; // a camera start waiting on the browser (its permission prompt)
let capturing = false;
const CAMERA_HIDDEN = 'Camera off — the tab was hidden.';

function cancelCameraStart() {
  if (camOpening) camOpening.cancelled = true;
}

camBtn.addEventListener('click', async (event) => {
  // One action per double click; the button is disabled while the camera
  // opens, and a synthetic click then stops here.
  if (event.detail > 1 || camOpening) return;
  if (stream) {
    stopCamera('Camera off.');
    return;
  }
  const opening = { cancelled: false };
  camOpening = opening;
  camBtn.disabled = true;
  let s;
  try {
    s = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment' },
    });
  } catch (err) {
    if (!opening.cancelled) status(`Camera: ${errText(err)}`);
    return;
  } finally {
    camOpening = null;
    camBtn.disabled = false;
  }
  if (opening.cancelled || document.visibilityState === 'hidden') {
    // A photo arrived, or the tab was hidden, while the browser was asking.
    s.getTracks().forEach((track) => track.stop());
    if (!opening.cancelled) status(CAMERA_HIDDEN);
    return;
  }
  stream = s;
  for (const track of s.getVideoTracks()) {
    // The browser or the OS took the camera away (unplugged, permission revoked).
    track.addEventListener('ended', () => {
      if (stream === s) stopCamera('Camera off — the browser ended the camera stream.');
    });
  }
  videoEl.srcObject = s;
  showStage('video');
  shutterBtn.style.display = 'inline-block';
  camBtn.textContent = 'Stop camera';
  status('Camera on — press Capture to use the current frame.');
});

/** Stop the camera (if on) and show the photo again; `why` replaces the
 * status line (none when a new photo takes over). */
function stopCamera(why = null) {
  if (!stream) return;
  stream.getTracks().forEach((track) => track.stop());
  stream = null;
  videoEl.srcObject = null;
  shutterBtn.style.display = 'none';
  camBtn.textContent = 'Use camera';
  showStage(photo ? 'photo' : 'placeholder');
  if (why) status(why);
}

shutterBtn.addEventListener('click', async () => {
  if (capturing || !stream) return; // one frame per press, even a double click
  if (videoEl.readyState < 2 || !videoEl.videoWidth) {
    status('The camera has no picture yet — press Capture again in a moment.');
    return;
  }
  capturing = true;
  let frame;
  try {
    // Snapshot to a bitmap so the frame survives stopCamera().
    frame = await createImageBitmap(videoEl);
  } catch (err) {
    status(`Capture failed: ${errText(err)}`);
    return;
  } finally {
    capturing = false;
  }
  await usePhoto(frame, 'camera');
});

// A hidden tab or a page being left gives the camera back (its light goes
// off); the photo on screen stays.
function releaseCamera() {
  if (camOpening) {
    cancelCameraStart();
    status(CAMERA_HIDDEN);
  }
  stopCamera(CAMERA_HIDDEN);
}
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden') releaseCamera();
});
window.addEventListener('pagehide', releaseCamera);

// --- backend switch (webgpu / wasm) ---------------------------------------

for (const button of backendButtons) {
  button.addEventListener('click', () => {
    const acc = button.dataset.backend;
    if (button.disabled || !models || acc === activeBackend()) return;
    syncBackendButtons(true);
    enqueue(async () => {
      let failure = null;
      try {
        await setBackends({ encoder: acc, decoder: acc });
      } catch (err) {
        failure = `Backend switch failed: ${errText(err)}`;
        console.error('[sam2] backend switch failed:', err);
      }
      if (!models) {
        // Neither the new backends nor the previous ones could be built.
        fatal = failure;
        status(failure);
      } else {
        // The photo's features went with the old graphs.
        if (photo) await encodeAndSegment(lastPoint);
        else status(READY);
        if (failure) status(`${failure} — still on ${activeBackend() === 'webgpu' ? 'WebGPU' : 'WASM'}.`);
      }
      syncBackendButtons();
    });
  });
}

// --- boot -----------------------------------------------------------------

function parsePoint(text) {
  if (!text) return null;
  const parts = text.split(',').map(Number);
  if (parts.length !== 2 || parts.some((v) => !(v >= 0 && v < 1))) {
    console.warn(`[sam2] ignoring ?point=${text}: expected x,y as fractions in [0, 1)`);
    return null;
  }
  return parts;
}

const median = (values) => {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

/** ?debug=1&bench=N: time N encoder runs, N decoder runs and N encoder runs
 * that also read the three feature maps back (an upper bound on the encoder
 * time if WebGPU queue timing undercounts). */
async function bench(n, point) {
  const encoderRuns = [];
  const decoderRuns = [];
  const readbackRuns = [];
  for (let i = 0; i < n; i++) {
    const result = await encode(models, photo.nchw);
    releaseFeatures(features);
    features = result.features;
    encoderRuns.push(result.ms);
  }
  const sparse = encodePoint(consts, point[0] * SIZE, point[1] * SIZE);
  for (let i = 0; i < n; i++) decoderRuns.push((await decode(models, features, sparse)).ms);
  for (let i = 0; i < n; i++) {
    const start = performance.now();
    const result = await encode(models, photo.nchw);
    for (const key of FEATURE_KEYS) await result.features[key].data();
    readbackRuns.push(performance.now() - start);
    releaseFeatures(result.features);
  }
  const report = {
    n,
    backends: Object.fromEntries(GRAPHS.map((g) => [g, placement(models[g])])),
    wasmThreads: !!wasmOpts?.threads,
    numThreads: models.encoder.options.cpuOptions?.numThreads ?? null,
    precision: PRECISION,
    visibility: document.visibilityState,
    focused: document.hasFocus(),
    median: {
      encoderMs: round(median(encoderRuns)),
      decoderMs: round(median(decoderRuns)),
      encoderReadbackMs: round(median(readbackRuns)),
    },
    encoderMs: encoderRuns.map((v) => round(v)),
    decoderMs: decoderRuns.map((v) => round(v)),
    encoderReadbackMs: readbackRuns.map((v) => round(v)),
    compileMs: buildInfo.compileMs,
    warmupMs: buildInfo.warmupMs,
    download: downloadInfo,
  };
  console.log('SAM2_BENCH ' + JSON.stringify(report));
  status(`Benchmark: encoder ${report.median.encoderMs} ms · decoder ${report.median.decoderMs} ms (median of ${n}).`);
}

async function boot() {
  try {
    status('Loading runtime…');
    // `threads` and `jspi` are mutually exclusive in LiteRT.js, and threads
    // only work on a cross-origin-isolated page — ask for what can succeed,
    // then fall back to plain. ?debug=1&threads=0 forces the single-thread build.
    const wantThreads = !(DEBUG && params.get('threads') === '0') && window.crossOriginIsolated;
    const rungs = wantThreads ? [{ threads: true }, { threads: false }] : [{ threads: false }];
    for (const [index, opts] of rungs.entries()) {
      try {
        await loadLiteRt(WASM_DIR, opts);
        wasmOpts = opts;
        break;
      } catch (err) {
        if (index === rungs.length - 1) {
          throw new Error(`LiteRT.js runtime did not load from ${WASM_DIR} (${errText(err)})`);
        }
      }
    }

    // navigator.gpu can exist without a usable adapter (headless browsers,
    // blocklisted GPUs); LiteRT.js then has no device and WebGPU is out.
    gpuAvailable = isWebGPUSupported() && !!getWebGpuDevice();
    const pick = (value, fallback) => (value === 'wasm' || value === 'webgpu' ? value : fallback);
    const both = pick(params.get('backend'), 'webgpu');
    const want = {};
    for (const [g, key] of [['encoder', 'enc'], ['decoder', 'dec']]) {
      want[g] = pick(DEBUG ? params.get(key) : null, both);
      if (want[g] === 'webgpu' && !gpuAvailable) want[g] = 'wasm';
    }

    bootStage = 'download';
    await downloadModels();
    consts = parsePromptConstants(bytes.prompt);

    await setBackends(want);
    status(READY);

    const n = DEBUG ? Number(params.get('bench')) : 0;
    const img = params.get('img') ?? (n > 0 ? 'example' : null);
    if (img) {
      bootStage = 'boot image';
      const url = img === 'example' ? EXAMPLE_URL : img;
      const point = parsePoint(params.get('point')) ?? (img === 'example' ? EXAMPLE_POINT : null);
      await usePhoto(await fetchBitmap(url), img === 'example' ? 'example' : 'url', point);
      if (n > 0 && photo) await enqueue(() => bench(n, point ?? [0.5, 0.5]));
    } else if (pendingPhoto) {
      const { bitmap, source, point } = pendingPhoto;
      pendingPhoto = null;
      await usePhoto(bitmap, source, point);
    }
  } catch (err) {
    const hint = bootStage === 'download'
      ? ' The weights stream from Hugging Face: check that huggingface.co is reachable, or pass ?models=<url> to load them from elsewhere.'
      : '';
    const text = `Failed to start (${bootStage}): ${errText(err).replace(/\.$/, '')}.${hint}`;
    if (!models) {
      fatal = text;
      pendingPhoto?.bitmap.close();
      pendingPhoto = null;
    }
    status(text);
    console.error(`[sam2] boot failed at stage "${bootStage}":`, err);
  } finally {
    syncBackendButtons();
  }
}

showStage('placeholder');
syncBackendButtons();
boot();
