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
 * Photo + typed words → a mask of what the words describe, fully client-side.
 *
 * CLIPSeg (rd64-refined) as three LiteRT graphs plus a little host work
 * (see the model card):
 *   photo → square resize to 352×352 (Pillow-exact bilinear, no crop)
 *     → x/255, ImageNet mean/std → NCHW float32
 *     → vision graph → t3, t6, t9 [1,485,768] (CLIP ViT-B/16 layers 3/6/9)
 *   prompt → CLIP BPE tokenizer (tokenizer.js) → 77 ids
 *     → rows of the float16 token-embedding table → [1,77,512]
 *     → text graph → hidden [1,77,512] → the EOT row @ text_projection
 *     → cond [1,512]
 *   (t3, t6, t9, cond) → decoder graph → logits [1,352,352]
 *     → sigmoid → stretched back over the photo; > 0.5 is the mask.
 * The vision graph runs once per photo: a new prompt on the same photo
 * reruns only the text graph and the decoder. Tensors are bound by
 * signature name (output_0/1/2 = tensor index order → t3/t6/t9, decoder
 * args_0..3 → t3/t6/t9/cond) and checked by shape.
 *
 * Every run logs one CLIPSEG_STATS JSON line to the console (tools/check.mjs
 * reads it). Debug URL params: ?img=<url>|example (segment that image at
 *   boot) · &prompt=<text> (default "a dog") · &bench=N (then N timed runs;
 *   logs CLIPSEG_BENCH) · ?backend=wasm (all graphs on WASM) · ?vision= /
 *   ?text= / ?decoder=webgpu|wasm (each graph's backend, to reproduce the
 *   comparison) · ?precision=fp16|fp32 (the two encoders' WebGPU precision;
 *   the decoder stays fp32) · ?resident=0 (read t3/t6/t9 back to the CPU
 *   between the GPU graphs) · ?threads=0 (single-thread WASM runtime)
 *   · ?raw=1 (also keep the model input and t3/t6/t9) · ?models=<base url>
 *   (fetch the files from somewhere other than Hugging Face). With ?img= or
 *   ?raw=1 the page also logs CLIPSEG_BOOT, adds an input digest to the stats
 *   and keeps the last result in window.__lastResult.
 */
import {
  Tensor,
  getWebGpuDevice,
  isWebGPUSupported,
  loadAndCompile,
  loadLiteRt,
} from '@litertjs/core';
import {
  COND_DIM,
  MASK_THRESHOLD,
  PATCHES,
  SIZE,
  TEXT_DIM,
  VISION_DIM,
  embedTokens,
  fnv1a,
  halfArray,
  maskPixels,
  project,
  resizeToInput,
  toInputTensor,
  widen,
} from './host.js';
import { CONTEXT_LENGTH, ClipTokenizer, MAX_TOKENS } from './tokenizer.js';

const params = new URLSearchParams(location.search);

// Weights stream from the model card's repo on Hugging Face and are cached
// with the Cache API after the first visit. ?models=<base url> points the
// page at another copy (a local directory, a mirror) — same file names.
const DEFAULT_MODEL_BASE = 'https://huggingface.co/litert-community/CLIPSeg-rd64-LiteRT/resolve/main/';
const MODEL_BASE = (params.get('models') ?? DEFAULT_MODEL_BASE).replace(/\/?$/, '/');
const FILES = {
  vision: 'clipseg_vision_fp16.tflite',
  text: 'clipseg_text_fp16.tflite',
  decoder: 'clipseg_decoder.tflite', // fp32
  embeddings: 'token_embedding_f16.bin', // 49408 x 512 float16
  projection: 'text_projection_f16.bin', // 512 x 512 float16
  vocab: 'vocab.json',
  merges: 'merges.txt',
};
const CACHE_NAME = 'clipseg-demo-v1';
// Where each graph runs when WebGPU works. Each of the three matches the
// Python LiteRT reference on WebGPU as on WASM (mask IoU ≥ 0.9996 on the
// example) and is faster on WebGPU. The WASM button moves every graph to WASM.
const DEFAULT_BACKENDS = { vision: 'webgpu', text: 'webgpu', decoder: 'webgpu' };
const GRAPHS = ['vision', 'text', 'decoder'];
const GRAPH_LABELS = { vision: 'image encoder', text: 'text encoder', decoder: 'decoder' };
// The LiteRT.js WASM runtime is served from litert-wasm/ at the site root
// (vite.config.js copies it there from node_modules). Resolve against this
// module's own URL: in dev it is <root>/clipseg/main.js, in the build
// <root>/assets/<hash>.js — one level below the runtime dir either way.
const WASM_DIR = new URL(/* @vite-ignore */ '../litert-wasm/', import.meta.url).href;
const EXAMPLE_URL = new URL('./example.jpg', import.meta.url).href;
const DEFAULT_PROMPT = 'a dog';
// Sources larger than this (beyond 12 MP phone photos) are first scaled by
// the browser to bound memory; up to it the page resizes the exact pixels.
const MAX_SIDE = 4096;
// The automation hooks (tools/check.mjs, the comparison scripts) run only
// when asked for, so a normal visit computes nothing it does not show.
const DEBUG = params.has('img') || params.get('raw') === '1';
const KEEP_RAW = params.get('raw') === '1';
// WebGPU compute precision per graph (LiteRT.js computes in fp32 unless told
// otherwise). Checked on the example against the Python reference: fp16
// leaves the two encoders' masks unchanged (IoU ≥ 0.9996) and makes the
// image encoder faster. On WebGPU with fp16 compute the decoder's logits
// were all 0 for the example prompt; fp32 keeps them, so the decoder stays
// fp32. ?precision=fp16|fp32 sets the two encoders.
const ENCODER_PRECISION = ['fp16', 'fp32'].includes(params.get('precision')) ? params.get('precision') : 'fp16';
const precisionOf = (key) => (key === 'decoder' ? 'fp32' : ENCODER_PRECISION);
// With the image encoder and the decoder both on WebGPU, t3/t6/t9 (4.5 MB)
// stay on the GPU between them instead of a read-back per photo and an
// upload per prompt. ?resident=0 reads them back.
const RESIDENT = params.get('resident') !== '0';

// Tensor shapes per graph. Slots are found by name (args_N / output_N, the
// converter's signature names) and the shapes are checked.
const IO = {
  vision: {
    inputs: { image: { slot: 0, shape: [1, 3, SIZE, SIZE] } },
    outputs: {
      t3: { slot: 0, shape: [1, PATCHES, VISION_DIM] },
      t6: { slot: 1, shape: [1, PATCHES, VISION_DIM] },
      t9: { slot: 2, shape: [1, PATCHES, VISION_DIM] },
    },
  },
  text: {
    inputs: { tokens: { slot: 0, shape: [1, CONTEXT_LENGTH, TEXT_DIM] } },
    outputs: { hidden: { slot: 0, shape: [1, CONTEXT_LENGTH, TEXT_DIM] } },
  },
  decoder: {
    inputs: {
      t3: { slot: 0, shape: [1, PATCHES, VISION_DIM] },
      t6: { slot: 1, shape: [1, PATCHES, VISION_DIM] },
      t9: { slot: 2, shape: [1, PATCHES, VISION_DIM] },
      cond: { slot: 3, shape: [1, COND_DIM] },
    },
    outputs: { logits: { slot: 0, shape: [1, SIZE, SIZE] } },
  },
};

const statusEl = document.getElementById('status');
const latencyEl = document.getElementById('latency');
const summaryEl = document.getElementById('summary');
const envEl = document.getElementById('env');
const backendButtons = [...document.querySelectorAll('#backend-switch button')];
const viewButtons = [...document.querySelectorAll('#view-switch button')];
const fileEl = document.getElementById('file');
const exampleBtn = document.getElementById('example');
const camBtn = document.getElementById('cam');
const shutterBtn = document.getElementById('shutter');
const promptForm = document.getElementById('prompt-form');
const promptEl = document.getElementById('prompt');
const stageEl = document.getElementById('stage');
const videoEl = document.getElementById('video');
const viewEl = document.getElementById('view');
const placeholderEl = document.getElementById('placeholder');
const dropOverlay = document.getElementById('drop-overlay');

const graphs = { vision: null, text: null, decoder: null }; // { model, acc, io, compileMs }
let assets = null; // { tokenizer, embeddings, projection }
let wasmOpts = null; // which loadLiteRt attempt succeeded
let webgpuOk = false;
const gpuRefused = new Set(); // graphs WebGPU could not take here: they stay on WASM
let mode = 'webgpu'; // backend switch position: 'webgpu' (defaults) or 'wasm'
let ready = false;
let warmSeconds = 0;
let bootFailure = null; // the status line of a failed boot, shown again when an input is used
let view = 'mask'; // 'mask' | 'heat'

let photo = null; // { bitmap, width, height, id, source: 'image' | 'camera' }
let photoCount = 0;
// The image features of one photo on one backend: { photoId, acc, resident,
// feats: t3/t6/t9 as typed arrays or (resident) GPU Tensors, raw, rgbDigest, ms }.
let vision = null;
const FEATURES = ['t3', 't6', 't9'];
const condCache = new Map(); // `${acc}\n${prompt}` → text-path result
// The result on screen: { photo, prompt, logits, stats, prob: canvas 352×352 }.
// Only a finished run replaces it, and only over its own photo.
let shown = null;

/** Whether t3/t6/t9 stay on the GPU between the image encoder and the decoder. */
const residentFeatures = () => RESIDENT && graphs.vision.acc === 'webgpu' && graphs.decoder.acc === 'webgpu';

/** WebGPU compute precision of each graph on WebGPU (null on WASM). */
const precisions = () => Object.fromEntries(
  GRAPHS.map((key) => [key, graphs[key].acc === 'webgpu' ? precisionOf(key) : null]));

/** Forget the cached image features, freeing them if they live on the GPU. */
function dropVision() {
  if (vision?.resident) for (const name of FEATURES) vision.feats[name].delete();
  vision = null;
}

if (params.has('prompt')) promptEl.value = params.get('prompt');

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

// --- downloads -------------------------------------------------------------

const mb = (bytes) => (bytes / 1e6).toFixed(0);

/** One file, from the Cache API when it holds it, else from the network
 * (then stored). Where the Cache API is missing or fails (storage blocked,
 * quota), the page still runs and downloads on every visit.
 * onProgress(received, total, cached): total from Content-Length, 0 if not
 * sent. */
async function fetchCached(url, onProgress, signal) {
  let cache = null;
  try {
    cache = 'caches' in window ? await caches.open(CACHE_NAME) : null;
    const hit = cache && (await cache.match(url));
    if (hit) {
      const bytes = new Uint8Array(await hit.arrayBuffer());
      onProgress?.(bytes.length, bytes.length, true);
      return bytes;
    }
  } catch (err) {
    console.warn(`[clipseg] Cache API unavailable (${errText(err)}); downloading without it.`);
    cache = null;
  }
  // The browser reports a blocked or unreachable host, or a connection lost
  // mid-download, as a bare "Failed to fetch" / "network error" — name the
  // URL so the failure is diagnosable.
  const fetchError = (err, after = '') => {
    const offline = navigator.onLine === false ? ', browser is offline' : '';
    return new Error(`could not fetch ${url} (${errText(err)}${after}${offline})`);
  };
  let response;
  try {
    response = await fetch(url, { signal });
  } catch (err) {
    throw fetchError(err);
  }
  if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`);
  const total = Number(response.headers.get('Content-Length')) || 0;
  onProgress?.(0, total, false);
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      onProgress?.(received, total, false);
    }
  } catch (err) {
    throw fetchError(err, ` after ${mb(received)}${total ? ` of ${mb(total)}` : ''} MB`);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  if (cache) {
    // A full disk or a storage quota only costs the next visit a download.
    try {
      await cache.put(url, new Response(bytes));
    } catch (err) {
      console.warn(`[clipseg] could not cache ${url} (${errText(err)}); the next visit downloads it again.`);
    }
  }
  return bytes;
}

/** All seven files in parallel, with one progress line. The first file that
 * fails stops the others. */
async function downloadAll() {
  const keys = Object.keys(FILES);
  const received = {};
  const sizes = {}; // from Content-Length; 0 = not sent
  let fromCache = true;
  const abort = new AbortController();
  const report = (key) => (n, size, cached) => {
    // The failure line of the file that failed must stay on screen.
    if (abort.signal.aborted) return;
    received[key] = n;
    sizes[key] = size;
    fromCache &&= cached;
    if (fromCache) {
      status('Loading model from cache…');
      return;
    }
    const done = Object.values(received).reduce((a, b) => a + b, 0);
    // Hugging Face sends the two tokenizer files compressed, without a
    // Content-Length: those count as what has arrived so far (1.5 MB of 278).
    const total = keys.every((k) => k in sizes)
      ? keys.reduce((a, k) => a + Math.max(sizes[k], received[k]), 0)
      : 0;
    if (total) {
      status(`Downloading model (one-time)… ${mb(done)} / ${mb(total)} MB`, Math.min(done / total, 1));
    } else {
      status(`Downloading model (one-time)… ${mb(done)} MB`);
    }
  };
  let files;
  try {
    files = await Promise.all(keys.map((key) => fetchCached(MODEL_BASE + FILES[key], report(key), abort.signal)));
  } catch (err) {
    abort.abort();
    throw err;
  }
  const bytes = Object.fromEntries(keys.map((key, i) => [key, files[i]]));
  const total = files.reduce((a, b) => a + b.length, 0);
  return { bytes, total, fromCache };
}

// --- graphs ------------------------------------------------------------------

const sameShape = (shape, want) => shape.length === want.length && want.every((d, i) => shape[i] === d);

/** Slot number in a signature name: "args_2", "serving_default_output_1_output" … */
function slotOf(name) {
  const m = /(?:args|output)_(\d+)/.exec(name);
  return m ? Number(m[1]) : null;
}

/** Map each named tensor of IO[key] to its position in the compiled model. */
function bindIO(key, model) {
  const bind = (details, wanted, what) => {
    const names = Object.keys(wanted);
    if (details.length !== names.length) {
      throw new Error(`${key} graph: expected ${names.length} ${what}, got ${details.length}`);
    }
    const bySlot = details.map((d) => slotOf(d.name));
    const named = bySlot.every((s) => s !== null) && new Set(bySlot).size === bySlot.length;
    const positions = {};
    for (const [name, { slot, shape }] of Object.entries(wanted)) {
      const at = named ? bySlot.indexOf(slot) : slot;
      if (at < 0 || !sameShape(details[at].shape, shape)) {
        throw new Error(`${key} graph: no ${what.slice(0, -1)} ${name} of shape [${shape}]`);
      }
      positions[name] = at;
    }
    return { positions, names: details.map((d) => d.name) };
  };
  const inputs = bind(model.getInputDetails(), IO[key].inputs, 'inputs');
  const outputs = bind(model.getOutputDetails(), IO[key].outputs, 'outputs');
  return { in: inputs.positions, out: outputs.positions, names: { in: inputs.names, out: outputs.names } };
}

// Which boot stage is in flight, so a failure names the culprit
// (runtime / download / compile vision webgpu / warm-up / …).
let bootStage = 'runtime';

async function compileGraph(key, acc, bytes) {
  bootStage = `compile ${key} ${acc}`;
  status(`Compiling the ${GRAPH_LABELS[key]} for ${acc === 'webgpu' ? 'WebGPU' : 'WASM'}…`);
  const start = performance.now();
  const options = { accelerator: acc };
  if (acc === 'webgpu') options.gpuOptions = { precision: precisionOf(key) };
  const model = await loadAndCompile(bytes, options);
  try {
    // In a browser without JSPI, LiteRT.js compiles a graph that WebGPU
    // cannot fully take on WASM instead, without an error: keep the backend
    // the model got, not the one asked for.
    const got = model.options?.accelerator ?? acc;
    return { model, acc: got, io: bindIO(key, model), compileMs: performance.now() - start };
  } catch (err) {
    model.delete();
    throw err;
  }
}

/**
 * Run one graph. Feeds are typed arrays (wrapped here) or Tensors (passed
 * through, still owned by the caller). Outputs come back as typed arrays,
 * except those named in `keep`, which come back as Tensors on the graph's
 * device for the caller to delete.
 */
async function runGraph(key, feeds, keep = []) {
  const { model, io } = graphs[key];
  const inputs = [];
  const owned = [];
  for (const [name, at] of Object.entries(io.in)) {
    const feed = feeds[name];
    if (feed instanceof Tensor) {
      inputs[at] = feed;
    } else {
      inputs[at] = Tensor.fromTypedArray(feed, IO[key].inputs[name].shape);
      owned.push(inputs[at]);
    }
  }
  try {
    const outputs = await model.run(inputs);
    const result = {};
    try {
      for (const [name, at] of Object.entries(io.out)) {
        if (keep.includes(name)) {
          result[name] = outputs[at];
          outputs[at] = null;
        } else {
          result[name] = await outputs[at].data();
        }
      }
      return result;
    } catch (err) {
      for (const name of keep) if (result[name] instanceof Tensor) result[name].delete();
      throw err;
    } finally {
      for (const output of outputs) output?.delete();
    }
  } finally {
    for (const input of owned) input.delete();
  }
}

const wasmLabel = () => (wasmOpts?.threads ? 'wasm' : 'wasm·1-thread');
const accLabel = (acc) => (acc === 'webgpu' ? 'webgpu' : wasmLabel());

/** Mirrors what actually loaded, so a page that silently lost threads (or
 * WebGPU) is visible at a glance. */
function updateEnv() {
  const label = (key) => (graphs[key].acc === 'webgpu' ? `webgpu ${precisionOf(key)}` : wasmLabel());
  envEl.textContent = `CLIPSeg · vision ${label('vision')} · text ${label('text')} · decoder ${label('decoder')}` +
    ` · warm-up ${warmSeconds.toFixed(1)} s`;
  envEl.style.display = 'block';
  for (const b of backendButtons) b.classList.toggle('active', b.dataset.backend === mode);
}

function disableWebGpuButton(reason) {
  for (const b of backendButtons) {
    if (b.dataset.backend === 'webgpu') {
      b.disabled = true;
      b.title = reason;
    }
  }
}

/** navigator.gpu exists in more browsers than can hand out a device
 * (headless Chromium without flags, some mobile browsers). */
async function hasWebGpuAdapter() {
  if (!isWebGPUSupported()) return false;
  try {
    return !!(await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }));
  } catch {
    return false;
  }
}

function backendsFor(position) {
  const want = {};
  for (const key of GRAPHS) {
    const pick = params.get(key);
    want[key] = position === 'wasm' || !webgpuOk || gpuRefused.has(key)
      ? 'wasm'
      : pick === 'webgpu' || pick === 'wasm' ? pick : DEFAULT_BACKENDS[key];
  }
  return want;
}

/** A WebGPU that took none of the graphs is off for this page. */
function checkWebGpu() {
  if (webgpuOk && GRAPHS.every((key) => gpuRefused.has(key))) {
    disableWebGpuButton('WebGPU could not run these graphs in this browser');
    mode = 'wasm';
  }
}

/** One throwaway run per graph: the first run after a WebGPU compile carries
 * the shader warm-up (seconds) and must never land on a user photo or in the
 * latency display. The image encoder is skipped on WASM, where there is no
 * shader to build and one run costs seconds. */
async function warmUp(keys) {
  const zeros = new Float32Array(PATCHES * VISION_DIM);
  let feats = { t3: zeros, t6: zeros, t9: zeros };
  const gpuVision = keys.includes('vision') && graphs.vision.acc === 'webgpu';
  // Warm the decoder up on the same kind of input it will get: GPU tensors
  // when the features stay resident.
  const resident = gpuVision && residentFeatures();
  try {
    if (gpuVision) {
      feats = await runGraph('vision', { image: new Float32Array(3 * SIZE * SIZE) }, resident ? FEATURES : []);
    }
    let cond = new Float32Array(COND_DIM);
    if (keys.includes('text')) {
      const { ids, eot } = assets.tokenizer.encode('a photo');
      const { hidden } = await runGraph('text', { tokens: embedTokens(ids, assets.embeddings) });
      cond = project(hidden, eot, assets.projection);
    }
    if (keys.includes('decoder')) await runGraph('decoder', { ...feats, cond });
  } finally {
    for (const name of FEATURES) if (feats[name] instanceof Tensor) feats[name].delete();
  }
}

/** Compile `keys` for the accelerators in `want` from the bytes given (or
 * the cache), falling back to WASM per graph when WebGPU fails to compile. */
async function compileGraphs(want, keys, bytes) {
  for (const key of keys) {
    const modelBytes = bytes?.[key] ?? await fetchCached(MODEL_BASE + FILES[key], (n, total, cached) => {
      status(cached
        ? 'Loading model from cache…'
        : `Downloading the ${GRAPH_LABELS[key]}… ${mb(n)}${total ? ` / ${mb(total)}` : ''} MB`);
    });
    try {
      graphs[key] = await compileGraph(key, want[key], modelBytes);
    } catch (err) {
      // WebGPU exists on paper in more browsers than it works in (mobile
      // WebKit in particular) — fall back to WASM instead of dying.
      if (want[key] !== 'webgpu') throw err;
      status(`WebGPU failed for the ${GRAPH_LABELS[key]} (${errText(err)}) — compiling on WASM…`);
      console.warn(`[clipseg] ${key} on WebGPU failed to compile:`, err);
      graphs[key] = await compileGraph(key, 'wasm', modelBytes);
    }
    if (want[key] === 'webgpu' && graphs[key].acc !== 'webgpu') gpuRefused.add(key);
  }
}

async function boot() {
  try {
    status('Loading runtime…');
    // `threads` and `jspi` are mutually exclusive in LiteRT.js, and threads
    // only work on a cross-origin-isolated page — ask for what can succeed,
    // then fall back to plain. ?threads=0 forces the single-thread build.
    const wantThreads = params.get('threads') !== '0' && window.crossOriginIsolated;
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

    webgpuOk = await hasWebGpuAdapter();
    if (!webgpuOk) disableWebGpuButton('WebGPU is not available in this browser');
    if (params.get('backend') === 'wasm' || !webgpuOk) mode = 'wasm';

    bootStage = 'download';
    const t0 = performance.now();
    const download = await downloadAll();
    const downloadMs = performance.now() - t0;

    bootStage = 'tokenizer';
    const vocab = JSON.parse(new TextDecoder().decode(download.bytes.vocab));
    assets = {
      tokenizer: new ClipTokenizer(vocab, new TextDecoder().decode(download.bytes.merges)),
      embeddings: halfArray(download.bytes.embeddings),
      projection: widen(halfArray(download.bytes.projection)),
    };

    await compileGraphs(backendsFor(mode), GRAPHS, download.bytes);
    checkWebGpu();
    // The graph bytes now live inside the runtime; a backend switch reads
    // them back from the cache.
    for (const key of GRAPHS) download.bytes[key] = null;

    bootStage = 'warm-up';
    status('Warming up (one throwaway run)…');
    let start = performance.now();
    try {
      await warmUp(GRAPHS);
    } catch (err) {
      // WebGPU can pass compile and still fail in use — move the GPU graphs
      // to WASM and warm up again.
      const onGpu = GRAPHS.filter((key) => graphs[key].acc === 'webgpu');
      if (!onGpu.length) throw err;
      status(`WebGPU failed at warm-up (${errText(err)}) — retrying on WASM…`);
      console.warn('[clipseg] WebGPU warm-up failed:', err);
      disableWebGpuButton('WebGPU failed on this device');
      mode = 'wasm';
      for (const key of onGpu) graphs[key].model.delete();
      await compileGraphs(backendsFor('wasm'), onGpu);
      bootStage = 'warm-up';
      start = performance.now();
      await warmUp(GRAPHS);
    }
    warmSeconds = (performance.now() - start) / 1000;
    updateEnv();
    ready = true;
    if (DEBUG) {
      const bootStats = {
        backends: Object.fromEntries(GRAPHS.map((key) => [key, graphs[key].acc])),
        precision: precisions(),
        wasmThreads: !!wasmOpts?.threads,
        crossOriginIsolated: window.crossOriginIsolated,
        downloadMB: +(download.total / 1e6).toFixed(1),
        fromCache: download.fromCache,
        downloadMs: Math.round(downloadMs),
        compileMs: Object.fromEntries(GRAPHS.map((key) => [key, Math.round(graphs[key].compileMs)])),
        warmupMs: Math.round(warmSeconds * 1000),
        io: Object.fromEntries(GRAPHS.map((key) => [key, graphs[key].io.names])),
        fullyAccelerated: Object.fromEntries(GRAPHS.map((key) => [key, graphs[key].model.isFullyAccelerated])),
      };
      window.__bootStats = bootStats;
      console.log('CLIPSEG_BOOT ' + JSON.stringify(bootStats));
    }
    status('Ready — choose a photo or try the example, then type what to segment.');
  } catch (err) {
    const hint = bootStage === 'download'
      ? ' The weights stream from Hugging Face: check that huggingface.co is reachable, or pass ?models=<url> to load them from elsewhere.'
      : '';
    bootFailure = `Failed to start (${bootStage}): ${errText(err).replace(/\.$/, '')}.${hint}`;
    status(bootFailure);
    console.error(`[clipseg] boot failed at stage "${bootStage}":`, err);
    return;
  }

  // ?img= runs once the page is up: an image that does not load is an input
  // failure, not a boot failure.
  const img = params.get('img');
  if (img) {
    const url = img === 'example' ? EXAMPLE_URL : img;
    let bitmap;
    try {
      bitmap = await loadBitmap(url);
    } catch (err) {
      status(`Failed: could not load ${url} (${errText(err)})`);
      return;
    }
    await setPhoto(bitmap);
    const bench = Math.max(0, Math.min(50, Number(params.get('bench')) || 0));
    if (bench) await runBench(bench);
  }
}

// --- one segmentation run ----------------------------------------------------

// Runs never overlap: a new photo, a new prompt and a backend switch take
// turns.
let queue = Promise.resolve();
function exclusive(task) {
  const run = queue.then(task);
  queue = run.catch(() => {});
  return run;
}

/** Resolves once the browser has painted what the page shows now. A hidden
 * tab runs no animation frames: there, and in a tab hidden while waiting, a
 * timeout stands in. */
function nextPaint() {
  return new Promise((resolve) => {
    const hidden = document.visibilityState === 'hidden';
    setTimeout(resolve, hidden ? 0 : 100);
    if (!hidden) requestAnimationFrame(() => setTimeout(resolve, 0));
  });
}

const pixelCanvas = document.createElement('canvas');
const pixelCtx = pixelCanvas.getContext('2d', { willReadFrequently: true });

function readPixels(source, width, height) {
  const scale = Math.min(1, MAX_SIDE / Math.max(width, height));
  const w = Math.max(1, Math.round(width * scale));
  const h = Math.max(1, Math.round(height * scale));
  if (pixelCanvas.width !== w || pixelCanvas.height !== h) {
    pixelCanvas.width = w;
    pixelCanvas.height = h;
  }
  // The canvas is shared: without this, the transparent parts of a PNG
  // would show the previous photo of the same size.
  pixelCtx.clearRect(0, 0, w, h);
  pixelCtx.imageSmoothingQuality = 'high';
  pixelCtx.drawImage(source, 0, 0, w, h);
  return { rgba: pixelCtx.getImageData(0, 0, w, h).data, width: w, height: h };
}

/** Whether the image features of `target` from the current graphs are at hand. */
const hasVision = (target) => vision?.photoId === target.id && vision.acc === graphs.vision.acc &&
  vision.resident === residentFeatures();

/** The vision graph on a photo, unless it already ran on it. */
async function encodePhoto(target, fresh = false) {
  const acc = graphs.vision.acc;
  const resident = residentFeatures();
  if (!fresh && hasVision(target)) return { ...vision, cached: true };
  const t0 = performance.now();
  const { rgba, width, height } = readPixels(target.bitmap, target.width, target.height);
  const rgb = resizeToInput(rgba, width, height);
  const input = toInputTensor(rgb);
  const t1 = performance.now();
  const feats = await runGraph('vision', { image: input }, resident ? FEATURES : []);
  // Outputs left on the GPU come back before the GPU has finished; wait for
  // it, so the time is the encoder's own.
  if (resident) await getWebGpuDevice()?.queue.onSubmittedWorkDone();
  const t2 = performance.now();
  let raw = null;
  if (KEEP_RAW) {
    raw = { input };
    for (const name of FEATURES) raw[name] = resident ? await feats[name].data() : feats[name];
  }
  dropVision();
  vision = {
    photoId: target.id, acc, resident, feats, raw, rgbDigest: DEBUG ? fnv1a(rgb) : null,
    ms: { prep: t1 - t0, vision: t2 - t1 },
  };
  return { ...vision, cached: false };
}

/** Prompt → cond [512] through the tokenizer, the text graph and the
 * projection; remembered per prompt and backend. */
async function encodePrompt(prompt, fresh = false) {
  const acc = graphs.text.acc;
  const key = `${acc}\n${prompt}`;
  const hit = !fresh && condCache.get(key);
  if (hit) return { ...hit, cached: true };
  const t0 = performance.now();
  const { ids, eot, count } = assets.tokenizer.encode(prompt);
  const t1 = performance.now();
  const tokens = embedTokens(ids, assets.embeddings);
  const t2 = performance.now();
  const { hidden } = await runGraph('text', { tokens });
  const t3 = performance.now();
  const cond = project(hidden, eot, assets.projection);
  const t4 = performance.now();
  const result = {
    prompt, ids, eot, count, cond,
    ms: { tokenize: t1 - t0, embed: t2 - t1, textGraph: t3 - t2, project: t4 - t3, text: t4 - t1 },
  };
  condCache.set(key, result);
  if (condCache.size > 64) condCache.delete(condCache.keys().next().value);
  return { ...result, cached: false };
}

const round = (v, digits = 1) => (v === null ? null : +v.toFixed(digits));

/** Photo features + prompt → logits, published for automation. */
async function segment(target, prompt, fresh = false) {
  const t0 = performance.now();
  const img = await encodePhoto(target, fresh);
  const txt = await encodePrompt(prompt, fresh);
  const t1 = performance.now();
  const { logits } = await runGraph('decoder', { ...img.feats, cond: txt.cond });
  const t2 = performance.now();
  const pixels = maskPixels(logits);
  const stats = {
    source: target.source,
    prompt,
    ids: Array.from(txt.ids.subarray(0, txt.eot + 1)),
    eot: txt.eot,
    tokens: txt.count,
    backends: Object.fromEntries(GRAPHS.map((key) => [key, graphs[key].acc])),
    precision: precisions(),
    residentFeatures: img.resident,
    wasmThreads: !!wasmOpts?.threads,
    visionCached: img.cached,
    visionMs: img.cached ? null : round(img.ms.vision),
    prepMs: img.cached ? null : round(img.ms.prep),
    textCached: txt.cached,
    tokenizeMs: txt.cached ? null : round(txt.ms.tokenize, 2),
    textMs: txt.cached ? null : round(txt.ms.text),
    textGraphMs: txt.cached ? null : round(txt.ms.textGraph),
    decoderMs: round(t2 - t1),
    totalMs: round(t2 - t0),
    maskPixels: pixels,
    maskFraction: +(pixels / logits.length).toFixed(4),
    imageSize: [target.width, target.height],
  };
  if (DEBUG) {
    stats.input = img.rgbDigest;
    stats.cond8 = Array.from(txt.cond.subarray(0, 8), (v) => +v.toFixed(6));
    window.__lastResult = {
      ...stats,
      ids: Array.from(txt.ids),
      logits,
      cond: txt.cond,
      ...(img.raw ?? {}),
    };
  }
  return { stats, logits, img, txt, decoderMs: t2 - t1 };
}

/** A result's numbers, with the backends it ran on. */
function showResult(stats) {
  const part = (label, ms, key, cached) => (cached
    ? `${label} <b>cached</b>`
    : `${label} <b>${ms.toFixed(0)} ms</b> (${accLabel(stats.backends[key])})`);
  latencyEl.innerHTML = [
    part('image', stats.visionMs, 'vision', stats.visionCached),
    part('text', stats.textMs, 'text', stats.textCached),
    part('decoder', stats.decoderMs, 'decoder', false),
  ].join(' · ');
  latencyEl.style.display = 'block';
  const pct = stats.maskFraction * 100;
  const share = pct < 0.1 ? '<0.1' : pct < 1 ? pct.toFixed(1) : pct.toFixed(0);
  summaryEl.textContent = stats.maskPixels
    ? `“${stats.prompt}” covers ${share}% of the photo.`
    : `Nothing in this photo scores above ${MASK_THRESHOLD * 100}% for “${stats.prompt}”.`;
  summaryEl.style.display = 'block';
}

/** Hide the previous result's numbers (a new photo, the camera, a failure). */
function clearResult() {
  latencyEl.style.display = 'none';
  summaryEl.style.display = 'none';
}

/** An input used before the model is ready: say so, or repeat why it never
 * will be. */
function notReady() {
  status(bootFailure ?? 'Still loading the model…');
}

const currentPrompt = () => promptEl.value.trim();

/** Run the current photo with the current prompt (vision only when no
 * prompt is typed yet) and show the result. */
async function runCurrent() {
  if (!photo) {
    status('Choose a photo or try the example first.');
    return;
  }
  const prompt = currentPrompt();
  const target = photo;
  await exclusive(async () => {
    // A newer photo is queued behind this one, or the camera took the stage.
    if (photo !== target || stream) return;
    try {
      const encode = !hasVision(target);
      status(encode || !prompt ? 'Reading the photo…' : 'Segmenting…');
      if (encode && graphs.vision.acc !== 'webgpu') {
        // The image encoder on WASM holds the main thread for seconds: let
        // the new photo and the line above paint before it starts.
        await nextPaint();
        if (photo !== target || stream) return;
      }
      if (!prompt) {
        await encodePhoto(target);
        if (photo === target) status('Type what to segment, then press Enter.');
        return;
      }
      const { stats, logits } = await segment(target, prompt);
      if (photo !== target) return; // a newer photo is queued
      // The finished result replaces the view in one step, over its own photo.
      shown = { photo: target, prompt, logits, stats, prob: probabilityCanvas(logits) };
      draw();
      showResult(stats);
      const cut = stats.tokens > MAX_TOKENS ? ` (prompt cut at ${MAX_TOKENS} tokens)` : '';
      status(`Done${cut}. Type something else to segment the same photo.`);
      console.log('CLIPSEG_STATS ' + JSON.stringify(stats));
    } catch (err) {
      if (photo === target) {
        // No earlier mask or numbers stay on screen next to a failure.
        shown = null;
        draw();
        clearResult();
      }
      status(`Failed: ${errText(err)}`);
      console.error('[clipseg] segmentation failed:', err);
    }
  });
}

/** A new photo from any input: it replaces the camera and the photo before
 * it, then runs with the current words. */
async function setPhoto(bitmap, source = 'image') {
  if (!ready) {
    bitmap.close();
    notReady();
    return;
  }
  cancelCameraStart(); // a photo asked for while the camera opens wins
  stopCamera();
  // A run reads its photo's pixels as soon as it starts, and a queued run
  // for an older photo skips it, so the old bitmap can go now.
  photo?.bitmap.close();
  photo = { bitmap, width: bitmap.width, height: bitmap.height, id: ++photoCount, source };
  placeholderEl.style.display = 'none';
  clearResult();
  draw();
  await runCurrent();
}

const median = (values) => {
  const s = values.slice().sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
};

/** ?bench=N: N timed passes over the current photo and prompt, each running
 * every graph afresh (no cached photo features or prompt). */
async function runBench(n) {
  const prompt = currentPrompt() || DEFAULT_PROMPT;
  await exclusive(async () => {
    const runs = [];
    try {
      for (let i = 0; i < n; i++) {
        status(`Benchmark run ${i + 1} / ${n}…`);
        const { stats } = await segment(photo, prompt, true);
        runs.push(stats);
      }
    } catch (err) {
      status(`Failed: ${errText(err)}`);
      console.error('[clipseg] benchmark failed:', err);
      return;
    }
    const pick = (field) => runs.map((r) => r[field]);
    const bench = {
      prompt,
      runs: n,
      backends: runs[0].backends,
      precision: runs[0].precision,
      residentFeatures: runs[0].residentFeatures,
      wasmThreads: runs[0].wasmThreads,
      visibility: document.visibilityState,
      median: {
        visionMs: round(median(pick('visionMs'))),
        textMs: round(median(pick('textMs'))),
        textGraphMs: round(median(pick('textGraphMs'))),
        decoderMs: round(median(pick('decoderMs'))),
        tokenizeMs: round(median(pick('tokenizeMs')), 2),
        prepMs: round(median(pick('prepMs'))),
      },
      visionMs: pick('visionMs'),
      textMs: pick('textMs'),
      decoderMs: pick('decoderMs'),
      maskPixels: pick('maskPixels'),
    };
    window.__lastBench = bench;
    console.log('CLIPSEG_BENCH ' + JSON.stringify(bench));
    status(`Benchmark done: image ${bench.median.visionMs} ms · text ${bench.median.textMs} ms · decoder ${bench.median.decoderMs} ms (median of ${n}).`);
  });
}

/** Fetch and decode an image URL (the example, ?img=). Callers name the URL
 * in the failure line. */
async function loadBitmap(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return createImageBitmap(await response.blob());
}

// --- drawing -------------------------------------------------------------------

const viewCtx = viewEl.getContext('2d');
const overlayCanvas = document.createElement('canvas');
const overlayCtx = overlayCanvas.getContext('2d', { willReadFrequently: true });

/** sigmoid(logits) as a 352×352 grayscale canvas, ready to be stretched over
 * the photo (the model saw the photo squashed to a square). */
function probabilityCanvas(logits) {
  const canvas = document.createElement('canvas');
  canvas.width = SIZE;
  canvas.height = SIZE;
  const ctx = canvas.getContext('2d');
  const image = ctx.createImageData(SIZE, SIZE);
  const px = image.data;
  for (let i = 0; i < logits.length; i++) {
    const p = Math.round(255 / (1 + Math.exp(-logits[i])));
    px[i * 4] = p;
    px[i * 4 + 1] = p;
    px[i * 4 + 2] = p;
    px[i * 4 + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
  return canvas;
}

/** Where a w×h picture lands when contain-fitted into the stage. */
function contentRect(stageW, stageH, w, h) {
  const s = Math.min(stageW / w, stageH / h);
  const rw = Math.max(1, Math.round(w * s));
  const rh = Math.max(1, Math.round(h * s));
  return { x: Math.round((stageW - rw) / 2), y: Math.round((stageH - rh) / 2), w: rw, h: rh };
}

// Heatmap colors: transparent dark blue → cyan → yellow as probability rises.
const HEAT = [[20, 40, 140], [40, 170, 255], [120, 255, 200], [255, 240, 60]];
function heatColor(p) {
  const x = p * (HEAT.length - 1);
  const i = Math.min(HEAT.length - 2, Math.floor(x));
  const f = x - i;
  return HEAT[i].map((c, k) => c + (HEAT[i + 1][k] - c) * f);
}

/** The probability map upsampled to the displayed photo size, then colored:
 * 'mask' tints p > 0.5, outlines it and dims the rest; 'heat' colors every
 * pixel by its probability. */
function drawOverlay(r) {
  if (overlayCanvas.width !== r.w || overlayCanvas.height !== r.h) {
    overlayCanvas.width = r.w;
    overlayCanvas.height = r.h;
  }
  overlayCtx.imageSmoothingEnabled = true;
  overlayCtx.imageSmoothingQuality = 'high';
  overlayCtx.clearRect(0, 0, r.w, r.h);
  overlayCtx.drawImage(shown.prob, 0, 0, r.w, r.h);
  const image = overlayCtx.getImageData(0, 0, r.w, r.h);
  const px = image.data;
  const n = r.w * r.h;
  if (view === 'heat') {
    for (let i = 0; i < n; i++) {
      const p = px[i * 4] / 255;
      const [cr, cg, cb] = heatColor(p);
      px[i * 4] = cr;
      px[i * 4 + 1] = cg;
      px[i * 4 + 2] = cb;
      px[i * 4 + 3] = Math.round(60 + 150 * p);
    }
  } else {
    const level = Math.round(255 * MASK_THRESHOLD);
    const inside = new Uint8Array(n);
    for (let i = 0; i < n; i++) inside[i] = px[i * 4] >= level ? 1 : 0;
    for (let y = 0; y < r.h; y++) {
      for (let x = 0; x < r.w; x++) {
        const i = y * r.w + x;
        const m = inside[i];
        const edge = m && (
          (x > 0 && !inside[i - 1]) || (x < r.w - 1 && !inside[i + 1]) ||
          (y > 0 && !inside[i - r.w]) || (y < r.h - 1 && !inside[i + r.w]));
        const o = i * 4;
        if (edge) {
          px[o] = 255; px[o + 1] = 255; px[o + 2] = 255; px[o + 3] = 255;
        } else if (m) {
          px[o] = 124; px[o + 1] = 196; px[o + 2] = 255; px[o + 3] = 120;
        } else {
          px[o] = 6; px[o + 1] = 8; px[o + 2] = 10; px[o + 3] = 150;
        }
      }
    }
  }
  overlayCtx.putImageData(image, 0, 0);
  viewCtx.drawImage(overlayCanvas, r.x, r.y);
}

function draw() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const W = Math.round(stageEl.clientWidth * dpr);
  const H = Math.round(stageEl.clientHeight * dpr);
  if (viewEl.width !== W || viewEl.height !== H) {
    viewEl.width = W;
    viewEl.height = H;
  }
  viewCtx.clearRect(0, 0, W, H);
  if (!photo || stream) return;
  const r = contentRect(W, H, photo.width, photo.height);
  viewCtx.imageSmoothingQuality = 'high';
  viewCtx.drawImage(photo.bitmap, r.x, r.y, r.w, r.h);
  if (shown && shown.photo === photo) drawOverlay(r);
}

new ResizeObserver(() => draw()).observe(stageEl);

for (const button of viewButtons) {
  button.addEventListener('click', () => {
    view = button.dataset.view;
    for (const b of viewButtons) b.classList.toggle('active', b === button);
    draw();
  });
}

// --- inputs: file, example, drop, paste, camera, prompt ---------------------

/** A chosen, dropped or pasted file. One the browser cannot decode (not an
 * image, cut short) ends in a status line, not an unhandled rejection. */
async function runFile(file) {
  let bitmap;
  try {
    bitmap = await createImageBitmap(file);
  } catch (err) {
    status(`Failed: could not read ${file.name || 'the image'} as an image (${errText(err)})`);
    return;
  }
  await setPhoto(bitmap);
}

fileEl.addEventListener('change', async () => {
  const file = fileEl.files?.[0];
  fileEl.value = '';
  if (file) await runFile(file);
});

exampleBtn.addEventListener('click', async () => {
  if (!ready) {
    notReady();
    return;
  }
  if (!currentPrompt()) promptEl.value = DEFAULT_PROMPT;
  let bitmap;
  try {
    bitmap = await loadBitmap(EXAMPLE_URL);
  } catch (err) {
    status(`Failed: could not load the example photo (${errText(err)})`);
    return;
  }
  await setPhoto(bitmap);
});

promptForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!ready) {
    notReady();
    return;
  }
  if (!currentPrompt()) {
    status('Type what to segment first, e.g. “a dog”.');
    return;
  }
  // With the camera on, the words apply to what it shows now.
  if (stream) {
    await capture();
    return;
  }
  await runCurrent();
});

// Only a drag that carries files is a photo: dragged text or a link is left
// to the browser (into the prompt field, for one).
const carriesFiles = (event) => [...(event.dataTransfer?.types ?? [])].includes('Files');
window.addEventListener('dragover', (event) => {
  if (!carriesFiles(event)) return;
  event.preventDefault();
  dropOverlay.style.display = 'flex';
});
window.addEventListener('dragleave', (event) => {
  if (event.relatedTarget === null) dropOverlay.style.display = 'none';
});
window.addEventListener('drop', async (event) => {
  if (!carriesFiles(event)) return;
  event.preventDefault();
  dropOverlay.style.display = 'none';
  const file = event.dataTransfer?.files?.[0];
  if (file) await runFile(file);
});

window.addEventListener('paste', async (event) => {
  // Text pasted into the prompt field stays text, even when the clipboard
  // also holds an image (copying from a web page can put both on it).
  if (event.target === promptEl && [...(event.clipboardData?.types ?? [])].includes('text/plain')) return;
  const item = [...(event.clipboardData?.items ?? [])].find((entry) =>
    entry.type.startsWith('image/'),
  );
  const file = item?.getAsFile();
  if (file) {
    event.preventDefault();
    await runFile(file);
  }
});

let stream = null; // the camera on the stage
let camOpening = null; // { cancelled } from the click until the video is on the stage

const CAMERA_ON = 'Camera on — press Capture to segment the current frame (one frame per Capture).';
const CAMERA_HIDDEN = 'Camera off — the tab was hidden.';

/** Give up a camera start that has not reached the stage: a photo asked for
 * meanwhile, or a hidden tab, wins. `why` replaces the status line. */
function cancelCameraStart(why = null) {
  if (!camOpening || camOpening.cancelled) return;
  camOpening.cancelled = true;
  if (why) status(why);
}

camBtn.addEventListener('click', async () => {
  if (camOpening) return; // the button is disabled while opening; this is a synthetic click
  if (stream) {
    stopCamera('Camera off.');
    return;
  }
  if (!ready) {
    notReady();
    return;
  }
  const opening = { cancelled: false };
  camOpening = opening;
  camBtn.disabled = true;
  status('Opening the camera…');
  try {
    let s;
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('not available on this page');
      s = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' },
        audio: false,
      });
    } catch (err) {
      if (!opening.cancelled) status(`Camera: ${errText(err)}`);
      return;
    }
    // In the queue: a run still in flight shows its result first, then the
    // video takes the stage.
    await exclusive(async () => {
      if (opening.cancelled || document.visibilityState === 'hidden') {
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
      videoEl.style.display = 'block';
      placeholderEl.style.display = 'none';
      shutterBtn.style.display = 'inline-block';
      camBtn.textContent = 'Stop camera';
      clearResult();
      draw();
      status(CAMERA_ON);
    });
  } finally {
    camOpening = null;
    camBtn.disabled = false;
  }
  if (stream && videoEl.srcObject === stream) {
    try {
      await videoEl.play();
    } catch { /* autoplay attribute covers it */ }
  }
});

/** Turn the camera off. With `why`, that is the status line and the photo
 * from before the camera comes back with its result. */
function stopCamera(why = null) {
  const s = stream;
  if (!s) return;
  stream = null;
  s.getTracks().forEach((track) => track.stop());
  videoEl.srcObject = null;
  videoEl.style.display = 'none';
  shutterBtn.style.display = 'none';
  camBtn.textContent = 'Use camera';
  if (!photo) placeholderEl.style.display = 'flex';
  draw();
  if (why) {
    status(why);
    if (shown && shown.photo === photo) showResult(shown.stats);
  }
}

/** The camera's current frame becomes the photo (Capture, or Segment while
 * the camera is on). */
async function capture() {
  if (!stream) return;
  if (!videoEl.videoWidth) {
    status('The camera has no picture yet — try again in a moment.');
    return;
  }
  // Snapshot to a bitmap so the frame survives stopCamera() and backend
  // switches can re-run it.
  let frame;
  try {
    frame = await createImageBitmap(videoEl);
  } catch (err) {
    stopCamera();
    status(`Failed: could not capture a camera frame (${errText(err)})`);
    return;
  }
  await setPhoto(frame, 'camera');
}

shutterBtn.addEventListener('click', capture);

// A hidden tab or a page being left gives the camera back: the camera light
// goes off, and nobody is shown a frame they cannot see.
function releaseCamera() {
  cancelCameraStart(CAMERA_HIDDEN);
  stopCamera(CAMERA_HIDDEN);
}
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden') releaseCamera();
});
window.addEventListener('pagehide', releaseCamera);

// --- backend switch (webgpu defaults / all wasm) ------------------------------

for (const button of backendButtons) {
  button.addEventListener('click', async () => {
    const target = button.dataset.backend;
    if (!ready || target === mode || button.disabled) return;
    const states = backendButtons.map((b) => b.disabled);
    for (const b of backendButtons) b.disabled = true;
    const photoAtClick = photo;
    try {
      await exclusive(async () => {
        const want = backendsFor(target);
        const keys = GRAPHS.filter((key) => graphs[key].acc !== want[key]);
        const prev = Object.fromEntries(keys.map((key) => [key, graphs[key]]));
        try {
          await compileGraphs(want, keys);
          status('Warming up (one throwaway run)…');
          const start = performance.now();
          await warmUp(keys);
          warmSeconds = (performance.now() - start) / 1000;
        } catch (err) {
          for (const key of keys) {
            if (graphs[key] !== prev[key]) graphs[key].model.delete();
            graphs[key] = prev[key];
          }
          throw err;
        }
        // Features made by an old graph go before the graph does.
        dropVision();
        for (const key of keys) prev[key].model.delete();
        mode = target;
        condCache.clear();
        updateEnv();
      });
      if (stream) {
        status(CAMERA_ON);
      } else if (photo && photo === photoAtClick) {
        // With no words typed yet, this only reads the photo. A photo chosen
        // during the switch already ran on the new backends.
        await runCurrent();
      } else if (!photo) {
        status('Ready — choose a photo or try the example, then type what to segment.');
      }
    } catch (err) {
      status(`Backend switch failed: ${errText(err)}`);
    } finally {
      backendButtons.forEach((b, i) => { b.disabled = states[i]; });
      checkWebGpu();
      updateEnv();
    }
  });
}

boot();
