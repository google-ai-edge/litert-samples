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
 * Photo / webcam → object boxes and labels, fully client-side.
 *
 * RF-DETR-Nano is a two-stage DETR split into two graphs with a small host
 * step between them (see the model card):
 *   image → square resize to 384×384 (Pillow-exact bilinear) → ImageNet
 *   mean/std → NCHW float32
 *   → Graph A (backbone + encoder + proposal heads): enc_class [1,576,91],
 *     enc_coord [1,576,4], memory [1,576,256]
 *   → host: top-300 proposals by max class logit → gather enc_coord
 *     → refpoint_ts [1,300,4]
 *   → Graph B (two-stage combine + decoder + heads): boxes [1,300,4] cxcywh
 *     in [0,1], logits [1,300,91] (index = COCO category id)
 *   → host: sigmoid, score > 0.45, per-class NMS at IoU 0.6 → boxes + labels.
 * Tensors are bound by shape from getInputDetails() / getOutputDetails();
 * the converter's slot order is not guaranteed.
 *
 * Every photo run logs one RFDETR_STATS JSON line to the console (the camera
 * one per second). Debug URL params: ?img=<url>|example (detect on that image
 *   at boot) · &repeat=N (run it N times) · ?backend=wasm (graph A on WASM)
 *   · ?a=webgpu|wasm&b=webgpu|wasm (each graph's backend, to reproduce the
 *   comparison: graph B on WebGPU does not match the Python reference, numbers
 *   in the README) · ?raw=1 (keep the raw model outputs in window.__lastRaw)
 *   · ?threads=0 (single-thread WASM runtime) · ?models=<base url> (fetch the
 *   weights from somewhere other than Hugging Face). With ?img= or ?raw=1 the
 *   page also logs RFDETR_BOOT, adds input / output digests to the photo
 *   stats and keeps the last photo result in window.__lastResult.
 */
import {
  Tensor,
  isWebGPUSupported,
  loadAndCompile,
  loadLiteRt,
} from '@litertjs/core';
import { colorOf, labelOf } from './coco.js';
import {
  HID,
  NCLS,
  NPROP,
  NQ,
  SCORE_THRESH,
  SIZE,
  decode,
  fnv1a,
  resizeToInput,
  selectQueries,
  toInputTensor,
} from './host.js';

const params = new URLSearchParams(location.search);

// Weights stream from the model card's repo on Hugging Face and are cached
// with the Cache API after the first visit. ?models=<base url> points the
// page at another copy (a local directory, a mirror) — same file names.
const DEFAULT_MODEL_BASE = 'https://huggingface.co/litert-community/RF-DETR-Nano-LiteRT/resolve/main/';
const MODEL_BASE = (params.get('models') ?? DEFAULT_MODEL_BASE).replace(/\/?$/, '/');
const FILES = { A: 'rfdetr_graphA_fp16.tflite', B: 'rfdetr_graphB_fp16.tflite' };
const CACHE_NAME = 'rfdetr-demo-v1';
// Graph B (the decoder) runs on WASM even where WebGPU works. Checked against
// the Python LiteRT reference on five photos (same input pixels): graph A on
// WebGPU matches it (final logits max |Δ| ≤ 3e-4, same boxes), graph B on
// WebGPU does not (logits max |Δ| 6.4–8.5; extra boxes, moved boxes or
// scores off by more than 0.05 on every photo), with the default or fp32
// precision alike. The backend switch moves graph A only.
const B_BACKEND = 'wasm';
// The LiteRT.js WASM runtime is served from litert-wasm/ at the site root
// (vite.config.js copies it there from node_modules). Resolve against this
// module's own URL: in dev it is <root>/rfdetr/main.js, in the build
// <root>/assets/<hash>.js — one level below the runtime dir either way.
const WASM_DIR = new URL(/* @vite-ignore */ '../litert-wasm/', import.meta.url).href;
const EXAMPLE_URL = new URL('./example.jpg', import.meta.url).href;
// Sources larger than this (beyond 12 MP phone photos) are first scaled by
// the browser to bound memory; up to it the page resizes the exact pixels.
const MAX_SIDE = 4096;
// The automation hooks (tools/check.mjs, the comparison scripts) run only
// when asked for, so a normal visit computes nothing it does not show.
const DEBUG = params.has('img') || params.get('raw') === '1';
const KEEP_RAW = params.get('raw') === '1';

// Tensor shapes per graph, used to bind inputs/outputs by shape.
const IO = {
  A: {
    inputs: { image: [1, 3, SIZE, SIZE] },
    outputs: { encClass: [1, NPROP, NCLS], encCoord: [1, NPROP, 4], memory: [1, NPROP, HID] },
  },
  B: {
    inputs: { memory: [1, NPROP, HID], refpoints: [1, NQ, 4] },
    outputs: { boxes: [1, NQ, 4], logits: [1, NQ, NCLS] },
  },
};

const statusEl = document.getElementById('status');
const latencyEl = document.getElementById('latency');
const summaryEl = document.getElementById('summary');
const envEl = document.getElementById('env');
const backendButtons = [...document.querySelectorAll('#backend-switch button')];
const fileEl = document.getElementById('file');
const exampleBtn = document.getElementById('example');
const camBtn = document.getElementById('cam');
const stageEl = document.getElementById('stage');
const videoEl = document.getElementById('video');
const viewEl = document.getElementById('view');
const placeholderEl = document.getElementById('placeholder');
const dropOverlay = document.getElementById('drop-overlay');

const graphs = { A: null, B: null }; // { model, acc, io }
const modelBytes = {};
let wasmOpts = null; // which loadLiteRt attempt succeeded
let ready = false;
let warmSeconds = 0;
let shown = null; // what the stage shows: { kind, bitmap?, width, height, detections }
let lastImage = null; // last photo, re-run on backend switch
let imageRequests = 0; // photos asked for so far (a run may wait behind a backend switch)
let bootFailure = null; // the status line of a failed boot, shown again when an input is used

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

// --- model loading --------------------------------------------------------

/** One model file, from the Cache API when it holds it, else from the
 * network (then stored). Where the Cache API is missing or fails (storage
 * blocked, quota), the page still runs and downloads on every visit. */
async function fetchCached(url, onProgress) {
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
    console.warn(`[rfdetr] Cache API unavailable (${errText(err)}); downloading without it.`);
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
  const total = Number(response.headers.get('Content-Length')) || 0;
  onProgress?.(0, total, false);
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress?.(received, total, false);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  if (cache) {
    try {
      await cache.put(url, new Response(bytes));
    } catch (err) {
      console.warn(`[rfdetr] could not cache ${url} (${errText(err)}); the next visit downloads it again.`);
    }
  }
  return bytes;
}

async function downloadModels() {
  const received = { A: 0, B: 0 };
  const sizes = { A: 0, B: 0 }; // from Content-Length; 0 = not known (yet)
  let fromCache = true;
  const report = (key) => (n, size, cached) => {
    // Once one file has failed, the other may still be streaming: its
    // progress must not cover the failure line.
    if (bootFailure) return;
    received[key] = n;
    sizes[key] = size;
    fromCache &&= cached;
    if (fromCache) {
      status('Loading model from cache…');
      return;
    }
    const mb = (bytes) => (bytes / 1e6).toFixed(0);
    const done = received.A + received.B;
    const total = sizes.A && sizes.B ? sizes.A + sizes.B : 0;
    if (total) {
      status(`Downloading model (one-time)… ${mb(done)} / ${mb(total)} MB`, Math.min(done / total, 1));
    } else {
      status(`Downloading model (one-time)… ${mb(done)} MB`);
    }
  };
  const [a, b] = await Promise.all(['A', 'B'].map((key) => fetchCached(MODEL_BASE + FILES[key], report(key))));
  modelBytes.A = a;
  modelBytes.B = b;
  return { bytes: a.length + b.length, fromCache };
}

const sameShape = (shape, want) => shape.length === want.length && want.every((d, i) => shape[i] === d);

/** Map each named tensor of IO[key] to its slot in the compiled model. */
function bindIO(key, model) {
  const bind = (details, wanted, what) => {
    if (details.length !== Object.keys(wanted).length) {
      throw new Error(`graph ${key}: expected ${Object.keys(wanted).length} ${what}, got ${details.length}`);
    }
    const slots = {};
    for (const [name, shape] of Object.entries(wanted)) {
      slots[name] = details.findIndex((d) => sameShape(d.shape, shape));
      if (slots[name] < 0) throw new Error(`graph ${key}: no ${what.slice(0, -1)} of shape [${shape}]`);
    }
    return slots;
  };
  return {
    in: bind(model.getInputDetails(), IO[key].inputs, 'inputs'),
    out: bind(model.getOutputDetails(), IO[key].outputs, 'outputs'),
  };
}

// Which boot stage is in flight, so a failure names the culprit
// (runtime / download / compile A webgpu / warm-up / …).
let bootStage = 'runtime';

async function compileGraph(key, acc) {
  bootStage = `compile ${key} ${acc}`;
  status(`Compiling graph ${key} for ${acc === 'webgpu' ? 'WebGPU' : 'WASM'}…`);
  const start = performance.now();
  const model = await loadAndCompile(modelBytes[key], { accelerator: acc });
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

async function runGraph(key, feeds) {
  const { model, io } = graphs[key];
  const inputs = [];
  for (const [name, slot] of Object.entries(io.in)) {
    inputs[slot] = Tensor.fromTypedArray(feeds[name], IO[key].inputs[name]);
  }
  try {
    const outputs = await model.run(inputs);
    try {
      const result = {};
      for (const [name, slot] of Object.entries(io.out)) result[name] = await outputs[slot].data();
      return result;
    } finally {
      for (const output of outputs) output.delete();
    }
  } finally {
    for (const input of inputs) input.delete();
  }
}

/** Both graphs and the host steps on one RGBA frame. */
async function detectPixels(rgba, width, height) {
  const t0 = performance.now();
  const rgb = resizeToInput(rgba, width, height);
  const t1 = performance.now();
  const a = await runGraph('A', { image: toInputTensor(rgb) });
  const t2 = performance.now();
  const { top, refpoints } = selectQueries(a.encClass, a.encCoord);
  const t3 = performance.now();
  const b = await runGraph('B', { memory: a.memory, refpoints });
  const t4 = performance.now();
  const detections = decode(b.boxes, b.logits);
  const t5 = performance.now();
  return {
    rgb, a, top, b, detections,
    ms: { prep: t1 - t0, a: t2 - t1, topk: t3 - t2, b: t4 - t3, post: t5 - t4 },
  };
}

/** One throwaway run through both graphs: the first run after compile
 * carries shader/kernel warm-up (~5 s on WebGPU) and must never land on a
 * user photo or in the latency display. */
async function warmUp() {
  const gray = new Uint8ClampedArray(SIZE * SIZE * 4).fill(128);
  await detectPixels(gray, SIZE, SIZE);
}

const wasmLabel = () => (wasmOpts?.threads ? 'wasm' : 'wasm·1-thread');
const accLabel = (acc) => (acc === 'webgpu' ? 'webgpu' : wasmLabel());

/** Mirrors what actually loaded, so a page that silently lost threads (or
 * WebGPU) is visible at a glance. */
function updateEnv() {
  envEl.textContent = `RF-DETR-Nano fp16 · A ${accLabel(graphs.A.acc)} · B ${accLabel(graphs.B.acc)}` +
    ` · warm-up ${warmSeconds.toFixed(1)} s`;
  envEl.style.display = 'block';
  for (const b of backendButtons) b.classList.toggle('active', b.dataset.backend === graphs.A.acc);
}

function disableWebGpuButton(reason) {
  for (const b of backendButtons) {
    if (b.dataset.backend === 'webgpu') {
      b.disabled = true;
      b.title = reason;
    }
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

    const gpu = isWebGPUSupported();
    const pick = (v) => (v === 'webgpu' || v === 'wasm' ? v : null);
    const want = {
      A: pick(params.get('a')) ?? (params.get('backend') === 'wasm' ? 'wasm' : 'webgpu'),
      B: pick(params.get('b')) ?? B_BACKEND,
    };
    if (!gpu) {
      want.A = 'wasm';
      want.B = 'wasm';
      disableWebGpuButton('WebGPU is not available in this browser');
    }

    bootStage = 'download';
    const t0 = performance.now();
    const download = await downloadModels();
    const downloadMs = performance.now() - t0;

    for (const key of ['A', 'B']) {
      try {
        graphs[key] = await compileGraph(key, want[key]);
        if (key === 'A' && want.A === 'webgpu' && graphs.A.acc !== 'webgpu') {
          disableWebGpuButton('WebGPU could not take all of graph A in this browser');
        }
      } catch (err) {
        // WebGPU exists on paper in more browsers than it works in (mobile
        // WebKit in particular) — fall back to WASM instead of dying.
        if (want[key] !== 'webgpu') throw err;
        status(`WebGPU failed for graph ${key} (${errText(err)}) — compiling on WASM…`);
        if (key === 'A') disableWebGpuButton('WebGPU failed on this device');
        graphs[key] = await compileGraph(key, 'wasm');
      }
    }

    bootStage = 'warm-up';
    status('Warming up (one throwaway run)…');
    let start = performance.now();
    try {
      await warmUp();
    } catch (err) {
      // WebGPU can pass compile and still fail in use — move the GPU graphs
      // to WASM and warm up again.
      const onGpu = ['A', 'B'].filter((key) => graphs[key].acc === 'webgpu');
      if (!onGpu.length) throw err;
      status(`WebGPU failed at warm-up (${errText(err)}) — retrying on WASM…`);
      disableWebGpuButton('WebGPU failed on this device');
      for (const key of onGpu) {
        graphs[key].model.delete();
        graphs[key] = await compileGraph(key, 'wasm');
      }
      bootStage = 'warm-up';
      start = performance.now();
      await warmUp();
    }
    warmSeconds = (performance.now() - start) / 1000;
    updateEnv();
    ready = true;
    if (DEBUG) {
      const bootStats = {
        backends: { A: graphs.A.acc, B: graphs.B.acc },
        wasmThreads: !!wasmOpts?.threads,
        crossOriginIsolated: window.crossOriginIsolated,
        downloadMB: +(download.bytes / 1e6).toFixed(1),
        fromCache: download.fromCache,
        downloadMs: Math.round(downloadMs),
        compileMs: { A: Math.round(graphs.A.compileMs), B: Math.round(graphs.B.compileMs) },
        warmupMs: Math.round(warmSeconds * 1000),
      };
      window.__bootStats = bootStats;
      console.log('RFDETR_BOOT ' + JSON.stringify(bootStats));
    }
    status('Ready — choose a photo, try the example, or use the camera.');
  } catch (err) {
    const hint = bootStage === 'download'
      ? ' The weights stream from Hugging Face: check that huggingface.co is reachable, or pass ?models=<url> to load them from elsewhere.'
      : '';
    bootFailure = `Failed to start (${bootStage}): ${errText(err).replace(/\.$/, '')}.${hint}`;
    status(bootFailure);
    console.error(`[rfdetr] boot failed at stage "${bootStage}":`, err);
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
    const repeat = Math.max(1, Math.min(50, Number(params.get('repeat')) || 1));
    for (let i = 0; i < repeat; i++) await runImage(bitmap);
  }
}

// --- one detection run ----------------------------------------------------

// Runs never overlap: the camera loop, a dropped photo and a backend switch
// take turns.
let queue = Promise.resolve();
function exclusive(task) {
  const run = queue.then(task);
  queue = run.catch(() => {});
  return run;
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
  pixelCtx.imageSmoothingQuality = 'high';
  pixelCtx.drawImage(source, 0, 0, w, h);
  return { rgba: pixelCtx.getImageData(0, 0, w, h).data, width: w, height: h };
}

function sum(array) {
  let s = 0;
  for (let i = 0; i < array.length; i++) s += array[i];
  return s;
}

function absMax(array) {
  let m = 0;
  for (let i = 0; i < array.length; i++) m = Math.max(m, Math.abs(array[i]));
  return m;
}

const round = (v, digits = 4) => +v.toFixed(digits);

/** Detect on a frame (a photo, or the camera's current frame). */
async function runOnSource(source, width, height) {
  const { rgba, width: w, height: h } = readPixels(source, width, height);
  const r = await detectPixels(rgba, w, h);
  const detections = r.detections.map((d) => ({
    cls: d.cls, label: labelOf(d.cls), score: d.score, xyxy: d.xyxy,
  }));
  return { ...r, detections, width, height, hostMs: r.ms.prep + r.ms.topk + r.ms.post };
}

/** The RFDETR_STATS console line, read by tools/check.mjs and the comparison
 * scripts. The input / output digests and window.__lastResult only exist
 * with ?img= or ?raw=1, and never for camera frames. */
function logStats(r, source, extra = {}) {
  const stats = {
    source,
    backends: { A: graphs.A.acc, B: graphs.B.acc },
    fullyAccelerated: { A: graphs.A.model.isFullyAccelerated, B: graphs.B.model.isFullyAccelerated },
    wasmThreads: !!wasmOpts?.threads,
    aMs: round(r.ms.a, 1),
    bMs: round(r.ms.b, 1),
    hostMs: round(r.hostMs, 1),
    prepMs: round(r.ms.prep, 1),
    topkMs: round(r.ms.topk, 1),
    postMs: round(r.ms.post, 1),
    imageSize: [r.width, r.height],
    detections: r.detections.length,
    top5: r.detections.slice(0, 5).map((d) => ({
      cls: d.cls, label: d.label, score: round(d.score), xyxy: d.xyxy.map((v) => round(v)),
    })),
    ...extra,
  };
  if (DEBUG && source === 'image') {
    stats.input = fnv1a(r.rgb);
    stats.raw = {
      top300First10: Array.from(r.top.subarray(0, 10)),
      boxesSum: round(sum(r.b.boxes), 6),
      logitsSum: round(sum(r.b.logits), 4),
      logitsAbsMax: round(absMax(r.b.logits), 4),
      memorySum: round(sum(r.a.memory), 4),
    };
    window.__lastResult = { ...stats, detections: r.detections };
    if (KEEP_RAW) {
      window.__lastRaw = { rgb: r.rgb, top: r.top, boxes: r.b.boxes, logits: r.b.logits };
    }
  }
  console.log('RFDETR_STATS ' + JSON.stringify(stats));
}

/** Hide the previous run's numbers (a new photo, the camera, a failure). */
function clearResult() {
  latencyEl.style.display = 'none';
  summaryEl.style.display = 'none';
}

function showLatency(ms, hostMs, fps = null) {
  const lat = (label, v, acc) => `${label} <b>${v.toFixed(0)} ms</b> (${accLabel(acc)})`;
  latencyEl.innerHTML =
    (fps !== null ? `camera <b>${fps.toFixed(1)} fps</b> · ` : '') +
    `${lat('A', ms.a, graphs.A.acc)} · ${lat('B', ms.b, graphs.B.acc)} · host <b>${hostMs.toFixed(0)} ms</b>`;
  latencyEl.style.display = 'block';
}

function showSummary(detections) {
  const counts = new Map();
  for (const d of detections) counts.set(d.label, (counts.get(d.label) ?? 0) + 1);
  const parts = [...counts].sort((a, b) => b[1] - a[1]).map(([label, n]) => (n > 1 ? `${label} ×${n}` : label));
  summaryEl.textContent = detections.length
    ? `${detections.length} object${detections.length > 1 ? 's' : ''}: ${parts.join(', ')}`
    : `No objects above ${Math.round(SCORE_THRESH * 100)}% confidence.`;
  summaryEl.style.display = 'block';
}

/** An input used before the model is ready: say so, or repeat why it never
 * will be. */
function notReady() {
  status(bootFailure ?? 'Still loading the model…');
}

async function runImage(bitmap) {
  if (!ready) {
    notReady();
    return;
  }
  imageRequests++;
  cancelCameraStart(null); // a photo asked for while the camera opens wins
  await stopCamera(false);
  await exclusive(async () => {
    lastImage = bitmap;
    // The photo shows at once; its boxes are published only over this same
    // photo, so a run never writes into a view that replaced it.
    const pending = { kind: 'image', bitmap, width: bitmap.width, height: bitmap.height, detections: [] };
    shown = pending;
    clearResult();
    placeholderEl.style.display = 'none';
    draw();
    status('Detecting…');
    try {
      const r = await runOnSource(bitmap, bitmap.width, bitmap.height);
      if (shown === pending) {
        shown = { ...pending, detections: r.detections };
        draw();
        showLatency(r.ms, r.hostMs);
        showSummary(r.detections);
      }
      status('Done.');
      logStats(r, 'image');
    } catch (err) {
      status(`Failed: ${errText(err)}`);
      console.error('[rfdetr] detection failed:', err);
    }
  });
}

/** Fetch and decode an image URL (the example, ?img=). Callers name the URL
 * in the failure line. */
async function loadBitmap(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return createImageBitmap(await response.blob());
}

// --- drawing --------------------------------------------------------------

const viewCtx = viewEl.getContext('2d');

/** Where a w×h picture lands when contain-fitted into the stage (the same
 * rule as the video's object-fit: contain). */
function contentRect(stageW, stageH, w, h) {
  const s = Math.min(stageW / w, stageH / h);
  return { x: (stageW - w * s) / 2, y: (stageH - h * s) / 2, w: w * s, h: h * s };
}

const clamp01 = (v) => Math.min(1, Math.max(0, v));

function draw() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const W = Math.round(stageEl.clientWidth * dpr);
  const H = Math.round(stageEl.clientHeight * dpr);
  if (viewEl.width !== W || viewEl.height !== H) {
    viewEl.width = W;
    viewEl.height = H;
  }
  viewCtx.clearRect(0, 0, W, H);
  if (!shown || !shown.width) return;
  const r = contentRect(W, H, shown.width, shown.height);
  if (shown.kind === 'image') viewCtx.drawImage(shown.bitmap, r.x, r.y, r.w, r.h);

  viewCtx.lineWidth = 2 * dpr;
  viewCtx.font = `600 ${12 * dpr}px -apple-system, "Segoe UI", Roboto, sans-serif`;
  viewCtx.textBaseline = 'middle';
  // Weakest first, so the most confident boxes and labels end up on top.
  for (const d of [...shown.detections].reverse()) {
    const x0 = r.x + clamp01(d.xyxy[0]) * r.w;
    const y0 = r.y + clamp01(d.xyxy[1]) * r.h;
    const x1 = r.x + clamp01(d.xyxy[2]) * r.w;
    const y1 = r.y + clamp01(d.xyxy[3]) * r.h;
    const color = colorOf(d.cls);
    viewCtx.strokeStyle = color;
    viewCtx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    const text = `${d.label} ${Math.round(d.score * 100)}%`;
    const tw = viewCtx.measureText(text).width + 10 * dpr;
    const th = 18 * dpr;
    const tx = Math.max(r.x, Math.min(x0, r.x + r.w - tw));
    const ty = y0 - th >= r.y ? y0 - th : y0; // inside the box at the top edge
    viewCtx.fillStyle = color;
    viewCtx.fillRect(tx, ty, tw, th);
    viewCtx.fillStyle = '#0b0d10';
    viewCtx.fillText(text, tx + 5 * dpr, ty + th / 2);
  }
}

new ResizeObserver(() => draw()).observe(stageEl);

// --- inputs: file, example, drop, paste, webcam ----------------------------

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
  await runImage(bitmap);
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
  let bitmap;
  try {
    bitmap = await loadBitmap(EXAMPLE_URL);
  } catch (err) {
    status(`Failed: could not load ${EXAMPLE_URL} (${errText(err)})`);
    return;
  }
  await runImage(bitmap);
});

window.addEventListener('dragover', (event) => {
  event.preventDefault();
  dropOverlay.style.display = 'flex';
});
window.addEventListener('dragleave', (event) => {
  if (event.relatedTarget === null) dropOverlay.style.display = 'none';
});
window.addEventListener('drop', async (event) => {
  event.preventDefault();
  dropOverlay.style.display = 'none';
  const file = event.dataTransfer?.files?.[0];
  if (file) await runFile(file);
});

window.addEventListener('paste', async (event) => {
  const item = [...(event.clipboardData?.items ?? [])].find((entry) =>
    entry.type.startsWith('image/'),
  );
  const file = item?.getAsFile();
  if (file) await runFile(file);
});

let stream = null;
let camToken = null; // identifies the running camera loop
let camOpening = null; // a camera start waiting on the browser (its permission prompt)

const CAMERA_HIDDEN = 'Camera off — the tab was hidden.';

/** Drop a camera start that is still waiting on the browser: a photo asked
 * for meanwhile, or a hidden tab, wins. `why` replaces the status line. */
function cancelCameraStart(why) {
  if (!camOpening || camOpening.cancelled) return;
  camOpening.cancelled = true;
  if (why) status(why);
}

camBtn.addEventListener('click', async () => {
  if (camOpening) return; // the button is disabled while opening; this is a synthetic click
  if (stream) {
    await stopCamera(true);
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
  let s;
  try {
    s = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment' },
      audio: false,
    });
  } catch (err) {
    if (!opening.cancelled) status(`Camera: ${errText(err)}`);
    return;
  } finally {
    camOpening = null;
    camBtn.disabled = false;
  }
  if (opening.cancelled || stream || document.visibilityState === 'hidden') {
    s.getTracks().forEach((track) => track.stop());
    if (!opening.cancelled && !stream) status(CAMERA_HIDDEN);
    return;
  }
  stream = s;
  for (const track of s.getVideoTracks()) {
    // The browser or the OS took the camera away (unplugged, permission revoked).
    track.addEventListener('ended', () => {
      if (stream === s) stopCamera(false, 'Camera off — the browser ended the camera stream.');
    });
  }
  videoEl.srcObject = s;
  camBtn.textContent = 'Stop camera';
  // In the queue: a photo still being detected finishes before the video
  // takes the stage.
  await exclusive(async () => {
    if (stream !== s) return;
    shown = null; // the previous photo must not cover the video
    draw();
    clearResult();
    videoEl.style.display = 'block';
    placeholderEl.style.display = 'none';
    status('Camera on — detecting continuously.');
  });
  if (stream !== s) return;
  try {
    await videoEl.play();
  } catch { /* autoplay attribute covers it */ }
  if (stream === s) cameraLoop();
});

/** Continuous detection: one inference at a time — the next frame is taken
 * only after the previous result is drawn — with the achieved rate shown. */
async function cameraLoop() {
  const token = {};
  camToken = token;
  let frames = 0;
  let windowStart = performance.now();
  let fps = null;
  while (camToken === token) {
    if (videoEl.readyState >= 2 && videoEl.videoWidth) {
      await exclusive(async () => {
        if (camToken !== token) return;
        const w = videoEl.videoWidth;
        const h = videoEl.videoHeight;
        try {
          const r = await runOnSource(videoEl, w, h);
          if (camToken !== token) return;
          shown = { kind: 'camera', width: w, height: h, detections: r.detections };
          draw();
          frames++;
          const now = performance.now();
          if (now - windowStart >= 1000) {
            fps = (frames * 1000) / (now - windowStart);
            frames = 0;
            windowStart = now;
            // One line per second, not per frame, to keep the console usable.
            logStats(r, 'camera', { fps: round(fps, 1) });
          }
          showLatency(r.ms, r.hostMs, fps);
          showSummary(r.detections);
        } catch (err) {
          if (camToken !== token) return;
          await stopCamera(false);
          clearResult();
          status(`Failed: ${errText(err)}`);
          console.error('[rfdetr] camera detection failed:', err);
        }
      });
    }
    await new Promise((resolve) => requestAnimationFrame(resolve));
  }
}

/** Stop the camera; with `keepFrame`, the last frame stays on screen as a
 * photo (detected once more, so its boxes match it exactly). `why` is the
 * status line when no photo takes the stage. */
async function stopCamera(keepFrame, why = 'Camera off.') {
  const s = stream;
  if (!s) return;
  camToken = null;
  let frame = null;
  if (keepFrame && videoEl.videoWidth) {
    try {
      frame = await createImageBitmap(videoEl);
    } catch { /* no frame to keep */ }
  }
  if (stream !== s) return; // a second stop got here first
  s.getTracks().forEach((track) => track.stop());
  stream = null;
  videoEl.srcObject = null;
  videoEl.style.display = 'none';
  camBtn.textContent = 'Use camera';
  if (frame) {
    await runImage(frame);
  } else if (shown?.kind !== 'image') {
    shown = null;
    draw();
    placeholderEl.style.display = 'flex';
    status(why);
  }
}

// A hidden tab or a page being left gives the camera back: nobody sees the
// detections, and the camera light goes off.
function releaseCamera() {
  cancelCameraStart(CAMERA_HIDDEN);
  stopCamera(false, CAMERA_HIDDEN);
}
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden') releaseCamera();
});
window.addEventListener('pagehide', releaseCamera);

// --- backend switch (graph A: webgpu / wasm) --------------------------------

for (const button of backendButtons) {
  button.addEventListener('click', async () => {
    const acc = button.dataset.backend;
    if (!ready || acc === graphs.A.acc || button.disabled) return;
    const states = backendButtons.map((b) => b.disabled);
    for (const b of backendButtons) b.disabled = true;
    const requestsAtClick = imageRequests;
    let fellBack = false; // asked for WebGPU, LiteRT.js compiled on WASM
    try {
      await exclusive(async () => {
        const prev = graphs.A;
        graphs.A = await compileGraph('A', acc);
        fellBack = graphs.A.acc !== acc;
        status('Warming up (one throwaway run)…');
        const start = performance.now();
        try {
          await warmUp();
        } catch (err) {
          graphs.A.model.delete();
          graphs.A = prev;
          throw err;
        }
        warmSeconds = (performance.now() - start) / 1000;
        prev.model.delete();
        updateEnv();
      });
      // A photo asked for during the switch waits behind it and runs on the
      // new backend; otherwise the last photo runs again.
      const newerPhoto = imageRequests !== requestsAtClick;
      if (stream) {
        status('Camera on — detecting continuously.');
      } else if (lastImage && !newerPhoto) {
        await runImage(lastImage);
      } else if (!newerPhoto) {
        status('Ready — choose a photo, try the example, or use the camera.');
      }
    } catch (err) {
      status(`Backend switch failed: ${errText(err)}`);
    } finally {
      backendButtons.forEach((b, i) => { b.disabled = states[i]; });
      if (fellBack) disableWebGpuButton('WebGPU could not take all of graph A in this browser');
    }
  });
}

boot();
