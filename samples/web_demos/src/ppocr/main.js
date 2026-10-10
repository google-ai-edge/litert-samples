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
 * Image → text with PP-OCRv5, fully client-side.
 *
 * Pipeline (the Page Text Chrome extension's engine, on a page):
 *   image → 640×640 stretch + ImageNet norm → detector (WebGPU, WASM
 *   fallback) → DB prob map → line boxes → each line drawn at h=48 from the
 *   full-resolution image and split on its own ink profile → recognizer
 *   (fp32 on WASM) → CTC greedy decode, with geometry retries for
 *   low-margin windows. Apart from the model runs and the 640×640 draw,
 *   every step is the extension's ocr-pipeline.js, imported as is
 *   (recognizeLines is the per-line loop).
 *
 * The recognizer stays on WASM: the WebGPU delegate flips some of its
 * argmaxes on real text crops (https://github.com/google-ai-edge/LiteRT/issues/9661),
 * and XNNPACK declines the fp16 recognizer graph, so the WASM path loads
 * the fp32 file (~20 ms per window vs ~430 ms for fp16).
 *
 * Debug URL params: ?img=<url>|sample (read that image at boot) ·
 *   ?backend=wasm (detector on WASM too) · ?models=<base url> (fetch the
 *   files from somewhere other than HF). Every read logs one OCR_STATS JSON
 *   line (tools/check.mjs reads it).
 */
import { Tensor, isWebGPUSupported, loadAndCompile, loadLiteRt } from '@litertjs/core';
import {
  DET_SIZE, REC_H, REC_W, buildCharTable, detPreprocess, groupLines,
  probToBoxes, recognizeLines,
} from '../../chrome_extension/src-ocr/ocr-pipeline.js';

const params = new URLSearchParams(location.search);

// Model files stream from the model card's repo on Hugging Face and are
// cached with the Cache API after the first visit. ?models=<base url> points
// the page at another copy (a local directory, a mirror) — same file names.
const DEFAULT_MODEL_BASE = 'https://huggingface.co/litert-community/PP-OCRv5-LiteRT/resolve/main/';
const MODEL_BASE = (params.get('models') ?? DEFAULT_MODEL_BASE).replace(/\/?$/, '/');
const FILES = {
  det: 'ppocr_det_fp16.tflite',
  rec: 'ppocr_rec_fp32.tflite',
  dict: 'ppocrv5_dict.txt',
};
const CACHE_NAME = 'ppocr-demo-v1';
// The LiteRT.js WASM runtime is served from litert-wasm/ at the site root.
// Resolve against this module's own URL: in dev it is <root>/ppocr/main.js,
// in the build <root>/assets/<hash>.js — one level below the runtime dir.
const WASM_DIR = new URL(/* @vite-ignore */ '../litert-wasm/', import.meta.url).href;
// "Try a sample": a menu card drawn on a canvas and saved as a PNG (README).
// tools/check.mjs reads it too (?img=sample), so the button, the check and
// every platform see the same pixels.
const SAMPLE_URL = new URL('./sample.png', import.meta.url).href;

const statusEl = document.getElementById('status');
const envEl = document.getElementById('env');
const latencyEl = document.getElementById('latency');
const resultEl = document.getElementById('result');
const countEl = document.getElementById('count');
const linesEl = document.getElementById('lines');
const copyBtn = document.getElementById('copy');
const fileEl = document.getElementById('file');
const fileBtn = document.getElementById('filebtn');
const camBtn = document.getElementById('cam');
const shutterBtn = document.getElementById('shutter');
const sampleBtn = document.getElementById('sample');
const videoEl = document.getElementById('video');
const dropOverlay = document.getElementById('drop-overlay');
const viewCanvas = document.getElementById('view');
const emptyEl = document.getElementById('empty');

let detModel = null;
let recModel = null;
let chars = null;
let detBackend = null; // 'webgpu' | 'wasm': what the detector compiled for
let wasmOpts = null; // which loadLiteRt rung succeeded (threads or plain)
let ready = false;
let bootFailure = null; // the status line of a failed boot, shown again when an input is used
let reading = 0; // reads asked for and not finished yet
let fetching = 0; // images being fetched for a read (the sample, ?img=)
let shown = null; // {bitmap, groups} of the last read, redrawn on resize/hover
let hot = -1; // hovered line group
let stream = null; // the camera preview
let camOpening = null; // { cancelled } from the camera click until the browser answers

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

/** wasm / wasm·1-thread — mirrors what actually loaded, so a page that
 * silently lost threads is visible at a glance. */
function wasmLabel() {
  return wasmOpts?.threads ? 'wasm' : 'wasm·1-thread';
}

// --- model loading --------------------------------------------------------

/** One model file, from the Cache API when it holds it, else from the
 * network (then stored). Where the Cache API is missing or fails (storage
 * blocked, quota), the page still runs and downloads on every visit.
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
    console.warn(`[ppocr] Cache API unavailable (${errText(err)}); downloading without it.`);
    cache = null;
  }
  let response;
  try {
    response = await fetch(url, { signal });
  } catch (err) {
    // The browser reports a blocked or unreachable host as a bare
    // "Failed to fetch" — name the URL so the failure is diagnosable.
    const offline = navigator.onLine === false ? ', browser is offline' : '';
    throw new Error(`could not fetch ${url} (${errText(err)}${offline})`);
  }
  if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`);
  const total = Number(response.headers.get('Content-Length')) || 0;
  onProgress?.(0, total, false);
  const chunks = [];
  let received = 0;
  try {
    const reader = response.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      onProgress?.(received, total, false);
    }
  } catch (err) {
    // A connection that drops mid-file also fails with a bare message.
    throw new Error(`download of ${url} stopped after ${(received / 1e6).toFixed(1)} MB (${errText(err)})`);
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
      console.warn(`[ppocr] could not cache ${url} (${errText(err)}); the next visit downloads it again.`);
    }
  }
  return bytes;
}

/** The three files in parallel, with one progress line. The first file that
 * fails stops the others, so its failure line stays on screen. */
async function downloadModels() {
  const keys = Object.keys(FILES);
  const received = {};
  const sizes = {}; // from Content-Length; 0 = not sent
  let fromCache = true;
  const abort = new AbortController();
  const report = (key) => (n, size, cached) => {
    if (abort.signal.aborted) return;
    received[key] = n;
    sizes[key] = size;
    fromCache &&= cached;
    if (fromCache) {
      status('Loading models from cache…');
      return;
    }
    const mb = (bytes) => (bytes / 1e6).toFixed(0);
    const done = Object.values(received).reduce((a, b) => a + b, 0);
    // A file sent compressed has no Content-Length: it counts as what has
    // arrived so far.
    const total = keys.every((k) => k in sizes)
      ? keys.reduce((a, k) => a + Math.max(sizes[k], received[k]), 0)
      : 0;
    if (total) {
      status(`Downloading models (one-time)… ${mb(done)} / ${mb(total)} MB`, Math.min(done / total, 1));
    } else {
      status(`Downloading models (one-time)… ${mb(done)} MB`);
    }
  };
  try {
    const files = await Promise.all(keys.map((key) => fetchCached(MODEL_BASE + FILES[key], report(key), abort.signal)));
    return Object.fromEntries(keys.map((key, i) => [key, files[i]]));
  } catch (err) {
    abort.abort();
    throw err;
  }
}

async function runModel(model, nchw, shape) {
  const input = Tensor.fromTypedArray(nchw, shape);
  try {
    const start = performance.now();
    const outputs = await model.run([input]);
    try {
      const data = await outputs[0].data();
      return { data, ms: performance.now() - start };
    } finally {
      for (const output of outputs) output.delete();
    }
  } finally {
    input.delete();
  }
}

// Which boot stage is in flight, so a failure names the culprit
// (runtime / download / compile / warm-up).
let bootStage = 'runtime';

/** Compile the detector for `acc` and burn one throwaway run: the first run
 * after compile carries shader/kernel warm-up and must not land on a user
 * image or in the latency line. */
async function compileDet(bytes, acc, wasmCompile) {
  bootStage = `compile det ${acc}`;
  const model = await loadAndCompile(bytes, acc === 'webgpu' ? { accelerator: 'webgpu' } : wasmCompile);
  // In a browser without JSPI, LiteRT.js compiles a graph that WebGPU cannot
  // fully take on WASM instead, without an error: keep the backend the model
  // got, not the one asked for.
  const got = model.options?.accelerator ?? acc;
  bootStage = `warm-up det ${got}`;
  try {
    const warm = await runModel(model, new Float32Array(3 * DET_SIZE * DET_SIZE), [1, 3, DET_SIZE, DET_SIZE]);
    return { model, acc: got, warmMs: warm.ms };
  } catch (err) {
    model.delete();
    throw err;
  }
}

async function boot() {
  try {
    status('Loading runtime…');
    // `threads` and `jspi` are mutually exclusive in LiteRT.js, and threads
    // only work on a cross-origin-isolated page — ask for what can succeed,
    // then fall back to plain.
    const rungs = window.crossOriginIsolated
      ? [{ threads: true }, { threads: false }]
      : [{ threads: false }];
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

    // ~43 MB on the first visit; Cache API afterwards.
    bootStage = 'download';
    const bytes = await downloadModels();
    chars = buildCharTable(new TextDecoder().decode(bytes.dict));

    status('Compiling…');
    const numThreads = Math.min(8, navigator.hardwareConcurrency || 4);
    const wasmCompile = { accelerator: 'wasm', cpuOptions: { numThreads } };
    bootStage = 'compile rec wasm';
    recModel = await loadAndCompile(bytes.rec, wasmCompile);
    status('Warming up (one throwaway run)…');
    bootStage = 'warm-up rec wasm';
    const recWarm = await runModel(recModel, new Float32Array(3 * REC_H * REC_W).fill(-1), [1, 3, REC_H, REC_W]);
    let det = null;
    if (isWebGPUSupported() && params.get('backend') !== 'wasm') {
      try {
        det = await compileDet(bytes.det, 'webgpu', wasmCompile);
      } catch (err) {
        // WebGPU exists on paper in more browsers than it works in (mobile
        // WebKit in particular) — fall back to WASM instead of dying.
        console.warn(`[ppocr] detector on WebGPU failed (${errText(err)}); using WASM`);
      }
    }
    if (!det) det = await compileDet(bytes.det, 'wasm', wasmCompile);
    detModel = det.model;
    detBackend = det.acc;
    const warmSeconds = (recWarm.ms + det.warmMs) / 1000;

    envEl.textContent = `PP-OCRv5 · detector fp16 ${detBackend === 'webgpu' ? 'webgpu' : wasmLabel()} · recognizer fp32 ${wasmLabel()} · warm-up ${warmSeconds.toFixed(1)} s`;
    envEl.style.display = 'block';
    ready = true;
    syncInputs();
    status('Ready — choose an image, use the camera, drop or paste one, or try the sample.');
  } catch (err) {
    const hint = bootStage === 'download'
      ? ' The models stream from Hugging Face: check that huggingface.co is reachable, or pass ?models=<url> to load them from elsewhere.'
      : '';
    bootFailure = `Failed to start (${bootStage}): ${errText(err).replace(/\.$/, '')}.${hint}`;
    status(bootFailure);
    console.error(`[ppocr] boot failed at stage "${bootStage}":`, err);
    return;
  }

  // ?img= runs once the page is up: an image that does not load is an input
  // failure, not a boot failure.
  const img = params.get('img');
  if (img) await readUrl(img === 'sample' ? SAMPLE_URL : img);
}

/** Fetch, decode and read an image URL (the sample, ?img=); one that does
 * not load ends in a status line naming the URL. The inputs are off from
 * the start, as during the read itself: a second click on "Try a sample"
 * does not fetch the image again. */
async function readUrl(url) {
  let bitmap;
  fetching++;
  syncInputs();
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    bitmap = await createImageBitmap(await response.blob(), { imageOrientation: 'from-image' });
  } catch (err) {
    status(`Failed: could not load ${url} (${errText(err)})`);
    return;
  } finally {
    fetching--;
    syncInputs();
  }
  await read(bitmap);
}

/** The inputs work once the models are ready and no read is waiting. */
function syncInputs() {
  const on = ready && reading === 0 && fetching === 0;
  fileEl.disabled = !on;
  fileBtn.classList.toggle('disabled', !on);
  // A camera that is opening keeps its button off until the browser answers.
  camBtn.disabled = !on || camOpening !== null;
  sampleBtn.disabled = !on;
}

/** An input used before the models are ready: say so, or repeat why they
 * never will be. */
function notReady() {
  status(bootFailure ?? 'Still loading the models…');
}

// --- recognition ------------------------------------------------------------

/** For an image with transparent pixels (`rgba` = the image drawn on a
 * transparent canvas), the color to read and show it on: white under dark
 * content, black under light content — on transparent black, dark text on a
 * transparent background would read as black on black. Null when every
 * pixel is opaque. */
function backdropOf(rgba) {
  let i = 3;
  while (i < rgba.length && rgba[i] === 255) i += 4;
  if (i >= rgba.length) return null;
  // Mean luminance of what is visible, each pixel weighted by its alpha
  // (getImageData returns unpremultiplied color).
  let sum = 0;
  let weight = 0;
  for (let p = 0; p < rgba.length; p += 4) {
    const a = rgba[p + 3];
    sum += a * (0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2]);
    weight += a;
  }
  return weight && sum / weight >= 128 ? '#000' : '#fff';
}

/** The image drawn on `backdrop`, as a new bitmap; closes the original. */
function flatten(bitmap, backdrop) {
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = backdrop;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  return canvas.transferToImageBitmap();
}

/** ImageBitmap → {bitmap, lines, boxes, windows, detMs, recMs}. `bitmap` is
 * the image as read: the input, or for an image with transparent pixels a
 * copy flattened onto its backdrop (the input is then closed). Line rects
 * are fractions of the image size. */
async function ocr(input) {
  const nw = input.width;
  const nh = input.height;
  const detCtx = new OffscreenCanvas(DET_SIZE, DET_SIZE).getContext('2d', { willReadFrequently: true });
  detCtx.drawImage(input, 0, 0, nw, nh, 0, 0, DET_SIZE, DET_SIZE);
  let rgba = detCtx.getImageData(0, 0, DET_SIZE, DET_SIZE).data;
  let bitmap = input;
  const backdrop = backdropOf(rgba);
  if (backdrop) {
    bitmap = flatten(input, backdrop);
    detCtx.drawImage(bitmap, 0, 0, nw, nh, 0, 0, DET_SIZE, DET_SIZE);
    rgba = detCtx.getImageData(0, 0, DET_SIZE, DET_SIZE).data;
  }
  try {
    const { nchw } = detPreprocess(rgba, nw, nh);
    const det = await runModel(detModel, nchw, [1, 3, DET_SIZE, DET_SIZE]);
    const boxes = probToBoxes(det.data);
    const { lines, windows, recMs } = await recognizeLines(bitmap, boxes, {
      chars,
      recognize: (nchwRec) => runModel(recModel, nchwRec, [1, 3, REC_H, REC_W]),
      makeCanvas: (w, h) => new OffscreenCanvas(w, h),
      onLine: (i, n) => status(`Reading… line ${i + 1}/${n}`),
    });
    return { bitmap, lines, boxes: boxes.length, windows, detMs: det.ms, recMs };
  } catch (err) {
    if (bitmap !== input) bitmap.close();
    throw err;
  }
}

/** Resolves after the browser has painted pending DOM changes (capped, as
 * requestAnimationFrame does not fire in a hidden tab). */
function nextPaint() {
  return new Promise((resolve) => {
    setTimeout(resolve, 100);
    requestAnimationFrame(() => setTimeout(resolve, 0));
  });
}

// Reads never overlap. An image asked for during a read waits for it; of
// several such images, only the newest is read.
let queue = Promise.resolve();
let requests = 0; // images asked for so far

/** Read an image from any input. It replaces the camera, and waits for a
 * read already in progress. */
function read(bitmap) {
  if (!ready) {
    bitmap.close();
    notReady();
    return Promise.resolve();
  }
  const id = ++requests;
  cancelCameraStart(); // an image asked for while the camera opens wins
  stopCamera();
  reading++;
  syncInputs();
  const run = queue.then(async () => {
    try {
      if (id === requests) await readNow(bitmap);
      else bitmap.close(); // a newer image is waiting
    } finally {
      reading--;
      syncInputs();
    }
  });
  queue = run.catch(() => {});
  return run;
}

async function readNow(bitmap) {
  status('Reading…');
  // The detector on WASM holds the main thread for about a second: let the
  // status line show first.
  if (detBackend !== 'webgpu') await nextPaint();
  try {
    const t0 = performance.now();
    const r = await ocr(bitmap);
    const totalMs = performance.now() - t0;
    const groups = groupLines(r.lines);
    show(r.bitmap, groups);
    const { width, height } = r.bitmap;
    const env = (b) => `<span class="env">(${b === 'wasm' ? wasmLabel() : b})</span>`;
    latencyEl.innerHTML =
      `detector <b>${r.detMs.toFixed(0)} ms</b> ${env(detBackend)} · ` +
      `recognizer <b>${r.recMs.toFixed(0)} ms</b> ${env('wasm')} for ${r.windows} window${r.windows === 1 ? '' : 's'}<br>` +
      `total <b>${totalMs.toFixed(0)} ms</b> · ${width}×${height} px`;
    latencyEl.style.display = 'block';
    status(groups.length ? 'Done. Hover or tap a line to find it in the image.' : 'No text found in this image.');
    // machine-readable result for automation
    console.log('OCR_STATS ' + JSON.stringify({
      natural: { w: width, h: height },
      backends: { det: detBackend, rec: 'wasm' },
      wasmThreads: !!wasmOpts?.threads,
      boxes: r.boxes,
      windows: r.windows,
      lines: groups.map((g) => g.text),
      scores: r.lines.map((l) => l.score),
      timings: { det: +r.detMs.toFixed(1), rec: +r.recMs.toFixed(1), total: +totalMs.toFixed(1) },
    }));
  } catch (err) {
    bitmap.close();
    clearShown();
    status(`Failed: ${errText(err)}`);
    console.error('[ppocr] read failed:', err);
  }
}

// --- display ------------------------------------------------------------------

function show(bitmap, groups) {
  shown?.bitmap.close?.();
  shown = { bitmap, groups };
  hot = -1;
  linesEl.textContent = '';
  for (const [i, g] of groups.entries()) {
    const li = document.createElement('li');
    li.textContent = g.text;
    li.addEventListener('mouseenter', () => setHot(i));
    li.addEventListener('mouseleave', () => setHot(-1));
    linesEl.appendChild(li);
  }
  countEl.textContent = `${groups.length} line${groups.length === 1 ? '' : 's'}`;
  resultEl.style.display = groups.length ? 'flex' : 'none';
  emptyEl.style.display = 'none';
  draw();
}

/** Empty the stage and the lines: after a failed read, no earlier image or
 * text stays next to the failure line. */
function clearShown() {
  shown?.bitmap.close();
  shown = null;
  hot = -1;
  linesEl.textContent = '';
  resultEl.style.display = 'none';
  latencyEl.style.display = 'none';
  emptyEl.style.display = '';
  draw();
}

function setHot(i) {
  hot = i;
  for (const [k, li] of [...linesEl.children].entries()) li.classList.toggle('hot', k === i);
  draw();
}

/** Image contain-fit in the stage, each recognized window outlined; the
 * hovered line filled. */
function draw() {
  const rect = viewCanvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  viewCanvas.width = Math.max(1, Math.round(rect.width * dpr));
  viewCanvas.height = Math.max(1, Math.round(rect.height * dpr));
  const ctx = viewCanvas.getContext('2d');
  ctx.clearRect(0, 0, viewCanvas.width, viewCanvas.height);
  if (!shown) return;
  const { bitmap, groups } = shown;
  const margin = 16 * dpr;
  const scale = Math.min(
    (viewCanvas.width - 2 * margin) / bitmap.width,
    (viewCanvas.height - 2 * margin) / bitmap.height,
  );
  const dw = bitmap.width * scale;
  const dh = bitmap.height * scale;
  const ox = (viewCanvas.width - dw) / 2;
  const oy = (viewCanvas.height - dh) / 2;
  ctx.drawImage(bitmap, ox, oy, dw, dh);
  ctx.lineWidth = Math.max(1, 1.5 * dpr);
  for (const [i, g] of groups.entries()) {
    for (const p of g.pieces) {
      const x = ox + p.x * dw;
      const y = oy + p.y * dh;
      const w = p.w * dw;
      const h = p.h * dh;
      if (i === hot) {
        ctx.fillStyle = 'rgba(124, 196, 255, 0.35)';
        ctx.fillRect(x, y, w, h);
      }
      ctx.strokeStyle = i === hot ? 'rgba(124, 196, 255, 1)' : 'rgba(124, 196, 255, 0.85)';
      ctx.strokeRect(x, y, w, h);
    }
  }
}

window.addEventListener('resize', draw);

// --- inputs: file, drop, paste, webcam, sample -----------------------------------

/** A chosen, dropped or pasted file. One the browser cannot decode (not an
 * image, cut short) ends in a status line, not an unhandled rejection. */
async function runFile(file) {
  if (!ready) {
    notReady();
    return;
  }
  let bitmap;
  try {
    bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
  } catch (err) {
    status(`Failed: could not read ${file.name || 'the image'} as an image (${errText(err)})`);
    return;
  }
  await read(bitmap);
}

fileEl.addEventListener('change', async () => {
  const file = fileEl.files?.[0];
  fileEl.value = '';
  if (file) await runFile(file);
});

// Only a drag that carries files shows the overlay (not text dragged out of
// the list of lines).
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
  event.preventDefault();
  dropOverlay.style.display = 'none';
  const file = event.dataTransfer?.files?.[0];
  if (file) await runFile(file);
});

window.addEventListener('paste', async (event) => {
  const items = [...(event.clipboardData?.items ?? [])].filter((entry) => entry.kind === 'file');
  const item = items.find((entry) => entry.type.startsWith('image/')) ?? items[0];
  const file = item?.getAsFile();
  if (file) await runFile(file);
});

const CAMERA_ON = 'Camera on — press Capture to read the current frame.';
const CAMERA_HIDDEN = 'Camera off — the tab was hidden.';

/** Give up a camera start the browser has not answered yet: an image asked
 * for meanwhile, or a hidden tab, wins. `why` replaces the status line. */
function cancelCameraStart(why = null) {
  if (!camOpening || camOpening.cancelled) return;
  camOpening.cancelled = true;
  if (why) status(why);
}

camBtn.addEventListener('click', async () => {
  // The button is disabled before the models are ready, while the camera
  // opens and during a read; a click that arrives anyway is a synthetic one.
  if (camBtn.disabled) return;
  if (stream) {
    stopCamera('Camera off.');
    return;
  }
  const opening = { cancelled: false };
  camOpening = opening;
  syncInputs();
  status('Opening the camera…');
  let s;
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('not available on this page');
    s = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } },
    });
  } catch (err) {
    if (!opening.cancelled) status(`Camera: ${errText(err)}`);
    return;
  } finally {
    camOpening = null;
    syncInputs();
  }
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
  shutterBtn.style.display = 'inline-block';
  camBtn.textContent = 'Stop camera';
  status(CAMERA_ON);
});

/** Turn the camera off; `why` is the status line, if any. */
function stopCamera(why = null) {
  const s = stream;
  if (!s) return;
  stream = null;
  s.getTracks().forEach((track) => track.stop());
  videoEl.srcObject = null;
  videoEl.style.display = 'none';
  shutterBtn.style.display = 'none';
  camBtn.textContent = 'Use camera';
  if (why) status(why);
}

shutterBtn.addEventListener('click', async () => {
  if (!stream) return;
  if (!videoEl.videoWidth) {
    status('The camera has no picture yet — try again in a moment.');
    return;
  }
  let frame;
  try {
    frame = await createImageBitmap(videoEl);
  } catch (err) {
    stopCamera();
    status(`Failed: could not capture a camera frame (${errText(err)})`);
    return;
  }
  await read(frame); // stops the camera
});

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

sampleBtn.addEventListener('click', () => {
  // Disabled before the models are ready and during a read; a click that
  // arrives anyway is a synthetic one.
  if (sampleBtn.disabled) return;
  readUrl(SAMPLE_URL);
});

copyBtn.addEventListener('click', async () => {
  if (!shown) return;
  const text = shown.groups.map((g) => g.text).join('\n');
  try {
    await navigator.clipboard.writeText(text);
    copyBtn.textContent = 'Copied';
  } catch {
    // Clipboard blocked (permissions policy, no focus): select the list so
    // the user can copy it by hand.
    const range = document.createRange();
    range.selectNodeContents(linesEl);
    const selection = getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    copyBtn.textContent = 'Selected — press ⌘C / Ctrl+C';
  }
  setTimeout(() => { copyBtn.textContent = 'Copy all'; }, 1600);
});

boot();
