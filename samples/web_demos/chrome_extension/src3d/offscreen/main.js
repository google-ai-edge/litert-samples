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
 * Offscreen document: the depth engine. Loads LiteRT.js (wasm bundled with
 * the extension), fetches MoGe-2 from Hugging Face (Cache API after the first
 * download), and turns image URLs into a packed depth map + a re-encoded
 * photo the content script can texture without canvas tainting.
 *
 * Pipeline per request (input as in the MoGe web demo, web_demos/src/moge/):
 *   url → bytes (fetch here; service-worker relay if this COEP-isolated
 *   context can't) → letterbox 448×448 NCHW [0,1] → MoGe-2 (WebGPU, wasm
 *   fallback) → per-pixel point map + confidence mask → relative disparity,
 *   16-bit packed into a PNG data URL (content rect only, letterbox cropped).
 *
 * Messages in  (target 'offscreen'): {type:'depth', url} | {type:'status'}.
 * Messages out (target 'ui'): status broadcasts for the popup.
 */
import { Tensor, isWebGPUSupported, loadAndCompile, loadLiteRt } from '@litertjs/core';

const MODEL_URL =
  'https://huggingface.co/litert-community/MoGe-2-LiteRT/resolve/main/moge.tflite';
const CACHE_NAME = 'page3d-models-v1';
const SIZE = 448;
const MASK_THRESHOLD = 0.5;
// Disparity (median-relative) clamp range. MUST match the shader decode in
// src3d/content.js: q = mix(DISP_MIN, DISP_MAX, n16/65535).
const DISP_MIN = 0.5;
const DISP_MAX = 2.2;
// Close-ups span a narrow disparity band (p95/p05 ≈ 1.1–1.5) and extrude
// weakly under the fixed clamp range. Amplify narrow-range images with a
// per-image exponent q^γ (median plane stays at q = 1, monotonic), aiming
// for RANGE_TARGET; near-flat content (graphics, screenshots) is left alone
// so noise isn't blown up, and wide-range scenes are never compressed.
const RANGE_TARGET = 2.0;
const RANGE_FLAT = 1.06;
const GAMMA_MAX = 3;
const PHOTO_MAX_SIDE = 1024;
const CACHE_ENTRIES = 8;

let state = 'loading'; // loading → downloading → compiling → ready | error
let error = null;
let backend = null; // 'webgpu' | 'wasm'
let wasmOpts = null;
let model = null;
let downloadedMB = 0;
let lastStats = null;
let runCount = 0;
let lastResult = null; // debug hook: full payload of the last depth run
const resultCache = new Map(); // url → payload (LRU, CACHE_ENTRIES)

// Register the listener before any async work so no early message is lost.
chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (!msg || msg.target !== 'offscreen') return;
  if (msg.type === 'depth') {
    requestDepth(msg.url).then(sendResponse);
    return true;
  }
  if (msg.type === 'status') {
    sendResponse(statusPayload());
  }
  return false;
});

function statusPayload() {
  return {
    target: 'ui',
    type: 'status',
    state,
    error,
    env: backend === 'webgpu' ? 'webgpu' : wasmOpts?.threads ? 'wasm' : 'wasm·1-thread',
    backend,
    flags: {
      webgpu: backend ? backend === 'webgpu' : null,
      threads: wasmOpts?.threads ?? null,
      jspi: wasmOpts?.jspi ?? null,
      crossOriginIsolated: globalThis.crossOriginIsolated,
    },
    stats: lastStats,
    downloadedMB,
    runs: runCount,
  };
}

let lastBroadcast = 0;
function broadcast(force = false) {
  const now = performance.now();
  if (!force && now - lastBroadcast < 250) return;
  lastBroadcast = now;
  chrome.runtime.sendMessage(statusPayload()).catch(() => {});
}

// --- model download (Cache API) ---------------------------------------------

function errText(err) {
  return err instanceof Error ? err.message : String(err);
}

async function fetchCached(url, onProgress) {
  const cache = 'caches' in globalThis ? await caches.open(CACHE_NAME) : null;
  if (cache) {
    const hit = await cache.match(url);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  }
  let response;
  try {
    response = await fetch(url);
  } catch (err) {
    // A blocked or unreachable host surfaces as a bare "Failed to fetch";
    // name the URL so the error says which request failed.
    throw new Error(`could not fetch ${url} (${errText(err)})`);
  }
  if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`);
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress?.(received);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  if (cache) await cache.put(url, new Response(bytes.slice().buffer));
  return bytes;
}

// --- boot ---------------------------------------------------------------------

// Which boot stage is in flight, so a failure names it (runtime / download /
// compile), as the web demos' boot errors do.
let bootStage = 'runtime';

async function boot() {
  try {
    const coi = globalThis.crossOriginIsolated;
    // `threads` and `jspi` are mutually exclusive in LiteRT.js — asking for
    // both throws, so the old first attempt failed on every cross-origin
    // isolated page and we silently fell through to the second. Threads are
    // what this workload wants, so ask for exactly that, then plain wasm.
    const attempts = [
      { threads: coi },
      { threads: false },
    ];
    let loaded = false;
    let lastErr = null;
    for (const opts of attempts) {
      try {
        await loadLiteRt('litert-wasm/', opts);
        wasmOpts = opts;
        loaded = true;
        break;
      } catch (err) {
        lastErr = err;
      }
    }
    if (!loaded) throw lastErr;

    backend = isWebGPUSupported() ? 'webgpu' : 'wasm';
    state = 'downloading';
    bootStage = 'download';
    broadcast(true);

    const bytes = await fetchCached(MODEL_URL, (received) => {
      downloadedMB = Math.round(received / 1048576);
      broadcast();
    });

    state = 'compiling';
    bootStage = 'compile';
    broadcast(true);
    const numThreads = Math.min(8, navigator.hardwareConcurrency || 4);
    const wasmCompile = { accelerator: 'wasm', cpuOptions: { numThreads } };
    if (backend === 'wasm') {
      model = await loadAndCompile(bytes, wasmCompile);
    } else {
      try {
        model = await loadAndCompile(bytes, { accelerator: 'webgpu' });
      } catch {
        backend = 'wasm';
        model = await loadAndCompile(bytes, wasmCompile);
      }
    }
    state = 'ready';
    broadcast(true);
  } catch (err) {
    state = 'error';
    error = `Failed to start (${bootStage}): ${errText(err)}`;
    console.error(`[page-3d] boot failed at stage "${bootStage}":`, err);
    broadcast(true);
  }
}

// --- image fetch (CORS strategy) ---------------------------------------------

/** Fetch image bytes. Extension host_permissions let this offscreen page
 * fetch cross-origin, but its COEP:require-corp can still block responses
 * from hosts that send neither CORS nor CORP headers — those fall back to a
 * relay fetch in the service worker (no COEP there). */
async function fetchImageBytes(url, { forceRelay = false } = {}) {
  const t0 = performance.now();
  if (!forceRelay) {
    try {
      const response = await fetch(url, { credentials: 'omit' });
      if (response.ok) {
        const bytes = new Uint8Array(await response.arrayBuffer());
        return { bytes, via: 'offscreen', fetchMs: performance.now() - t0 };
      }
    } catch { /* fall through to relay */ }
  }
  const relay = await chrome.runtime.sendMessage({ target: 'bg', type: 'fetch-image', url });
  if (!relay?.ok) throw new Error(`fetch failed: ${relay?.error ?? 'no relay response'}`);
  const bin = atob(relay.b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return { bytes, via: 'sw-relay', fetchMs: performance.now() - t0 };
}

// --- preprocessing / inference ---------------------------------------------------

const cropCanvas = new OffscreenCanvas(SIZE, SIZE);
const cropCtx = cropCanvas.getContext('2d', { willReadFrequently: true });

/** Contain-fit (letterbox) into 448×448; the pad is excluded from the output
 * depth map via the recorded content rect. */
function preprocess(bitmap) {
  const scale = Math.min(SIZE / bitmap.width, SIZE / bitmap.height);
  const drawW = Math.round(bitmap.width * scale);
  const drawH = Math.round(bitmap.height * scale);
  const offX = Math.floor((SIZE - drawW) / 2);
  const offY = Math.floor((SIZE - drawH) / 2);
  cropCtx.fillStyle = '#7f7f7f';
  cropCtx.fillRect(0, 0, SIZE, SIZE);
  cropCtx.drawImage(bitmap, 0, 0, bitmap.width, bitmap.height, offX, offY, drawW, drawH);
  const { data } = cropCtx.getImageData(0, 0, SIZE, SIZE);
  const plane = SIZE * SIZE;
  const nchw = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    nchw[i] = data[i * 4] / 255;
    nchw[plane + i] = data[i * 4 + 1] / 255;
    nchw[2 * plane + i] = data[i * 4 + 2] / 255;
  }
  return { nchw, rect: { offX, offY, drawW, drawH } };
}

function sampleAbsMax(array) {
  let max = 0;
  const step = Math.max(1, Math.floor(array.length / 5000));
  for (let i = 0; i < array.length; i += step) {
    const v = Math.abs(array[i]);
    if (v > max) max = v;
  }
  return max;
}

/** Output order in the .tflite is not guaranteed — identify by shape and
 * range: points is the [h,w,3] map with values beyond [-1,1] (normals are
 * unit vectors), mask is [h,w,1] sigmoid. */
function resolveOutputs(buffers) {
  const plane = SIZE * SIZE;
  const big = buffers.filter((b) => b.length === plane * 3);
  const mask = buffers.find((b) => b.length === plane);
  if (big.length !== 2 || !mask) throw new Error('unexpected model outputs');
  const pointsFirst = sampleAbsMax(big[0]) > 2;
  return {
    points: pointsFirst ? big[0] : big[1],
    normals: pointsFirst ? big[1] : big[0],
    mask,
  };
}

async function infer(nchw) {
  const input = Tensor.fromTypedArray(nchw, [1, 3, SIZE, SIZE]);
  const start = performance.now();
  const outputs = await model.run([input]);
  const buffers = [];
  for (const output of outputs) buffers.push(await output.data());
  const inferMs = performance.now() - start;
  for (const output of outputs) output.delete();
  input.delete();
  return { ...resolveOutputs(buffers), inferMs };
}

// --- depth map packing ----------------------------------------------------------

/** points z (camera frame, forward-positive) → median-relative disparity,
 * clamped to [DISP_MIN, DISP_MAX], normalized to 16-bit, packed into a PNG
 * (R = high byte, G = low byte, B = confidence mask). Invalid pixels are
 * treated as far. Only the letterbox content rect is emitted. */
function packDepth(points, mask, rect) {
  const { offX, offY, drawW, drawH } = rect;
  const depths = [];
  for (let y = offY; y < offY + drawH; y += 2) {
    for (let x = offX; x < offX + drawW; x += 2) {
      const i = y * SIZE + x;
      const d = points[i * 3 + 2];
      if (mask[i] > MASK_THRESHOLD && Number.isFinite(d) && d > 0) depths.push(d);
    }
  }
  if (depths.length < 32) throw new Error('no confident depth in this image');
  depths.sort((a, b) => a - b);
  const median = depths[depths.length >> 1];
  // Depth-percentile ratio == disparity-percentile ratio (q = median / d).
  const p05 = depths[Math.floor(depths.length * 0.05)];
  const p95 = depths[Math.min(depths.length - 1, Math.floor(depths.length * 0.95))];
  const ratio = p95 / Math.max(p05, 1e-6);
  let gamma = 1;
  if (ratio > RANGE_FLAT && ratio < RANGE_TARGET) {
    gamma = Math.min(GAMMA_MAX, Math.log(RANGE_TARGET) / Math.log(ratio));
  }

  const out = new ImageData(drawW, drawH);
  const px = out.data;
  let validCount = 0;
  let nMin = 1;
  let nMax = 0;
  for (let y = 0; y < drawH; y++) {
    for (let x = 0; x < drawW; x++) {
      const i = (y + offY) * SIZE + (x + offX);
      const d = points[i * 3 + 2];
      const valid = mask[i] > MASK_THRESHOLD && Number.isFinite(d) && d > 0;
      let n = 0; // invalid → far
      if (valid) {
        const q = Math.min(DISP_MAX, Math.max(DISP_MIN, (median / d) ** gamma));
        n = (q - DISP_MIN) / (DISP_MAX - DISP_MIN);
        validCount++;
        if (n < nMin) nMin = n;
        if (n > nMax) nMax = n;
      }
      const v = Math.round(n * 65535);
      const o = (y * drawW + x) * 4;
      px[o] = v >> 8;
      px[o + 1] = v & 255;
      px[o + 2] = valid ? 255 : 0;
      px[o + 3] = 255;
    }
  }
  return {
    imageData: out,
    stats: {
      validRatio: +(validCount / (drawW * drawH)).toFixed(3),
      nMin: +nMin.toFixed(3),
      nMax: +nMax.toFixed(3),
      dispRatio: +ratio.toFixed(2),
      gamma: +gamma.toFixed(2),
    },
  };
}

/** MoGe normals (camera frame, unit vectors) → RGB PNG, (n + 1) / 2 per
 * channel, content rect only. Invalid pixels get the camera-facing normal so
 * relighting stays neutral there. The mean z of confident normals is reported
 * so the shader-side frame convention can be checked empirically (OpenCV
 * camera frame: z forward-positive → camera-facing normals have z < 0). */
function packNormals(normals, mask, rect) {
  const { offX, offY, drawW, drawH } = rect;
  const out = new ImageData(drawW, drawH);
  const px = out.data;
  let sumZ = 0;
  let sumLen = 0;
  let count = 0;
  for (let y = 0; y < drawH; y++) {
    for (let x = 0; x < drawW; x++) {
      const i = (y + offY) * SIZE + (x + offX);
      const nx = normals[i * 3];
      const ny = normals[i * 3 + 1];
      const nz = normals[i * 3 + 2];
      const len = Math.hypot(nx, ny, nz);
      const valid = mask[i] > MASK_THRESHOLD && len > 0.5 && len < 1.5;
      const o = (y * drawW + x) * 4;
      if (valid) {
        px[o] = Math.round(((nx / len) * 0.5 + 0.5) * 255);
        px[o + 1] = Math.round(((ny / len) * 0.5 + 0.5) * 255);
        px[o + 2] = Math.round(((nz / len) * 0.5 + 0.5) * 255);
        sumZ += nz / len;
        sumLen += len;
        count++;
      } else {
        px[o] = 128;
        px[o + 1] = 128;
        px[o + 2] = 0; // z = -1: straight at the camera
      }
      px[o + 3] = 255;
    }
  }
  return {
    imageData: out,
    stats: {
      normalMeanZ: count ? +(sumZ / count).toFixed(3) : null,
      normalMeanLen: count ? +(sumLen / count).toFixed(3) : null,
    },
  };
}

async function canvasToDataUrl(imageDataOrBitmap, type, quality) {
  const w = imageDataOrBitmap.width;
  const h = imageDataOrBitmap.height;
  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext('2d');
  if (imageDataOrBitmap instanceof ImageData) ctx.putImageData(imageDataOrBitmap, 0, 0);
  else ctx.drawImage(imageDataOrBitmap, 0, 0, w, h);
  const blob = await canvas.convertToBlob({ type, quality });
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

async function encodePhoto(bitmap) {
  const scale = Math.min(1, PHOTO_MAX_SIDE / Math.max(bitmap.width, bitmap.height));
  const w = Math.max(1, Math.round(bitmap.width * scale));
  const h = Math.max(1, Math.round(bitmap.height * scale));
  const canvas = new OffscreenCanvas(w, h);
  canvas.getContext('2d').drawImage(bitmap, 0, 0, w, h);
  const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality: 0.88 });
  const url = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
  return { url, w, h };
}

// --- request queue ----------------------------------------------------------------

let chain = Promise.resolve();

function requestDepth(url, opts) {
  const run = chain.then(() => runDepth(url, opts)).catch((err) => ({
    ok: false,
    error: String(err instanceof Error ? err.message : err),
  }));
  chain = run.then(() => {});
  return run;
}

async function runDepth(url, { forceRelay = false } = {}) {
  if (state !== 'ready') {
    // Boot may still be in flight (or previously failed) — wait for it here so
    // a hover during the one-time model download still resolves.
    const deadline = Date.now() + 5 * 60 * 1000;
    while (state !== 'ready' && state !== 'error' && Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 300));
    }
    if (state !== 'ready') return { ok: false, error: error ?? 'engine not ready' };
  }
  const cached = resultCache.get(url);
  if (cached && !forceRelay) {
    resultCache.delete(url);
    resultCache.set(url, cached); // LRU refresh
    return cached;
  }

  const { bytes, via, fetchMs } = await fetchImageBytes(url, { forceRelay });
  const bitmap = await createImageBitmap(new Blob([bytes]), { imageOrientation: 'from-image' });
  const { nchw, rect } = preprocess(bitmap);
  const { points, normals, mask, inferMs } = await infer(nchw);
  const { imageData, stats: depthStats } = packDepth(points, mask, rect);
  const depthUrl = await canvasToDataUrl(imageData, 'image/png');
  const packedNormals = packNormals(normals, mask, rect);
  const normalUrl = await canvasToDataUrl(packedNormals.imageData, 'image/png');
  const photo = await encodePhoto(bitmap);
  bitmap.close();

  runCount++;
  lastStats = {
    fetchMs: +fetchMs.toFixed(0),
    inferMs: +inferMs.toFixed(0),
    via,
    ...depthStats,
    ...packedNormals.stats,
  };
  broadcast(true);

  const payload = {
    ok: true,
    depth: { url: depthUrl, w: imageData.width, h: imageData.height },
    normal: { url: normalUrl, w: imageData.width, h: imageData.height },
    photo,
    stats: { ...lastStats, backend, env: statusPayload().env },
  };
  lastResult = payload;
  resultCache.set(url, payload);
  if (resultCache.size > CACHE_ENTRIES) {
    resultCache.delete(resultCache.keys().next().value);
  }
  return payload;
}

// Debug/smoke hook, dev builds only: lets CDP automation (tools3d/) poll
// engine state and run depth.
if (__DEV__) {
  globalThis.__p3 = {
    get status() { return statusPayload(); },
    get lastResult() { return lastResult; },
    async depthSummary(url, opts) {
      const r = await requestDepth(url, opts);
      if (!r.ok) return r;
      return { ok: true, depthW: r.depth.w, depthH: r.depth.h, photoW: r.photo.w,
        photoH: r.photo.h, normalW: r.normal?.w, normalH: r.normal?.h, stats: r.stats };
    },
  };
}

boot();
