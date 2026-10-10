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
 * Offscreen document: the OCR engine. Loads LiteRT.js (wasm bundled with the
 * extension), fetches PP-OCRv5 from Hugging Face (Cache API after the first
 * download), and turns an image URL into positioned text lines.
 *
 * Backends:
 *   det fp16 → webgpu (wasm fallback)
 *   rec fp32 → wasm (XNNPACK); tools-ocr/verify.mjs compares its decoded
 *     text on webgpu and wasm, as does
 *     https://github.com/google-ai-edge/LiteRT/issues/9661. On wasm the
 *     fp16 recognizer is ~20× slower (430–460 vs 20 ms per line).
 *
 * Pipeline per request:
 *   url → bytes (fetch here; service-worker relay if COEP blocks it) →
 *   640×640 stretch + ImageNet norm → det prob map → threshold + connected
 *   components + line merge (ocr-pipeline.js) → per-line valley split →
 *   crops from the FULL-RES bitmap at h=48 → rec → CTC greedy decode
 *   (recognizeLines in ocr-pipeline.js) →
 *   [{x, y, w, h, text}] normalized to the natural image size.
 *
 * Messages in  (target 'offscreen'): {type:'ocr', url} | {type:'status'}.
 * Messages out (target 'ui'): status broadcasts for the popup.
 */
import { Tensor, isWebGPUSupported, loadAndCompile, loadLiteRt } from '@litertjs/core';
import {
  DET_SIZE, EDGE_PAD, REC_H, REC_W, buildCharTable, columnInkProfile,
  ctcDecode, detPreprocess, probToBoxes, recPreprocess, recognizeLines,
  splitByInk,
} from '../ocr-pipeline.js';

const HF = 'https://huggingface.co/litert-community/PP-OCRv5-LiteRT/resolve/main';
const DET_URL = `${HF}/ppocr_det_fp16.tflite`;
const REC_URL = `${HF}/ppocr_rec_fp32.tflite`;
const DICT_URL = `${HF}/ppocrv5_dict.txt`;
const CACHE_NAME = 'pagetext-models-v1';
const CACHE_ENTRIES = 8;

let state = 'loading'; // loading → downloading → compiling → ready | error
let error = null;
let detBackend = null; // 'webgpu' | 'wasm'
let wasmOpts = null;
let detModel = null;
let recModel = null;
let chars = null;
let downloadedMB = 0;
let lastStats = null;
let runCount = 0;
let lastResult = null; // debug hook: full payload of the last OCR run
const resultCache = new Map(); // url → payload (LRU, CACHE_ENTRIES)

// Register the listener before any async work so no early message is lost.
chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (!msg || msg.target !== 'offscreen') return;
  if (msg.type === 'ocr') {
    requestOcr(msg.url, { background: Boolean(msg.background) }).then(sendResponse);
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
    env: detBackend ? `det ${detBackend} · rec wasm` : null,
    flags: {
      webgpu: detBackend ? detBackend === 'webgpu' : null,
      threads: wasmOpts?.threads ?? null,
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

    state = 'downloading';
    bootStage = 'download';
    broadcast(true);
    let base = 0;
    const progress = (received) => {
      downloadedMB = Math.round((base + received) / 1048576);
      broadcast();
    };
    const detBytes = await fetchCached(DET_URL, progress);
    base += detBytes.length;
    const recBytes = await fetchCached(REC_URL, progress);
    base += recBytes.length;
    const dictBytes = await fetchCached(DICT_URL, progress);
    chars = buildCharTable(new TextDecoder().decode(dictBytes));

    state = 'compiling';
    bootStage = 'compile';
    broadcast(true);
    const numThreads = Math.min(8, navigator.hardwareConcurrency || 4);
    const wasmCompile = { accelerator: 'wasm', cpuOptions: { numThreads } };
    recModel = await loadAndCompile(recBytes, wasmCompile);
    detBackend = isWebGPUSupported() ? 'webgpu' : 'wasm';
    if (detBackend === 'wasm') {
      detModel = await loadAndCompile(detBytes, wasmCompile);
    } else {
      try {
        detModel = await loadAndCompile(detBytes, { accelerator: 'webgpu' });
      } catch {
        detBackend = 'wasm';
        detModel = await loadAndCompile(detBytes, wasmCompile);
      }
    }
    state = 'ready';
    broadcast(true);
  } catch (err) {
    state = 'error';
    error = `Failed to start (${bootStage}): ${errText(err)}`;
    console.error(`[page-text] boot failed at stage "${bootStage}":`, err);
    broadcast(true);
  }
}

// --- image fetch (CORS strategy, same as Page 3D) ------------------------------

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

// --- inference ------------------------------------------------------------------

async function runModel(model, nchw, shape) {
  const input = Tensor.fromTypedArray(nchw, shape);
  const start = performance.now();
  const outputs = await model.run([input]);
  const data = await outputs[0].data();
  const ms = performance.now() - start;
  for (const output of outputs) output.delete();
  input.delete();
  return { data, ms };
}

const detCanvas = new OffscreenCanvas(DET_SIZE, DET_SIZE);
const detCtx = detCanvas.getContext('2d', { willReadFrequently: true });

async function runOcr(url, { forceRelay = false } = {}) {
  if (state !== 'ready') {
    // Boot may still be in flight (or previously failed) — wait for it here
    // so a request during the one-time model download still resolves.
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
  if (!forceRelay) {
    const stored = await dbGet(url);
    if (stored) {
      resultCache.set(url, stored);
      return stored;
    }
  }

  const { bytes, via, fetchMs } = await fetchImageBytes(url, { forceRelay });
  const bitmap = await createImageBitmap(new Blob([bytes]), { imageOrientation: 'from-image' });
  const nw = bitmap.width;
  const nh = bitmap.height;

  detCtx.drawImage(bitmap, 0, 0, nw, nh, 0, 0, DET_SIZE, DET_SIZE);
  const rgba = detCtx.getImageData(0, 0, DET_SIZE, DET_SIZE).data;
  const { nchw } = detPreprocess(rgba, nw, nh);
  const det = await runModel(detModel, nchw, [1, 3, DET_SIZE, DET_SIZE]);
  const boxes = probToBoxes(det.data);

  const { lines, recMs } = await recognizeLines(bitmap, boxes, {
    chars,
    recognize: (input) => runModel(recModel, input, [1, 3, REC_H, REC_W]),
    makeCanvas: (w, h) => new OffscreenCanvas(w, h),
  });
  bitmap.close();

  runCount++;
  lastStats = {
    fetchMs: +fetchMs.toFixed(0),
    detMs: +det.ms.toFixed(0),
    recMs: +recMs.toFixed(0),
    lineCount: lines.length,
    via,
  };
  broadcast(true);

  const payload = { ok: true, natural: { w: nw, h: nh }, lines, stats: { ...lastStats } };
  lastResult = payload;
  resultCache.set(url, payload);
  dbPut(url, payload);
  if (resultCache.size > CACHE_ENTRIES) {
    resultCache.delete(resultCache.keys().next().value);
  }
  return payload;
}

// --- persistent result cache -------------------------------------------------------

const DB_NAME = 'pagetext-index';
const STORE = 'reads';
let dbPromise = null;

function openDb() {
  dbPromise ??= new Promise((resolve) => {
    let req;
    try {
      req = indexedDB.open(DB_NAME, 1);
    } catch {
      resolve(null);
      return;
    }
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => resolve(null);
  });
  return dbPromise;
}

async function dbGet(url) {
  const db = await openDb();
  if (!db) return null;
  return new Promise((resolve) => {
    const req = db.transaction(STORE, 'readonly').objectStore(STORE).get(url);
    req.onsuccess = () => resolve(req.result ?? null);
    req.onerror = () => resolve(null);
  });
}

async function dbPut(url, payload) {
  const db = await openDb();
  if (!db) return;
  try {
    db.transaction(STORE, 'readwrite').objectStore(STORE).put(payload, url);
  } catch { /* quota or closed db — the memory LRU still covers this session */ }
}

// --- request queue ----------------------------------------------------------------

// Interactive reads (a right-click) must not wait behind a page's worth of
// background indexing, so there are two queues and background work only
// runs when the foreground queue is empty.
const queues = { fg: [], bg: [] };
let pumping = false;

function requestOcr(url, opts = {}) {
  return new Promise((resolve) => {
    queues[opts.background ? 'bg' : 'fg'].push({ url, opts, resolve });
    pump();
  });
}

async function pump() {
  if (pumping) return;
  pumping = true;
  for (;;) {
    const job = queues.fg.shift() ?? queues.bg.shift();
    if (!job) break;
    let out;
    try {
      out = await runOcr(job.url, job.opts);
    } catch (err) {
      out = { ok: false, error: String(err instanceof Error ? err.message : err) };
    }
    job.resolve(out);
  }
  pumping = false;
}

// Debug/smoke hook, dev builds only: lets CDP automation (tools-ocr/) poll
// engine state and run OCR.
if (__DEV__) {
  globalThis.__pt = {
    get status() { return statusPayload(); },
    get lastResult() { return lastResult; },
    async ocrSummary(url, opts) {
      const r = await requestOcr(url, opts);
      if (!r.ok) return r;
      return {
        ok: true,
        natural: r.natural,
        lineCount: r.lines.length,
        texts: r.lines.map((l) => l.text),
        scores: r.lines.map((l) => l.score),
        stats: r.stats,
      };
    },
    /** Dev-only: run the recognizer on a caller-built NCHW tensor. Lets
     * tools-ocr/probe-window.mjs sweep preprocessing choices for one crop. */
    async recRaw(nchw) {
      if (!__DEV__) return { text: '', score: 0 };
      const rec = await runModel(recModel, Float32Array.from(nchw), [1, 3, REC_H, REC_W]);
      const C = chars.length;
      const d = ctcDecode(rec.data, rec.data.length / C, C, chars);
      return { text: d.text, score: d.score };
    },
    /** Dev-only introspection: per det box, the rec strip as a data URL with
     * piece boundaries burned in as red lines, plus each piece's decode. */
    async debugOcr(url) {
      if (!__DEV__) return { ok: false, error: 'dev builds only' };
      const { bytes } = await fetchImageBytes(url);
      const bitmap = await createImageBitmap(new Blob([bytes]), { imageOrientation: 'from-image' });
      const nw = bitmap.width;
      const nh = bitmap.height;
      detCtx.drawImage(bitmap, 0, 0, nw, nh, 0, 0, DET_SIZE, DET_SIZE);
      const rgba = detCtx.getImageData(0, 0, DET_SIZE, DET_SIZE).data;
      const { nchw, scaleX, scaleY } = detPreprocess(rgba, nw, nh);
      const det = await runModel(detModel, nchw, [1, 3, DET_SIZE, DET_SIZE]);
      const boxes = probToBoxes(det.data);
      const out = [];
      for (const box of boxes) {
        const sx = box.x0 * scaleX;
        const sy = box.y0 * scaleY;
        const sw = (box.x1 - box.x0 + 1) * scaleX;
        const sh = (box.y1 - box.y0 + 1) * scaleY;
        const lw = Math.min(4096, Math.max(1, Math.round(REC_H * (sw / sh))));
        const strip = new OffscreenCanvas(lw, REC_H);
        const stripCtx = strip.getContext('2d', { willReadFrequently: true });
        stripCtx.drawImage(bitmap, sx, sy, sw, sh, 0, 0, lw, REC_H);
        const stripRgba = stripCtx.getImageData(0, 0, lw, REC_H).data;
        const { profile, bg } = columnInkProfile(stripRgba, lw, REC_H);
        const pieces = splitByInk(profile, lw, {
          maxW: REC_W - 2 * EDGE_PAD,
          squashLimit: (REC_W - 2 * EDGE_PAD) * 2.2,
        });
        const toUrl = async (canvas) => {
          const blob = await canvas.convertToBlob({ type: 'image/png' });
          return await new Promise((res, rej) => {
            const rd = new FileReader();
            rd.onload = () => res(rd.result);
            rd.onerror = rej;
            rd.readAsDataURL(blob);
          });
        };
        // exercise the exact runOcr window path per piece; run each input
        // twice (state-leak probe) and once padded to the full 320 with bg
        // instead of the preprocessor's −1 fill (padding-value probe)
        const pieceInfo = [];
        for (const { from, to } of pieces) {
          const pw = to - from;
          if (pw < 3) continue;
          const drawnW = Math.min(REC_W - 2 * EDGE_PAD, pw);
          const contentW = drawnW + 2 * EDGE_PAD;
          const win = new OffscreenCanvas(contentW, REC_H);
          const winCtx = win.getContext('2d', { willReadFrequently: true });
          winCtx.fillStyle = `rgb(${bg[0]},${bg[1]},${bg[2]})`;
          winCtx.fillRect(0, 0, contentW, REC_H);
          winCtx.drawImage(strip, from, 0, pw, REC_H, EDGE_PAD, 0, drawnW, REC_H);
          const rgbaPiece = winCtx.getImageData(0, 0, contentW, REC_H).data;
          const C = chars.length;
          const decode1 = ctcDecode((await runModel(
            recModel, recPreprocess(rgbaPiece, contentW), [1, 3, REC_H, REC_W])).data,
            40, C, chars);
          const decode2 = ctcDecode((await runModel(
            recModel, recPreprocess(rgbaPiece, contentW), [1, 3, REC_H, REC_W])).data,
            40, C, chars);
          const winFull = new OffscreenCanvas(REC_W, REC_H);
          const wfCtx = winFull.getContext('2d', { willReadFrequently: true });
          wfCtx.fillStyle = `rgb(${bg[0]},${bg[1]},${bg[2]})`;
          wfCtx.fillRect(0, 0, REC_W, REC_H);
          wfCtx.drawImage(win, 0, 0);
          const rgbaFull = wfCtx.getImageData(0, 0, REC_W, REC_H).data;
          const decodeBgPad = ctcDecode((await runModel(
            recModel, recPreprocess(rgbaFull, REC_W), [1, 3, REC_H, REC_W])).data,
            40, C, chars);
          // reconstruct the exact tensor the model saw, as an image
          const nchw = recPreprocess(rgbaPiece, contentW);
          const tImg = new ImageData(REC_W, REC_H);
          for (let i = 0; i < REC_H * REC_W; i++) {
            tImg.data[i * 4] = Math.round((nchw[i] + 1) * 127.5);
            tImg.data[i * 4 + 1] = Math.round((nchw[REC_H * REC_W + i] + 1) * 127.5);
            tImg.data[i * 4 + 2] = Math.round((nchw[2 * REC_H * REC_W + i] + 1) * 127.5);
            tImg.data[i * 4 + 3] = 255;
          }
          const tCanvas = new OffscreenCanvas(REC_W, REC_H);
          tCanvas.getContext('2d').putImageData(tImg, 0, 0);
          pieceInfo.push({ from, to, contentW,
            text: decode1.text, score: +decode1.score.toFixed(3),
            text2: decode2.text, textBgPad: decodeBgPad.text,
            winUrl: await toUrl(win), tensorUrl: await toUrl(tCanvas) });
        }
        stripCtx.fillStyle = 'rgba(255,0,0,.85)';
        for (const p of pieces.slice(1)) stripCtx.fillRect(p.from, 0, 2, REC_H);
        out.push({
          det: [box.x0, box.y0, box.x1, box.y1],
          src: [Math.round(sx), Math.round(sy), Math.round(sw), Math.round(sh)],
          lw,
          pieces: pieceInfo,
          stripUrl: await toUrl(strip),
        });
      }
      bitmap.close();
      return { ok: true, natural: { w: nw, h: nh }, boxes: out };
    },
  };
}

boot();
