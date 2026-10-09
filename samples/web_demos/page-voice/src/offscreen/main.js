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
 * Offscreen document: the TTS engine. Loads LiteRT.js (wasm bundled with the
 * extension), fetches Matcha-TTS model files from Hugging Face (Cache API
 * after the first download), synthesizes queued texts chunk by chunk and
 * plays them gaplessly through an AudioContext.
 *
 * Pipeline per chunk (≤127 phonemes), as in the Matcha-TTS web demo
 * (web_demos/src/matcha-tts/, which shares g2p.js and synth.js):
 *   text → G2P (dict + DeepPhonemizer on WASM) → symbol ids
 *   → text encoder (WebGPU) → durations → length-regulate
 *   → flow-matching decoder ×4 Euler steps (WASM — GPU mis-fuses this graph)
 *   → HiFi-GAN vocoder (WebGPU) → 22.05 kHz waveform.
 *
 * Messages in  (target 'offscreen'): {type:'speak', text} | {type:'stop'} |
 *   {type:'status'} (sync response).
 * Messages out (target 'ui'): status broadcasts — background relays them to
 *   content scripts; the popup receives them directly.
 */
import { isWebGPUSupported, loadAndCompile, loadLiteRt } from '@litertjs/core';
import { G2P, phonemize } from './g2p.js';
import { Synthesizer } from './synth.js';

const HF = 'https://huggingface.co/litert-community/Matcha-TTS/resolve/main/';
const FILES = {
  textenc: 'matcha_textenc_fp16.tflite',
  decoder: 'matcha_decoder_fp16.tflite',
  vocoder: 'matcha_vocoder_fp16.tflite',
  g2p: 'dp_g2p_matcha_fp16.tflite',
  emb: 'emb.bin',
  dict: 'g2p_dict.txt.gz',
  config: 'config.json',
  g2pMeta: 'g2p_meta.json',
};
const CACHE_NAME = 'page-voice-models-v1';
// 4 Euler steps: ear-approved quality at ~1/3 the latency of the full 10
// (validated in the matcha-tts demo).
const STEPS = 4;
const SEED = 0;

let state = 'loading'; // loading → downloading → compiling → ready | error
let error = null;
let backends = null;
let wasmOpts = null; // which loadLiteRt attempt succeeded
let g2p = null;
let synth = null;
let cfg = null;
let symToId = null;
let audioCtx = null;
let playCursor = null;
let liveSources = [];
let queue = [];
let processing = false;
let generation = 0; // bumped by stop() to cancel in-flight synthesis
let lastStats = null;
let downloadedMB = 0;
let spokenCount = 0; // total texts accepted into the queue (smoke/debug)

// Register the listener before any async work so no early message is lost.
chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (!msg || msg.target !== 'offscreen') return;
  if (msg.type === 'speak') {
    enqueue(msg.text);
    sendResponse({ ok: true });
  } else if (msg.type === 'stop') {
    stopAll();
    sendResponse({ ok: true });
  } else if (msg.type === 'status') {
    sendResponse(statusPayload());
  }
  return false;
});

function speaking() {
  return processing || queue.length > 0 ||
    (playCursor !== null && audioCtx && playCursor > audioCtx.currentTime);
}

function statusPayload() {
  return {
    target: 'ui',
    type: 'status',
    state,
    speaking: speaking(),
    error,
    env: envLabel(),
    flags: {
      webgpu: backends ? backends.textenc === 'webgpu' : null,
      threads: wasmOpts?.threads ?? null,
      jspi: wasmOpts?.jspi ?? null,
      crossOriginIsolated: globalThis.crossOriginIsolated,
    },
    backends,
    stats: lastStats,
    queued: queue.length,
    downloadedMB,
    spoken: spokenCount,
  };
}

function envLabel() {
  if (!backends) return null;
  const gpu = backends.textenc === 'webgpu' || backends.vocoder === 'webgpu';
  return gpu ? 'webgpu+wasm' : wasmOpts?.threads ? 'wasm' : 'wasm·1-thread';
}

let lastBroadcast = 0;
function broadcast(force = false) {
  const now = performance.now();
  if (!force && now - lastBroadcast < 250) return;
  lastBroadcast = now;
  chrome.runtime.sendMessage(statusPayload()).catch(() => {});
}

// --- asset loading ---------------------------------------------------------

function errText(err) {
  return err instanceof Error ? err.message : String(err);
}

async function fetchCached(name, onProgress) {
  const url = HF + name;
  const cache = 'caches' in globalThis ? await caches.open(CACHE_NAME) : null;
  if (cache) {
    const hit = await cache.match(url);
    if (hit) {
      const bytes = new Uint8Array(await hit.arrayBuffer());
      onProgress?.(bytes.length);
      return bytes;
    }
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

async function gunzip(bytes) {
  if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return bytes; // already plain
  const ds = new DecompressionStream('gzip');
  const stream = new Blob([bytes]).stream().pipeThrough(ds);
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function parseDict(bytes) {
  const text = new TextDecoder().decode(bytes);
  const dict = new Map();
  let start = 0;
  while (start < text.length) {
    let end = text.indexOf('\n', start);
    if (end === -1) end = text.length;
    const tab = text.indexOf('\t', start);
    if (tab > start && tab < end) {
      dict.set(text.slice(start, tab), text.slice(tab + 1, end));
    }
    start = end + 1;
  }
  return dict;
}

// --- boot -------------------------------------------------------------------

// Which boot stage is in flight, so a failure names it (runtime / download /
// compile), as the web demos' boot errors do.
let bootStage = 'runtime';

async function boot() {
  try {
    // Threaded wasm needs cross-origin isolation (COEP/COOP set in the
    // manifest); single-thread makes the CPU decoder ~5-50x slower.
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

    const gpu = isWebGPUSupported() ? 'webgpu' : 'wasm';
    backends = { g2p: 'wasm', textenc: gpu, decoder: 'wasm', vocoder: gpu };
    state = 'downloading';
    bootStage = 'download';
    broadcast(true);

    // ~94 MB total on first use; Cache API afterwards.
    const progress = {};
    const grab = (name) =>
      fetchCached(FILES[name], (received) => {
        progress[name] = received;
        let done = 0;
        for (const k in progress) done += progress[k];
        downloadedMB = Math.round(done / 1048576);
        broadcast();
      });
    const [teB, deB, voB, g2pB, embB, dictB, cfgB, metaB] = await Promise.all([
      grab('textenc'), grab('decoder'), grab('vocoder'), grab('g2p'),
      grab('emb'), grab('dict'), grab('config'), grab('g2pMeta'),
    ]);

    state = 'compiling';
    bootStage = 'compile';
    broadcast(true);
    cfg = JSON.parse(new TextDecoder().decode(cfgB));
    const meta = JSON.parse(new TextDecoder().decode(metaB));
    symToId = new Map();
    cfg.symbols.forEach((s, i) => {
      if (s.length === 1) symToId.set(s, i);
    });
    const dict = parseDict(await gunzip(dictB));
    const emb = new Float32Array(embB.buffer, embB.byteOffset, embB.byteLength / 4);

    const numThreads = Math.min(8, navigator.hardwareConcurrency || 4);
    const wasmCompile = { accelerator: 'wasm', cpuOptions: { numThreads } };
    const models = {};
    for (const [key, bytes] of [['textenc', teB], ['decoder', deB], ['vocoder', voB], ['g2p', g2pB]]) {
      if (backends[key] === 'wasm') {
        models[key] = await loadAndCompile(bytes, wasmCompile);
        continue;
      }
      try {
        models[key] = await loadAndCompile(bytes, { accelerator: backends[key] });
      } catch {
        // WebGPU advertised but compile failed (e.g. no adapter in this
        // context) — fall back to wasm and record the downgrade.
        backends[key] = 'wasm';
        models[key] = await loadAndCompile(bytes, wasmCompile);
      }
    }

    g2p = new G2P(dict, meta, models.g2p);
    synth = new Synthesizer(models, emb, cfg);
    state = 'ready';
    broadcast(true);
    processQueue();
  } catch (err) {
    state = 'error';
    error = `Failed to start (${bootStage}): ${errText(err)}`;
    console.error(`[page-voice] boot failed at stage "${bootStage}":`, err);
    broadcast(true);
  }
}

// --- speak queue + playback --------------------------------------------------

function enqueue(text) {
  text = (text ?? '').trim();
  if (!text) return;
  spokenCount++;
  queue.push(text);
  broadcast(true);
  processQueue();
}

function stopAll() {
  generation++;
  queue = [];
  for (const src of liveSources) {
    try { src.stop(); } catch { /* already ended */ }
  }
  liveSources = [];
  playCursor = null;
  broadcast(true);
}

async function processQueue() {
  if (processing || state !== 'ready') return;
  processing = true;
  try {
    audioCtx ??= new AudioContext({ sampleRate: cfg.sample_rate });
    if (audioCtx.state === 'suspended') {
      await audioCtx.resume().catch(() => {});
      if (audioCtx.state === 'suspended') {
        error = 'AudioContext suspended — autoplay blocked in offscreen document';
        broadcast(true);
      }
    }
    while (queue.length) {
      const gen = generation;
      const text = queue.shift();
      broadcast(true);
      await speakText(text, gen);
    }
  } finally {
    processing = false;
    broadcast(true);
  }
}

async function speakText(text, gen) {
  const t0 = performance.now();
  const chunks = await phonemize(g2p, symToId, text);
  const tG2p = performance.now() - t0;
  if (!chunks.length) return;
  const timings = { g2p: tG2p, textenc: 0, decoder: 0, vocoder: 0 };
  let audioSeconds = 0;
  for (const chunk of chunks) {
    if (gen !== generation) return; // stopped
    const r = await synth.run(chunk.ids, { steps: STEPS, seed: SEED });
    if (gen !== generation) return;
    for (const k of ['textenc', 'decoder', 'vocoder']) timings[k] += r.timings[k];
    audioSeconds += r.wav.length / cfg.sample_rate;
    playWav(r.wav);
    const totalMs = timings.g2p + timings.textenc + timings.decoder + timings.vocoder;
    lastStats = {
      totalMs: +totalMs.toFixed(0),
      audioSeconds: +audioSeconds.toFixed(1),
      rtf: +(totalMs / 1000 / audioSeconds).toFixed(2),
      steps: STEPS,
    };
    broadcast(true);
  }
}

function playWav(wav) {
  const buf = audioCtx.createBuffer(1, wav.length, cfg.sample_rate);
  buf.copyToChannel(wav, 0);
  const src = audioCtx.createBufferSource();
  src.buffer = buf;
  src.connect(audioCtx.destination);
  const at = Math.max(playCursor ?? 0, audioCtx.currentTime + 0.05);
  src.start(at);
  playCursor = at + buf.duration;
  liveSources.push(src);
  src.onended = () => {
    liveSources = liveSources.filter((s) => s !== src);
    broadcast(); // lets the HUD drop out of "speaking" when playback drains
  };
}

// Debug/smoke hook, dev builds only: lets CDP automation (tools/) poll
// engine state and speak.
if (__DEV__) {
  globalThis.__pv = {
    get status() { return statusPayload(); },
    speak: (text) => enqueue(text),
    stop: stopAll,
  };
}

boot();
