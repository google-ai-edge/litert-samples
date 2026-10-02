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

// SAM 2 web demo on the C++ ModelChain pipeline (wasm). The UI matches the
// LiteRT.js demo in ../../web; what runs underneath is different: every
// computation — frame preprocessing, the image encoder, per-object prompt /
// track steps with the memory bank, and the mask composite on screen — is a
// C++ Tensor API model inside the wasm module, chained by LiteRT's ModelChain
// and executed on WebGPU through LiteRT.js. This file handles the UI, moves
// frames into the pipeline (GPU copies) and blits its output.

import {runTurn, type ToolContext, type ToolEvent} from './agent';
import {ChainRuntime, clearModelCache, type Effect, type ProfileMode, type RunProfile} from './chain/runtime';
import {type DraftBox, GpuView, type Marker} from './display';
import {DEFAULT_SERVER, detect as gemmaDetect, type GemmaBox, type GemmaResult, listModels as gemmaModels} from './gemma';
import {detectWeb, isWebModel, type LlmInference, loadWebGemma, WEB_MODELS, webEngine, webGemmaLoaded, webGemmaSupported, webModel} from './gemma_web';
import {listening, speechSupported, startDictation, stopDictation} from './speech';
import {type JsonValue} from './toolcalls';
import {type AppActions, makeTools} from './tools';
import {type Clip, decodeVideo, demoClip} from './video';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const q = new URLSearchParams(location.search);

/** Object colours; object k uses pipeline slot k (the composite graph's palette). */
const COLORS: Array<[number, number, number]> = [[76, 141, 255], [255, 158, 44], [62, 207, 142],
  [240, 82, 156], [170, 110, 255], [250, 204, 21]];
// One colour per object; the loaded wasm pipeline may compile fewer slots.
const maxObjects = () => Math.min(COLORS.length, state.engine?.maxObjects ?? COLORS.length);
const MAX_CLICKS = 8;

// Static files are resolved against the page URL, so the built app also works
// from a sub-path (e.g. a GCS bucket folder), not only from the server root.
// Absolute URLs also keep dynamic import() of the wasm loaders page-relative.
const asset = (path: string) => new URL(path, document.baseURI).href;

const MODELS = {
  384: {url: asset('models/sam2_chain_384.tflite'), mb: 164},
  512: {url: asset('models/sam2_chain_512.tflite'), mb: 169},
  1024: {url: asset('models/sam2_chain_1024.tflite'), mb: 208},
} as const;
type ModelSize = keyof typeof MODELS;

/** A prompt point, normalized frame coords. label 1 / 0: positive / negative
 * click; 2 / 3: a box's top-left / bottom-right corner (SAM 2 encodes a box as
 * these two points, placed before any clicks). */
interface Click {
  nx: number;
  ny: number;
  label: 0 | 1 | 2 | 3;
}

/** Positive / negative clicks, or drag a box. */
type Tool = 'pos' | 'neg' | 'box';

/** "box + 2 clicks", "3 clicks". */
function describePrompt(pts: Click[]): string {
  const clicks = pts.filter((c) => c.label <= 1).length;
  const c = `${clicks} click${clicks === 1 ? '' : 's'}`;
  return pts.some((p) => p.label === 2) ? (clicks ? `box + ${c}` : 'box') : c;
}

/**
 * The object's next prompt: a new box replaces its previous one (boxes come
 * first), clicks accumulate after it. Returns a message instead when the
 * change isn't allowed.
 */
function composePrompt(prev: Click[], add: {click?: Click; box?: [Click, Click]}): Click[] | string {
  const box = add.box ?? prev.filter((c) => c.label >= 2);
  const clicks = [...prev.filter((c) => c.label <= 1), ...(add.click ? [add.click] : [])];
  if (add.click?.label === 0 && !box.length && !clicks.some((c) => c.label === 1)) {
    return 'Start with a positive click or a box; negative clicks then remove areas.';
  }
  const pts = [...box, ...clicks];
  if (pts.length > MAX_CLICKS) return `Up to ${MAX_CLICKS} points per object (a box counts as 2). Press Reset to start over.`;
  return pts;
}

interface TrackedObject {
  id: number;
  slot: number;  // pipeline object index = palette index
  color: [number, number, number];
  /** The prompt: 1..8 clicks (normalized coords) on one frame. */
  point: {frame: number; pts: Click[]} | null;
  /** What Gemma called it ("soccer ball"), when it came from Ask Gemma. */
  label?: string;
}

/** Camera mode: each processed frame is shown with its own masks. */
interface Live {
  stream: MediaStream;
  video: HTMLVideoElement;
  width: number;
  height: number;
  t: number;
  frameTimes: number[];
  // Per processed frame: deliver = capture -> the page sees it (rVFC), wait =
  // seen -> the pipeline takes it, model = pipeline, toScreen = capture -> done.
  timings: Array<{model: number; toScreen: number; deliver: number; wait: number}>;
  lastFrame: {captureTime?: number; presented: number; seen: number} | null;
  processed: number;  // presentedFrames of the last frame the pipeline took
  maskT: number;      // newest frame with masks (smooth display composites it)
  maskBorn: number;   // its camera capture time
  displayPending: boolean;
  draws: Array<{at: number; toScreen: number; maskLag: number}>;
  // Resolved by the frame callback after it updates lastFrame, so a loop that
  // waited for a frame always reads that frame's timestamps.
  waiters: Array<() => void>;
}

/**
 * Camera display. smooth: every camera frame is composited (in-graph, display
 * chain) with the newest masks as it arrives, so the video runs at camera
 * rate and masks trail by about one pipeline step. aligned: each processed
 * frame is shown with its own masks (exact, at the pipeline's rate).
 */
type LiveDisplay = 'smooth' | 'aligned';

const state = {
  tool: 'pos' as Tool,
  engine: null as ChainRuntime | null,
  clip: null as Clip | null,
  objects: [] as TrackedObject[],
  selected: 0,
  nextId: 1,
  results: new Map<number, Map<number, Float32Array>>(),  // frame -> object id -> low-res logits
  frame: 0,
  effect: 'overlay' as Effect,
  stroke: 3,
  nmm: (q.get('nmm') === '7' ? 7 : 2) as 7 | 2,
  size: (Number(q.get('size')) in MODELS ? Number(q.get('size')) : 384) as ModelSize,
  loading: false,
  live: null as Live | null,
  liveDisplay: (q.get('livedisplay') === 'aligned' ? 'aligned' : 'smooth') as LiveDisplay,
  busy: false,
  /** A Gemma request is in flight. */
  asking: false,
  /** Gemma models on offer (browser and LiteRT-LM server ids), or null when there are none. */
  gemma: null as string[] | null,
  tracking: false,
  stop: false,
  playing: false,
};
/** The object the chips box last scrolled into view. */
let shownSelected = -1;

const view = $<HTMLCanvasElement>('view');
const marks = $<HTMLCanvasElement>('marks');
const scrubber = $<HTMLInputElement>('scrubber');
const timeline = $<HTMLCanvasElement>('timeline');
let gpuView: GpuView | null = null;

// ---------------------------------------------------------------- overlay / status

function showOverlay(text: string, opts: {spinner?: boolean; progress?: number} = {}) {
  $('overlayMsg').hidden = false;
  $('overlayText').textContent = text;
  $('spinner').hidden = !opts.spinner;
  $('loadProgress').hidden = opts.progress === undefined;
  if (opts.progress !== undefined) $('loadBar').style.width = `${(opts.progress * 100).toFixed(1)}%`;
}
const hideOverlay = () => ($('overlayMsg').hidden = true);

type Metric = [label: string, value: string, w?: number];
function showMetrics(items: Metric[]) {
  $('stats').replaceChildren(...items.map(([label, value, w]) => {
    const cell = document.createElement('span');
    cell.className = 'metric';
    const k = document.createElement('span');
    k.textContent = label;
    const v = document.createElement('b');
    v.textContent = value;
    if (w) v.style.minWidth = `${w}ch`;
    cell.append(k, v);
    return cell;
  }));
}
const ms = (x: number) => `${x.toFixed(0)} ms`;
// Models are kept in the Cache API; window.__clearModelCache() empties it.
(window as unknown as {__clearModelCache: typeof clearModelCache}).__clearModelCache = clearModelCache;
// ?cam=WxH: requested camera size (upload, preprocess and composite scale with it).
const CAM = ((q.get('cam') ?? '').match(/^(\d+)x(\d+)$/)?.slice(1).map(Number) ?? [1280, 720]) as number[];
// ---------------------------------------------------------------- profiling
// ?profile=cpu: per-signature time for LiteRT.js run() to return, plus stage
// wall times, with the normal (overlapped) queue. ?profile=gpu: also waits for
// the GPU after the upload and after every signature, so each row is that
// stage's CPU + GPU time (the total gets longer; use it to split the time).
const PROFILE = (['cpu', 'gpu'].includes(q.get('profile') ?? '') ? q.get('profile') : null) as ProfileMode;
const profFrames: Array<Record<string, number>> = [];
function recordProfile(stages: {upload: number; encode: number; track: number; read: number; total: number},
                       runs: RunProfile[]) {
  const f: Record<string, number> = {upload: stages.upload};
  let inRuns = 0;
  for (const r of runs) {
    const ms = PROFILE === 'gpu' ? r.gpu : r.run + r.copy;
    f[r.key] = (f[r.key] ?? 0) + ms;
    inRuns += ms;
  }
  // Time in encode()/track() outside LiteRT.js run(): C++ ModelChain, JSPI, bindings.
  f['c++/jspi'] = stages.encode + stages.track - inRuns;
  f.read = stages.read;
  f.total = stages.total;
  profFrames.push(f);
  if (profFrames.length > 60) profFrames.shift();
}
/** Medians over the last 60 live frames, e.g. for pasting into a bug. */
function profileSummary(): string {
  if (!profFrames.length) return '';
  const keys = [...new Set(profFrames.flatMap((f) => Object.keys(f)))];
  const order = (k: string) => ['upload', 'preprocess', 'encode'].indexOf(k) + 1 ||
      (k.startsWith('track') || k.startsWith('prompt') ? 4 : 0) ||
      ({composite: 5, 'c++/jspi': 6, read: 7, total: 8} as Record<string, number>)[k] || 9;
  keys.sort((a, b) => order(a) - order(b));
  const med = (k: string) => {
    const v = profFrames.map((f) => f[k]).filter((x) => x !== undefined && !Number.isNaN(x)).sort((a, b) => a - b);
    return v.length ? v[Math.floor(v.length / 2)] : NaN;
  };
  return `profile=${PROFILE} (${profFrames.length} frames, median ms): ` +
      keys.map((k) => `${k} ${med(k).toFixed(1)}`).join(' · ');
}
(window as unknown as {__profile: () => string}).__profile = profileSummary;

const setupText = (n: number) =>
  `${n} object${n === 1 ? '' : 's'} · ${state.engine?.imageSize ?? state.size}px · ${state.nmm}-frame memory`;

/** Recent per-frame times in file mode (prompt and track steps), for the measure tool. */
const lastTrackMs: number[] = [];

function setStats(frameMs: number, nObjects: number) {
  lastTrackMs.push(frameMs);
  if (lastTrackMs.length > 60) lastTrackMs.shift();
  showMetrics([['Last frame', ms(frameMs), 7], ['', setupText(nObjects)]]);
}

// ---------------------------------------------------------------- pipeline access

/** Serializes pipeline work (the C++ pipeline is one stateful object). */
let queue: Promise<unknown> = Promise.resolve();
function exclusive<T>(fn: () => Promise<T>): Promise<T> {
  const p = queue.then(fn, fn);
  queue = p.catch(() => undefined);
  return p;
}

const source = (): {width: number; height: number} | null => state.live ?? state.clip;

/** Frame geometry -> the pipeline authors + compiles its frame model (preprocess + composite). */
async function applyGeometry() {
  const rt = state.engine, src = source();
  if (!rt || !src) return;
  await exclusive(() => rt.setGeometry(src.width, src.height));
  gpuView ??= new GpuView(view, marks, rt.device);
  gpuView.resize(src.width, src.height);
  rt.setEffect(state.effect, state.stroke);
  rt.setMemorySize(state.nmm);
  rt.setResultRetention(state.live ? 3 : -1);
}

function markersFor(t: number): Marker[] {
  if (state.playing) return [];
  return state.objects.flatMap((o) => o.point && o.point.frame === t
    ? o.point.pts.map((p) => ({...p, color: o.color})) : []);
}

/** Draws the pipeline's current composite + markers. */
function present(t: number) {
  const rt = state.engine;
  if (!rt || !gpuView) return;
  gpuView.show(rt.outputBuffer);
  gpuView.drawMarkers(markersFor(t));
}

async function loadEngine(size: ModelSize) {
  state.loading = true;
  refresh();
  const old = state.engine;
  state.engine = null;
  const precision = (q.get('precision') as 'fp16' | 'fp32') ?? 'fp16';
  const label = `C++ ModelChain (wasm) · LiteRT.js WebGPU ${precision} · ${size}px`;
  $('backendPill').textContent = `${label} · loading`;
  try {
    await queue;
    const {url, mb} = MODELS[size];
    const rt = await ChainRuntime.create({
      modelUrl: url,
      hostConstsUrl: asset('models/sam2_host_consts.safetensors'),
      weightsUrl: q.get('build') === 'browser' ? asset(`models/sam2_tiny_${size}_video.safetensors`) : undefined,
      imageSize: size,
      litertWasm: asset('litert-wasm/'),
      chainWasm: asset('wasm/'),
      nmm: state.nmm,
      precision,
      cache: q.get('cache') !== '0',  // ?cache=0: always download
      onProgress: (stage, f) => showOverlay(
          f === undefined ? `${stage}…` : `${stage} (${size}px) · ${(f * mb).toFixed(0)} / ${mb} MB`,
          {spinner: f === undefined, progress: f}),
    });
    old?.dispose();
    rt.setProfile(PROFILE);
    state.engine = rt;
    $('backendPill').textContent = label;
    state.loading = false;
    if (source()) {
      await applyGeometry();
      hideOverlay();
      await repromptAll();
    } else {
      showOverlay('Upload a video or use the sample');
    }
    refresh();
  } catch (e) {
    state.loading = false;
    console.error(e);
    $('backendPill').textContent = 'Pipeline failed to load';
    $('backendPill').classList.add('warn');
    showOverlay(`Could not load the pipeline: ${(e as Error).message}`);
  }
}

/** After a model switch: re-run every placed prompt on the new pipeline. */
async function repromptAll() {
  state.results.clear();
  if (state.live) {
    for (const o of state.objects) o.point = null;
    return refresh();
  }
  const placed = state.objects.filter((o) => o.point);
  state.busy = true;
  refresh();
  try {
    for (const o of placed) await runPrompt(o);
    if (placed.length) $('trackInfo').textContent = `Switched to the ${state.engine?.imageSize}px model. Press Track to re-run tracking.`;
  } finally {
    state.busy = false;
    refresh();
    render();
  }
}

/** Model-space clicks for the loaded model (SAM 2 squashes frames to S x S). */
function promptClicks(pts: Click[]) {
  const S = state.engine!.imageSize;
  return pts.map((p) => ({x: Math.min(Math.max(p.nx * S, 0), S - 1), y: Math.min(Math.max(p.ny * S, 0), S - 1),
    label: p.label}));
}

/** Uploads frame t of the clip into the pipeline's frame tensor. */
function uploadClipFrame(t: number) {
  state.engine!.uploadFrame(state.clip!.frames[t]);
}

/** Reads object masks of frame t back into state.results in one GPU readback. */
async function collect(t: number, objs: TrackedObject[]) {
  const rt = state.engine!;
  const res = state.results.get(t) ?? new Map<number, Float32Array>();
  state.results.set(t, res);
  const masks = await rt.readMasks(objs.map((o) => o.slot), t);
  objs.forEach((o, i) => {
    const m = masks[i];
    if (m) res.set(o.id, m); else res.delete(o.id);
  });
}

/** Yields to the macrotask queue without Chrome's 4 ms nested-setTimeout clamp. */
const yieldChannel = new MessageChannel();
const yieldToMain = () => new Promise<void>((r) => {
  yieldChannel.port1.onmessage = () => r();
  yieldChannel.port2.postMessage(null);
});

/** Prompt object `o` on its clicked frame: preprocess + encode + prompt{k} + composite. */
async function runPrompt(o: TrackedObject) {
  const rt = state.engine!, t = o.point!.frame;
  return exclusive(async () => {
    uploadClipFrame(t);
    await rt.encode(t);
    await rt.prompt(o.slot, t, promptClicks(o.point!.pts));
    present(t);
    await collect(t, [o]);
    return rt.readScores(o.slot, t);
  });
}

// ---------------------------------------------------------------- clip / objects

function resetObjects() {
  const rt = state.engine, slots = state.objects.map((o) => o.slot);
  if (rt) void exclusive(async () => slots.forEach((k) => rt.clearObject(k)));
  state.objects = [];
  state.nextId = 1;
  state.results.clear();
  addObject();
}

function addObject() {
  if (state.objects.length >= maxObjects()) return;
  const used = new Set(state.objects.map((o) => o.slot));
  const slot = COLORS.findIndex((_, k) => !used.has(k));
  const o: TrackedObject = {id: state.nextId++, slot, color: COLORS[slot], point: null};
  state.objects.push(o);
  state.selected = o.id;
  refresh();
}

function forget(o: TrackedObject) {
  const rt = state.engine;
  // Through the queue: in camera mode a pipeline step may be running.
  if (rt) void exclusive(async () => rt.clearObject(o.slot));
  for (const m of state.results.values()) m.delete(o.id);
}

function resetSelected() {
  if (state.tracking || state.busy) return;
  const obj = state.objects.find((o) => o.id === state.selected);
  if (!obj) return;
  obj.point = null;
  forget(obj);
  $('trackInfo').textContent = 'Object reset. Click it again to add points.';
  refresh();
  if (state.live) void exclusive(async () => {
    // Show the change right away (smooth mode redraws on the next camera frame).
    if (state.live && state.liveDisplay === 'aligned') {
      await state.engine?.composite(state.live.maskT);
      present(state.live.maskT);
    }
  });
  else render();
}

function removeObject(id: number) {
  const o = state.objects.find((x) => x.id === id);
  if (o) forget(o);
  state.objects = state.objects.filter((x) => x.id !== id);
  if (!state.objects.length) addObject();
  if (!state.objects.some((x) => x.id === state.selected)) state.selected = state.objects[0].id;
  refresh();
  render();
}

async function setClip(clip: Clip) {
  stopPlayback();
  state.clip?.frames.forEach((f) => f.close());
  state.clip = clip;
  state.frame = 0;
  scrubber.max = String(clip.frames.length - 1);
  scrubber.value = '0';
  $('clipInfo').textContent =
      `${clip.name} · ${clip.frames.length} frames @ ${clip.fps} fps · ${clip.width}×${clip.height}`;
  resetObjects();
  if (state.engine) {
    await applyGeometry();
    hideOverlay();
  }
  render();
  showTab('objects');
}

// ---------------------------------------------------------------- rendering

/** Frame t through the pipeline's composite (coalesced: only the latest request runs). */
let renderWanted = false;
function render() {
  if (state.live || !state.clip) return;
  scrubber.value = String(state.frame);
  $('counter').textContent = `${state.frame + 1} / ${state.clip.frames.length}`;
  drawTimeline();
  if (!state.engine || !gpuView || renderWanted) return;
  renderWanted = true;
  void exclusive(async () => {
    renderWanted = false;
    const rt = state.engine;
    if (!rt || !state.clip || state.live) return;
    const t = state.frame;
    uploadClipFrame(t);
    await rt.composite(t);
    present(t);
  }).catch((e) => console.error(e));
}

function drawTimeline() {
  const clip = state.clip;
  if (!clip) return;
  const w = timeline.clientWidth || 600;
  if (timeline.width !== w) timeline.width = w;
  const g = timeline.getContext('2d')!;
  g.clearRect(0, 0, w, timeline.height);
  const n = clip.frames.length;
  const cw = w / n;
  for (let t = 0; t < n; t++) {
    const r = state.results.get(t);
    if (!r || !r.size) continue;
    g.fillStyle = 'rgba(123,140,255,0.85)';
    g.fillRect(Math.floor(t * cw), 0, Math.ceil(cw), timeline.height);
  }
  for (const o of state.objects) {
    if (!o.point) continue;
    g.fillStyle = `rgb(${o.color.join(',')})`;
    g.fillRect(Math.floor(o.point.frame * cw), 0, Math.max(3, Math.ceil(cw)), timeline.height);
  }
}

function refresh() {
  const list = $('objectList');
  const chips = list.parentElement!;
  const scrollTop = chips.scrollTop;  // rebuilding the list would reset it
  list.innerHTML = '';
  state.objects.forEach((o, i) => {
    const li = document.createElement('li');
    li.className = 'obj' + (o.id === state.selected ? ' sel' : '');
    li.style.setProperty('--c', `rgb(${o.color.join(',')})`);
    const n = o.point?.pts.length ?? 0;
    const status = !o.point ? 'click or box it' : state.live ? `live · ${describePrompt(o.point.pts)}`
      : `${describePrompt(o.point.pts)} · f${o.point.frame + 1}`;
    const name = o.label ? `${i + 1} · ${o.label}` : `Object ${i + 1}`;
    li.innerHTML = `<span class="dot"></span><span class="name">${name.replace(/</g, '&lt;')}</span>` +
        `<span class="state">${status}</span>`;
    if (state.objects.length > 1) {
      const x = document.createElement('button');
      x.className = 'x';
      x.title = 'Remove object';
      x.textContent = '×';
      x.onclick = (ev) => {
        ev.stopPropagation();
        if (!state.tracking) removeObject(o.id);
      };
      li.appendChild(x);
    }
    li.onclick = () => {
      state.selected = o.id;
      refresh();
    };
    list.appendChild(li);
  });
  chips.scrollTop = scrollTop;
  if (state.selected !== shownSelected) {  // e.g. a new object: scroll it into the box
    shownSelected = state.selected;
    const sel = list.querySelector<HTMLElement>('.sel');
    if (sel && sel.offsetTop < chips.scrollTop) chips.scrollTop = sel.offsetTop;
    else if (sel && sel.offsetTop + sel.offsetHeight > chips.scrollTop + chips.clientHeight) {
      chips.scrollTop = sel.offsetTop + sel.offsetHeight - chips.clientHeight;
    }
  }
  const live = !!state.live;
  const ready = !!state.engine && !!state.clip && !state.loading && !live;
  $<HTMLButtonElement>('addObjBtn').disabled = state.objects.length >= maxObjects() || state.tracking;
  $<HTMLButtonElement>('resetObjBtn').disabled = state.tracking || state.busy ||
      !state.objects.find((o) => o.id === state.selected)?.point;
  for (const b of $('polSeg').querySelectorAll('button')) {
    b.disabled = state.tracking;
    b.classList.toggle('on', b.dataset.v === state.tool);
  }
  const trackBtn = $<HTMLButtonElement>('trackBtn');
  trackBtn.disabled = !ready || (!state.tracking && !state.objects.some((o) => o.point));
  trackBtn.textContent = live ? 'Tracking live (automatic)' : state.tracking ? 'Stop tracking' : 'Track objects';
  trackBtn.classList.toggle('danger', state.tracking);
  trackBtn.classList.toggle('primary', !state.tracking);
  $<HTMLButtonElement>('playBtn').disabled = !state.clip || state.tracking || live;
  scrubber.disabled = !state.clip || state.tracking || live;
  const canAsk = !!state.gemma && !!state.engine && !state.loading && !state.tracking && !state.busy &&
      !state.asking && (!!state.clip || live);
  $<HTMLInputElement>('gemmaAsk').disabled = !state.gemma || state.asking;
  $<HTMLButtonElement>('gemmaBtn').disabled = !canAsk;
  $<HTMLButtonElement>('gemmaBtn').textContent = state.asking ? 'Working…' : useAgent() ? 'Ask' : 'Find';
  const sel = $<HTMLSelectElement>('gemmaModel');
  sel.disabled = !state.gemma || state.asking;
  for (const o of sel.options) o.disabled = !!state.gemma && !state.gemma.includes(o.value);
  const mic = $<HTMLButtonElement>('micBtn');
  mic.hidden = !speechSupported();
  mic.disabled = !state.gemma || state.asking;
  mic.classList.toggle('listening', listening());
  $('liveSeg').hidden = !live;
  for (const b of $('liveSeg').querySelectorAll('button')) b.classList.toggle('on', b.dataset.v === state.liveDisplay);
  const cam = $<HTMLButtonElement>('camBtn');
  cam.textContent = live ? 'Stop camera' : 'Use camera';
  cam.classList.toggle('live', live);
  cam.disabled = state.tracking;
  for (const b of $('memSeg').querySelectorAll('button')) {
    b.disabled = state.tracking;
    b.classList.toggle('on', Number(b.dataset.v) === state.nmm);
  }
  for (const b of $('sizeSeg').querySelectorAll('button')) {
    // The live loop runs on the current pipeline; switching would dispose it mid-frame.
    b.disabled = state.tracking || state.loading || state.busy || live;
    b.dataset.title ??= b.title;
    b.title = live ? 'Stop the camera to switch models' : b.dataset.title;
    b.classList.toggle('on', Number(b.dataset.v) === state.size);
  }
  $<HTMLButtonElement>('demoBtn').disabled = state.tracking;
  $<HTMLSelectElement>('sampleSel').disabled = state.tracking;
  $('uploadBtn').classList.toggle('disabled', state.tracking);
  view.classList.toggle('busy', state.busy || state.tracking);
}

// ---------------------------------------------------------------- interaction

/** Canvas event -> normalized [0,1] frame coords (object-fit: contain). */
function eventToFrame(ev: MouseEvent): [number, number] | null {
  const r = view.getBoundingClientRect();
  const scale = Math.min(r.width / view.width, r.height / view.height);
  const w = view.width * scale, h = view.height * scale;
  const x = (ev.clientX - r.left - (r.width - w) / 2) / w;
  const y = (ev.clientY - r.top - (r.height - h) / 2) / h;
  return x < 0 || x > 1 || y < 0 || y > 1 ? null : [x, y];
}

/** Label of a click: the tool, or the other type with Shift. */
function clickLabel(ev: MouseEvent): 0 | 1 {
  const pos = state.tool !== 'neg';
  return (ev.shiftKey ? !pos : pos) ? 1 : 0;
}

view.addEventListener('click', (ev) => {
  if (state.tool === 'box' || state.asking) return;  // boxes are dragged (pointer handlers below)
  const p = eventToFrame(ev);
  if (p) void addPrompt({click: {nx: p[0], ny: p[1], label: clickLabel(ev)}});
});

// ---- box: drag on the video
let drag: DraftBox | null = null;
function eventToFrameClamped(ev: MouseEvent): [number, number] {
  const r = view.getBoundingClientRect();
  const scale = Math.min(r.width / view.width, r.height / view.height);
  const w = view.width * scale, h = view.height * scale;
  const x = (ev.clientX - r.left - (r.width - w) / 2) / w;
  const y = (ev.clientY - r.top - (r.height - h) / 2) / h;
  return [Math.min(Math.max(x, 0), 1), Math.min(Math.max(y, 0), 1)];
}
function redrawMarks() {
  const t = state.live ? state.live.maskT : state.frame;
  gpuView?.drawMarkers(markersFor(t), drag ?? undefined);
}
view.addEventListener('pointerdown', (ev) => {
  if (state.tool !== 'box' || !eventToFrame(ev) || state.busy || state.tracking || state.asking) return;
  const [x, y] = eventToFrameClamped(ev);
  const obj = state.objects.find((o) => o.id === state.selected)!;
  drag = {x0: x, y0: y, x1: x, y1: y, color: obj.color};
  view.setPointerCapture(ev.pointerId);
  ev.preventDefault();
});
view.addEventListener('pointermove', (ev) => {
  if (!drag) return;
  [drag.x1, drag.y1] = eventToFrameClamped(ev);
  redrawMarks();
});
view.addEventListener('pointerup', (ev) => {
  if (!drag) return;
  const d = drag;
  drag = null;
  view.releasePointerCapture(ev.pointerId);
  redrawMarks();
  const x0 = Math.min(d.x0, d.x1), x1 = Math.max(d.x0, d.x1), y0 = Math.min(d.y0, d.y1), y1 = Math.max(d.y0, d.y1);
  if ((x1 - x0) * view.width < 8 || (y1 - y0) * view.height < 8) {
    $('trackInfo').textContent = 'Drag to draw a box around the object.';
    return;
  }
  void addPrompt({box: [{nx: x0, ny: y0, label: 2}, {nx: x1, ny: y1, label: 3}]});
});
view.addEventListener('pointercancel', () => {
  drag = null;
  redrawMarks();
});

/** Adds a click or a box to the selected object and re-segments it. */
async function addPrompt(add: {click?: Click; box?: [Click, Click]}, target?: TrackedObject) {
  if (state.live) return livePrompt(state.live, add, target);
  if (!state.engine || !state.clip || state.tracking || state.busy) return;
  stopPlayback();
  const obj = target ?? state.objects.find((o) => o.id === state.selected)!;
  const t = state.frame;
  // Same frame: add to the object's prompt; another frame: start over there.
  const prev = obj.point && obj.point.frame === t ? obj.point.pts : [];
  const pts = composePrompt(prev, add);
  if (typeof pts === 'string') {
    $('trackInfo').textContent = pts;
    return;
  }
  obj.point = {frame: t, pts};
  // A new prompt replaces everything tracked for this object.
  for (const m of state.results.values()) m.delete(obj.id);
  state.busy = true;
  refresh();
  try {
    const t0 = performance.now();
    const scores = await runPrompt(obj);
    const took = performance.now() - t0;
    const appearing = !!scores && scores[0] > 0;
    $('trackInfo').textContent = appearing
      ? `Segmented with ${describePrompt(pts)} in ${took.toFixed(0)} ms (predicted IoU ${scores![1].toFixed(2)}). ` +
        'Refine with clicks, add objects, or press Track.'
      : add.box ? 'No object found in that box. Try a tighter box.' : 'No object found at that point. Try clicking closer to its center.';
    setStats(took, 1);
  } catch (e) {
    console.error(e);
    $('trackInfo').textContent = `Error: ${(e as Error).message}`;
  } finally {
    state.busy = false;
    refresh();
    drawTimeline();
  }
}

// ---------------------------------------------------------------- Ask Gemma

const GEMMA_SERVER = q.get('llm') ?? DEFAULT_SERVER;
// tools/gemma_server.sh: Gemma 4 on LiteRT-LM (language model and vision encoder on the GPU).
const GEMMA_START = `tools/gemma_server.sh ${location.origin}`;
const objHint = (text: string) => ($('objHint').textContent = text);
const OBJ_HINT = 'Click to add points, or choose Box and drag around the object. Tracking starts from that frame.';

/**
 * Gemma models on offer: Gemma 4 in the browser (MediaPipe on WebGPU), and
 * whatever the LiteRT-LM server has imported. Re-checked on focus.
 */
async function probeGemma() {
  const ids = await gemmaModels(GEMMA_SERVER);
  const server = ids ? ids.filter((id) => /gemma-4-e[24]b/i.test(id)) : [];
  const web = webGemmaSupported() ? WEB_MODELS.map((m) => m.id) : [];
  state.gemma = web.length || server.length ? [...web, ...server] : null;
  if (!state.gemma) {
    objHint(ids
      ? 'LiteRT-LM is running but has no Gemma 4 model: litert-lm import --from-huggingface-repo ' +
        'litert-community/gemma-4-E2B-it-litert-lm gemma-4-E2B-it.litertlm gemma-4-e2b'
      : `Ask Gemma needs WebGPU, or LiteRT-LM on this machine: ${GEMMA_START}`);
  } else if (!state.asking && /^Ask Gemma needs|^LiteRT-LM is running/.test($('objHint').textContent ?? '')) {
    objHint(OBJ_HINT);
  }
  // Default: E4B in the browser (no server needed; tighter boxes than E2B), else what the server has.
  const sel = $<HTMLSelectElement>('gemmaModel');
  if (state.gemma && (!sel.dataset.chosen || !state.gemma.includes(sel.value))) {
    sel.value = state.gemma.find((id) => /e4b/i.test(id)) ?? state.gemma[0];
  }
  refresh();
}

/** The chosen Gemma: its id, display name, and (browser models) the loaded runtime. */
interface Chosen {
  modelId: string;
  name: string;
  llm?: LlmInference;
}

/** Loads the chosen model if it runs in the browser (downloaded once, then cached). */
async function chooseGemma(): Promise<Chosen> {
  const modelId = $<HTMLSelectElement>('gemmaModel').value;
  const web = isWebModel(modelId) ? webModel(modelId) : undefined;
  const name = web ? `${web.name} (browser)`
    : modelId.replace(/^gemma-4-/i, 'Gemma 4 ').toUpperCase().replace('GEMMA 4', 'Gemma 4');
  if (!web) return {modelId, name};
  if (!webGemmaLoaded(modelId)) objHint(`Starting ${name}…`);
  try {
    const llm = await loadWebGemma(modelId, (text, f) =>
      objHint(f !== undefined && f < 1 ? `${text} (${(f * 100).toFixed(0)}%)` : text));
    return {modelId, name, llm};
  } catch (e) {
    throw new Error(`${(e as Error).message}. Needs WebGPU with shader-f16, and ~4 GB free disk for the model cache.`);
  }
}

/**
 * Gemma 4 finds what the user described in the frame on screen, and the
 * request replaces the selection: all objects are reset and each box (up to
 * `max`) becomes a fresh object with a SAM 2 box prompt. Boxes are used as
 * Gemma streams them, so the first mask shows while the rest are still being
 * written. If Gemma finds nothing, the objects are kept.
 */
async function findObjects(g: Chosen, what: string, max = maxObjects()): Promise<{found: number; labels: string[]; seconds: number}> {
  stopPlayback();
  // The frame on screen now (loading a model can take a while).
  const L = state.live;
  const frame: CanvasImageSource | undefined = L ? L.video : state.clip?.frames[state.frame];
  if (!frame) throw new Error('no frame on screen');
  objHint(`${g.name} is looking for “${what}”…`);
  const t0 = performance.now();
  const labels: string[] = [];
  const limit = Math.min(max, maxObjects());
  // One box at a time, in order: the first resets the objects, the others add one each.
  let chain = Promise.resolve();
  const onBox = (b: GemmaBox, i: number) => (chain = chain.then(async () => {
    if (i === 0) resetObjects();
    else addObject();
    const target = state.objects[state.objects.length - 1];
    target.label = b.label || what;
    state.selected = target.id;
    await addPrompt({box: [{nx: b.x0, ny: b.y0, label: 2}, {nx: b.x1, ny: b.y1, label: 3}]}, target);
    if (b.label) labels.push(b.label);
    objHint(`${g.name}: ${i + 1} found in ${((performance.now() - t0) / 1000).toFixed(1)} s, looking for more…`);
  }));
  let found: GemmaResult;
  try {
    found = g.llm
      ? await detectWeb(g.llm, frame, what, g.modelId, onBox, limit)
      : await gemmaDetect(frame, what, `${g.modelId},gpu`, GEMMA_SERVER, onBox, limit);
    await chain;
  } catch (e) {
    throw new Error(`${(e as Error).message}${g.llm ? '' : `. Is LiteRT-LM running? ${GEMMA_START}`}`);
  }
  refresh();
  if (!found.boxes.length) {
    objHint(`${g.name} found no “${what}” in this frame (${found.seconds.toFixed(1)} s); your objects are unchanged.`);
    console.info(`${g.name} replied:`, found.raw);
    return {found: 0, labels, seconds: found.seconds};
  }
  const used = Math.min(found.boxes.length, limit);
  objHint(`${g.name} found ${used} (${[...new Set(labels)].join(', ') || what}) in ${found.seconds.toFixed(1)} s` +
      (used === maxObjects() ? ` (${maxObjects()} objects max)` : '') +
      '. Refine with clicks or press Track.');
  return {found: used, labels: [...new Set(labels)], seconds: found.seconds};
}

/** Find only (the server models, or ?agent=0): the text is what to look for. */
async function askGemma(what: string) {
  if (!state.gemma || state.asking || state.busy || state.tracking) return;
  if (!(state.live ? state.live.video : state.clip)) return;
  state.asking = true;
  refresh();
  try {
    const g = await chooseGemma();
    await findObjects(g, what);
  } catch (e) {
    console.error(e);
    objHint(`Gemma failed: ${(e as Error).message}`);
  }
  state.asking = false;
  refresh();
}

// ---------------------------------------------------------------- the agent
//
// With a browser model the request is a plan: Gemma answers with tool calls
// (find objects, change the effect, remove objects, playback, camera, model
// quality, describe, measure) that run as they stream in. The calls show up
// as chips under the box, so what the model decided is visible.

/** Browser models run the agent; the server path and ?agent=0 keep plain Find. */
function useAgent(): boolean {
  return q.get('agent') !== '0' && isWebModel($<HTMLSelectElement>('gemmaModel').value);
}

const EFFECT_NAMES: Record<string, Effect> = {overlay: 'overlay', spotlight: 'spotlight', cutout: 'cutout'};

/** What the tools can do to the app. */
const appActions: AppActions = {
  maxObjects,
  findObjects: (what, max, ctx) => agentTurn!.gemma.then(async (g) => {
    await ctx.afterGeneration;  // one model: the plan must finish streaming before the vision call
    if (ctx.signal.aborted) throw new Error('cancelled');
    return findObjects(g, what, max);
  }),
  setEffect(effect, outline) {
    state.effect = EFFECT_NAMES[effect] ?? state.effect;
    if (outline !== undefined) state.stroke = outline;
    for (const b of $('fxSeg').querySelectorAll('button')) b.classList.toggle('on', b.dataset.v === state.effect);
    for (const b of $('outlineSeg').querySelectorAll('button')) b.classList.toggle('on', Number(b.dataset.v) === state.stroke);
    state.engine?.setEffect(state.effect, state.stroke);
    rerenderStill();
  },
  objects: () => state.objects.map((o) => ({id: o.id, label: o.label ?? '', prompted: !!o.point})),
  removeObjects(ids) {
    if (state.tracking) throw new Error('tracking is running; stop it first');
    for (const id of ids) removeObject(id);
  },
  removeAll() {
    if (state.tracking) throw new Error('tracking is running; stop it first');
    resetObjects();
    render();
  },
  async playback(action) {
    if (state.live && action !== 'stop') return 'the camera is live: masks follow automatically';
    if (!state.clip) throw new Error('no video loaded');
    switch (action) {
      case 'track':
        if (state.tracking) return 'already tracking';
        if (!state.objects.some((o) => o.point)) throw new Error('nothing to track: find or click an object first');
        void track();
        return 'tracking through the video';
      case 'play':
        if (!state.playing) togglePlay();
        return 'playing';
      case 'pause':
        stopPlayback();
        return 'paused';
      case 'restart':
        stopPlayback();
        state.frame = 0;
        render();
        return 'at the first frame';
      case 'stop':
        if (state.tracking) state.stop = true;
        stopPlayback();
        return 'stopped';
    }
  },
  async useCamera(on) {
    if (on === !!state.live) return on ? 'the camera is already on' : 'the camera is already off';
    if (on) await startCamera();
    else await stopCamera();
    return state.live ? `camera on, ${state.live.width}×${state.live.height}` : 'camera off, back to the video';
  },
  async setQuality(size, memory) {
    const notes: string[] = [];
    if (memory && memory !== state.nmm) {
      state.nmm = memory;
      state.engine?.setMemorySize(memory);
      for (const b of $('memSeg').querySelectorAll('button')) b.classList.toggle('on', Number(b.dataset.v) === memory);
      notes.push(`${memory}-frame memory`);
    }
    if (size && size !== state.size) {
      if (state.live) throw new Error('stop the camera to switch models');
      state.size = size;
      for (const b of $('sizeSeg').querySelectorAll('button')) b.classList.toggle('on', Number(b.dataset.v) === size);
      await loadEngine(size);
      notes.push(`${size} px model loaded`);
    }
    return notes.join(', ') || 'already set';
  },
  describeScene(): JsonValue {
    const L = state.live;
    return {
      source: L ? {camera: true, width: L.width, height: L.height}
        : state.clip ? {video: state.clip.name, frames: state.clip.frames.length, fps: state.clip.fps, frame: state.frame + 1} : null,
      objects: state.objects.filter((o) => o.point).map((o) => ({id: o.id, label: o.label ?? 'unnamed', prompt: describePrompt(o.point!.pts)})),
      effect: state.effect,
      model: `${state.size} px, ${state.nmm}-frame memory, WebGPU fp16`,
      tracking: state.tracking,
      playing: state.playing,
    };
  },
  measure(): JsonValue {
    const L = state.live;
    const med = (xs: number[]) => (xs.length ? [...xs].sort((a, b) => a - b)[xs.length >> 1] : null);
    const perFrame = L ? med(L.timings.map((t) => t.model)) : med(lastTrackMs);
    const toScreen = L ? med(L.timings.map((t) => t.toScreen)) : null;
    return {
      ms_per_frame: perFrame !== null ? Math.round(perFrame) : null,
      fps: perFrame ? Math.round(1000 / perFrame) : null,
      camera_to_screen_ms: toScreen !== null ? Math.round(toScreen) : null,
      objects: state.objects.filter((o) => o.point).length,
      model: `${state.size} px, ${state.nmm}-frame memory`,
      profile: PROFILE ? profileSummary() : 'open with ?profile=gpu for per-stage GPU times',
      note: perFrame === null ? 'nothing has run yet: track objects or start the camera first' : null,
    };
  },
};
const tools = makeTools(appActions);

/** The turn in flight. */
let agentTurn: {gemma: Promise<Chosen>; abort: AbortController} | null = null;

/** A summary line for the prompt, so the model knows what is already true. */
function sceneLine(): string {
  const objs = state.objects.filter((o) => o.point);
  return (state.live ? 'camera is on' : state.clip ? `video "${state.clip.name}" at frame ${state.frame + 1}` : 'no video') +
      `; ${objs.length ? `tracked: ${objs.map((o) => o.label ?? `object ${o.id}`).join(', ')}` : 'no objects yet'}` +
      `; effect ${state.effect}; model ${state.size} px.`;
}

const fmtArgs = (a: Record<string, JsonValue>) => Object.entries(a).map(([k, v]) => `${k}: ${JSON.stringify(v)}`).join(', ');

/** One chip per call, updated in place: name(args) and its status. */
function showCall(e: ToolEvent) {
  const log = $('agentLog');
  log.hidden = false;
  let li = log.querySelector<HTMLElement>(`[data-id="${e.id}"]`);
  if (!li) {
    li = document.createElement('li');
    li.dataset.id = String(e.id);
    log.appendChild(li);
  }
  li.className = `call ${e.status}`;
  const tail = e.status === 'started' ? '…' : e.status === 'done' ? ` ✓${e.ms !== undefined && e.ms >= 100 ? ` ${(e.ms / 1000).toFixed(1)} s` : ''}`
    : ` ✗ ${e.error ?? ''}`;
  li.textContent = `${e.name}(${fmtArgs(e.arguments)})${tail}`;
  li.title = e.result !== undefined ? JSON.stringify(e.result) : e.error ?? '';
}

/** Runs the request as an agent turn: Gemma plans with tool calls, the app executes them as they arrive. */
async function runAgent(utterance: string) {
  if (!state.gemma || state.asking || state.busy || state.tracking) return;
  if (!(state.live ? state.live.video : state.clip)) return;
  state.asking = true;
  refresh();
  const log = $('agentLog');
  log.innerHTML = '';
  log.hidden = true;
  const abort = new AbortController();
  const gemma = chooseGemma();
  agentTurn = {gemma, abort};
  try {
    const g = await gemma;
    if (!g.llm) throw new Error('the agent needs a browser model');
    objHint(`${g.name} is working on “${utterance}”…`);
    const result = await runTurn(utterance, {
      engine: webEngine(g.llm),
      tools,
      scene: sceneLine,
      onEvent: showCall,
      onText: (text) => objHint(`${g.name}: ${text}`),
      signal: abort.signal,
    });
    console.info(`${g.name} planned:`, result.raw);
    if (!result.events.length && !result.text) objHint(`${g.name} had nothing to do for “${utterance}”.`);
    else if (!result.text && result.events.every((e) => e.status !== 'done')) {
      objHint(`${g.name} could not do that: ${result.events.map((e) => e.error).filter(Boolean).join('; ')}`);
    }
  } catch (e) {
    console.error(e);
    objHint(`Gemma failed: ${(e as Error).message}`);
  }
  agentTurn = null;
  state.asking = false;
  refresh();
}

$('gemmaModel').addEventListener('change', (ev) => {
  (ev.target as HTMLElement).dataset.chosen = '1';
  refresh();
});
$<HTMLFormElement>('gemmaForm').addEventListener('submit', (ev) => {
  ev.preventDefault();
  const what = $<HTMLInputElement>('gemmaAsk').value.trim();
  if (!what) return;
  if (useAgent()) void runAgent(what);
  else void askGemma(what);
});
window.addEventListener('keydown', (ev) => {
  if (ev.code === 'Escape' && agentTurn) agentTurn.abort.abort();
});

// Voice: Chrome speech recognition fills the box, and a final phrase runs it.
$('micBtn').addEventListener('click', async () => {
  if (listening()) return stopDictation();
  const input = $<HTMLInputElement>('gemmaAsk');
  const before = $('objHint').textContent ?? '';
  try {
    const mode = await startDictation({
      onText(text, final) {
        input.value = text.replace(/[.!?]+$/, '');
        if (final && input.value && !$<HTMLButtonElement>('gemmaBtn').disabled) {
          $<HTMLFormElement>('gemmaForm').requestSubmit();
        }
      },
      onStatus: objHint,
      onEnd(error) {
        if (error) objHint(`Speech: ${error}.`);
        else if (/^Listening/.test($('objHint').textContent ?? '')) objHint(before);
        refresh();
      },
    });
    objHint(`Listening (${mode} speech recognition)… ` + (useAgent()
      ? 'say what to find or do, e.g. “find all the players and cut them out”.' : 'say what to find, e.g. “the ball”.'));
  } catch (e) {
    objHint(`Speech: ${(e as Error).message}.`);
  }
  refresh();
});
window.addEventListener('focus', () => void probeGemma());

/** Tracks every prompted object through the clip: one ModelChain run per frame. */
async function track() {
  const clip = state.clip!, rt = state.engine!;
  const objs = state.objects.filter((o) => o.point);
  if (!objs.length) return;
  stopPlayback();
  state.tracking = true;
  state.stop = false;
  refresh();
  const start = Math.min(...objs.map((o) => o.point!.frame));
  const n = clip.frames.length;
  for (const o of objs) for (const [t, m] of state.results) if (t !== o.point!.frame) m.delete(o.id);
  $('progress').hidden = false;
  const times: number[] = [];
  try {
    await exclusive(async () => {
      for (let t = start; t < n && !state.stop; t++) {
        const t0 = performance.now();
        uploadClipFrame(t);
        await rt.encode(t);
        // An object's own frame keeps its prompt (the pipeline holds it since
        // the click); re-prompt only if the pipeline lost it (model switch).
        for (const o of objs) {
          if (o.point!.frame === t && rt.condFrame(o.slot) !== t) {
            await rt.prompt(o.slot, t, promptClicks(o.point!.pts));
          }
        }
        await rt.track(t);  // track{n} for every object prompted before t, then composite
        state.frame = t;
        present(t);
        const active = objs.filter((o) => o.point!.frame <= t);
        await collect(t, active);
        const took = performance.now() - t0;
        times.push(took);
        scrubber.value = String(t);
        $('counter').textContent = `${t + 1} / ${n}`;
        drawTimeline();
        setStats(took, active.length);
        const done = t - start + 1, total = n - start;
        const avg = times.slice(-5).reduce((a, b) => a + b, 0) / Math.min(times.length, 5);
        $('progressBar').style.width = `${(100 * done / total).toFixed(1)}%`;
        $('perfPill').textContent = `${took.toFixed(0)} ms / frame`;
        $('trackInfo').textContent = `Tracking ${objs.length} object${objs.length > 1 ? 's' : ''}: frame ` +
            `${done} / ${total} · ${(1000 / avg).toFixed(1)} fps · ~${Math.ceil((total - done) * avg / 1000)} s left`;
        await yieldToMain();
      }
    });
    const avg = times.reduce((a, b) => a + b, 0) / Math.max(times.length, 1);
    $('trackInfo').textContent = (state.stop ? 'Stopped. ' : 'Done! ') +
        `${times.length} frames at ${avg.toFixed(0)} ms/frame average. Press play to review.`;
  } catch (e) {
    console.error(e);
    $('trackInfo').textContent = `Tracking failed: ${(e as Error).message}`;
  } finally {
    state.tracking = false;
    $('progress').hidden = true;
    refresh();
    render();
  }
}

// ---------------------------------------------------------------- playback

let playTimer = 0;
function stopPlayback() {
  state.playing = false;
  clearTimeout(playTimer);
  $('playBtn').textContent = '▶';
}
function togglePlay() {
  if (!state.clip || state.tracking) return;
  if (state.playing) {
    stopPlayback();
    render();
    return;
  }
  state.playing = true;
  $('playBtn').textContent = '❚❚';
  if (state.frame >= state.clip.frames.length - 1) state.frame = 0;
  const step = () => {
    if (!state.playing || !state.clip) return;
    render();
    if (state.frame >= state.clip.frames.length - 1) {
      stopPlayback();
      render();
      return;
    }
    state.frame++;
    playTimer = window.setTimeout(step, 1000 / state.clip.fps);
  };
  step();
}

// ---------------------------------------------------------------- camera

/** Next camera frame, after onFrame has recorded its timestamps. */
const nextLiveFrame = (L: Live) => new Promise<void>((r) => L.waiters.push(r));

async function startCamera() {
  if (state.tracking || state.live || !state.engine) return;
  stopPlayback();
  let stream: MediaStream;
  try {
    // 60 fps when the camera offers it: frames are half as old when the
    // pipeline picks up the newest one, and sensor-to-page delivery is shorter.
    stream = await navigator.mediaDevices.getUserMedia({
      video: {width: {ideal: CAM[0]}, height: {ideal: CAM[1]}, frameRate: {ideal: 60}, facingMode: 'user'},
      audio: false,
    });
  } catch (e) {
    $('trackInfo').textContent = `Camera unavailable: ${(e as Error).message}`;
    return;
  }
  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true;
  video.srcObject = stream;
  await video.play();
  const L: Live = {stream, video, width: video.videoWidth, height: video.videoHeight, t: 0,
    frameTimes: [], timings: [], lastFrame: null, processed: -1, maskT: 0, maskBorn: 0, displayPending: false,
    draws: [], waiters: []};
  const onFrame = (_: number, meta: VideoFrameCallbackMetadata) => {
    if (state.live !== L) return;
    L.lastFrame = {captureTime: meta.captureTime, presented: meta.presentedFrames, seen: performance.now()};
    for (const wake of L.waiters.splice(0)) wake();
    if (state.liveDisplay === 'smooth') displaySmooth(L);
    video.requestVideoFrameCallback(onFrame);
  };
  video.requestVideoFrameCallback(onFrame);
  state.live = L;
  resetObjects();
  await applyGeometry();
  const camFps = stream.getVideoTracks()[0]?.getSettings().frameRate;
  $('clipInfo').textContent = `Live camera · ${L.width}×${L.height}` + (camFps ? ` @ ${Math.round(camFps)} fps` : '');
  $('counter').textContent = 'LIVE';
  $('trackInfo').textContent = 'Click an object in the camera view to start tracking it.';
  hideOverlay();
  refresh();
  showTab('objects');
  void liveLoop(L);
}

async function stopCamera() {
  const L = state.live;
  if (!L) return;
  state.live = null;
  for (const wake of L.waiters.splice(0)) wake();  // let liveLoop see the stop
  L.stream.getTracks().forEach((t) => t.stop());
  showMetrics([]);
  $('trackInfo').textContent = '';
  $('perfPill').textContent = '– ms / frame';
  resetObjects();
  if (state.clip) {
    $('clipInfo').textContent = `${state.clip.name} · ${state.clip.frames.length} frames @ ${state.clip.fps} fps · ` +
        `${state.clip.width}×${state.clip.height}`;
    state.frame = 0;
    await applyGeometry();
  }
  refresh();
  render();
}

/**
 * Camera frames through the pipeline until the camera stops: upload -> encode
 * -> track{n} per clicked object -> composite, each frame shown with its own
 * masks (the composite needs the frame it segmented).
 */
/**
 * Smooth display of the newest camera frame: display chain (composite only)
 * with the newest masks. At most one queued; runs between pipeline steps.
 */
function displaySmooth(L: Live) {
  const rt = state.engine;
  if (!rt || L.displayPending || state.loading) return;
  L.displayPending = true;
  const born = L.lastFrame?.captureTime ?? L.lastFrame?.seen ?? performance.now();
  void exclusive(async () => {
    L.displayPending = false;
    if (state.live !== L || state.liveDisplay !== 'smooth') return;
    rt.uploadFrame(L.video);
    await rt.composite(L.maskT);
    present(L.maskT);
    const now = performance.now();
    L.draws.push({at: now, toScreen: now - born, maskLag: L.maskT ? now - L.maskBorn : NaN});
    if (L.draws.length > 60) L.draws.shift();
  }).catch((e) => console.error(e));
}

/** Timestamps of one live pipeline frame, for the metrics panel. */
interface LiveFrame {t0: number; seen: number; captured?: number; born: number; tracked: number}

async function liveLoop(L: Live) {
  while (state.live === L) {
    // Take the newest camera frame right away; wait only if we have it already.
    if (!L.lastFrame || L.lastFrame.presented === L.processed) await nextLiveFrame(L);
    const rt = state.engine;
    if (!rt || state.loading || state.live !== L) continue;
    L.processed = L.lastFrame?.presented ?? -1;
    const t0 = performance.now();
    const seen = L.lastFrame?.seen ?? t0;
    const captured = L.lastFrame?.captureTime;  // absent on some platforms
    const born = captured ?? seen;
    let tracked = 0;
    try {
      await exclusive(async () => {
        if (state.live !== L) return;
        const t = ++L.t;
        if (PROFILE) rt.takeProfile();  // drop runs from displays / prompts in between
        const s0 = performance.now();
        rt.uploadFrame(L.video);
        if (PROFILE === 'gpu') await rt.gpuIdle();
        const s1 = performance.now();
        tracked = rt.trackedObjects(t).length;
        if (tracked) {
          await rt.encode(t);
          const s2 = performance.now();
          await rt.track(t);
          const s3 = performance.now();
          // Queue the blit before the readback: the GPU runs it right after
          // track's composite instead of after a readback round trip.
          if (state.liveDisplay === 'aligned') present(t);
          await rt.readScores(rt.trackedObjects(t)[0], t);  // wait for the GPU
          const s4 = performance.now();
          if (PROFILE) {
            recordProfile({upload: s1 - s0, encode: s2 - s1, track: s3 - s2, read: s4 - s3, total: s4 - s0},
                          rt.takeProfile());
          }
          L.maskT = t;
          L.maskBorn = born;
        } else if (state.liveDisplay === 'aligned') {
          await rt.composite(t);
          present(t);
        }
      });
      // Smooth mode without objects: the camera callback draws; just idle.
      if (!tracked && state.liveDisplay === 'smooth') {
        L.frameTimes = [];
        await nextLiveFrame(L);
        continue;
      }
    } catch (e) {
      console.error(e);
      $('trackInfo').textContent = `Live tracking error: ${(e as Error).message}`;
      continue;
    }
    liveMetrics(L, {t0, seen, captured, born, tracked});
  }
}

function liveMetrics(L: Live, f: LiveFrame) {
  if (state.live !== L) return;
  const {t0, seen, captured, born, tracked} = f;
  const now = performance.now();
  L.timings.push({model: now - t0, toScreen: now - born,
    deliver: captured === undefined ? NaN : seen - captured, wait: t0 - seen});
  if (L.timings.length > 30) L.timings.shift();
  L.frameTimes.push(now);
  if (L.frameTimes.length > 20) L.frameTimes.shift();
  const n = L.frameTimes.length;
  const fps = n > 1 ? (1000 * (n - 1)) / (L.frameTimes[n - 1] - L.frameTimes[0]) : 0;
  const med = (k: 'model' | 'toScreen' | 'deliver' | 'wait') => {
    const v = L.timings.map((x) => x[k]).filter((x) => !Number.isNaN(x)).sort((a, b) => a - b);
    return v.length ? v[Math.floor(v.length / 2)] : NaN;
  };
  const msOr = (x: number) => (Number.isNaN(x) ? '–' : ms(x));
  // camera → screen (pipeline path) = delivery + wait + pipeline.
  const breakdown: Metric[] = [['Delivery', msOr(med('deliver')), 5], ['Wait', msOr(med('wait')), 5],
    ['Pipeline', msOr(med('model')), 6]];
  if (PROFILE && tracked && !state.busy) $('trackInfo').textContent = profileSummary();
  const setup = setupText(tracked);
  if (state.liveDisplay === 'smooth') {
    const d = L.draws.slice(-30);
    const dm = (k: 'toScreen' | 'maskLag') => {
      const v = d.map((x) => x[k]).filter((x) => !Number.isNaN(x)).sort((a, b) => a - b);
      return v.length ? v[Math.floor(v.length / 2)] : 0;
    };
    const vfps = d.length > 1 ? (1000 * (d.length - 1)) / (d[d.length - 1].at - d[0].at) : 0;
    $('perfPill').textContent = `${dm('toScreen').toFixed(0)} ms camera → screen`;
    showMetrics([['Video', `${vfps.toFixed(0)} fps`, 6], ['Camera → screen', ms(dm('toScreen')), 6],
      ['Masks', `${fps.toFixed(1)} fps`, 8], ['Mask lag', ms(dm('maskLag')), 6], ...breakdown, ['', setup]]);
  } else {
    $('perfPill').textContent = `${med('toScreen').toFixed(0)} ms camera → screen`;
    showMetrics([['Rate', `${fps.toFixed(1)} fps`, 8], ['Camera → screen', msOr(med('toScreen')), 6],
      ...breakdown, ['', setup]]);
  }
}

/** A click in camera mode prompts on the newest camera frame. */
/**
 * A click in camera mode. Like file mode, clicks accumulate on the selected
 * object (positive / negative, up to 8); each click re-prompts the object with
 * all its clicks on the newest camera frame, which becomes its prompt frame.
 */
async function livePrompt(L: Live, add: {click?: Click; box?: [Click, Click]}, target?: TrackedObject) {
  const rt = state.engine;
  if (!rt || state.loading || state.busy) return;
  const obj = target ?? state.objects.find((o) => o.id === state.selected)!;
  const prev = obj.point?.pts ?? [];
  const composed = composePrompt(prev, add);
  if (typeof composed === 'string') {
    $('trackInfo').textContent = composed;
    return;
  }
  const pts = composed;
  state.busy = true;
  refresh();
  try {
    const t0 = performance.now();
    const {scores, kept} = await exclusive(async () => {
      const t = ++L.t;
      rt.uploadFrame(L.video);
      await rt.encode(t);
      await rt.prompt(obj.slot, t, promptClicks(pts));
      let scores = await rt.readScores(obj.slot, t);
      let kept = pts;
      if (!(scores && scores[0] > 0) && prev.length) {
        // Nothing there with the new click: keep the object's previous clicks.
        kept = prev;
        await rt.prompt(obj.slot, t, promptClicks(prev));
        scores = await rt.readScores(obj.slot, t);
      }
      const appearing = !!scores && scores[0] > 0;
      obj.point = appearing ? {frame: t, pts: kept} : null;
      if (!appearing) rt.clearObject(obj.slot);
      L.maskT = t;
      L.maskBorn = L.lastFrame?.captureTime ?? L.lastFrame?.seen ?? performance.now();
      present(t);
      return {scores, kept};
    });
    $('trackInfo').textContent = !obj.point
      ? 'No object found at that point. Try clicking closer to its center.'
      : kept !== pts
        ? 'That prompt found nothing; kept the previous one. Tracking live.'
        : `Segmented with ${describePrompt(obj.point.pts)} in ${(performance.now() - t0).toFixed(0)} ms ` +
          `(predicted IoU ${scores![1].toFixed(2)}); tracking live. Click again to refine, or Reset.`;
  } catch (e) {
    console.error(e);
    $('trackInfo').textContent = `Error: ${(e as Error).message}`;
  } finally {
    state.busy = false;
    refresh();
  }
}

// ---------------------------------------------------------------- sample

const SAMPLES = {
  football: {url: asset('assets/football_ai_studio.mp4'), name: 'Football (sample video)', fps: 24, seconds: 8},
  flowers: {url: asset('assets/flowers.mp4'), name: 'Flowers (sample video)', fps: 24, seconds: 8},
};
type SampleId = keyof typeof SAMPLES;
const sampleSel = $<HTMLSelectElement>('sampleSel');
if (q.get('sample') && q.get('sample')! in SAMPLES) sampleSel.value = q.get('sample')!;
const SYNTHETIC = q.get('demo') === 'synthetic';

function loadSample(onProgress: (f: number) => void): Promise<Clip> {
  if (SYNTHETIC) return demoClip();
  const sample = SAMPLES[sampleSel.value as SampleId];
  return decodeVideo(sample, sample.fps, sample.seconds, 1280, onProgress);
}

// ---------------------------------------------------------------- wiring

function showTab(tab: string) {
  $('dock').dataset.tab = tab;
  for (const b of $('dockTabs').querySelectorAll('button')) b.classList.toggle('on', b.dataset.tab === tab);
}
$('dockTabs').addEventListener('click', (ev) => {
  const b = (ev.target as HTMLElement).closest('button');
  if (b?.dataset.tab) showTab(b.dataset.tab);
});

$('camBtn').addEventListener('click', () => (state.live ? void stopCamera() : void startCamera()));

$<HTMLInputElement>('fileInput').addEventListener('change', async (ev) => {
  const f = (ev.target as HTMLInputElement).files?.[0];
  if (!f || state.tracking) return;
  await stopCamera();
  const fps = Number($<HTMLSelectElement>('fpsSel').value);
  const secs = Number($<HTMLSelectElement>('secSel').value);
  try {
    showOverlay('Decoding video…', {progress: 0});
    const clip = await decodeVideo(f, fps, secs, 1280, (p) => showOverlay('Decoding video…', {progress: p}));
    await setClip(clip);
    if (!state.engine) showOverlay('Loading the pipeline…', {spinner: true});
  } catch (e) {
    showOverlay(`Could not decode this video: ${(e as Error).message}`);
  }
  (ev.target as HTMLInputElement).value = '';
});

async function openSample() {
  if (state.tracking) return;
  await stopCamera();
  try {
    await setClip(await loadSample((p) => showOverlay('Loading sample video…', {progress: p})));
    if (!state.engine) showOverlay('Loading the pipeline…', {spinner: true});
  } catch (e) {
    showOverlay(`Could not load the sample video: ${(e as Error).message}`);
  }
}
$('demoBtn').addEventListener('click', () => void openSample());
sampleSel.addEventListener('change', () => void openSample());
$('addObjBtn').addEventListener('click', () => addObject());
$('resetObjBtn').addEventListener('click', () => resetSelected());
$('trackBtn').addEventListener('click', () => {
  if (state.tracking) state.stop = true;
  else void track();
});
$('playBtn').addEventListener('click', togglePlay);
scrubber.addEventListener('input', () => {
  stopPlayback();
  state.frame = Number(scrubber.value);
  render();
});
for (const [segId, apply] of [
  ['memSeg', (v: string) => {
    state.nmm = Number(v) as 7 | 2;
    state.engine?.setMemorySize(state.nmm);
  }],
  ['polSeg', (v: string) => {
    state.tool = v as Tool;
    $('trackInfo').textContent = v === 'box' ? 'Drag a box around the object; clicks can then refine it.' : '';
  }],
  ['liveSeg', (v: string) => (state.liveDisplay = v as LiveDisplay)],
  ['sizeSeg', (v: string) => {
    const size = Number(v) as ModelSize;
    if (size === state.size || state.live) return;
    state.size = size;
    void loadEngine(size);
  }],
  ['outlineSeg', (v: string) => {
    state.stroke = Number(v);
    state.engine?.setEffect(state.effect, state.stroke);
    rerenderStill();
  }],
  ['fxSeg', (v: string) => {
    state.effect = v as Effect;
    state.engine?.setEffect(state.effect, state.stroke);
    rerenderStill();
  }],
] as const) {
  $(segId).addEventListener('click', (ev) => {
    const b = (ev.target as HTMLElement).closest('button');
    if (!b || b.disabled) return;
    for (const x of $(segId).querySelectorAll('button')) x.classList.toggle('on', x === b);
    apply(b.dataset.v!);
  });
}
/** Effect changes: re-composite the frame on screen (live mode picks it up next frame). */
function rerenderStill() {
  if (!state.live && !state.tracking) render();
}
window.addEventListener('keydown', (ev) => {
  if (!state.clip || state.tracking || state.live || (ev.target as HTMLElement).tagName === 'INPUT') return;
  if (ev.code === 'Space') {
    ev.preventDefault();
    togglePlay();
  } else if (ev.code === 'ArrowRight' || ev.code === 'ArrowLeft') {
    stopPlayback();
    state.frame = Math.min(Math.max(state.frame + (ev.code === 'ArrowRight' ? 1 : -1), 0), state.clip.frames.length - 1);
    render();
  }
});
window.addEventListener('resize', drawTimeline);

// Read-only handle for the UI tests.
(window as unknown as {__sam2: typeof state}).__sam2 = state;

void probeGemma();
showOverlay('Loading the pipeline…', {spinner: true});
void loadSample(() => undefined)
    .catch((e) => {
      console.error('sample video failed, using the synthetic clip', e);
      return demoClip();
    })
    .then(async (clip) => {
      if (!state.clip) await setClip(clip);
      if (!state.engine) showOverlay('Loading the pipeline…', {spinner: true});
    });
void loadEngine(state.size);
