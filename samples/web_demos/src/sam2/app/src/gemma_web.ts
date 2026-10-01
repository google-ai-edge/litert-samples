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

// "Ask Gemma" in the browser: Gemma 4 (E2B / E4B) with its vision encoder,
// running on WebGPU through the MediaPipe LLM Inference task. Same prompt and
// box parsing as the server path (gemma.ts), no local server needed.
//
// The models are the hand-written GPU ("hw") .litertlm builds. Each file holds
// the text decoder, then the vision encoder and adapter as separate sections.
// The published @mediapipe/tasks-genai skips everything but the decoder, so
// it can't take images from these files. public/genai/ has a MediaPipe GenAI
// build that also reads the vision sections:
//   genai_js.js             LlmInference / FilesetResolver (ES module)
//   genai_wasm_internal.js  Emscripten loader
// The 33.7 MB .wasm is fetched from the same public bucket as the models.
//
// The model is downloaded once into the Cache API, then streamed from disk
// into the runtime. Streaming the download straight into the runtime while
// also caching it (tee) would buffer gigabytes in memory, because loading is
// much slower than writing to disk.

import {detectionPrompt, type FrameSource, frameToCanvas, type GemmaBox, type GemmaResult, parseBoxes} from './gemma';

export interface WebModel {
  id: string;
  name: string;
  url: string;
  bytes: number;
}

const BUCKET = 'https://storage.googleapis.com/litertjs_demos/';
export const WEB_MODELS: WebModel[] = [
  {id: 'web:gemma4-e4b', name: 'Gemma 4 E4B', bytes: 3201116788,
    url: `${BUCKET}gemma_4_models/gemma4-e4b-hw-int4-20260622.litertlm`},
  {id: 'web:gemma4-e2b', name: 'Gemma 4 E2B', bytes: 2237340276,
    url: `${BUCKET}gemma_4_models/gemma4-e2b-hw.litertlm`},
];
const GENAI_WASM = `${BUCKET}wasm/genai_wasm_internal.wasm`;
const CACHE_NAME = 'sam2-gemma4-web';

export const isWebModel = (id: string) => id.startsWith('web:');
export const webModel = (id: string) => WEB_MODELS.find((m) => m.id === id);
export const webGemmaSupported = () => 'gpu' in navigator && typeof caches !== 'undefined';

// ---- the parts of the MediaPipe GenAI API used here
/** safevalues TrustedResourceUrl as built by FilesetResolver; `h` holds the URL. */
interface TrustedUrl {h: unknown}
interface WasmFileset {wasmLoaderPath: TrustedUrl; wasmBinaryPath: TrustedUrl}
type PromptPart = string | {imageSource: CanvasImageSource | string};
interface LlmInference {
  generateResponse(query: PromptPart[], progress?: (partial: string, done: boolean) => void): Promise<string>;
  cancelProcessing(): void;
  close(): void;
}
interface GenAiModule {
  FilesetResolver: {forGenAiTasks(): Promise<WasmFileset>};
  LlmInference: {createFromOptions(fileset: WasmFileset, options: Record<string, unknown>): Promise<LlmInference>};
}

/** Status text and, while bytes move, the fraction done. */
export type Progress = (text: string, fraction?: number) => void;

const gb = (n: number) => (n / 1e9).toFixed(2);

/** Passes a stream through, reporting bytes seen (at most every 0.5%). */
function counted(body: ReadableStream<Uint8Array>, total: number, report: (n: number) => void) {
  let n = 0, last = -1;
  return body.pipeThrough(new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, ctl) {
      n += chunk.byteLength;
      const step = Math.floor((n / total) * 200);
      if (step !== last) {
        last = step;
        report(n);
      }
      ctl.enqueue(chunk);
    },
  }));
}

/** The model as a stream: from the cache, or downloaded into it first. */
async function modelStream(m: WebModel, progress: Progress): Promise<ReadableStream<Uint8Array>> {
  const cache = await caches.open(CACHE_NAME).catch(() => null);
  const fromDisk = (body: ReadableStream<Uint8Array>, total: number) => counted(body, total,
      (n) => progress(`Loading ${m.name} onto the GPU: ${gb(n)} / ${gb(total)} GB`, n / total));
  const hit = await cache?.match(m.url).catch(() => undefined);
  if (hit?.ok && hit.body) return fromDisk(hit.body, Number(hit.headers.get('Content-Length')) || m.bytes);

  progress(`Downloading ${m.name} (${gb(m.bytes)} GB, once; it is kept in the browser cache)…`, 0);
  const res = await fetch(m.url);
  if (!res.ok || !res.body) throw new Error(`HTTP ${res.status} for ${m.url}`);
  const total = Number(res.headers.get('Content-Length')) || m.bytes;
  const download = counted(res.body, total,
      (n) => progress(`Downloading ${m.name}: ${gb(n)} / ${gb(total)} GB`, n / total));
  if (cache) {
    try {
      await cache.put(m.url, new Response(download, {headers: {
        'Content-Type': 'application/octet-stream', 'Content-Length': String(total)}}));
      const back = await cache.match(m.url);
      if (back?.body) return fromDisk(back.body, total);
    } catch (e) {
      console.warn(`Could not cache ${m.name} (disk quota?); streaming it from the network`, e);
    }
  }
  // Not cacheable: download again, straight into the runtime.
  const again = await fetch(m.url);
  if (!again.ok || !again.body) throw new Error(`HTTP ${again.status} for ${m.url}`);
  return counted(again.body, total, (n) => progress(`Downloading ${m.name} (not cached): ${gb(n)} / ${gb(total)} GB`, n / total));
}

let loaded: {id: string; llm: Promise<LlmInference>} | null = null;

/** Loads a browser model (once; switching models closes the previous one). */
export function loadWebGemma(id: string, progress: Progress): Promise<LlmInference> {
  if (loaded?.id === id) return loaded.llm;
  const prev = loaded;
  const llm = (async () => {
    if (prev) (await prev.llm.catch(() => null))?.close();
    const m = webModel(id);
    if (!m) throw new Error(`unknown browser model ${id}`);
    const base = new URL('genai/', document.baseURI).href;
    const genai = await import(/* @vite-ignore */ `${base}genai_js.js`) as GenAiModule;
    const fileset = await genai.FilesetResolver.forGenAiTasks();
    fileset.wasmLoaderPath.h = `${base}genai_wasm_internal.js`;
    fileset.wasmBinaryPath.h = GENAI_WASM;
    const stream = await modelStream(m, progress);
    const t0 = performance.now();
    const inference = await genai.LlmInference.createFromOptions(fileset, {
      baseOptions: {modelAssetBuffer: stream.getReader()},
      // Image (~260 tokens) + prompt + six boxes (~30 tokens each), with room to spare.
      maxTokens: 2048,
      topK: 1,
      temperature: 0,
      randomSeed: 1,
      maxNumImages: 1,  // loads the vision encoder and adapter
    });
    progress(`${m.name} ready in the browser (${((performance.now() - t0) / 1000).toFixed(0)} s to load).`, 1);
    return inference;
  })();
  loaded = {id, llm};
  llm.catch(() => {
    if (loaded?.llm === llm) loaded = null;
  });
  return llm;
}

/** True once `id` is loaded (or loading) in this page. */
export const webGemmaLoaded = (id: string) => loaded?.id === id;

/**
 * Asks the browser model for boxes of `what` in `frame`; same contract as
 * gemma.ts detect(): boxes are streamed to `onBox`, and generation is
 * cancelled after `maxBoxes`.
 */
export async function detectWeb(llm: LlmInference, frame: FrameSource, what: string, model: string,
                                onBox?: (box: GemmaBox, index: number) => void | Promise<void>,
                                maxBoxes = Infinity): Promise<GemmaResult> {
  const image = frameToCanvas(frame);
  const query: PromptPart[] = ['<|turn>user\n', {imageSource: image},
    `${detectionPrompt(what)}<turn|>\n<|turn>model\n`];
  let raw = '', emitted = 0, stopped = false;
  const pending: Array<void | Promise<void>> = [];
  const emit = (boxes: GemmaBox[]) => {
    while (emitted < Math.min(boxes.length, maxBoxes)) {
      pending.push(onBox?.(boxes[emitted], emitted));
      emitted++;
    }
  };
  const t0 = performance.now();
  try {
    await llm.generateResponse(query, (partial) => {
      raw += partial;
      if (stopped) return;
      emit(parseBoxes(raw.slice(0, raw.lastIndexOf('}') + 1)));  // complete entries only
      if (emitted >= maxBoxes) {
        stopped = true;
        llm.cancelProcessing();  // enough objects: stop Gemma writing more
      }
    });
  } catch (e) {
    if (!stopped) throw e;
  }
  const seconds = (performance.now() - t0) / 1000;
  const boxes = parseBoxes(raw);
  emit(boxes);  // a last entry without a closing brace
  await Promise.all(pending);
  return {boxes, seconds, model, raw};
}
