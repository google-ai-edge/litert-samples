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

// TypeScript layer over the wasm build of the C++ SAM 2 ModelChain pipeline
// (web-tensorapi/cc + wasm/). Everything that computes — preprocess, encode,
// prompt / track steps, the memory bank, composite — is C++ Tensor API
// models chained by LiteRT's ModelChain inside the wasm module, executed on
// WebGPU through LiteRT.js. This file only loads things, moves video frames
// into the pipeline's frame tensor (a GPU copy) and forwards calls.

import * as litert from '@litertjs/core';

export type Effect = 'overlay' | 'spotlight' | 'cutout';
const EFFECT_ID: Record<Effect, number> = {overlay: 0, spotlight: 1, cutout: 2};

export interface ChainClick {
  x: number;  // model space (S x S)
  y: number;
  label: 0 | 1 | 2 | 3;  // 1 / 0 positive / negative click; 2 / 3 box top-left / bottom-right
}

export interface ChainOptions {
  /** SAM 2 model authored by the C++ Tensor API (native build). */
  modelUrl: string;
  hostConstsUrl: string;
  /** Instead of modelUrl: author the SAM 2 model in the browser from these weights. */
  weightsUrl?: string;
  imageSize?: number;
  litertWasm: string;  // LiteRT.js runtime files
  chainWasm: string;   // sam2_chain.mjs / .wasm
  nmm: 2 | 7;
  precision?: 'fp16' | 'fp32';
  /** Keep downloads in the Cache API (default true; see fetchBytes). */
  cache?: boolean;
  onProgress?: (msg: string, frac?: number) => void;
}

interface Sam2ChainNative {
  buildSam2Model(weights: string, size: number, consts: string, out: string): boolean;
  init(model: string, consts: string, nmm: number): Promise<boolean>;
  imageSize(): number;
  setMemorySize(nmm: number): void;
  setResultRetention(frames: number): void;
  setEffect(effect: number, stroke: number): void;
  clearObject(object: number): void;
  condFrame(object: number): number;
  setGeometry(width: number, height: number): Promise<boolean>;
  frameStride(): number;
  frameBuffer(): number;
  outputBuffer(): number;
  encode(t: number): Promise<boolean>;
  prompt(object: number, t: number, clicks: ChainClick[]): Promise<boolean>;
  track(t: number): Promise<boolean>;
  composite(t: number): Promise<boolean>;
  trackedObjects(t: number): number[];
  hasResult(object: number, t: number): boolean;
  readMask(object: number, t: number): Promise<Float32Array | null>;
  readMasks(objects: number[], t: number): Promise<Array<Float32Array | null>>;
  readScores(object: number, t: number): Promise<Float32Array | null>;
  readPixels(): Promise<Float32Array | null>;
  readOutput(): Promise<Float32Array | null>;
  times(): {encode: number; step: number; composite: number};
  lastError(): string;
}

interface ChainModule {
  Sam2Chain: new () => Sam2ChainNative;
  FS: {writeFile(path: string, data: Uint8Array): void; unlink(path: string): void};
  lrt: {buffers: Map<number, GPUBuffer>; prof: {mode: ProfileMode; records: RunProfile[]}};
  maxObjects?(): number;
}

/** 'cpu': time for each LiteRT.js run() to return; 'gpu': also wait for the GPU after each run. */
export type ProfileMode = 'cpu' | 'gpu' | null;
export interface RunProfile {
  key: string;   // signature
  run: number;   // ms until LiteRT.js run() returned
  copy: number;  // ms to queue the output copies
  gpu: number;   // 'gpu' mode: ms until the GPU finished this run (NaN otherwise)
}

async function readBody(res: Response, onProgress?: (f: number) => void): Promise<Uint8Array<ArrayBuffer>> {
  const total = res.headers.get('content-encoding') ? 0 : Number(res.headers.get('content-length') ?? 0);
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const out = new Uint8Array(total);
  const reader = res.body.getReader();
  let n = 0;
  for (;;) {
    const {done, value} = await reader.read();
    if (done) break;
    out.set(value, n);
    n += value.length;
    onProgress?.(n / total);
  }
  return out;
}

// Downloads (models are 100-200 MB) are kept with the Cache API. A cached
// copy is used when a HEAD request says the file is unchanged (ETag,
// Last-Modified and size), or when the server can't be reached.
const CACHE_NAME = 'sam2-tensorapi-models-v1';
const VERSION_HEADERS = ['etag', 'last-modified', 'content-length'];
const SRC = 'x-src-';  // server's version headers, as stored with the cached copy

async function fetchBytes(url: string, onProgress?: (f: number) => void,
                          useCache = true): Promise<Uint8Array<ArrayBuffer>> {
  const cache = useCache && 'caches' in self ? await caches.open(CACHE_NAME).catch(() => null) : null;
  const cached = await cache?.match(url);
  if (cache && cached) {
    const head = await fetch(url, {method: 'HEAD', cache: 'no-store'}).catch(() => null);
    const same = !head?.ok ||
        VERSION_HEADERS.every((h) => (head.headers.get(h) ?? '') === (cached.headers.get(SRC + h) ?? ''));
    if (same) return readBody(cached, onProgress);
  }
  const res = await fetch(url, {cache: 'no-store'});
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  // A dev server answers a missing file with its index page (200, text/html).
  if (res.headers.get('content-type')?.startsWith('text/html')) {
    throw new Error(`${url} not found (the server returned an HTML page); see tools/build_models.sh`);
  }
  const bytes = await readBody(res, onProgress);
  if (cache) {
    const headers = new Headers({'content-length': String(bytes.byteLength)});
    for (const h of VERSION_HEADERS) headers.set(SRC + h, res.headers.get(h) ?? '');
    // Not awaited: storing 100+ MB shouldn't delay the start. Quota errors are harmless.
    cache.put(url, new Response(bytes, {headers})).catch((e) => console.warn(`cache ${url}:`, e));
  }
  return bytes;
}

/** Deletes every cached download (see fetchBytes). */
export async function clearModelCache(): Promise<boolean> {
  return 'caches' in self ? caches.delete(CACHE_NAME) : false;
}

let liteRtLoaded: Promise<unknown> | null = null;

export class ChainRuntime {
  width = 0;
  height = 0;
  private uploadTex: GPUTexture | null = null;

  private constructor(private readonly mod: ChainModule, private readonly chain: Sam2ChainNative,
                      readonly buildMs: number) {}

  static async create(o: ChainOptions): Promise<ChainRuntime> {
    if (!('gpu' in navigator)) throw new Error('WebGPU is not available in this browser');
    o.onProgress?.('Loading LiteRT.js');
    liteRtLoaded ??= litert.loadLiteRt(o.litertWasm);
    await liteRtLoaded;
    o.onProgress?.('Loading the C++ pipeline (wasm)');
    const factory = (await import(/* @vite-ignore */ `${o.chainWasm}sam2_chain.mjs`)).default;
    const mod: ChainModule = await factory({
      litert: {core: litert, precision: o.precision ?? 'fp16'},
      locateFile: (p: string) => `${o.chainWasm}${p}`,
    });
    const chain = new mod.Sam2Chain();
    mod.FS.writeFile('/host_consts.safetensors', await fetchBytes(o.hostConstsUrl, undefined, o.cache ?? true));
    let buildMs = 0;
    if (o.weightsUrl) {
      const w = await fetchBytes(o.weightsUrl, (f) => o.onProgress?.('Downloading weights', f),
                                 o.cache ?? true);
      mod.FS.writeFile('/weights.safetensors', w);
      o.onProgress?.('Building the SAM 2 model with the Tensor API (in the browser)');
      await new Promise((r) => setTimeout(r, 0));
      const t0 = performance.now();
      if (!chain.buildSam2Model('/weights.safetensors', o.imageSize ?? 384, '/host_consts.safetensors', '/sam2.tflite')) {
        throw new Error(`building the model failed: ${chain.lastError()}`);
      }
      buildMs = performance.now() - t0;
      mod.FS.unlink('/weights.safetensors');
    } else {
      const bytes = await fetchBytes(o.modelUrl, (f) => o.onProgress?.('Loading model', f),
                                     o.cache ?? true);
      // TFLite flatbuffer file identifier: parsing anything else traps in wasm.
      if (new TextDecoder().decode(bytes.subarray(4, 8)) !== 'TFL3') {
        throw new Error(`${o.modelUrl} is not a .tflite model`);
      }
      mod.FS.writeFile('/sam2.tflite', bytes);
    }
    o.onProgress?.('Compiling for WebGPU');
    const ok = await chain.init('/sam2.tflite', '/host_consts.safetensors', o.nmm);
    mod.FS.unlink('/sam2.tflite');
    if (!ok) throw new Error(`pipeline init failed: ${chain.lastError()}`);
    return new ChainRuntime(mod, chain, buildMs);
  }

  /** Frees the C++ pipeline (its WebGPU buffers and compiled models). */
  dispose() {
    this.uploadTex?.destroy();
    this.uploadTex = null;
    (this.chain as unknown as {delete(): void}).delete();
  }

  get imageSize(): number { return this.chain.imageSize(); }
  /** Object slots of this wasm build (kMaxObjects); older builds had 3. */
  get maxObjects(): number { return this.mod.maxObjects?.() ?? 3; }
  get device(): GPUDevice { return litert.getWebGpuDevice()!; }
  get stride(): number { return this.chain.frameStride(); }

  private check(ok: boolean, what: string) {
    if (!ok) throw new Error(`${what}: ${this.chain.lastError()}`);
  }

  /** Builds (Tensor API, in C++) and compiles the frame model for W x H frames. Clears objects. */
  async setGeometry(width: number, height: number) {
    this.check(await this.chain.setGeometry(width, height), 'setGeometry');
    this.width = width;
    this.height = height;
    this.uploadTex?.destroy();
    this.uploadTex = this.device.createTexture({
      size: [width, height], format: 'rgba32float',
      usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC | GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  get frameBuffer(): GPUBuffer { return this.mod.lrt.buffers.get(this.chain.frameBuffer())!; }
  get outputBuffer(): GPUBuffer { return this.mod.lrt.buffers.get(this.chain.outputBuffer())!; }

  /** Video frame -> the pipeline's RGBA frame tensor, on the GPU (no CPU pixels). */
  uploadFrame(source: GPUCopyExternalImageSource) {
    const tex = this.uploadTex!;
    this.device.queue.copyExternalImageToTexture({source}, {texture: tex}, [this.width, this.height]);
    const enc = this.device.createCommandEncoder();
    enc.copyTextureToBuffer({texture: tex}, {buffer: this.frameBuffer, bytesPerRow: this.stride * 16,
      rowsPerImage: this.height}, [this.width, this.height]);
    this.device.queue.submit([enc.finish()]);
  }

  /** Raw RGBA floats [H, stride, 4] in [0,1] -> frame tensor (tests). */
  uploadFloats(data: Float32Array) {
    this.device.queue.writeBuffer(this.frameBuffer, 0, data.buffer, data.byteOffset, data.byteLength);
  }

  async encode(t: number) { this.check(await this.chain.encode(t), `encode ${t}`); }
  async prompt(object: number, t: number, clicks: ChainClick[]) {
    this.check(await this.chain.prompt(object, t, clicks), `prompt ${object}@${t}`);
  }
  async track(t: number) { this.check(await this.chain.track(t), `track ${t}`); }
  async composite(t: number) { this.check(await this.chain.composite(t), `composite ${t}`); }

  setEffect(effect: Effect, stroke: number) { this.chain.setEffect(EFFECT_ID[effect], stroke); }
  setMemorySize(nmm: 2 | 7) { this.chain.setMemorySize(nmm); }
  setResultRetention(frames: number) { this.chain.setResultRetention(frames); }
  clearObject(object: number) { this.chain.clearObject(object); }
  condFrame(object: number): number { return this.chain.condFrame(object); }
  trackedObjects(t: number): number[] { return this.chain.trackedObjects(t); }
  hasResult(object: number, t: number): boolean { return this.chain.hasResult(object, t); }
  readMask(object: number, t: number) { return this.chain.readMask(object, t); }
  readMasks(objects: number[], t: number) { return this.chain.readMasks(objects, t); }
  readScores(object: number, t: number) { return this.chain.readScores(object, t); }
  readPixels() { return this.chain.readPixels(); }
  readOutput() { return this.chain.readOutput(); }
  times() { return this.chain.times(); }

  /** Per-run profiling in the LiteRT.js bridge (see litert_js_bridge.js). */
  setProfile(mode: ProfileMode) {
    this.mod.lrt.prof.mode = mode;
    this.mod.lrt.prof.records = [];
  }
  get profileMode(): ProfileMode { return this.mod.lrt.prof.mode; }
  /** Runs recorded since the last call. */
  takeProfile(): RunProfile[] {
    const r = this.mod.lrt.prof.records;
    this.mod.lrt.prof.records = [];
    return r;
  }
  /** Resolves when the GPU has finished all submitted work. */
  gpuIdle(): Promise<void> { return this.device.queue.onSubmittedWorkDone(); }
}
