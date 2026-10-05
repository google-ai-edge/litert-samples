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

// Browser half of the wasm e2e test: runs the C++ ModelChain pipeline (wasm,
// WebGPU via LiteRT.js) over a raw RGBA clip with multi-object / multi-click
// prompts, exactly like sam2_chain_main does natively, and POSTs the same
// dump layout (pixels, masks, scores, composites, meta.json) to the test
// server for tools/verify_chain.py.

import {ChainRuntime, type ChainClick, type Effect} from '../../src/chain/runtime';

const q = new URLSearchParams(location.search);
const status = document.getElementById('status')!;

async function post(name: string, data: Float32Array | string) {
  const body = typeof data === 'string' ? data : new Blob([data.buffer as ArrayBuffer]);
  const res = await fetch(`/__dump/${q.get('dump')}/${name}`, {method: 'POST', body});
  if (!res.ok) throw new Error(`dump ${name}: ${res.status}`);
}

function parsePrompts(spec: string) {
  return spec.split('|').filter(Boolean).map((item) => {
    const [head, pts] = item.split(':');
    const [object, frame] = head.split('@').map(Number);
    return {object, frame, points: pts.split(';').map((p) => p.split(',').map(Number))};
  });
}

async function run() {
  const size = Number(q.get('size') ?? 384);
  const nmm = Number(q.get('nmm') ?? 7) as 2 | 7;
  const W = Number(q.get('W')), H = Number(q.get('H')), T = Number(q.get('T'));
  const effect = (q.get('effect') ?? 'overlay') as Effect;
  const stroke = Number(q.get('stroke') ?? 3);
  const rgbFrames = (q.get('rgb') ?? '0').split(',').map(Number);
  const prompts = parsePrompts(q.get('prompts') ?? '');

  const t0 = performance.now();
  const rt = await ChainRuntime.create({
    modelUrl: `/models/sam2_chain_${size}.tflite`,
    hostConstsUrl: '/models/sam2_host_consts.safetensors',
    weightsUrl: q.get('build') === 'browser' ? `/@fs${q.get('weights')}` : undefined,
    imageSize: size,
    litertWasm: '/litert-wasm/', chainWasm: '/wasm/', nmm,
    precision: (q.get('precision') ?? 'fp16') as 'fp16' | 'fp32',
    onProgress: (m) => { status.textContent = m; },
  });
  const loadMs = performance.now() - t0;
  await rt.setGeometry(W, H);
  rt.setEffect(effect, stroke);

  const clip = new Uint8Array(await (await fetch(`/@fs${q.get('clip')}`)).arrayBuffer());
  const stride = rt.stride;
  const frame = new Float32Array(H * stride * 4);
  const frameMs: number[] = [];
  const stages: Array<{encode: number; step: number}> = [];
  for (let t = 0; t < T; t++) {
    status.textContent = `frame ${t}`;
    const src = clip.subarray(t * W * H * 4, (t + 1) * W * H * 4);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W * 4; x++) frame[y * stride * 4 + x] = src[y * W * 4 + x] / 255;
    }
    const f0 = performance.now();
    rt.uploadFloats(frame);
    await rt.encode(t);
    const enc = rt.times().encode;
    for (const p of prompts.filter((p) => p.frame === t)) {
      const clicks: ChainClick[] = p.points.map(([x, y, l]) => ({
        x: Math.min(Math.max(x * size, 0), size - 1), y: Math.min(Math.max(y * size, 0), size - 1),
        label: l as 0 | 1}));
      await rt.prompt(p.object, t, clicks);
    }
    await rt.track(t);
    // Wait for the GPU: read one small result (what the app does per frame).
    const tracked = rt.trackedObjects(t);
    const any = tracked.length ? tracked[0] : prompts.find((p) => p.frame <= t)?.object;
    if (any !== undefined) await rt.readScores(any, t);
    frameMs.push(performance.now() - f0);
    stages.push({encode: enc, step: rt.times().step});

    for (let k = 0; k < 5; k++) {
      if (!rt.hasResult(k, t)) continue;
      await post(`mask_o${k}_f${t}.f32`, (await rt.readMask(k, t))!);
      await post(`score_o${k}_f${t}.f32`, (await rt.readScores(k, t))!);
    }
    await post(`pixels_f${t}.f32`, (await rt.readPixels())!);
    if (rgbFrames.includes(t)) await post(`rgb_f${t}.f32`, (await rt.readOutput())!);
  }
  await post('meta.json', JSON.stringify({size, width: W, height: H, frames: T, nmm,
    prompts: q.get('prompts'), effect, stroke}));
  const med = (v: number[]) => [...v].sort((a, b) => a - b)[Math.floor(v.length / 2)];
  status.textContent = 'done';
  return {loadMs, buildMs: rt.buildMs, frameMs: med(frameMs.slice(1)),
    frameMsAll: frameMs, stages};
}

(window as unknown as {runChain: () => Promise<unknown>}).runChain = run;
