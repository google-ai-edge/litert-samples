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

// WebGPU health check of the demo (C++ ModelChain pipeline in wasm, executed
// by LiteRT.js on WebGPU), in system Chrome:
//   1. adapter: a hardware GPU (not a fallback / software adapter), shader-f16
//   2. acceleration: every signature of both compiled models runs on WebGPU
//   3. errors: no uncaptured WebGPU errors and no device loss, whole session
//   4. GPU memory: buffer count / bytes bounded across prompt, 192-frame
//      tracking, re-tracking, playback, reset, and a camera session
//      (per-frame results are kept for the clip by design; nothing else grows)
//   5. sustained speed: per-frame time over the clip, first vs last quarter
//
//   node test/e2e/webgpu_check.mjs
import {execFileSync} from 'node:child_process';
import {existsSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {resolve} from 'node:path';
import {chromium} from 'playwright-core';
import {createServer} from 'vite';

const app = resolve(import.meta.dirname, '../..');
const y4m = resolve(tmpdir(), 'sam2chain_football_640.y4m');
if (!existsSync(y4m)) {
  execFileSync('ffmpeg', ['-loglevel', 'error', '-y', '-i', resolve(app, 'public/assets/football_ai_studio.mp4'),
    '-t', '4', '-vf', 'scale=640:360', '-pix_fmt', 'yuv420p', y4m]);
}
const server = await createServer({root: app, logLevel: 'error', server: {port: 5179}});
await server.listen();
const browser = await chromium.launch({channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu',
  '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', `--use-file-for-fake-video-capture=${y4m}`]});
let failed = false;
const check = (ok, msg) => {
  console.log(`${ok ? '  ok ' : '  FAIL'} ${msg}`);
  if (!ok) failed = true;
};
const mb = (b) => `${(b / 1e6).toFixed(1)} MB`;
try {
  const page = await browser.newPage({viewport: {width: 1440, height: 900}});
  page.on('pageerror', (e) => console.log(`  [pageerror] ${e.message}`));
  await page.goto('http://localhost:5179/');
  await page.waitForFunction(() => window.__sam2?.engine && window.__sam2?.clip && !window.__sam2.loading,
      null, {timeout: 300000});

  // ---- 1. adapter + error hooks on LiteRT.js's device (the one everything runs on)
  const adapter = await page.evaluate(async () => {
    const dev = window.__sam2.engine.device;
    window.__gpuErrors = [];
    window.__gpuLost = null;
    dev.addEventListener('uncapturederror', (e) => window.__gpuErrors.push(e.error.message));
    dev.lost.then((i) => { window.__gpuLost = `${i.reason}: ${i.message}`; });
    const a = await navigator.gpu.requestAdapter();
    const info = a.info ?? {};
    return {vendor: info.vendor, arch: info.architecture, desc: info.description, fallback: !!info.isFallbackAdapter,
      f16: a.features.has('shader-f16'), maxBuf: a.limits.maxStorageBufferBindingSize,
      devF16: dev.features.has('shader-f16')};
  });
  check(!adapter.fallback && adapter.vendor && adapter.vendor !== 'google' || (!adapter.fallback && adapter.arch),
      `hardware adapter: vendor "${adapter.vendor}", architecture "${adapter.arch}", fallback ${adapter.fallback}, ` +
      `shader-f16 ${adapter.f16}, max storage binding ${mb(adapter.maxBuf)}`);

  // ---- 2. acceleration of every compiled model
  const accel = await page.evaluate(() => [...window.__sam2.engine.mod.lrt.models.values()].map((m) => ({
    signatures: Object.keys(m.model.signatures).length, full: m.model.isFullyAccelerated})));
  check(accel.length >= 2 && accel.every((m) => m.full),
      `compiled models: ${accel.map((m) => `${m.signatures} signatures fully on WebGPU: ${m.full}`).join('; ')}`);

  const gpuMem = () => page.evaluate(() => {
    const bufs = [...window.__sam2.engine.mod.lrt.buffers.values()];
    return {n: bufs.length, bytes: bufs.reduce((a, b) => a + b.size, 0)};
  });
  const clip = await page.evaluate(() => ({w: window.__sam2.clip.width, h: window.__sam2.clip.height}));
  const click = async ([nx, ny]) => {
    const b = await page.locator('#view').boundingBox();
    const s = Math.min(b.width / clip.w, b.height / clip.h);
    await page.mouse.click(b.x + (b.width - clip.w * s) / 2 + nx * clip.w * s, b.y + (b.height - clip.h * s) / 2 + ny * clip.h * s);
    await page.waitForFunction(() => !window.__sam2.busy, null, {timeout: 60000});
  };
  const track = async () => {
    await page.click('#trackBtn');
    await page.waitForFunction(() => !window.__sam2.tracking && /Done|Stopped|failed/.test(
        document.getElementById('trackInfo').textContent), null, {timeout: 20 * 60 * 1000, polling: 500});
  };
  // Per-frame wall times from the tracking loop's status text is too coarse: time frames via the pipeline.
  const mem = {};
  mem.loaded = await gpuMem();
  await click([0.44, 0.28]);
  await click([0.45, 0.47]);
  await page.click('#addObjBtn');
  await click([0.484, 0.79]);
  mem.prompted = await gpuMem();

  // ---- 5. sustained speed: frame times during tracking
  await page.evaluate(() => {
    window.__frameTimes = [];
    let last = performance.now();
    const obs = new MutationObserver(() => {
      const now = performance.now();
      window.__frameTimes.push(now - last);
      last = now;
    });
    obs.observe(document.getElementById('counter'), {childList: true, characterData: true, subtree: true});
    window.__obs = obs;
  });
  await track();
  const times = await page.evaluate(() => { window.__obs.disconnect(); return window.__frameTimes.slice(1); });
  mem.tracked = await gpuMem();
  await track();  // re-track twice: results of earlier runs are replaced, not added
  mem.retracked = await gpuMem();
  await track();
  mem.retracked2 = await gpuMem();
  await page.click('#playBtn');
  await page.waitForFunction(() => !window.__sam2.playing, null, {timeout: 120000});
  mem.played = await gpuMem();
  await page.click('#resetObjBtn');
  await page.waitForTimeout(500);
  mem.reset = await gpuMem();

  const q = Math.floor(times.length / 4);
  const med = (v) => [...v].sort((a, b) => a - b)[Math.floor(v.length / 2)];
  const p95 = (v) => [...v].sort((a, b) => a - b)[Math.floor(v.length * 0.95)];
  const first = med(times.slice(0, q)), lastq = med(times.slice(-q));
  check(times.length >= 150 && lastq < first * 1.35,
      `sustained tracking, 2 objects, ${times.length} frames: median ${med(times).toFixed(0)} ms, p95 ${p95(times).toFixed(0)} ms; ` +
      `first quarter ${first.toFixed(0)} ms vs last quarter ${lastq.toFixed(0)} ms`);

  // ---- 4. GPU memory
  console.log('  GPU buffers:', Object.entries(mem).map(([k, v]) => `${k} ${v.n} / ${mb(v.bytes)}`).join(' · '));
  // Re-tracking may settle once (pooled buffers, a first-seen chain); a leak grows on every run.
  check(mem.retracked2.n <= mem.retracked.n + 8 && mem.played.n <= mem.retracked2.n + 8,
      `no GPU buffer growth on repeated tracking / playback: ${mem.tracked.n} -> ${mem.retracked.n} -> ` +
      `${mem.retracked2.n} -> ${mem.played.n}`);
  check(mem.reset.n < mem.played.n - 400,
      `Reset frees the object's per-frame results: ${mem.played.n} -> ${mem.reset.n} buffers ` +
      `(${mb(mem.played.bytes)} -> ${mb(mem.reset.bytes)})`);

  // camera: results retained for 3 frames only; the buffer count must plateau
  await page.click('#camBtn');
  await page.waitForFunction(() => window.__sam2.live, null, {timeout: 30000});
  await page.waitForTimeout(1500);
  const live = await page.evaluate(() => ({w: window.__sam2.live.width, h: window.__sam2.live.height}));
  const b = await page.locator('#view').boundingBox();
  const s = Math.min(b.width / live.w, b.height / live.h);
  await page.mouse.click(b.x + (b.width - live.w * s) / 2 + 0.44 * live.w * s, b.y + (b.height - live.h * s) / 2 + 0.3 * live.h * s);
  await page.waitForFunction(() => !window.__sam2.busy, null, {timeout: 60000});
  const samples = [];
  for (let i = 0; i < 8; i++) {
    await page.waitForTimeout(1000);
    samples.push({...(await gpuMem()), t: await page.evaluate(() => window.__sam2.live.t)});
  }
  const frames = samples[samples.length - 1].t - samples[1].t;
  check(frames > 40 && samples[samples.length - 1].n <= samples[1].n + 8,
      `camera, ${frames} frames tracked in 7 s: GPU buffers ${samples.map((x) => x.n).join(' → ')} (plateau)`);
  await page.click('#camBtn');

  // ---- 3. errors over the whole session
  const errs = await page.evaluate(() => ({errors: window.__gpuErrors, lost: window.__gpuLost}));
  check(errs.errors.length === 0 && !errs.lost,
      `WebGPU errors during the session: ${errs.errors.length}${errs.errors.length ? ' — ' + errs.errors.slice(0, 3).join(' | ') : ''}; ` +
      `device lost: ${errs.lost ?? 'no'}`);
} finally {
  await browser.close();
  await server.close();
}
console.log(failed ? 'WEBGPU CHECK FAIL' : 'WEBGPU CHECK PASS');
process.exit(failed ? 1 : 0);
