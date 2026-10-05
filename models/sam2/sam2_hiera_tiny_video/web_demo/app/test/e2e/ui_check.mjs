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

// UI end-to-end check of the demo app (C++ ModelChain pipeline in wasm on
// WebGPU), in system Chrome via playwright-core:
//   1. the football sample opens by default and the pipeline loads;
//   2. clicks: positive grows the mask, negative shrinks it, Reset clears it;
//   3. two objects (player 2 clicks, ball) tracked over the whole clip: masks
//      on >= 90% of frames, sane areas, no identity jumps;
//   4. every effect renders (the composite graph's output on the canvas);
//   5. the page fits one screen at common sizes (no scrolling);
//   6. camera mode (fake camera): a click segments, then tracking runs live.
// Screenshots go to test/e2e/screenshots/.
//
//   npm run e2e:ui
import {execFileSync} from 'node:child_process';
import {existsSync, mkdirSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {resolve} from 'node:path';
import {chromium} from 'playwright-core';
import {createServer} from 'vite';

const app = resolve(import.meta.dirname, '../..');
const shots = resolve(app, 'test/e2e/screenshots');
mkdirSync(shots, {recursive: true});
const y4m = resolve(tmpdir(), 'sam2chain_football_640.y4m');
if (!existsSync(y4m)) {
  execFileSync('ffmpeg', ['-loglevel', 'error', '-y', '-i', resolve(app, 'public/assets/football_ai_studio.mp4'),
    '-t', '4', '-vf', 'scale=640:360', '-pix_fmt', 'yuv420p', y4m]);
}
const server = await createServer({root: app, logLevel: 'error', server: {port: 5177}});
await server.listen();
const browser = await chromium.launch({channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu',
  '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', `--use-file-for-fake-video-capture=${y4m}`]});
let failed = false;
const check = (ok, msg) => {
  console.log(`${ok ? '  ok ' : '  FAIL'} ${msg}`);
  if (!ok) failed = true;
};
try {
  const page = await browser.newPage({viewport: {width: 1440, height: 900}});
  page.on('pageerror', (e) => console.log(`  [pageerror] ${e.message}`));
  page.on('console', (m) => { if (m.type() === 'error') console.log(`  [console] ${m.text()}`); });
  const t0 = Date.now();
  await page.goto('http://localhost:5177/');
  await page.waitForFunction(() => window.__sam2?.engine && window.__sam2?.clip && !window.__sam2.loading,
      null, {timeout: 300000});
  const clip = await page.evaluate(() => ({name: window.__sam2.clip.name, n: window.__sam2.clip.frames.length,
    w: window.__sam2.clip.width, h: window.__sam2.clip.height, fps: window.__sam2.clip.fps}));
  check(clip.name === 'Football (sample video)' && clip.n === 192,
      `sample + pipeline ready in ${((Date.now() - t0) / 1000).toFixed(1)} s: ${clip.n} frames @ ${clip.fps} fps, ${clip.w}x${clip.h}`);
  console.log('  backend:', await page.textContent('#backendPill'));
  await page.waitForTimeout(500);
  check(await page.evaluate(() => document.getElementById('overlayMsg').hidden),
      'loading overlay hidden once the pipeline and the sample are ready');

  const click = async ([nx, ny], opts = {}) => {
    const b = await page.locator('#view').boundingBox();
    const s = Math.min(b.width / clip.w, b.height / clip.h);
    if (opts.shift) await page.keyboard.down('Shift');
    await page.mouse.click(b.x + (b.width - clip.w * s) / 2 + nx * clip.w * s, b.y + (b.height - clip.h * s) / 2 + ny * clip.h * s);
    if (opts.shift) await page.keyboard.up('Shift');
    await page.waitForFunction(() => !window.__sam2.busy, null, {timeout: 60000});
  };
  const area = (i = 0) => page.evaluate((i) => {
    const s = window.__sam2, o = s.objects[i];
    const m = s.results.get(s.frame)?.get(o.id);
    return {n: o.point?.pts.length ?? 0, labels: o.point?.pts.map((p) => p.label).join('') ?? '',
      area: m ? m.filter((v) => v > 0).length / m.length : 0};
  }, i);

  // ---- 2. clicks
  await click([0.44, 0.28]);
  const a1 = await area();
  await click([0.45, 0.47]);
  const a2 = await area();
  check(a1.area > 0.003 && a2.n === 2 && a2.area > a1.area * 1.2,
      `positive clicks grow the mask: ${(100 * a1.area).toFixed(2)}% -> ${(100 * a2.area).toFixed(2)}% (${await page.textContent('#trackInfo')})`);
  await page.click('#polSeg button[data-v="neg"]');
  await click([0.44, 0.28]);
  const a3 = await area();
  check(a3.labels === '110' && a3.area < a2.area * 0.8, `negative click shrinks it: ${(100 * a3.area).toFixed(2)}% (clicks ${a3.labels})`);
  await page.screenshot({path: `${shots}/ui_pos_neg.png`});
  await page.click('#resetObjBtn');
  const a4 = await area();
  check(a4.n === 0 && a4.area === 0, 'Reset clears the clicks and the mask');

  // ---- 2b. box: drag around the ball, then refine with a click
  const drag = async ([x0, y0], [x1, y1]) => {
    const b = await page.locator('#view').boundingBox();
    const s = Math.min(b.width / clip.w, b.height / clip.h);
    const px = (nx) => b.x + (b.width - clip.w * s) / 2 + nx * clip.w * s;
    const py = (ny) => b.y + (b.height - clip.h * s) / 2 + ny * clip.h * s;
    await page.mouse.move(px(x0), py(y0));
    await page.mouse.down();
    await page.mouse.move(px((x0 + x1) / 2), py((y0 + y1) / 2), {steps: 4});
    await page.mouse.move(px(x1), py(y1), {steps: 4});
    await page.mouse.up();
    await page.waitForFunction(() => !window.__sam2.busy, null, {timeout: 60000});
  };
  const boxStats = () => page.evaluate(() => {
    const s = window.__sam2, o = s.objects[0];
    const m = s.results.get(s.frame)?.get(o.id);
    if (!m) return null;
    const N = Math.sqrt(m.length);
    let n = 0, sx = 0, sy = 0;
    for (let i = 0; i < m.length; i++) if (m[i] > 0) { n++; sx += i % N; sy += Math.floor(i / N); }
    return {labels: o.point.pts.map((p) => p.label).join(''), area: n / m.length, cx: (sx / n + 0.5) / N, cy: (sy / n + 0.5) / N,
      status: document.querySelector('#objectList .obj.sel .state').textContent};
  });
  await page.click('#polSeg button[data-v="box"]');
  await drag([0.455, 0.70], [0.515, 0.87]);   // the ball
  const bx = await boxStats();
  check(bx && bx.labels === '23' && bx.area > 0.0005 && bx.area < 0.02 && bx.cx > 0.455 && bx.cx < 0.515 && bx.cy > 0.7 && bx.cy < 0.87,
      `box around the ball segments it: ${bx ? (100 * bx.area).toFixed(2) + '% of the frame, centroid (' + bx.cx.toFixed(3) + ', ' + bx.cy.toFixed(3) + ')' : 'no mask'} ` +
      `(${await page.textContent('#trackInfo')})`);
  await page.screenshot({path: `${shots}/ui_box.png`});
  await page.click('#polSeg button[data-v="neg"]');
  await click([0.49, 0.74]);                  // refine: a negative click inside the box
  const bx2 = await boxStats();
  check(bx2 && bx2.labels === '230' && /box \+ 1 click/.test(bx2.status),
      `a click refines the box: points ${bx2?.labels}, object shows "${bx2?.status}"`);
  await page.click('#polSeg button[data-v="box"]');
  await drag([0.40, 0.15], [0.52, 0.95]);     // a new box replaces the old one, the click stays
  const bx3 = await boxStats();
  check(bx3 && bx3.labels === '230' && bx3.area > bx.area * 3,
      `a new box replaces the previous one: ${(100 * bx3.area).toFixed(2)}% of the frame (player), points ${bx3.labels}`);
  await page.click('#resetObjBtn');
  await page.click('#polSeg button[data-v="pos"]');
  await page.click('#polSeg button[data-v="pos"]');

  // ---- 3. two objects over the whole clip
  await click([0.44, 0.28]);
  await click([0.45, 0.47]);
  await page.click('#addObjBtn');
  await click([0.484, 0.79]);
  await page.screenshot({path: `${shots}/ui_prompted.png`});
  const tt = Date.now();
  await page.click('#trackBtn');
  await page.waitForFunction(() => !window.__sam2.tracking && /Done|Stopped|failed/.test(
      document.getElementById('trackInfo').textContent), null, {timeout: 20 * 60 * 1000, polling: 1000});
  const info = await page.textContent('#trackInfo');
  console.log(`  ${info} (${((Date.now() - tt) / 1000).toFixed(1)} s wall)`);
  const tracks = await page.evaluate(() => {
    const s = window.__sam2;
    return s.objects.map((o) => {
      const per = [];
      for (let f = 0; f < s.clip.frames.length; f++) {
        const m = s.results.get(f)?.get(o.id);
        if (!m) { per.push(null); continue; }
        const N = Math.sqrt(m.length);
        let n = 0, sx = 0, sy = 0;
        for (let i = 0; i < m.length; i++) if (m[i] > 0) { n++; sx += i % N; sy += Math.floor(i / N); }
        per.push(n ? {area: n / m.length, cx: (sx / n + 0.5) / N, cy: (sy / n + 0.5) / N} : {area: 0});
      }
      return per;
    });
  });
  const targets = [{name: 'player', area: [0.01, 0.4]}, {name: 'ball', area: [0.0005, 0.05]}];
  tracks.forEach((per, k) => {
    const t = targets[k];
    const present = per.filter((p) => p && p.area > 0);
    const areas = present.map((p) => p.area).sort((a, b) => a - b);
    const med = areas[Math.floor(areas.length / 2)];
    // Identity switches: centroid jumps between consecutive frames where the
    // object is well visible. Frames whose mask collapses below 30% of the
    // median area (heavy occlusion: the mask fragments onto the visible bits)
    // are counted, not scored.
    const visible = (p) => p?.area >= 0.3 * med;
    let jump = 0;
    for (let i = 1; i < per.length; i++) {
      if (visible(per[i - 1]) && visible(per[i])) {
        jump = Math.max(jump, Math.hypot(per[i - 1].cx - per[i].cx, per[i - 1].cy - per[i].cy));
      }
    }
    const occluded = present.filter((p) => !visible(p)).length;
    check(present.length / per.length >= 0.9 && med >= t.area[0] && med <= t.area[1] && jump < 0.15 &&
        occluded <= 0.1 * per.length,
        `${t.name}: mask on ${present.length}/${per.length} frames, median area ${(100 * med).toFixed(2)}%, ` +
        `largest centroid jump ${(100 * jump).toFixed(1)}% (${occluded} occluded frames)`);
  });

  // ---- 4. effects: the canvas shows the composite graph's output buffer
  // (a WebGPU canvas cannot be read back once presented, so read the buffer
  // it was blitted from; the screenshots show the canvas itself).
  const canvasStats = () => page.evaluate(async () => {
    const d = await window.__sam2.engine.readOutput();
    const n = d.length / 3;
    let green = 0, dark = 0, mean = 0;
    for (let i = 0; i < d.length; i += 3) {
      const [r, g, b] = [d[i] * 255, d[i + 1] * 255, d[i + 2] * 255];
      if (r < 20 && g > 150 && b > 40 && b < 90) green++;
      if (r + g + b < 150) dark++;
      mean += r + g + b;
    }
    return {green: green / n, dark: dark / n, mean: mean / n / 3};
  });
  await page.evaluate(() => {
    const el = document.getElementById('scrubber');
    el.value = '96';
    el.dispatchEvent(new Event('input'));
  });
  await page.waitForTimeout(500);
  const fx = {};
  for (const e of ['overlay', 'spotlight', 'cutout']) {
    await page.click(`#fxSeg button[data-v="${e}"]`);
    await page.waitForTimeout(600);
    fx[e] = await canvasStats();
    await page.locator('#view').screenshot({path: `${shots}/ui_f96_${e}.png`});
  }
  check(fx.overlay.mean > 60 && fx.cutout.green > 0.8 && fx.spotlight.dark > 0.6,
      `effects render on the canvas: overlay mean ${fx.overlay.mean.toFixed(0)}, spotlight ${(100 * fx.spotlight.dark).toFixed(0)}% dark, ` +
      `cutout ${(100 * fx.cutout.green).toFixed(0)}% green`);
  await page.click('#fxSeg button[data-v="overlay"]');

  // ---- 5. layout
  for (const [w, h] of [[1440, 900], [1280, 720], [390, 844]]) {
    await page.setViewportSize({width: w, height: h});
    await page.waitForTimeout(300);
    const m = await page.evaluate(() => ({sh: document.scrollingElement.scrollHeight, sw: document.scrollingElement.scrollWidth,
      ih: innerHeight, iw: innerWidth, vh: document.getElementById('canvasWrap').getBoundingClientRect().height}));
    check(m.sh <= m.ih && m.sw <= m.iw && m.vh / m.ih > 0.4, `${w}x${h}: no page scroll, video ${Math.round(100 * m.vh / m.ih)}% of the height`);
  }
  await page.setViewportSize({width: 1440, height: 900});

  // ---- 6. camera
  await page.click('#camBtn');
  await page.waitForFunction(() => window.__sam2.live, null, {timeout: 30000});
  await page.waitForTimeout(1500);
  const live = await page.evaluate(() => ({w: window.__sam2.live.width, h: window.__sam2.live.height}));
  const b = await page.locator('#view').boundingBox();
  const s = Math.min(b.width / live.w, b.height / live.h);
  await page.mouse.click(b.x + (b.width - live.w * s) / 2 + 0.44 * live.w * s, b.y + (b.height - live.h * s) / 2 + 0.3 * live.h * s);
  await page.waitForFunction(() => !window.__sam2.busy, null, {timeout: 60000});
  const segInfo = await page.textContent('#trackInfo');
  await page.waitForTimeout(4000);
  const stats = await page.textContent('#stats');
  const vfps = Number((stats.match(/Video\s*(\d+)\s*fps/) ?? [])[1] ?? 0);
  check(/Segmented/.test(segInfo) && /Masks/.test(stats) && vfps >= 20,
      `camera, smooth (default): ${segInfo.split(';')[0]} · ${stats.replace(/\s+/g, ' ')}`);
  await page.screenshot({path: `${shots}/ui_camera_smooth.png`});
  // Clicks accumulate in camera mode too: + (shorts), − (head), then Reset.
  const liveClick = async ([nx, ny]) => {
    const vb = await page.locator('#view').boundingBox();
    const vs = Math.min(vb.width / live.w, vb.height / live.h);
    await page.mouse.click(vb.x + (vb.width - live.w * vs) / 2 + nx * live.w * vs, vb.y + (vb.height - live.h * vs) / 2 + ny * live.h * vs);
    await page.waitForFunction(() => !window.__sam2.busy, null, {timeout: 60000});
  };
  const liveLabels = () => page.evaluate(() => window.__sam2.objects[0].point?.pts.map((c) => c.label).join('') ?? '');
  const polEnabled = await page.evaluate(() => [...document.querySelectorAll('#polSeg button')].every((b) => !b.disabled));
  await liveClick([0.45, 0.47]);
  const l2 = await liveLabels();
  await page.click('#polSeg button[data-v="neg"]');
  await liveClick([0.44, 0.2]);
  const l3 = await liveLabels();
  const info3 = await page.textContent('#trackInfo');
  await page.click('#polSeg button[data-v="pos"]');
  const resetEnabled = await page.evaluate(() => !document.getElementById('resetObjBtn').disabled);
  await page.click('#resetObjBtn');
  await page.waitForTimeout(800);
  const afterReset = await page.evaluate(() => ({n: window.__sam2.objects[0].point?.pts.length ?? 0,
    tracked: window.__sam2.engine.trackedObjects(window.__sam2.live.t + 1).length}));
  check(polEnabled && l2 === '11' && (l3 === '110' || /kept the previous/.test(info3)) && resetEnabled &&
      afterReset.n === 0 && afterReset.tracked === 0,
      `camera clicks: +/− buttons enabled, clicks ${l2} -> ${l3}, Reset clears the object (${info3.split(';')[0]})`);
  // Box in camera mode: prompts the newest camera frame.
  await page.click('#polSeg button[data-v="box"]');
  {
    const vb = await page.locator('#view').boundingBox();
    const vs = Math.min(vb.width / live.w, vb.height / live.h);
    const px = (nx) => vb.x + (vb.width - live.w * vs) / 2 + nx * live.w * vs;
    const py = (ny) => vb.y + (vb.height - live.h * vs) / 2 + ny * live.h * vs;
    await page.mouse.move(px(0.36), py(0.12));
    await page.mouse.down();
    await page.mouse.move(px(0.54), py(0.95), {steps: 6});
    await page.mouse.up();
    await page.waitForFunction(() => !window.__sam2.busy, null, {timeout: 60000});
  }
  const liveBox = await liveLabels();
  const liveInfo = await page.textContent('#trackInfo');
  check(liveBox === '23' && /Segmented with box/.test(liveInfo), `camera box: points ${liveBox} (${liveInfo.split(';')[0]})`);
  await page.click('#polSeg button[data-v="pos"]');
  await page.click('#liveSeg button[data-v="aligned"]');
  await page.waitForTimeout(3000);
  const stats2 = await page.textContent('#stats');
  check(/Rate/.test(stats2), `camera, aligned: ${stats2.replace(/\s+/g, ' ')}`);
  await page.screenshot({path: `${shots}/ui_camera_aligned.png`});
  await page.click('#camBtn');
} finally {
  await browser.close();
  await server.close();
}
console.log(failed ? 'UI CHECK FAIL' : 'UI CHECK PASS');
process.exit(failed ? 1 : 0);
