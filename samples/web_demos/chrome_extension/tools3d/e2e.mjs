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
 * E2E hover-parallax test: serves a local gallery of generated test photos
 * (tools3d/fixtures.mjs), hovers the first one via CDP input, waits for the
 * WebGL overlay to activate, and captures screenshots at three mouse
 * positions (the parallax should visibly shift between them), then two with
 * Shift held (the cursor light).
 *
 *   node tools3d/e2e.mjs <chrome-binary> [--profile=<dir>]
 *       [--url=<page>] [--img=<css-selector>]
 *
 * With --url the local gallery server is skipped and the test runs against a
 * real page (e.g. a Wikipedia article), hovering the image picked by --img
 * (default: the largest visible <img>).
 * Screenshots land in out3d/e2e-{center,left,right,light-a,light-b}.png.
 * Exit code 0 = the overlay activated and stayed up through the screenshots
 * (it may be lost once, to a real pointer crossing the window: the test then
 * hovers again and logs "retry: overlay lost once, re-hovered").
 */
import { createServer } from 'node:http';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import {
  Cdp, attachTo, evalIn, findTarget, launchChrome, sleep, waitForEndpoint, waitForEngine,
} from '../tools/cdp.mjs';
import { FIXTURE_NAMES, fixturePng } from './fixtures.mjs';

const chromeBin = process.argv[2];
const profile = process.argv.find((a) => a.startsWith('--profile='))?.slice(10);
if (!chromeBin) {
  console.error('usage: node tools3d/e2e.mjs <chrome-binary> [--profile=<dir>]');
  process.exit(2);
}

const outDir = resolve(import.meta.dirname, '..', 'out3d');
mkdirSync(outDir, { recursive: true });

const urlArg = process.argv.find((a) => a.startsWith('--url='))?.slice(6);
const imgSelector = process.argv.find((a) => a.startsWith('--img='))?.slice(6);

const PORT_HTTP = 8917;
let server = null;
if (!urlArg) {
  const gallery = readFileSync(resolve(import.meta.dirname, 'gallery.html'));
  const images = Object.fromEntries(FIXTURE_NAMES.map((n) => [`/fixture-${n}.png`, fixturePng(n)]));
  server = createServer((req, res) => {
    const png = images[req.url];
    res.writeHead(200, { 'content-type': png ? 'image/png' : 'text/html' });
    res.end(png ?? gallery);
  }).listen(PORT_HTTP, '127.0.0.1');
}
const pageUrl = urlArg ?? `http://127.0.0.1:${PORT_HTTP}/gallery.html`;

const port = 9225;
launchChrome(chromeBin, {
  dist: resolve(import.meta.dirname, '..', 'dist3d'),
  port,
  profile,
  url: pageUrl,
});

async function shot(cdp, session, name, clip) {
  const { data } = await cdp.send('Page.captureScreenshot',
    clip ? { format: 'png', clip: { ...clip, scale: 2 } } : { format: 'png' }, session);
  writeFileSync(resolve(outDir, `e2e-${name}.png`), Buffer.from(data, 'base64'));
  console.log(`saved out3d/e2e-${name}.png`);
}

try {
  const cdp = await Cdp.connect(await waitForEndpoint(port));
  const { status, session: engineSession } = await waitForEngine(cdp, { hook: '__p3' });
  if (status?.state !== 'ready') {
    console.error('engine not ready:', JSON.stringify(status));
    process.exit(1);
  }

  const page = await findTarget(cdp, (t) =>
    t.type === 'page' && (urlArg ? t.url.startsWith(urlArg.slice(0, 24)) : t.url.includes('gallery')));
  const session = await attachTo(cdp, page);
  await cdp.send('Page.enable', {}, session);

  // The page opened with the browser can load before the extension has
  // registered its content script, and then never gets one — reload it and
  // wait for the content script's isolated world before hovering.
  let contentScriptReady = false;
  cdp.ws.addEventListener('message', (ev) => {
    const m = JSON.parse(ev.data);
    if (m.method === 'Runtime.executionContextCreated' && m.sessionId === session
      && (m.params.context.name || '').includes('Page 3D')) contentScriptReady = true;
  });
  await cdp.send('Runtime.enable', {}, session);
  contentScriptReady = false; // ignore the contexts replayed from before the reload
  await cdp.send('Page.reload', {}, session);
  for (let i = 0; i < 75 && !contentScriptReady; i++) await sleep(200);
  if (!contentScriptReady) throw new Error('content script never attached to the page');

  // Wait for the test image itself to be loaded before hovering it.
  // Default target: #img1 (gallery) or the largest loaded <img> in view.
  const pickImg = imgSelector
    ? `document.querySelector(${JSON.stringify(imgSelector)})`
    : urlArg
      ? `[...document.images]
          .filter((i) => i.complete && i.naturalWidth > 50)
          .map((i) => [i, i.getBoundingClientRect()])
          .filter(([, r]) => r.top >= 0 && r.bottom < innerHeight && r.width >= 150 && r.height >= 110)
          .sort((a, b) => b[1].width * b[1].height - a[1].width * a[1].height)[0]?.[0]`
      : `document.getElementById('img1')`;
  const deadline = Date.now() + 60 * 1000;
  let rect = null;
  while (Date.now() < deadline) {
    try {
      const raw = await evalIn(cdp, session, `(() => {
        const img = ${pickImg};
        if (!img || !img.complete || !img.naturalWidth) return null;
        const r = img.getBoundingClientRect();
        return JSON.stringify({ x: r.left, y: r.top, w: r.width, h: r.height });
      })()`);
      if (raw) { rect = JSON.parse(raw); break; }
    } catch { /* navigating */ }
    await sleep(300);
  }
  if (!rect) throw new Error('target image never loaded');

  const cx = rect.x + rect.w / 2;
  const cy = rect.y + rect.h / 2;

  // The browser window sits on a desktop someone may be using (launchChrome
  // keeps 40 px of it on screen), and a real pointer crossing it lands in the
  // page just like the test's moves. The content script then sees the
  // pointer off the photo and tears the overlay down 220 ms later (or never
  // shows it, if that happens during inference). So the page notes the last
  // pointer position and counts frames for the test: every wait puts a
  // pointer the test did not send back on the test's point, the overlay is
  // checked around each screenshot, and a lost overlay gets one re-hover.
  await evalIn(cdp, session, `(() => {
    window.__e2e = { frames: 0, pointer: null };
    addEventListener('pointermove', (e) => { __e2e.pointer = [e.clientX, e.clientY]; },
      { capture: true, passive: true });
    const tick = () => { __e2e.frames++; requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
  })()`);
  const look = async () => JSON.parse(await evalIn(cdp, session, `JSON.stringify({
    frames: __e2e.frames,
    pointer: __e2e.pointer,
    overlay: (() => {
      const host = document.querySelector('[data-page3d]');
      const state = host?.dataset.page3d ?? null;
      // teardown fades the host out before removing it
      return state === 'active' && host.style.opacity === '0' ? 'fading' : state;
    })(),
  })`));
  const sent = []; // every point the test has moved the pointer to
  let want = null; // where the test's pointer should be now
  const dispatch = ([x, y]) =>
    cdp.send('Input.dispatchMouseEvent', { type: 'mouseMoved', x, y }, session);
  const move = (x, y) => {
    want = [x, y];
    sent.push(want);
    return dispatch(want);
  };
  let strays = 0;
  const check = async () => {
    const s = await look();
    const ours = !s.pointer || sent.some(([x, y]) =>
      Math.abs(x - s.pointer[0]) <= 1 && Math.abs(y - s.pointer[1]) <= 1);
    if (!ours) {
      strays++;
      console.log(`stray pointer at (${s.pointer.map(Math.round).join(', ')}), moved back`);
      await dispatch(want);
    }
    return s;
  };
  // The overlay eases per frame (rotation 0.12, light 0.16 of the way), so
  // wait for frames, not time: a loaded machine draws fewer of them.
  const settle = async (frames, minMs = 0) => {
    const start = Date.now();
    const goal = (await look()).frames + frames;
    for (;;) {
      await sleep(100);
      const s = await check();
      if (s.frames >= goal && Date.now() - start >= minMs) return;
      if (Date.now() - start > 10 * 1000) {
        console.log(`only ${frames - (goal - s.frames)} of ${frames} frames drawn in 10 s`);
        return;
      }
    }
  };
  const hover = async () => {
    await move(cx, cy);
    await sleep(120);
    await move(cx + 4, cy); // second move so dwell sees a settled pointer
  };
  const waitActive = async (timeoutMs) => {
    const hoverStart = Date.now();
    let lastState = '';
    while (Date.now() - hoverStart < timeoutMs) {
      const state = (await check()).overlay;
      if (state !== lastState) {
        console.log(`t+${((Date.now() - hoverStart) / 1000).toFixed(1)}s overlay: ${state}`);
        lastState = state;
      }
      if (state === 'active') return true;
      await sleep(400);
    }
    return false;
  };

  // Overlay appears after dwell + fetch + inference (model is already cached).
  await hover();
  const active = await waitActive(180 * 1000);
  const engineNow = JSON.parse(await evalIn(cdp, engineSession, 'JSON.stringify(__p3.status)'));
  console.log('engine stats:', JSON.stringify(engineNow.stats), 'runs:', engineNow.runs);
  if (!active) {
    await shot(cdp, session, 'failed');
    console.error('overlay never activated');
    process.exit(1);
  }

  // A stray real scroll can move the page mid-test too, so pin the scroll
  // position before each capture and clip to the image rect (margin for
  // pop-out).
  const pin = () => evalIn(cdp, session, 'window.scrollTo(0, 0), null');
  const clip = {
    x: Math.max(0, rect.x - 24), y: Math.max(0, rect.y - 24),
    width: rect.w + 48, height: rect.h + 48,
  };
  // Cursor light: hold Shift and capture the flashlight at two positions —
  // the lit pool must follow the cursor (compare light-a vs light-b, and
  // either against center for the ambient dim).
  const key = (type) => cdp.send('Input.dispatchKeyEvent',
    { type, key: 'Shift', windowsVirtualKeyCode: 16, modifiers: 8 }, session);
  let shiftDown = false;
  const beats = [ // name, pointer x, y, frames to settle, at least ms
    ['center', cx + 4, cy, 30, 800], // the extrude-in takes 450 ms
    ['left', rect.x + rect.w * 0.06, cy - rect.h * 0.2, 60],
    ['right', rect.x + rect.w * 0.94, cy + rect.h * 0.2, 60],
    ['light-a', rect.x + rect.w * 0.3, rect.y + rect.h * 0.35, 45],
    ['light-b', rect.x + rect.w * 0.72, rect.y + rect.h * 0.6, 45],
  ];
  // Every screenshot from one overlay; returns where it was lost, if it was.
  const capture = async () => {
    for (const [name, x, y, frames, minMs] of beats) {
      if (name.startsWith('light') && !shiftDown) {
        await key('rawKeyDown');
        shiftDown = true;
      }
      await move(x, y);
      await settle(frames, minMs);
      await pin();
      let { overlay } = await check();
      if (overlay !== 'active') return `before the ${name} screenshot: ${overlay}`;
      await shot(cdp, session, name, clip);
      ({ overlay } = await check());
      if (overlay !== 'active') return `during the ${name} screenshot: ${overlay}`;
    }
    await key('keyUp');
    shiftDown = false;
    const { overlay } = await check();
    return overlay === 'active' ? null : `during light beat: ${overlay}`;
  };

  let lost = await capture();
  const retried = Boolean(lost);
  if (lost) {
    if (shiftDown) await key('keyUp');
    shiftDown = false;
    console.log(`retry: overlay lost once, re-hovered (${lost})`);
    await hover();
    if (!(await waitActive(60 * 1000))) throw new Error(`overlay lost ${lost}, and not back on re-hover`);
    lost = await capture();
    if (lost) throw new Error(`overlay lost ${lost}`);
  }

  console.log('E2E_RESULT ' + JSON.stringify({
    active, light: true, retried, strayPointers: strays, rect,
  }, null, 2));
  await cdp.send('Browser.close').catch(() => {});
  server?.close();
  process.exit(0);
} catch (err) {
  console.error('e2e failed:', err.message);
  server?.close();
  process.exit(1);
}
