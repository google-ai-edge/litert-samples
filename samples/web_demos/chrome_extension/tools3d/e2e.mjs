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
 * positions (the parallax should visibly shift between them).
 *
 *   node tools3d/e2e.mjs <chrome-binary> [--profile=<dir>]
 *       [--url=<page>] [--img=<css-selector>]
 *
 * With --url the local gallery server is skipped and the test runs against a
 * real page (e.g. a Wikipedia article), hovering the image picked by --img
 * (default: the largest visible <img>).
 * Screenshots land in out3d/e2e-{center,left,right}.png.
 * Exit code 0 = overlay activated.
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
  const move = (x, y) =>
    cdp.send('Input.dispatchMouseEvent', { type: 'mouseMoved', x, y }, session);

  await move(cx, cy);
  await sleep(120);
  await move(cx + 4, cy); // second move so dwell sees a settled pointer

  // Overlay appears after dwell + fetch + inference (model is already cached).
  let active = false;
  const hoverStart = Date.now();
  const overlayDeadline = hoverStart + 180 * 1000;
  let lastState = '';
  while (Date.now() < overlayDeadline) {
    const state = await evalIn(cdp, session,
      `document.querySelector('[data-page3d]')?.dataset.page3d ?? null`);
    if (state !== lastState) {
      console.log(`t+${((Date.now() - hoverStart) / 1000).toFixed(1)}s overlay: ${state}`);
      lastState = state;
    }
    if (state === 'active') { active = true; break; }
    await sleep(400);
  }
  const engineNow = JSON.parse(await evalIn(cdp, engineSession, 'JSON.stringify(__p3.status)'));
  console.log('engine stats:', JSON.stringify(engineNow.stats), 'runs:', engineNow.runs);
  if (!active) {
    await shot(cdp, session, 'failed');
    console.error('overlay never activated');
    process.exit(1);
  }

  // The browser window is visible on the user's desktop — a stray real
  // scroll/input can move the page mid-test, so pin the scroll position
  // before each capture and clip to the image rect (margin for pop-out).
  const pin = () => evalIn(cdp, session, 'window.scrollTo(0, 0), null');
  const clip = {
    x: Math.max(0, rect.x - 24), y: Math.max(0, rect.y - 24),
    width: rect.w + 48, height: rect.h + 48,
  };
  await sleep(800); // let the extrude-in settle
  await pin();
  await shot(cdp, session, 'center', clip);
  await move(rect.x + rect.w * 0.06, cy - rect.h * 0.2);
  await sleep(900); // rot lerp settles in ~0.5 s
  await pin();
  await shot(cdp, session, 'left', clip);
  await move(rect.x + rect.w * 0.94, cy + rect.h * 0.2);
  await sleep(900);
  await pin();
  await shot(cdp, session, 'right', clip);

  // Cursor light: hold Shift and capture the flashlight at two positions —
  // the lit pool must follow the cursor (compare light-a vs light-b, and
  // either against center for the ambient dim).
  const key = (type) => cdp.send('Input.dispatchKeyEvent',
    { type, key: 'Shift', windowsVirtualKeyCode: 16, modifiers: 8 }, session);
  await key('rawKeyDown');
  await move(rect.x + rect.w * 0.3, rect.y + rect.h * 0.35);
  await sleep(700); // light fade-in ≈ 0.16/frame
  await pin();
  await shot(cdp, session, 'light-a', clip);
  await move(rect.x + rect.w * 0.72, rect.y + rect.h * 0.6);
  await sleep(700);
  await pin();
  await shot(cdp, session, 'light-b', clip);
  await key('keyUp');
  const lightState = await evalIn(cdp, session,
    `document.querySelector('[data-page3d]')?.dataset.page3d ?? null`);
  if (lightState !== 'active') throw new Error(`overlay lost during light beat: ${lightState}`);

  console.log('E2E_RESULT ' + JSON.stringify({ active, light: true, rect }, null, 2));
  await cdp.send('Browser.close').catch(() => {});
  server?.close();
  process.exit(0);
} catch (err) {
  console.error('e2e failed:', err.message);
  server?.close();
  process.exit(1);
}
