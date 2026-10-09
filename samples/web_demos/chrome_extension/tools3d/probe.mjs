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
 * One-off diagnostic for hover-activation failures on a real page: launches
 * the browser with dist3d/, navigates, dumps the hit-test stack over the
 * target image, dispatches hover moves, and reports what the content script
 * did (badge? overlay? engine runs?). Console/Log entries are echoed.
 *
 *   node tools3d/probe.mjs <chrome-binary> <url> [--profile=<dir>]
 */
import { resolve } from 'node:path';
import {
  Cdp, attachTo, evalIn, findTarget, launchChrome, sleep, waitForEndpoint, waitForEngine,
} from '../tools/cdp.mjs';

const [, , chromeBin, url] = process.argv;
const profile = process.argv.find((a) => a.startsWith('--profile='))?.slice(10);

const port = 9227;
launchChrome(chromeBin, {
  dist: resolve(import.meta.dirname, '..', 'dist3d'),
  port,
  profile,
  url,
});

const cdp = await Cdp.connect(await waitForEndpoint(port));
const { status, session: engineSession } = await waitForEngine(cdp, { hook: '__p3' });
console.log('engine:', status?.state, status?.env);

const page = await findTarget(cdp, (t) =>
  t.type === 'page' && t.url.startsWith(url.slice(0, 30)));
const session = await attachTo(cdp, page);
await cdp.send('Page.enable', {}, session);
await cdp.send('Target.activateTarget', { targetId: page.targetId });
await cdp.send('Page.bringToFront', {}, session).catch(() => {});
await cdp.send('Log.enable', {}, session).catch(() => {});
await cdp.send('Runtime.enable', {}, session).catch(() => {});
cdp.ws.addEventListener('message', (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.method === 'Log.entryAdded') {
    const e = msg.params.entry;
    console.log(`[log:${e.source}/${e.level}] ${e.text}`);
  } else if (msg.method === 'Runtime.consoleAPICalled') {
    const args = msg.params.args.map((a) => a.value ?? a.description ?? '').join(' ');
    console.log(`[console:${msg.params.type}] ${args}`);
  }
});
await sleep(4000);

const diag = await evalIn(cdp, session, `(() => {
  const img = [...document.images]
    .filter((i) => i.complete && i.naturalWidth > 50)
    .map((i) => [i, i.getBoundingClientRect()])
    .filter(([, r]) => r.top >= 0 && r.bottom < innerHeight && r.width >= 150 && r.height >= 110)
    .sort((a, b) => b[1].width * b[1].height - a[1].width * a[1].height)[0]?.[0];
  if (!img) return JSON.stringify({ error: 'no candidate image' });
  const r = img.getBoundingClientRect();
  const cx = r.left + r.width / 2;
  const cy = r.top + r.height / 2;
  const stack = document.elementsFromPoint(cx, cy).slice(0, 8)
    .map((el) => el.tagName + (el.className ? '.' + String(el.className).slice(0, 40) : ''));
  return JSON.stringify({
    rect: { x: r.left, y: r.top, w: r.width, h: r.height },
    complete: img.complete,
    naturalWidth: img.naturalWidth,
    src: (img.currentSrc || img.src).slice(0, 90),
    stack,
  }, null, 2);
})()`);
console.log('diag:', diag);

const { rect } = JSON.parse(diag);
if (rect) {
  const cx = rect.x + rect.w / 2;
  const cy = rect.y + rect.h / 2;
  for (let i = 0; i < 12; i++) {
    await cdp.send('Input.dispatchMouseEvent',
      { type: 'mouseMoved', x: cx + i, y: cy }, session);
    await sleep(120);
  }
  await sleep(2500);
  const after = await evalIn(cdp, session, `JSON.stringify({
    overlay: document.querySelector('[data-page3d]')?.dataset.page3d ?? null,
    pills: [...document.querySelectorAll('div')].filter((d) =>
      d.style.zIndex === '2147483647').length,
  })`);
  const engineNow = JSON.parse(await evalIn(cdp, engineSession, 'JSON.stringify(__p3.status)'));
  console.log('after hover:', after, 'engine runs:', engineNow.runs, 'stats:', JSON.stringify(engineNow.stats));
}
await cdp.send('Browser.close').catch(() => {});
process.exit(0);
