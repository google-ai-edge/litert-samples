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
 * Find-in-images E2E: serves the demo feed, lets the extension index it in the
 * background, then searches for a phrase that only exists INSIDE a screenshot
 * and checks that the right lines light up.
 *
 *   node tools-ocr/find-e2e.mjs <chrome-binary> [--profile=<dir>]
 *
 * Pass a persistent --profile: a freshly installed extension's service worker
 * cannot reach the content script yet.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, resolve } from 'node:path';
import { Cdp, attachTo, evalIn, findTarget, launchChrome, sleep, waitForEndpoint, waitForEngine } from '../tools/cdp.mjs';

const chromeBin = process.argv[2];
const profile = process.argv.find((a) => a.startsWith('--profile='))?.slice(10);
if (!chromeBin) {
  console.error('usage: node tools-ocr/find-e2e.mjs <chrome-binary> [--profile=<dir>]');
  process.exit(2);
}
const outDir = resolve(import.meta.dirname, '..', 'out-ocr');
const MIME = { '.html': 'text/html', '.png': 'image/png' };
const server = createServer((req, res) => {
  try {
    const path = req.url === '/' ? resolve(import.meta.dirname, 'demo.html')
      : resolve(outDir, decodeURIComponent(req.url.slice(1)));
    const body = readFileSync(path);
    res.writeHead(200, { 'Content-Type': MIME[extname(path)] ?? 'text/html' });
    res.end(body);
  } catch { res.writeHead(404).end(); }
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;

const port = 9240;
launchChrome(chromeBin, {
  dist: resolve(import.meta.dirname, '..', 'dist-ocr'),
  port,
  profile,
  url: `${base}/`,
});

// Phrases that exist ONLY inside the screenshots, never in the page's own DOM.
const CASES = [
  ['power outlets', 'fixture-en-dark'],
  ['PP-OCRv5', 'fixture-en-light'],
  ['lunch', 'fixture-en-dark'],
  // sits where a rec window ends: must not come back as "flashi ng"
  ['flashing', 'fixture-en-ship'],
];

try {
  const cdp = await Cdp.connect(await waitForEndpoint(port));
  const { status } = await waitForEngine(cdp, { hook: '__pt' });
  if (status?.state !== 'ready') throw new Error(`engine: ${JSON.stringify(status)}`);

  const page = await findTarget(cdp, (t) => t.type === 'page' && t.url.startsWith(base));
  const session = await attachTo(cdp, page);
  await cdp.send('Page.enable', {}, session);

  // The content script's globals live in its ISOLATED world; a plain
  // Runtime.evaluate lands in the page's main world and cannot see them.
  // Capture the extension's context id, then evaluate there.
  const worlds = [];
  cdp.ws.addEventListener('message', (ev) => {
    const m = JSON.parse(ev.data);
    if (m.method === 'Runtime.executionContextCreated') worlds.push(m.params.context);
  });
  await cdp.send('Runtime.enable', {}, session);
  worlds.length = 0; // contexts from before the reload are destroyed with it
  await cdp.send('Page.reload', {}, session);
  await sleep(3000);
  // take the newest matching context: reloads can leave stale ids behind
  const world = [...worlds].reverse().find((c) => (c.name || '').includes('Page Text'));
  if (!world) throw new Error(`content-script world not found; saw ${JSON.stringify(worlds.map((c) => c.name))}`);
  const evalCs = async (expression) => {
    const { result, exceptionDetails } = await cdp.send('Runtime.evaluate',
      { expression, contextId: world.id, returnByValue: true, awaitPromise: true }, session, 60000);
    if (exceptionDetails) throw new Error(exceptionDetails.exception?.description ?? 'eval failed');
    return result?.value;
  };

  // The page's own text must not contain the phrases — otherwise a "hit" would
  // prove nothing about reading the images.
  const inDom = JSON.parse(await evalIn(cdp, session,
    `JSON.stringify(${JSON.stringify(CASES.map((c) => c[0]))}.map((q) => document.body.innerText.includes(q)))`));
  if (inDom.some(Boolean)) throw new Error(`phrase leaked into page DOM: ${JSON.stringify(inDom)}`);

  // index the page
  await evalCs('__ptContent.scan(), null');
  let indexed = 0;
  const deadline = Date.now() + 120000;
  while (Date.now() < deadline) {
    const st = JSON.parse(await evalCs(
      'JSON.stringify({i: __ptContent.find.indexed, p: __ptContent.find.pending})'));
    indexed = st.i;
    if (st.p === 0 && st.i > 0) break;
    await sleep(1000);
  }
  console.log(`indexed ${indexed} images with text`);

  let allOk = indexed > 0;
  for (const [query, expectFixture] of CASES) {
    const n = await evalCs(`__ptContent.find.search(${JSON.stringify(query)})`);
    const hits = JSON.parse(await evalCs('JSON.stringify(__ptContent.find.matches)'));
    const boxes = await evalIn(cdp, session,
      'document.querySelectorAll("[data-pagetext-hit]").length');
    const ok = n > 0 && boxes > 0;
    if (!ok) allOk = false;
    console.log(`${ok ? 'ok  ' : 'FAIL'} "${query}" → ${n} match(es), ${boxes} highlight(s) ` +
      `[${expectFixture}] ${JSON.stringify(hits.slice(0, 2))}`);
  }

  // a query that matches nothing must clear the highlights
  await evalCs('__ptContent.find.search("zzzznotpresent")');
  const stale = await evalIn(cdp, session,
    'document.querySelectorAll("[data-pagetext-hit]").length');
  if (stale !== 0) { allOk = false; console.log(`FAIL stale highlights: ${stale}`); }
  else console.log('ok   empty query clears highlights');

  await evalCs('__ptContent.find.search("power outlets")');
  await sleep(300);
  const shot = await cdp.send('Page.captureScreenshot', { format: 'png' }, session, 30000);
  writeFileSync(resolve(outDir, 'find-e2e.png'), Buffer.from(shot.data, 'base64'));

  console.log(allOk ? 'FIND_E2E_OK' : 'FIND_E2E_FAIL');
  await cdp.send('Browser.close').catch(() => {});
  server.close();
  process.exit(allOk ? 0 : 1);
} catch (err) {
  console.error('find-e2e failed:', err.message);
  process.exit(1);
}
