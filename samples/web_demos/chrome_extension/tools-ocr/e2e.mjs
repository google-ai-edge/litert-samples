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
 * Overlay E2E over CDP: serves a page embedding the intro fixture, triggers the
 * real activation path (service worker → content script → engine → overlay),
 * then checks the selectable-text layer in the DOM and saves screenshots
 * (overlay boxes flashing, then faded) for visual inspection.
 *
 *   node tools-ocr/e2e.mjs <chrome-binary> [--profile=<dir>]
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, resolve } from 'node:path';
import { Cdp, attachTo, evalIn, findTarget, launchChrome, sleep, waitForEndpoint, waitForEngine } from '../tools/cdp.mjs';

const chromeBin = process.argv[2];
const profile = process.argv.find((a) => a.startsWith('--profile='))?.slice(10);
if (!chromeBin) {
  console.error('usage: node tools-ocr/e2e.mjs <chrome-binary> [--profile=<dir>]');
  process.exit(2);
}
const outDir = resolve(import.meta.dirname, '..', 'out-ocr');

const PAGE = `<!doctype html><meta charset="utf-8"><title>pagetext e2e</title>
<body style="margin:40px;background:#f0f2f5;font-family:sans-serif">
<h3>Page Text e2e</h3>
<img id="post" src="fixture-en-intro.png" style="width:560px;display:block;box-shadow:0 4px 24px rgba(0,0,0,.2);border-radius:12px">
</body>`;

const MIME = { '.png': 'image/png', '.html': 'text/html' };
const server = createServer((req, res) => {
  try {
    if (req.url === '/' || req.url === '/page.html') {
      res.writeHead(200, { 'Content-Type': 'text/html' }).end(PAGE);
      return;
    }
    const p = resolve(outDir, decodeURIComponent(req.url.slice(1)));
    const body = readFileSync(p);
    res.writeHead(200, { 'Content-Type': MIME[extname(p)] ?? 'application/octet-stream' });
    res.end(body);
  } catch {
    res.writeHead(404).end();
  }
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;

const port = 9235;
launchChrome(chromeBin, {
  dist: resolve(import.meta.dirname, '..', 'dist-ocr'),
  port,
  profile,
  url: `${base}/page.html`,
});

try {
  const cdp = await Cdp.connect(await waitForEndpoint(port));
  const { status } = await waitForEngine(cdp, { hook: '__pt' });
  if (status?.state !== 'ready') throw new Error(`engine: ${JSON.stringify(status)}`);

  const page = await findTarget(cdp, (t) => t.type === 'page' && t.url.startsWith(base));
  const pageSession = await attachTo(cdp, page);
  await cdp.send('Page.enable', {}, pageSession);
  // The page opened with the browser can load before the extension has
  // registered its content script, and then never gets one — reload it.
  await cdp.send('Page.reload', {}, pageSession);

  // real activation path: service worker sends the context-menu message.
  // Find our worker by our extension id (taken from the offscreen document):
  // Chrome's own component extensions run service workers in the profile too.
  const offscreen = await findTarget(cdp, (t) => t.url.includes('/offscreen.html'));
  const extensionId = new URL(offscreen.url).host;
  const sw = await findTarget(cdp, (t) =>
    t.type === 'service_worker' && t.url === `chrome-extension://${extensionId}/background.js`);
  const swSession = await attachTo(cdp, sw);
  // The content script attaches at document_idle; until then sendMessage
  // rejects with "Receiving end does not exist", so retry for a few seconds.
  const sent = await evalIn(cdp, swSession, `
    chrome.tabs.query({ url: ${JSON.stringify(`${base}/*`)} }).then(async (tabs) => {
      if (!tabs.length) return 'no tab';
      for (let i = 0; ; i++) {
        try {
          await chrome.tabs.sendMessage(tabs[0].id, {
            type: 'activate-ocr',
            srcUrl: ${JSON.stringify(`${base}/fixture-en-intro.png`)},
          });
          return 'sent to ' + tabs[0].id + (i ? ' after ' + i + ' retries' : '');
        } catch (err) {
          if (i >= 40) throw err;
          await new Promise((r) => setTimeout(r, 250));
        }
      }
    })`, true);
  console.log('activation:', sent);

  // wait for the overlay, snapshot while boxes still flash, then after fade
  let overlay = null;
  const deadline = Date.now() + 60000;
  while (Date.now() < deadline) {
    overlay = await evalIn(cdp, pageSession, `(() => {
      const host = document.querySelector('[data-pagetext]');
      if (!host) return null;
      const spans = [...host.querySelectorAll('span')].filter((s) => s.textContent);
      return {
        spans: spans.length,
        text: spans.map((s) => s.textContent).join('\\n'),
        boxes: host.querySelectorAll('.pt-box').length,
      };
    })()`);
    if (overlay) break;
    await sleep(400);
  }
  if (!overlay) throw new Error('overlay never appeared');
  console.log(`overlay: ${overlay.spans} spans, ${overlay.boxes} boxes`);
  console.log(overlay.text.split('\n').map((l) => `  | ${l}`).join('\n'));

  const shot1 = await cdp.send('Page.captureScreenshot', { format: 'png' }, pageSession, 30000);
  writeFileSync(resolve(outDir, 'e2e-overlay-flash.png'), Buffer.from(shot1.data, 'base64'));
  await sleep(2000);
  const shot2 = await cdp.send('Page.captureScreenshot', { format: 'png' }, pageSession, 30000);
  writeFileSync(resolve(outDir, 'e2e-overlay-faded.png'), Buffer.from(shot2.data, 'base64'));

  // programmatic selection over the overlay (what a drag-select produces)
  const selected = await evalIn(cdp, pageSession, `(() => {
    const host = document.querySelector('[data-pagetext]');
    const range = document.createRange();
    range.selectNodeContents(host);
    const sel = getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
    return sel.toString();
  })()`);
  console.log('selection sample:', JSON.stringify(selected.slice(0, 120)));

  // Match the selected text as is: a phrase across two rec windows or two
  // lines only matches if the layer puts Copy all's separators between them.
  const mustHave = ['screenshot,\nnobody can search it', 'search it or copy it later',
    'selectable text', 'Nothing is sent to a server'];
  const missing = mustHave.filter((s) => !selected.includes(s));
  console.log(missing.length ? `MISSING: ${missing.map((s) => JSON.stringify(s)).join(', ')}` : 'E2E_OK');
  await cdp.send('Browser.close').catch(() => {});
  server.close();
  process.exit(missing.length ? 1 : 0);
} catch (err) {
  console.error('e2e failed:', err);
  process.exit(1);
}
