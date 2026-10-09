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
 * End-to-end test of the chat auto-read watcher: serves a synthetic
 * streaming page (ChatGPT DOM shape), toggles the HUD's Auto-read button,
 * streams four sentences word-by-word, and verifies the engine received and
 * synthesized them (spoken counter + stats via the __pv debug hook).
 *
 *   node tools/stream-smoke.mjs <chrome-binary> [--profile=<dir>]
 *
 * The dev manifest matches http://127.0.0.1/* so the content script runs on
 * the test page. Exit code 0 = sentences flowed through the whole pipeline.
 */
import { createServer } from 'node:http';
import { readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import {
  Cdp, attachTo, evalIn, findTarget, launchChrome, sleep, waitForEndpoint, waitForEngine,
} from './cdp.mjs';

const chromeBin = process.argv[2];
const profile = process.argv.find((a) => a.startsWith('--profile='))?.slice(10);
if (!chromeBin) {
  console.error('usage: node tools/stream-smoke.mjs <chrome-binary> [--profile=<dir>]');
  process.exit(2);
}

const httpPort = 8907;
const html = readFileSync(join(import.meta.dirname, 'stream-test.html'));
const server = createServer((_req, res) => {
  res.setHeader('content-type', 'text/html');
  res.end(html);
});
server.listen(httpPort);

const port = 9223;
launchChrome(chromeBin, {
  dist: resolve(import.meta.dirname, '..', 'dist'),
  port,
  profile,
});

try {
  const cdp = await Cdp.connect(await waitForEndpoint(port));
  const { status: boot, session: engine } = await waitForEngine(cdp);
  if (boot?.state !== 'ready') throw new Error(`engine not ready: ${JSON.stringify(boot)}`);

  // open the synthetic chat page and wait for the content script's HUD;
  // evaluate can throw while the initial navigation swaps documents, so
  // retry (re-attaching if the session died)
  await cdp.send('Target.createTarget', { url: `http://127.0.0.1:${httpPort}/` });
  await sleep(1500);
  const findPage = (t) => t.type === 'page' && t.url.includes(`127.0.0.1:${httpPort}`);
  let pageSession = await attachTo(cdp, await findTarget(cdp, findPage));
  const findHud = `[...document.documentElement.children]
    .find((e) => e.shadowRoot && e.shadowRoot.getElementById('auto'))`;
  let hudUp = false;
  for (let i = 0; i < 30 && !hudUp; i++) {
    try {
      hudUp = await evalIn(cdp, pageSession, `!!(${findHud})`);
    } catch {
      try { pageSession = await attachTo(cdp, await findTarget(cdp, findPage)); } catch { /* retry */ }
    }
    if (!hudUp) await sleep(500);
  }
  if (!hudUp) throw new Error('HUD never appeared — content script not injected?');

  // enable auto-read via the HUD button (shadow DOM is open); it may already
  // be armed — the toggle persists per host in chrome.storage
  const isArmed = `(${findHud}).shadowRoot.getElementById('auto').classList.contains('on')`;
  if (!(await evalIn(cdp, pageSession, isArmed))) {
    await evalIn(cdp, pageSession, `(${findHud}).shadowRoot.getElementById('auto').click()`);
    await sleep(500);
  }
  const armed = await evalIn(cdp, pageSession, isArmed);
  if (!armed) throw new Error('Auto-read toggle did not arm');
  console.log('auto-read armed; streaming…');

  const spokenBefore = JSON.parse(await evalIn(cdp, engine, 'JSON.stringify(__pv.status)')).spoken ?? 0;
  await evalIn(cdp, pageSession, 'startStream()');

  // wait for: stream finished → quiet flush → engine synthesized and drained
  let status = null;
  const deadline = Date.now() + 120 * 1000;
  let lastLine = '';
  while (Date.now() < deadline) {
    const done = await evalIn(cdp, pageSession, 'window.streamDone');
    status = JSON.parse(await evalIn(cdp, engine, 'JSON.stringify(__pv.status)'));
    const spoken = (status.spoken ?? 0) - spokenBefore;
    const line = `streamDone=${done} spoken=${spoken} queued=${status.queued} speaking=${status.speaking}`;
    if (line !== lastLine) console.log(line);
    lastLine = line;
    if (done && spoken >= 2 && !status.speaking && status.queued === 0) break;
    await sleep(1000);
  }

  const spoken = (status?.spoken ?? 0) - spokenBefore;
  const pass = spoken >= 2 && !!status?.stats && !status?.error;
  console.log('STREAM_RESULT ' + JSON.stringify({
    pass,
    spokenUtterances: spoken,
    stats: status?.stats,
    error: status?.error,
  }, null, 2));
  await cdp.send('Browser.close').catch(() => {});
  process.exit(pass ? 0 : 1);
} catch (err) {
  console.error('stream smoke failed:', err.message);
  process.exit(1);
} finally {
  server.close();
}
