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
 * Boot + CORS + depth smoke test for the Page 3D offscreen engine, over CDP.
 * Needs a Chromium build that honors --load-extension (Chrome for Testing,
 * Chromium, Canary — branded stable/beta ignore it since M137).
 *
 *   node tools3d/smoke.mjs <chrome-binary> [--profile=<dir>]
 *
 * Launches the browser with the built dist3d/ extension (the background
 * worker creates the offscreen engine on startup), waits for the engine
 * (first run downloads the 136 MB MoGe-2 model — pass --profile to reuse a
 * previous profile's cache), then runs the depth pipeline against real image
 * CDNs (the CORS check) plus a forced service-worker-relay fetch, and
 * saves the last depth/photo data URLs to out3d/ for visual inspection.
 * Exit code 0 = engine ready and every case produced a depth map.
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { Cdp, evalIn, launchChrome, waitForEndpoint, waitForEngine } from '../tools/cdp.mjs';

const chromeBin = process.argv[2];
const profile = process.argv.find((a) => a.startsWith('--profile='))?.slice(10);
if (!chromeBin) {
  console.error('usage: node tools3d/smoke.mjs <chrome-binary> [--profile=<dir>]');
  process.exit(2);
}

const CASES = [
  ['x-cdn', 'https://pbs.twimg.com/profile_images/1683899100922511378/5lY42eHs_400x400.jpg'],
  ['wikipedia', 'https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg'],
  ['pexels-fox', 'https://images.pexels.com/photos/6724234/pexels-photo-6724234.jpeg?auto=compress&cs=tinysrgb&w=1200'],
  ['github-avatar', 'https://avatars.githubusercontent.com/u/9919?s=460&v=4'],
  // Same image, but forcing the service-worker relay: proves the fallback
  // path for hosts that send neither CORS nor CORP headers.
  ['sw-relay-forced', 'https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg', { forceRelay: true }],
];

const outDir = resolve(import.meta.dirname, '..', 'out3d');
mkdirSync(outDir, { recursive: true });

function saveDataUrl(dataUrl, file) {
  const b64 = dataUrl.slice(dataUrl.indexOf(',') + 1);
  writeFileSync(file, Buffer.from(b64, 'base64'));
}

const port = 9224;
launchChrome(chromeBin, {
  dist: resolve(import.meta.dirname, '..', 'dist3d'),
  port,
  profile,
});

try {
  const cdp = await Cdp.connect(await waitForEndpoint(port));
  const { status, session } = await waitForEngine(cdp, { hook: '__p3' });
  if (status?.state !== 'ready') {
    console.log('SMOKE_RESULT ' + JSON.stringify({ boot: status }, null, 2));
    process.exit(1);
  }

  const results = {};
  let allOk = true;
  for (const [name, url, opts] of CASES) {
    const expr = `__p3.depthSummary(${JSON.stringify(url)}, ${JSON.stringify(opts ?? {})})
      .then((r) => JSON.stringify(r))`;
    let summary;
    try {
      summary = JSON.parse(await evalIn(cdp, session, expr, true));
    } catch (err) {
      summary = { ok: false, error: err.message };
    }
    results[name] = summary;
    if (!summary.ok) allOk = false;
    console.log(`${summary.ok ? 'ok  ' : 'FAIL'} ${name}: ${JSON.stringify(summary)}`);
    if (summary.ok) {
      const depthUrl = await evalIn(cdp, session, '__p3.lastResult.depth.url');
      const photoUrl = await evalIn(cdp, session, '__p3.lastResult.photo.url');
      const normalUrl = await evalIn(cdp, session, '__p3.lastResult.normal?.url ?? null');
      saveDataUrl(depthUrl, resolve(outDir, `smoke-${name}-depth.png`));
      saveDataUrl(photoUrl, resolve(outDir, `smoke-${name}-photo.jpg`));
      if (normalUrl) saveDataUrl(normalUrl, resolve(outDir, `smoke-${name}-normal.png`));
    }
  }

  console.log('SMOKE_RESULT ' + JSON.stringify({
    state: status.state,
    env: status.env,
    flags: status.flags,
    cases: results,
  }, null, 2));
  await cdp.send('Browser.close').catch(() => {});
  process.exit(allOk ? 0 : 1);
} catch (err) {
  console.error('smoke failed:', err.message);
  process.exit(1);
}
