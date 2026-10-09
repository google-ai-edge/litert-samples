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
 * Boot smoke test for the offscreen TTS engine, over CDP.
 * Needs a Chromium build that honors --load-extension (Chrome for Testing,
 * Chromium, Canary — branded stable/beta ignore it since M137).
 *
 *   node tools/smoke.mjs <chrome-binary> [--speak] [--profile=<dir>]
 *
 * Launches the browser with the built dist/ extension (the background worker
 * creates the offscreen engine on startup), polls the engine's __pv.status
 * debug hook until ready (first run downloads ~94 MB of models — pass
 * --profile to reuse a previous profile's cache), optionally speaks a test
 * sentence, prints the probe results as JSON, and closes the browser.
 * Exit code 0 = engine ready.
 */
import { resolve } from 'node:path';
import { Cdp, evalIn, launchChrome, sleep, waitForEndpoint, waitForEngine } from './cdp.mjs';

const chromeBin = process.argv[2];
const doSpeak = process.argv.includes('--speak');
const profile = process.argv.find((a) => a.startsWith('--profile='))?.slice(10);
if (!chromeBin) {
  console.error('usage: node tools/smoke.mjs <chrome-binary> [--speak] [--profile=<dir>]');
  process.exit(2);
}

const port = 9223;
launchChrome(chromeBin, {
  dist: resolve(import.meta.dirname, '..', 'dist'),
  port,
  profile,
});

try {
  const cdp = await Cdp.connect(await waitForEndpoint(port));
  let { status, session } = await waitForEngine(cdp);

  if (doSpeak && status?.state === 'ready') {
    console.log('speaking test sentence…');
    try {
      await evalIn(cdp, session,
        `__pv.speak('This voice is generated locally, on this machine.')`);
    } catch {
      // A WASM-only synthesis holds the offscreen thread; the call returns when the sentence is done.
    }
    const deadline = Date.now() + 60 * 1000;
    while (Date.now() < deadline) {
      try {
        status = JSON.parse(await evalIn(cdp, session, 'JSON.stringify(__pv.status)'));
      } catch {
        // With WASM-only synthesis the offscreen thread answers nothing until the sentence is done.
        await sleep(500);
        continue;
      }
      if (status.stats && !status.speaking) break;
      await sleep(500);
    }
  }

  console.log('SMOKE_RESULT ' + JSON.stringify({
    state: status?.state,
    error: status?.error,
    env: status?.env,
    flags: status?.flags,
    backends: status?.backends,
    stats: status?.stats,
  }, null, 2));
  await cdp.send('Browser.close').catch(() => {});
  process.exit(status?.state === 'ready' ? 0 : 1);
} catch (err) {
  console.error('smoke failed:', err.message);
  process.exit(1);
}
