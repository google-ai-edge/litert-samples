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
 * Boot + pipeline smoke test for the Page Text offscreen engine, over CDP.
 * Needs a Chromium build that honors --load-extension (Chrome for Testing,
 * Chromium, Canary — branded stable/beta ignore it since M137).
 *
 *   node tools-ocr/smoke.mjs <chrome-binary> [--profile=<dir>]
 *
 * Launches the browser with the built dist-ocr/ extension, waits for the
 * engine (first run downloads ~43 MB of PP-OCRv5 from HF — pass --profile to
 * reuse a previous profile's cache), then OCRs the two mock screenshots the
 * verification harness saved to out-ocr/ and the make-fixtures.mjs images
 * (served over localhost with no CORS headers).
 * Exit code 0 = engine ready and every case found its expected substrings.
 */
import { existsSync, readFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { resolve } from 'node:path';
import { Cdp, evalIn, launchChrome, waitForEndpoint, waitForEngine } from '../tools/cdp.mjs';

const chromeBin = process.argv[2];
const profile = process.argv.find((a) => a.startsWith('--profile='))?.slice(10);
if (!chromeBin) {
  console.error('usage: node tools-ocr/smoke.mjs <chrome-binary> [--profile=<dir>]');
  process.exit(2);
}

const outDir = resolve(import.meta.dirname, '..', 'out-ocr');
// Expectations are the REGRESSION FLOOR: what the pipeline reliably reads
// today. Known model limits, deliberately not asserted: the period in
// ".js" reads as a comma at body-text sizes, and half-size (~13px font)
// Latin windows drop the odd character ("PP-CRv5").
const CASES = [
  // half-size stress mocks from the verify harness
  ['post-en-light', 'verify-B-post-en-light.png',
    ['Screenshots', 'no search, no translate', 'Right-click any', 'selectable']],
  ['post-ja-dark', 'verify-B-post-ja-dark.png',
    ['スクショ', 'コピー', 'PP-OCRv5']],
  // phone-Retina geometry fixtures (make-fixtures.mjs) — the quality bar
  ['fixture-ja-dark', 'fixture-ja-dark.png',
    ['長文を画像で貼る文化', '検索しても', 'コピーも翻訳もできない', 'PP-OCRv5',
      '右クリックひとつで', 'サーバーには何も送りません']],
  ['fixture-en-dark', 'fixture-en-dark.png',
    ['Notes for Saturday', 'at least 16 GB of RAM', 'We start at 10:00',
      '8 power outlets for 20 people', 'Handouts go out on Friday']],
  ['fixture-en-light', 'fixture-en-light.png',
    ['Screenshots are where text goes to die.', 'word inside it',
      'browser extension that runs', 'PP-OCRv5', 'Right-click any',
      'image and the text inside', 'becomes selectable, and copyable']],
];
for (const [, file] of CASES) {
  if (!existsSync(resolve(outDir, file))) {
    console.error(`missing ${file} — run tools-ocr/verify.mjs first`);
    process.exit(2);
  }
}

const server = createServer((req, res) => {
  try {
    res.writeHead(200, { 'Content-Type': 'image/png' });
    res.end(readFileSync(resolve(outDir, decodeURIComponent(req.url.slice(1)))));
  } catch {
    res.writeHead(404).end();
  }
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;

const port = 9232;
launchChrome(chromeBin, {
  dist: resolve(import.meta.dirname, '..', 'dist-ocr'),
  port,
  profile,
});

try {
  const cdp = await Cdp.connect(await waitForEndpoint(port));
  const { status, session } = await waitForEngine(cdp, { hook: '__pt' });
  if (status?.state !== 'ready') {
    console.log('SMOKE_RESULT ' + JSON.stringify({ boot: status }, null, 2));
    process.exit(1);
  }

  const results = {};
  let allOk = true;
  for (const [name, file, expected] of CASES) {
    const expr = `__pt.ocrSummary(${JSON.stringify(`${base}/${file}`)})
      .then((r) => JSON.stringify(r))`;
    let summary;
    try {
      summary = JSON.parse(await evalIn(cdp, session, expr, true));
    } catch (err) {
      summary = { ok: false, error: err.message };
    }
    if (summary.ok) {
      // whitespace-insensitive: rec windows may split lines and eat spaces
      // at cut points; character fidelity is what this gate checks
      const flat = summary.texts.join('').replace(/\s+/g, '');
      summary.missing = expected.filter((s) => !flat.includes(s.replace(/\s+/g, '')));
      if (summary.missing.length) summary.ok = false;
    }
    results[name] = summary;
    if (!summary.ok) allOk = false;
    console.log(`${summary.ok ? 'ok  ' : 'FAIL'} ${name}: ${summary.lineCount ?? '-'} lines, ` +
      `stats=${JSON.stringify(summary.stats ?? null)} missing=${JSON.stringify(summary.missing ?? null)}`);
    (summary.texts ?? []).forEach((t, i) =>
      console.log(`     | [${summary.scores?.[i] ?? '-'}] ${t}`));
  }

  console.log('SMOKE_RESULT ' + JSON.stringify({
    state: status.state,
    env: status.env,
    flags: status.flags,
    cases: Object.fromEntries(Object.entries(results).map(([k, v]) => [k, {
      ok: v.ok, lineCount: v.lineCount, stats: v.stats, missing: v.missing,
    }])),
  }, null, 2));
  await cdp.send('Browser.close').catch(() => {});
  server.close();
  process.exit(allOk ? 0 : 1);
} catch (err) {
  console.error('smoke failed:', err.message);
  process.exit(1);
}
