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

// Deploy-shaped end-to-end check of the built site (dist/).
//
// Serves dist/ under a sub-path with NO COOP/COEP headers — exactly what
// GitHub Pages does — opens each demo in a headless browser, and waits for a
// full run: coi-serviceworker must turn on cross-origin isolation, the
// threaded WASM runtime must load, the model files must come through
// Hugging Face, and one inference (MoGe) / one synthesis (Matcha) / one
// photo + one click (SAM 2.1, on the bundled example: the reference mask)
// must finish. Headless browsers without flags have no usable WebGPU, so the
// WASM fallback is what runs here; the fallback path is itself under test.
//
//   npm run build
//   npx playwright install chromium        # once
//   npm run check                          # every demo
//   npm run check -- sam2                  # one demo: moge | matcha | sam2
//   npm run check -- matcha --block-hf     # what a user sees when HF is unreachable
//   npm run check -- moge --webkit         # WebKit engine (npx playwright install webkit)
//
// Options: [moge|matcha|sam2|all] [--block-hf] [--webkit] [--prefix /some/path]
//          [--img <url>] [--timeout <ms>]
import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium, webkit } from 'playwright';

const args = process.argv.slice(2);
const opt = (name, def) => {
  const i = args.indexOf(name);
  return i === -1 ? def : args[i + 1];
};
const which = args.find((a) => ['moge', 'matcha', 'sam2', 'all'].includes(a)) ?? 'all';
const BLOCK_HF = args.includes('--block-hf');
const engine = args.includes('--webkit') ? webkit : chromium;
const PREFIX = opt('--prefix', '/litert-samples/samples/web_demos/dist');
const TIMEOUT = Number(opt('--timeout', 420_000));
// Any photo with a clear subject; the demo fetches it from inside the page.
const IMG = opt('--img', 'https://images.pexels.com/photos/1170986/pexels-photo-1170986.jpeg?w=640');
// SAM 2.1 runs on its bundled example unless --img names another photo, and
// clicks this point (fractions). On the example the mask must be the one a
// WASM run of the same files gives (WebGPU fp16 is within 0.1 % of it): the
// same candidate, its predicted IoU above 0.95, its pixel count within 2 %.
// Any photo: a mask that is neither empty nor everything, around the click.
const SAM2_IMG = args.includes('--img') ? IMG : 'example';
const SAM2_POINT = '0.6,0.5';
const SAM2_EXAMPLE = { best: 0, iou: 0.95, pixels: 17_657 };
const DIST = resolve(fileURLToPath(new URL('../dist/', import.meta.url)));
const PORT = 8931;

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript',
  '.wasm': 'application/wasm',
  '.json': 'application/json',
  '.jpg': 'image/jpeg',
};

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://localhost');
    if (!url.pathname.startsWith(PREFIX + '/')) throw new Error('outside prefix');
    const rel = normalize(url.pathname.slice(PREFIX.length + 1));
    if (rel.includes('..')) throw new Error('bad path');
    let file = join(DIST, rel);
    try {
      if ((await stat(file)).isDirectory()) file = join(file, 'index.html');
    } catch { /* fall through to readFile */ }
    const body = await readFile(file);
    // deliberately NO COOP/COEP — that is GitHub Pages
    res.setHeader('Content-Type', MIME[extname(file)] ?? 'application/octet-stream');
    res.setHeader('Content-Length', body.length);
    res.end(body);
  } catch {
    res.statusCode = 404;
    res.end('not found');
  }
});
await new Promise((r) => server.listen(PORT, r));

const launchArgs = BLOCK_HF && engine === chromium
  ? ['--host-resolver-rules=MAP huggingface.co ~NOTFOUND, MAP *.huggingface.co ~NOTFOUND, MAP *.hf.co ~NOTFOUND']
  : [];
if (BLOCK_HF && engine !== chromium) {
  console.log('--block-hf is Chromium-only (host-resolver-rules); ignoring');
}
const browser = await engine.launch({ args: launchArgs });

async function run(demo) {
  const context = await browser.newContext(); // fresh profile: no SW, no cache
  const page = await context.newPage();
  const consoleTail = [];
  let stats = null;
  page.on('console', (m) => {
    const t = m.text();
    consoleTail.push(t);
    if (t.startsWith('MATCHA_STATS ')) stats = JSON.parse(t.slice(13));
    if (t.startsWith('SAM2_STATS ')) stats = JSON.parse(t.slice(11));
  });
  page.on('pageerror', (e) => consoleTail.push('PAGEERROR ' + e.message));
  let navs = 0;
  page.on('framenavigated', (f) => { if (f === page.mainFrame()) navs++; });

  const url = {
    matcha: `http://localhost:${PORT}${PREFIX}/matcha-tts/?nosound=1&text=${encodeURIComponent('The quick brown fox jumps over the lazy dog.')}`,
    moge: `http://localhost:${PORT}${PREFIX}/moge/?img=${encodeURIComponent(IMG)}`,
    sam2: `http://localhost:${PORT}${PREFIX}/sam2/?img=${encodeURIComponent(SAM2_IMG)}&point=${SAM2_POINT}`,
  }[demo];
  console.log(`\n== ${demo}: ${url}`);
  const t0 = Date.now();
  await page.goto(url);

  let last = '';
  let ok = false;
  let failed = false;
  while (Date.now() - t0 < TIMEOUT) {
    await new Promise((r) => setTimeout(r, 1500));
    const s = await page.locator('#status').textContent().catch(() => '(no #status)');
    if (s !== last) {
      last = s;
      console.log(`  [${((Date.now() - t0) / 1000).toFixed(0).padStart(3)}s] ${s}`);
    }
    if (['Failed', 'Could not read', 'Example:'].some((p) => s.startsWith(p))) { failed = true; break; }
    ok = demo === 'moge' ? await page.locator('#latency').isVisible().catch(() => false) : !!stats;
    if (ok) break;
  }
  const finished = ok; // the run completed; the checks below can still fail it

  const state = await page.evaluate(() => ({
    crossOriginIsolated: window.crossOriginIsolated,
    serviceWorker: navigator.serviceWorker?.controller?.scriptURL ?? null,
  }));
  console.log(`  navigations: ${navs} (2 = one coi-serviceworker reload)`);
  console.log(`  isolation: ${JSON.stringify(state)}`);
  if (demo === 'matcha') {
    if (stats) {
      const t = stats.timings;
      const rtf = (t.g2p + t.textenc + t.decoder + t.vocoder) / 1000 / stats.seconds;
      console.log(`  backends: ${JSON.stringify(stats.backends)} threads: ${stats.wasmThreads}`);
      console.log(`  audio: ${stats.seconds}s rms ${stats.rms} peak ${stats.peak} nonFinite ${stats.nonFinite} · RTF ${rtf.toFixed(2)}`);
      ok = ok && stats.nonFinite === 0 && stats.rms > 0.01;
    }
  } else if (demo === 'sam2') {
    if (stats) {
      console.log(`  env: ${await page.locator('#env').textContent()}`);
      console.log(`  backends: ${JSON.stringify(stats.backends)} threads: ${stats.wasmThreads} (${stats.numThreads})`);
      console.log(`  point ${stats.point.fx},${stats.point.fy} · encoder ${stats.encoderMs} ms · decoder ${stats.decoderMs} ms · iou ${JSON.stringify(stats.iouScores)} · best ${stats.best} · mask ${stats.maskPixels} px of 65536 · hit ${stats.hit}`);
      const ref = SAM2_EXAMPLE;
      const asExpected = SAM2_IMG === 'example'
        ? stats.best === ref.best && stats.iouScores[ref.best] > ref.iou && Math.abs(stats.maskPixels - ref.pixels) <= ref.pixels * 0.02
        : stats.maskPixels > 0 && stats.maskPixels < 65536;
      if (!asExpected || !stats.hit) console.log(`  not the expected mask${SAM2_IMG === 'example' ? ` (best ${ref.best}, iou > ${ref.iou}, ${ref.pixels} px ± 2 %, around the click)` : ' (non-empty, around the click)'}`);
      ok = ok && asExpected && stats.hit;
    }
  } else if (ok) {
    console.log(`  env: ${await page.locator('#env').textContent()}`);
    console.log(`  latency: ${(await page.locator('#latency').innerText()).replace(/\n/g, ' | ')}`);
  }
  if (!ok) {
    console.log(failed ? '  FAILED' : finished ? '  CHECK FAILED' : '  TIMED OUT', '— console tail:');
    for (const line of consoleTail.slice(-8)) console.log('   |', line.slice(0, 240));
  }
  await context.close();
  return ok && state.crossOriginIsolated;
}

const demos = which === 'all' ? ['moge', 'matcha', 'sam2'] : [which];
const results = {};
for (const demo of demos) results[demo] = await run(demo);
await browser.close();
server.close();
console.log('\nRESULT', JSON.stringify(results));
process.exit(Object.values(results).every(Boolean) ? 0 : 1);
