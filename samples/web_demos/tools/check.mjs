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
// OCR read (PP-OCRv5) / two
// segmentations of the bundled example photo, "a dog" and then "the grass"
// typed into the page (CLIPSeg) / one
// photo + one click (SAM 2.1, on the bundled example: the reference mask)
// must finish. The OCR read is of the page's own "Try a
// sample" image (src/ppocr/sample.png, through ?img=sample) and must return
// its six lines exactly, with the recognizer on threaded WASM. Headless
// Chromium launched without flags has no usable WebGPU, so the WASM fallback
// is what runs here; the fallback path is itself under test. --webgpu runs
// the full Chromium build in its new headless mode, where WebGPU works.
//
//   npm run build
//   npx playwright install chromium        # once
//   npm run check                          # every demo
//   npm run check -- sam2                  # one demo: moge | matcha | ppocr | clipseg | sam2
//   npm run check -- matcha --block-hf     # what a user sees when HF is unreachable
//   npm run check -- moge --webkit         # WebKit engine (npx playwright install webkit)
//   npm run check -- ppocr --webgpu        # the WebGPU path
//   npm run check -- clipseg --webgpu      # the WebGPU path
//
// Options: [moge|matcha|ppocr|clipseg|sam2|all] [--block-hf] [--webkit] [--webgpu]
//          [--prefix /some/path] [--img <url>] [--timeout <ms>]
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
const which = args.find((a) => ['moge', 'matcha', 'ppocr', 'clipseg', 'sam2', 'all'].includes(a)) ?? 'all';
const BLOCK_HF = args.includes('--block-hf');
const engine = args.includes('--webkit') ? webkit : chromium;
const WEBGPU = args.includes('--webgpu') && engine === chromium;
const PREFIX = opt('--prefix', '/litert-samples/samples/web_demos/dist');
const TIMEOUT = Number(opt('--timeout', 420_000));
// Any photo with a clear subject; the demo fetches it from inside the page.
const IMG = opt('--img', 'https://images.pexels.com/photos/1170986/pexels-photo-1170986.jpeg?w=640');
const DIST = resolve(fileURLToPath(new URL('../dist/', import.meta.url)));
const PORT = 8931;
// PP-OCRv5 reads its sample (the same PNG as the "Try a sample" button)
// unless --img names another image; the sample must come back as exactly
// these lines.
const OCR_IMG = args.includes('--img') ? IMG : 'sample';
const OCR_LINES = [
  'Corner Cafe', 'Espresso 3.20', 'Cappuccino 4.10', 'Green tea 3.50',
  'Open 8:00-18:00', 'Free Wi-Fi inside',
];
// CLIPSeg runs on its bundled example unless --img names another photo, with
// "a dog" and then "the grass", which must reuse the photo's image features.
// These are the prompts' CLIP token ids and, on the example, the share of the
// 352x352 output their masks cover (Python LiteRT reference: 21444 and 62001
// of 123904 pixels). On another photo, "a dog" must mask something.
const CLIPSEG_IMG = args.includes('--img') ? IMG : 'example';
const CLIPSEG_RUNS = [
  { prompt: 'a dog', ids: [49406, 320, 1929, 49407], mask: [0.15, 0.25] },
  { prompt: 'the grass', ids: [49406, 518, 5922, 49407], mask: [0.45, 0.55] },
];
// SAM 2.1 runs on its bundled example unless --img names another photo, and
// clicks this point (fractions). On the example the mask must be the one a
// WASM run of the same files gives (WebGPU fp16 is within 0.1 % of it): the
// same candidate, its predicted IoU above 0.95, its pixel count within 2 %.
// Any photo: a mask that is neither empty nor everything, around the click.
const SAM2_IMG = args.includes('--img') ? IMG : 'example';
const SAM2_POINT = '0.6,0.5';
const SAM2_EXAMPLE = { best: 0, iou: 0.95, pixels: 17_657 };

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript',
  '.wasm': 'application/wasm',
  '.json': 'application/json',
  '.png': 'image/png',
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
const browser = await engine.launch({ args: launchArgs, ...(WEBGPU ? { channel: 'chromium' } : {}) });

async function run(demo) {
  const context = await browser.newContext(); // fresh profile: no SW, no cache
  const page = await context.newPage();
  const consoleTail = [];
  let stats = null;
  const clipsegRuns = [];
  page.on('console', (m) => {
    const t = m.text();
    consoleTail.push(t);
    if (t.startsWith('MATCHA_STATS ')) stats = JSON.parse(t.slice(13));
    if (t.startsWith('OCR_STATS ')) stats = JSON.parse(t.slice(10));
    if (t.startsWith('CLIPSEG_STATS ')) clipsegRuns.push(stats = JSON.parse(t.slice(14)));
    if (t.startsWith('SAM2_STATS ')) stats = JSON.parse(t.slice(11));
  });
  page.on('pageerror', (e) => consoleTail.push('PAGEERROR ' + e.message));
  let navs = 0;
  page.on('framenavigated', (f) => { if (f === page.mainFrame()) navs++; });

  const url = {
    matcha: `http://localhost:${PORT}${PREFIX}/matcha-tts/?nosound=1&text=${encodeURIComponent('The quick brown fox jumps over the lazy dog.')}`,
    moge: `http://localhost:${PORT}${PREFIX}/moge/?img=${encodeURIComponent(IMG)}`,
    ppocr: `http://localhost:${PORT}${PREFIX}/ppocr/?img=${encodeURIComponent(OCR_IMG)}`,
    clipseg: `http://localhost:${PORT}${PREFIX}/clipseg/?img=${encodeURIComponent(CLIPSEG_IMG)}&prompt=${encodeURIComponent(CLIPSEG_RUNS[0].prompt)}`,
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
  if (ok && demo === 'clipseg') {
    // The second prompt, typed and entered as a viewer would.
    await page.locator('#prompt').fill(CLIPSEG_RUNS[1].prompt);
    await page.locator('#prompt').press('Enter');
    ok = false;
    while (Date.now() - t0 < TIMEOUT) {
      await new Promise((r) => setTimeout(r, 500));
      const s = await page.locator('#status').textContent().catch(() => '(no #status)');
      if (s !== last) {
        last = s;
        console.log(`  [${((Date.now() - t0) / 1000).toFixed(0).padStart(3)}s] ${s}`);
      }
      if (s.startsWith('Failed')) { failed = true; break; }
      ok = clipsegRuns.length >= 2;
      if (ok) break;
    }
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
  } else if (demo === 'ppocr') {
    if (stats) {
      const t = stats.timings;
      console.log(`  env: ${await page.locator('#env').textContent()}`);
      console.log(`  backends: ${JSON.stringify(stats.backends)} threads: ${stats.wasmThreads}`);
      console.log(`  detector ${t.det} ms · recognizer ${t.rec} ms for ${stats.windows} windows · total ${t.total} ms`);
      for (const line of stats.lines) console.log(`   | ${line}`);
      const expected = OCR_IMG === IMG ? null : OCR_LINES;
      const same = !expected || JSON.stringify(stats.lines) === JSON.stringify(expected);
      if (!same) console.log(`  expected: ${JSON.stringify(expected)}`);
      if (!stats.wasmThreads) console.log('  the recognizer ran on single-thread WASM');
      if (WEBGPU && stats.backends.det !== 'webgpu') console.log(`  the detector ran on ${stats.backends.det}, not WebGPU`);
      ok = ok && stats.lines.length > 0 && same && stats.wasmThreads &&
        (!WEBGPU || stats.backends.det === 'webgpu');
    }
  } else if (demo === 'clipseg') {
    if (clipsegRuns.length) {
      console.log(`  env: ${await page.locator('#env').textContent()}`);
      CLIPSEG_RUNS.forEach((want, i) => {
        const r = clipsegRuns[i];
        if (!r) return;
        console.log(`  "${r.prompt}": ids ${JSON.stringify(r.ids)}, mask ${r.maskPixels} of ${352 * 352} pixels · ` +
          `image encoder ${r.visionCached ? 'cached' : `${r.visionMs} ms`} · text encoder ${r.textMs} ms · decoder ${r.decoderMs} ms`);
        const idsOk = JSON.stringify(r.ids) === JSON.stringify(want.ids);
        const [lo, hi] = want.mask;
        const maskOk = CLIPSEG_IMG === 'example'
          ? r.maskFraction >= lo && r.maskFraction <= hi
          : i > 0 || r.maskFraction > 0;
        const featuresOk = i === 0 || r.visionCached === true;
        const gpuOk = !WEBGPU || ['vision', 'text', 'decoder'].every((key) => r.backends[key] === 'webgpu');
        if (!idsOk) console.log(`  token ids differ from ${JSON.stringify(want.ids)}`);
        if (!maskOk) console.log(`  mask covers ${r.maskFraction} of the output, not ${CLIPSEG_IMG === 'example' ? `${lo}-${hi}` : '> 0'}`);
        if (!featuresOk) console.log('  the image encoder ran again for the same photo');
        if (!r.wasmThreads) console.log('  the WASM runtime runs single-threaded');
        if (!gpuOk) console.log(`  graphs ran on ${JSON.stringify(r.backends)}, not all on WebGPU`);
        ok = ok && r.source === 'image' && r.wasmThreads === true && idsOk && maskOk && featuresOk && gpuOk;
      });
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

const demos = which === 'all' ? ['moge', 'matcha', 'ppocr', 'clipseg', 'sam2'] : [which];
const results = {};
for (const demo of demos) results[demo] = await run(demo);
await browser.close();
server.close();
console.log('\nRESULT', JSON.stringify(results));
process.exit(Object.values(results).every(Boolean) ? 0 : 1);
