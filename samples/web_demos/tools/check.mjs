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
// Hugging Face, and one inference (MoGe) / one synthesis (Matcha) must
// finish. Headless browsers have no usable WebGPU, so the WASM fallback is
// what runs here; the fallback path is itself under test.
//
//   npm run build
//   npx playwright install chromium        # once
//   npm run check                          # both demos
//   npm run check -- matcha --block-hf     # what a user sees when HF is unreachable
//   npm run check -- moge --webkit         # WebKit engine (npx playwright install webkit)
//
// Options: [moge|matcha|all] [--block-hf] [--webkit] [--prefix /some/path]
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
const which = args.find((a) => ['moge', 'matcha', 'all'].includes(a)) ?? 'all';
const BLOCK_HF = args.includes('--block-hf');
const engine = args.includes('--webkit') ? webkit : chromium;
const PREFIX = opt('--prefix', '/litert-samples/samples/web_demos/dist');
const TIMEOUT = Number(opt('--timeout', 420_000));
// Any photo with a clear subject; the demo fetches it from inside the page.
const IMG = opt('--img', 'https://images.pexels.com/photos/1170986/pexels-photo-1170986.jpeg?w=640');
const DIST = resolve(fileURLToPath(new URL('../dist/', import.meta.url)));
const PORT = 8931;

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript',
  '.wasm': 'application/wasm',
  '.json': 'application/json',
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
  });
  page.on('pageerror', (e) => consoleTail.push('PAGEERROR ' + e.message));
  let navs = 0;
  page.on('framenavigated', (f) => { if (f === page.mainFrame()) navs++; });

  const url = demo === 'matcha'
    ? `http://localhost:${PORT}${PREFIX}/matcha-tts/?nosound=1&text=${encodeURIComponent('The quick brown fox jumps over the lazy dog.')}`
    : `http://localhost:${PORT}${PREFIX}/moge/?img=${encodeURIComponent(IMG)}`;
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
    if (s.startsWith('Failed')) { failed = true; break; }
    ok = demo === 'matcha' ? !!stats : await page.locator('#latency').isVisible().catch(() => false);
    if (ok) break;
  }

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
  } else if (ok) {
    console.log(`  env: ${await page.locator('#env').textContent()}`);
    console.log(`  latency: ${(await page.locator('#latency').innerText()).replace(/\n/g, ' | ')}`);
  }
  if (!ok) {
    console.log(failed ? '  FAILED' : '  TIMED OUT', '— console tail:');
    for (const line of consoleTail.slice(-8)) console.log('   |', line.slice(0, 240));
  }
  await context.close();
  return ok && state.crossOriginIsolated;
}

const demos = which === 'all' ? ['moge', 'matcha'] : [which];
const results = {};
for (const demo of demos) results[demo] = await run(demo);
await browser.close();
server.close();
console.log('\nRESULT', JSON.stringify(results));
process.exit(Object.values(results).every(Boolean) ? 0 : 1);
