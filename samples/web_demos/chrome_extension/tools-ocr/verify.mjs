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
 * Recognizer backend comparison: decodes the same text crops with the
 * PP-OCRv5 recognizer on WebGPU and on WASM, with fp16 and with fp32
 * weights, and prints how often the decoded strings agree. A tool, not a
 * test: it exits 0 whatever the agreement, 1 only if the harness fails.
 *
 *   node tools-ocr/verify.mjs <chrome-binary> [--profile=<dir>]
 *
 * Builds the harness page with esbuild, serves it over localhost with
 * COOP/COEP (with the models, downloaded once to out-ocr/models/, and the
 * fixture-half-*.png images from make-fixtures.mjs), drives a Chromium that
 * has WebGPU, and saves the crops and verify-report.json to out-ocr/.
 */
import { build } from 'esbuild';
import { createServer } from 'node:http';
import { mkdirSync, existsSync, writeFileSync, readFileSync, cpSync } from 'node:fs';
import { join, resolve, extname } from 'node:path';
import { spawn } from 'node:child_process';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { Cdp, attachTo, evalIn, findTarget, sleep, waitForEndpoint } from '../tools/cdp.mjs';

const chromeBin = process.argv[2];
if (!chromeBin) {
  console.error('usage: node tools-ocr/verify.mjs <chrome-binary> [--profile=<dir>]');
  process.exit(2);
}
const profile = process.argv.find((a) => a.startsWith('--profile='))?.slice(10)
  ?? mkdtempSync(join(tmpdir(), 'ocr-verify-'));

const HF = 'https://huggingface.co/litert-community/PP-OCRv5-LiteRT/resolve/main';
const MODELS = ['ppocr_det_fp16.tflite', 'ppocr_rec_fp16.tflite', 'ppocrv5_dict.txt',
  'ppocr_rec_fp32.tflite'];
const FIXTURES = ['fixture-half-en-light.png', 'fixture-half-ja-dark.png'];
const root = resolve(import.meta.dirname, '..');
const outDir = join(root, 'out-ocr');
for (const f of FIXTURES) {
  if (!existsSync(join(outDir, f))) {
    console.error(`missing ${f} — run tools-ocr/make-fixtures.mjs first`);
    process.exit(2);
  }
}
const cacheDir = join(outDir, 'models');
mkdirSync(cacheDir, { recursive: true });
for (const f of MODELS) {
  const p = join(cacheDir, f);
  if (existsSync(p)) continue;
  console.log(`downloading ${f} ...`);
  const res = await fetch(`${HF}/${f}`);
  if (!res.ok) throw new Error(`${f}: HTTP ${res.status}`);
  writeFileSync(p, Buffer.from(await res.arrayBuffer()));
}

// --- build the page -----------------------------------------------------------
const dist = join(outDir, 'verify-dist');
mkdirSync(dist, { recursive: true });
await build({
  entryPoints: { page: join(root, 'tools-ocr', 'verify-page.js') },
  bundle: true,
  format: 'iife',
  target: 'chrome128',
  outdir: dist,
  logLevel: 'silent',
});
writeFileSync(join(dist, 'index.html'),
  '<!doctype html><meta charset="utf-8"><title>ocr verify</title><script src="page.js"></script>');
cpSync(join(root, 'node_modules', '@litertjs', 'core', 'wasm'), join(dist, 'litert-wasm'),
  { recursive: true });
mkdirSync(join(dist, 'models'), { recursive: true });
for (const f of MODELS) cpSync(join(cacheDir, f), join(dist, 'models', f));
mkdirSync(join(dist, 'fixtures'), { recursive: true });
for (const f of FIXTURES) cpSync(join(outDir, f), join(dist, 'fixtures', f));

// --- serve with COI headers -----------------------------------------------------
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.wasm': 'application/wasm',
  '.tflite': 'application/octet-stream', '.txt': 'text/plain; charset=utf-8', '.png': 'image/png' };
const server = createServer((req, res) => {
  const path = join(dist, req.url === '/' ? 'index.html' : decodeURIComponent(req.url));
  try {
    const body = readFileSync(path);
    res.writeHead(200, {
      'Content-Type': MIME[extname(path)] ?? 'application/octet-stream',
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    });
    res.end(body);
  } catch {
    res.writeHead(404).end('nope');
  }
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const pageUrl = `http://127.0.0.1:${server.address().port}/`;
console.log(`serving ${pageUrl} profile=${profile}`);

// --- launch + drive --------------------------------------------------------------
const port = 9231;
const child = spawn(chromeBin, [
  `--user-data-dir=${profile}`,
  `--remote-debugging-port=${port}`,
  '--no-first-run', '--no-default-browser-check', '--disable-sync',
  '--use-mock-keychain', // see launchChrome in tools/cdp.mjs
  '--disable-backgrounding-occluded-windows', '--disable-renderer-backgrounding',
  '--disable-background-timer-throttling', '--hide-crash-restore-bubble',
  pageUrl,
], { stdio: 'ignore' });
process.on('exit', () => { try { child.kill(); } catch { /* gone */ } });

function saveDataUrl(dataUrl, file) {
  writeFileSync(file, Buffer.from(dataUrl.slice(dataUrl.indexOf(',') + 1), 'base64'));
}

function printTable(rows) {
  const widths = rows[0].map((_, c) => Math.max(...rows.map((row) => String(row[c]).length)));
  for (const row of rows) {
    console.log(row.map((v, c) => String(v).padEnd(widths[c])).join('  ').trimEnd());
  }
}

try {
  const cdp = await Cdp.connect(await waitForEndpoint(port));
  const target = await findTarget(cdp, (t) => t.type === 'page' && t.url.startsWith(pageUrl));
  const session = await attachTo(cdp, target);

  let last = '';
  const deadline = Date.now() + 5 * 60 * 1000;
  for (;;) {
    if (Date.now() > deadline) throw new Error('verification timed out');
    let s = null;
    try {
      s = await evalIn(cdp, session,
        `typeof __ocrv === 'undefined' ? null : JSON.stringify(__ocrv.status)`);
    } catch { /* not ready */ }
    if (s) {
      const st = JSON.parse(s);
      const line = `${st.state} ${st.progress ?? ''}`;
      if (line !== last) console.log(line);
      last = line;
      if (st.state === 'done' || st.state === 'error') break;
    }
    await sleep(500);
  }

  const raw = await evalIn(cdp, session, 'JSON.stringify(__ocrv.result)');
  const r = JSON.parse(raw);
  if (!r?.ok) {
    console.error('HARNESS ERROR:\n' + (r?.error ?? 'no result'));
    process.exit(1);
  }

  // save crops for eyeballing + full report
  for (const row of r.stageA) {
    saveDataUrl(row.cropUrl, join(outDir, `verify-A-${String(row.idx).padStart(2, '0')}-${row.label}.png`));
    delete row.cropUrl;
  }
  for (const post of r.stageB) {
    post.lines.forEach((l, i) => {
      saveDataUrl(l.cropUrl, join(outDir, `verify-B-${post.name}-line${String(i).padStart(2, '0')}.png`));
      delete l.cropUrl;
    });
  }
  writeFileSync(join(outDir, 'verify-report.json'), JSON.stringify(r, null, 2));

  const a = r.summary.stageA;
  const b = r.summary.stageB;
  console.log(`\nStage A: ${a.total} rendered lines, the same crop on both backends`);
  printTable([
    ['weights', 'WebGPU = WASM', 'WebGPU = truth', 'WASM = truth', 'WebGPU ms/line', 'WASM ms/line'],
    ['fp16', `${a.equal}/${a.total}`, `${a.gpuExactTruth}/${a.total}`, `${a.cpuExactTruth}/${a.total}`,
      a.gpuMsP50, a.cpuMsP50],
    ['fp32', `${a.equal32}/${a.total}`, `${a.gpu32ExactTruth}/${a.total}`, `${a.cpu32ExactTruth}/${a.total}`,
      a.gpu32MsP50, a.cpu32MsP50],
  ]);
  console.log(`\nStage B: det → rec on ${r.stageB.map((p) => p.name).join(', ')}; ` +
    `WebGPU = WASM on ${b.equal}/${b.lines} lines (fp16), ${b.equal32}/${b.lines} (fp32)`);
  const diffs = [
    ...r.stageA.filter((x) => !x.equal || !x.equal32)
      .map((x) => [`A ${x.label}`, x.gpuText, x.cpuText, x.gpu32Text, x.cpu32Text]),
    ...r.stageB.flatMap((p) => p.lines.filter((l) => !l.equal || !l.equal32)
      .map((l) => [`B ${p.name}`, l.gpuText, l.cpuText, l.gpu32Text, l.cpu32Text])),
  ];
  if (diffs.length) {
    console.log('\nCrops that decode differently (per-timestep detail in out-ocr/verify-report.json)');
    printTable([['crop', 'fp16 WebGPU', 'fp16 WASM', 'fp32 WebGPU', 'fp32 WASM'],
      ...diffs.map(([label, ...texts]) => [label, ...texts.map((t) => JSON.stringify(t))])]);
  }
  console.log(`\nSUMMARY agreement fp16 ${a.equal}/${a.total}, fp32 ${a.equal32}/${a.total}`);
  server.close();
  await cdp.send('Browser.close').catch(() => {});
  process.exit(0);
} catch (err) {
  console.error('verify failed:', err);
  process.exit(1);
}
