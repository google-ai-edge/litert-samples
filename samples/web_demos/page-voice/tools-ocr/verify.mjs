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
 * Recognizer backend check: the PP-OCRv5 rec fp16 model's webgpu logits
 * differ from wasm (max_abs_diff 0.084 on random input) — does that survive
 * CTC decode on real text crops? Passes when every crop decodes to the same
 * string on both backends, with the fp16 or with the fp32 model.
 *
 *   node tools-ocr/verify.mjs <chrome-binary> [--profile=<dir>]
 *
 * Builds the harness page with esbuild, serves it (with the models,
 * downloaded once to out-ocr/models/) over localhost with COOP/COEP,
 * drives a Chromium that has WebGPU, saves crops + report to out-ocr/.
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
const root = resolve(import.meta.dirname, '..');
const cacheDir = join(root, 'out-ocr', 'models');
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
const dist = join(root, 'out-ocr', 'verify-dist');
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

// --- serve with COI headers -----------------------------------------------------
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.wasm': 'application/wasm',
  '.tflite': 'application/octet-stream', '.txt': 'text/plain; charset=utf-8' };
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

const outDir = join(root, 'out-ocr');
mkdirSync(outDir, { recursive: true });

function saveDataUrl(dataUrl, file) {
  writeFileSync(file, Buffer.from(dataUrl.slice(dataUrl.indexOf(',') + 1), 'base64'));
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
    saveDataUrl(post.imageUrl, join(outDir, `verify-B-${post.name}.png`));
    delete post.imageUrl;
    post.lines.forEach((l, i) => {
      saveDataUrl(l.cropUrl, join(outDir, `verify-B-${post.name}-line${String(i).padStart(2, '0')}.png`));
      delete l.cropUrl;
    });
  }
  writeFileSync(join(outDir, 'verify-report.json'), JSON.stringify(r, null, 2));

  console.log('\n--- stage A (rec-only, identical crops, wasm vs webgpu) ---');
  for (const row of r.stageA) {
    const mark = row.equal ? (row.gpuText === row.truth ? 'ok ' : 'ok≠truth') : 'DIFF';
    console.log(`${mark.padEnd(8)} #${String(row.idx).padStart(2, '0')} ${row.label.padEnd(14)} flips=${
      row.idFlips} minMargin=${row.minMarginGpu}${row.gpuDeterministic ? '' : ' NONDET'} gpu="${row.gpuText}"${
      row.equal ? '' : ` cpu="${row.cpuText}"`}${row.gpuText === row.truth ? '' : ` truth="${row.truth}"`}`);
    for (const f of row.flipDetail ?? []) {
      console.log(`         t=${f.t} gpu'${f.gpu}'(m=${f.gpuMargin}) cpu'${f.cpu}'(m=${f.cpuMargin})`);
    }
  }
  console.log('\n--- stage B (det→split→rec) ---');
  for (const post of r.stageB) {
    console.log(`${post.name}: det ${post.detMs}ms, ${post.boxCount} boxes`);
    for (const l of post.lines) {
      console.log(`  ${l.equal ? 'ok  ' : 'DIFF'} flips=${l.idFlips} "${l.gpuText}"${
        l.equal ? '' : ` / cpu="${l.cpuText}"`}`);
    }
  }
  console.log('\nSUMMARY ' + JSON.stringify(r.summary, null, 2));
  const diff32 = [
    ...r.stageA.filter((x) => !x.equal32).map((x) => `A#${x.idx} gpu32="${x.gpu32Text}" cpu32="${x.cpu32Text}"`),
    ...r.stageB.flatMap((p) => p.lines.filter((l) => !l.equal32)
      .map((l) => `B(${p.name}) gpu32="${l.gpu32Text}" cpu32="${l.cpu32Text}"`)),
  ];
  console.log('\n--- fp32 model: webgpu vs wasm ---');
  console.log(diff32.length ? diff32.join('\n') : 'identical on every crop');
  const go = r.summary.stageA.equal === r.summary.stageA.total
    && r.summary.stageB.equal === r.summary.stageB.lines;
  const go32 = r.summary.stageA.equal32 === r.summary.stageA.total
    && r.summary.stageB.equal32 === r.summary.stageB.lines;
  console.log(go ? '\nGATE(fp16): GO — decode identical on both backends'
    : '\nGATE(fp16): CHECK — decoded text differs between backends (see DIFF rows)');
  console.log(go32 ? 'GATE(fp32 model): GO — webgpu decode identical to wasm on every crop'
    : 'GATE(fp32 model): CHECK — see fp32 diffs above');
  server.close();
  await cdp.send('Browser.close').catch(() => {});
  process.exit(go || go32 ? 0 : 1);
} catch (err) {
  console.error('verify failed:', err);
  process.exit(1);
}
