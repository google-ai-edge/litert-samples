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

// End-to-end test of the wasm build: runs test/e2e/chain_e2e.html in system
// Chrome (WebGPU, JSPI) via playwright-core — the C++ ModelChain pipeline on
// 24 frames of the football sample with 5 objects (2 clicks; 1 click; a box
// + a negative click joining at frame 6; a box; a box joining at frame 3) — then verifies the dumps with
// tools/verify_chain.py: preprocess vs numpy, every mask vs HF
// Sam2VideoModel, composites vs numpy.
//
//   npm run e2e [-- --size=384 --nmm=7 --effect=overlay --precision=fp16 --headed
//                   --build=browser]   (author the SAM 2 model in the browser)
import {execFileSync, spawnSync} from 'node:child_process';
import {existsSync, mkdirSync, rmSync, writeFileSync} from 'node:fs';
import {dirname, resolve} from 'node:path';
import {chromium} from 'playwright-core';
import {createServer} from 'vite';

const args = Object.fromEntries(process.argv.slice(2).map((a) => {
  const [k, v] = a.replace(/^--/, '').split('=');
  return [k, v ?? 'true'];
}));
const size = args.size ?? '384';
const nmm = args.nmm ?? '7';
const effect = args.effect ?? 'overlay';
const precision = args.precision ?? 'fp16';
const build = args.build ?? 'native';
const app = resolve(import.meta.dirname, '../..');
const proj = resolve(app, '..');
const root = proj;  // artifacts/ and .venv/ live in the sample directory
// External inputs (override with env vars; defaults: the repo root and ./artifacts, ./.venv).
const art = resolve(process.env.ARTIFACTS ?? resolve(root, 'artifacts'), 'chain');
const python = process.env.PYTHON ?? resolve(root, '.venv/bin/python');
const video = resolve(app, 'public/assets/football_ai_studio.mp4');
const W = 640, H = 360, T = Number(args.frames ?? 24);
const clip = resolve(art, 'football_640x360_24.rgba');
if (!existsSync(clip)) {
  execFileSync('ffmpeg', ['-loglevel', 'error', '-y', '-i', video,
    '-frames:v', '24', '-vf', 'scale=640:360', '-f', 'rawvideo', '-pix_fmt', 'rgba', clip]);
}
const PROMPTS = '0@0:0.44,0.28,1;0.46,0.40,1|1@0:0.484,0.79,1|2@6:0.14,0.32,2;0.215,0.645,3;0.15,0.34,0|3@0:0.194,0.342,2;0.253,0.632,3|4@3:0.594,0.352,2;0.658,0.632,3';
const tag = args.tag ?? `wasm${size}_${precision}_nmm${nmm}_${effect}${build === 'browser' ? '_browserbuild' : ''}`;
const dumpDir = resolve(art, tag);
rmSync(dumpDir, {recursive: true, force: true});
mkdirSync(dumpDir, {recursive: true});

// The page POSTs its dumps here.
const dumpPlugin = {
  name: 'dump',
  configureServer(server) {
    server.middlewares.use('/__dump/', (req, res) => {
      const chunks = [];
      req.on('data', (c) => chunks.push(c));
      req.on('end', () => {
        const file = resolve(art, decodeURIComponent(req.url.replace(/^\//, '')));
        mkdirSync(dirname(file), {recursive: true});
        writeFileSync(file, Buffer.concat(chunks));
        res.end('ok');
      });
    });
  },
};
const server = await createServer({root: app, logLevel: 'error', server: {port: 5176}, plugins: [dumpPlugin]});
await server.listen();
const browser = await chromium.launch({channel: 'chrome', headless: args.headed !== 'true',
  args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist']});
let result;
try {
  const page = await browser.newPage();
  page.on('console', (m) => { if (m.type() !== 'debug') console.log(`  [page] ${m.text()}`); });
  page.on('pageerror', (e) => console.log(`  [pageerror] ${e.message}`));
  const url = `http://localhost:5176/test/e2e/chain_e2e.html?size=${size}&nmm=${nmm}&effect=${effect}` +
    `&precision=${precision}&W=${W}&H=${H}&T=${T}&clip=${clip}&dump=${tag}&rgb=0,6,23` +
    `&prompts=${encodeURIComponent(PROMPTS)}&build=${build}` +
    `&weights=${resolve(art, '..', `sam2_tiny_${size}_video.safetensors`)}`;
  await page.goto(url);
  await page.waitForFunction(() => window.runChain, null, {timeout: 60000});
  result = await page.evaluate(() => window.runChain());
} finally {
  await browser.close();
  await server.close();
}
console.log(`  loaded in ${(result.loadMs / 1000).toFixed(1)} s` +
  (result.buildMs ? ` (SAM 2 model authored in-browser in ${(result.buildMs / 1000).toFixed(1)} s)` : '') +
  `; median frame (5 objects, readback of one score) ${result.frameMs.toFixed(1)} ms`);
const minIou = precision === 'fp32' ? '0.95' : '0.85';
const meanIou = precision === 'fp32' ? '0.99' : '0.98';
const refCache = args.ref_cache ? ['--ref_cache', args.ref_cache] : [];
const v = spawnSync(python, [resolve(proj, 'tools/verify_chain.py'), '--dump', dumpDir,
  '--rgba', clip, '--png', '--min_iou', minIou, '--min_mean_iou', meanIou, '--tag', tag, ...refCache], {encoding: 'utf8'});
const lines = (v.stdout + v.stderr).split('\n').filter((l) => /^ +(ok|FAIL|info|\()|^VERIFY/.test(l));
console.log(lines.join('\n'));
// Per-frame pixel dumps are large (S*S*3 floats); keep only when asked.
if (args.keep_pixels !== 'true') {
  for (let t = 0; t < T; t++) rmSync(resolve(dumpDir, `pixels_f${t}.f32`), {force: true});
}
process.exit(v.status ?? 1);
