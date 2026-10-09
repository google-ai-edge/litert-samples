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
 * Preprocessing matrix for one problem window. The recognizer has
 * deterministic failure pockets (a visually clean crop decoding to
 * confident-looking garbage); this asks which preprocessing choice moves a
 * given crop out of its pocket, instead of guessing one at a time.
 *
 *   node tools-ocr/probe-window.mjs <chrome-binary> <image> <x> <y> <w> <h> [--profile=<dir>]
 *
 * x/y/w/h are in SOURCE image pixels (take them from debug-strips output).
 * Prints decode + CTC margin for each condition.
 */
import { readFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { resolve } from 'node:path';
import { Cdp, evalIn, launchChrome, waitForEndpoint, waitForEngine } from '../tools/cdp.mjs';

const [chromeBin, file, x, y, w, h] = process.argv.slice(2);
const profile = process.argv.find((a) => a.startsWith('--profile='))?.slice(10);
if (!chromeBin || !file || !h) {
  console.error('usage: node tools-ocr/probe-window.mjs <chrome> <image> <x> <y> <w> <h> [--profile=]');
  process.exit(2);
}
const outDir = resolve(import.meta.dirname, '..', 'out-ocr');
const server = createServer((req, res) => {
  try {
    const body = readFileSync(resolve(outDir, decodeURIComponent(req.url.slice(1))));
    res.writeHead(200, { 'Content-Type': 'image/png' });
    res.end(body);
  } catch { res.writeHead(404).end(); }
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;

launchChrome(chromeBin, { dist: resolve(import.meta.dirname, '..', 'dist-ocr'), port: 9237, profile });
const cdp = await Cdp.connect(await waitForEndpoint(9237));
const { status, session } = await waitForEngine(cdp, { hook: '__pt' });
if (status?.state !== 'ready') { console.error('engine not ready'); process.exit(1); }

const expr = `(async () => {
  const REC_H = 48, REC_W = 320;
  const res = await fetch(${JSON.stringify(`${base}/${file}`)});
  const bmp = await createImageBitmap(await res.blob());
  const sx = ${x}, sy = ${y}, sw = ${w}, sh = ${h};
  const out = [];

  const decode = (rgba, contentW) => {
    const plane = REC_H * REC_W;
    const nchw = new Float32Array(3 * plane).fill(-1);
    for (let yy = 0; yy < REC_H; yy++) {
      for (let xx = 0; xx < contentW; xx++) {
        const s = (yy * contentW + xx) * 4, d = yy * REC_W + xx;
        nchw[d] = rgba[s] / 127.5 - 1;
        nchw[plane + d] = rgba[s + 1] / 127.5 - 1;
        nchw[2 * plane + d] = rgba[s + 2] / 127.5 - 1;
      }
    }
    return __pt.recRaw(nchw);
  };

  const draw = (drawnW, pad, bgFill, invert, padWithBg) => {
    const contentW = Math.min(REC_W, padWithBg ? REC_W : drawnW + 2 * pad);
    const c = new OffscreenCanvas(contentW, REC_H);
    const g = c.getContext('2d', { willReadFrequently: true });
    g.fillStyle = bgFill;
    g.fillRect(0, 0, contentW, REC_H);
    g.drawImage(bmp, sx, sy, sw, sh, pad, 0, drawnW, REC_H);
    const rgba = g.getImageData(0, 0, contentW, REC_H).data;
    if (invert) {
      for (let i = 0; i < rgba.length; i += 4) {
        rgba[i] = 255 - rgba[i]; rgba[i+1] = 255 - rgba[i+1]; rgba[i+2] = 255 - rgba[i+2];
      }
    }
    return decode(rgba, contentW);
  };

  // sample the source background at the crop's top-left corner
  const probe = new OffscreenCanvas(1, 1);
  const pg = probe.getContext('2d', { willReadFrequently: true });
  pg.drawImage(bmp, sx, sy, 1, 1, 0, 0, 1, 1);
  const p = pg.getImageData(0, 0, 1, 1).data;
  const bgFill = 'rgb(' + p[0] + ',' + p[1] + ',' + p[2] + ')';
  const nat = Math.min(REC_W - 16, Math.max(1, Math.round(REC_H * (sw / sh))));

  out.push(['native + black pad', await draw(nat, 8, bgFill, false, false)]);
  out.push(['native + bg pad to 320', await draw(nat, 8, bgFill, false, true)]);
  out.push(['inverted', await draw(nat, 8, bgFill, true, false)]);
  out.push(['inverted + bg pad', await draw(nat, 8, bgFill, true, true)]);
  out.push(['stretched to 304', await draw(304, 8, bgFill, false, false)]);
  out.push(['stretched + bg pad', await draw(304, 8, bgFill, false, true)]);
  out.push(['0.85 scale', await draw(Math.round(nat * 0.85), 12, bgFill, false, false)]);
  out.push(['1.15 scale', await draw(Math.min(REC_W - 16, Math.round(nat * 1.15)), 8, bgFill, false, false)]);
  return JSON.stringify({ bg: bgFill, nat, rows: out });
})()`;

const r = JSON.parse(await evalIn(cdp, session, expr, true, 120000));
console.log(`bg=${r.bg} natural width=${r.nat}`);
for (const [name, d] of r.rows) {
  console.log(`  ${name.padEnd(24)} [${d.score.toFixed(3)}] "${d.text}"`);
}
await cdp.send('Browser.close').catch(() => {});
server.close();
