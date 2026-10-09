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
 * Dev introspection: dump each det box's rec strip (piece boundaries burned
 * in red) for an image, to eyeball where the detector and the splitter
 * actually cut. Serves out-ocr/ like smoke.mjs does.
 *
 *   node tools-ocr/debug-strips.mjs <chrome-binary> <image-file-in-out-ocr> [--profile=<dir>]
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { resolve } from 'node:path';
import { Cdp, evalIn, launchChrome, waitForEndpoint, waitForEngine } from '../tools/cdp.mjs';

const chromeBin = process.argv[2];
const file = process.argv[3];
const profile = process.argv.find((a) => a.startsWith('--profile='))?.slice(10);
if (!chromeBin || !file) {
  console.error('usage: node tools-ocr/debug-strips.mjs <chrome-binary> <image> [--profile=<dir>]');
  process.exit(2);
}
const outDir = resolve(import.meta.dirname, '..', 'out-ocr');

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

launchChrome(chromeBin, { dist: resolve(import.meta.dirname, '..', 'dist-ocr'), port: 9233, profile });

const cdp = await Cdp.connect(await waitForEndpoint(9233));
const { status, session } = await waitForEngine(cdp, { hook: '__pt' });
if (status?.state !== 'ready') {
  console.error('engine not ready:', JSON.stringify(status));
  process.exit(1);
}
const raw = await evalIn(cdp, session,
  `__pt.debugOcr(${JSON.stringify(`${base}/${file}`)}).then((r) => JSON.stringify(r))`, true, 120000);
const r = JSON.parse(raw);
if (!r.ok) {
  console.error('debugOcr failed:', r.error);
  process.exit(1);
}
r.boxes.forEach((b, i) => {
  const name = `debug-strip-${String(i).padStart(2, '0')}.png`;
  writeFileSync(resolve(outDir, name),
    Buffer.from(b.stripUrl.slice(b.stripUrl.indexOf(',') + 1), 'base64'));
  console.log(`${name} det=[${b.det}] src=[${b.src}] lw=${b.lw}`);
  b.pieces.forEach((p, j) => {
    const wname = `debug-win-${String(i).padStart(2, '0')}-${j}.png`;
    writeFileSync(resolve(outDir, wname),
      Buffer.from(p.winUrl.slice(p.winUrl.indexOf(',') + 1), 'base64'));
    if (p.tensorUrl) {
      writeFileSync(resolve(outDir, `debug-tensor-${String(i).padStart(2, '0')}-${j}.png`),
        Buffer.from(p.tensorUrl.slice(p.tensorUrl.indexOf(',') + 1), 'base64'));
    }
    console.log(`  ${wname} [${p.from},${p.to}] cw=${p.contentW} [${p.score}] "${p.text}"` +
      (p.text2 !== p.text ? ` RUN2="${p.text2}"` : '') +
      (p.textBgPad !== p.text ? ` BGPAD="${p.textBgPad}"` : ' BGPAD=same'));
  });
});
await cdp.send('Browser.close').catch(() => {});
server.close();
