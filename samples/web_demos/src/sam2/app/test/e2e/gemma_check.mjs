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

// "Ask Gemma" end to end: the demo UI with Gemma 4 (E2B by default) on the real
// LiteRT-LM server (`litert-lm serve`), in system Chrome on WebGPU:
//   1. "the soccer ball" on frame 1 -> one object with a box prompt, and SAM 2's
//      mask of it lies on the ball;
//   2. "all players" -> up to six objects, each with a box prompt,
//      tracking then runs on all of them;
//   3. camera mode: "the player" -> an object on the newest camera frame;
//   4. without the server, only the in-browser models are offered.
//
//   tools/gemma_server.sh http://localhost:5183 &
//   node test/e2e/gemma_check.mjs [--model=gemma-4-e2b]   (default gemma-4-e4b)
import {execFileSync} from 'node:child_process';
import {existsSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {resolve} from 'node:path';
import {chromium} from 'playwright-core';
import {createServer} from 'vite';

const args = Object.fromEntries(process.argv.slice(2).map((a) => {
  const [k, v] = a.replace(/^--/, '').split('=');
  return [k, v ?? 'true'];
}));
const model = args.model ?? 'gemma-4-e4b';
const app = resolve(import.meta.dirname, '../..');
const shots = resolve(app, 'test/e2e/screenshots');
const y4m = resolve(tmpdir(), 'sam2chain_football_640.y4m');
if (!existsSync(y4m)) {
  execFileSync('ffmpeg', ['-loglevel', 'error', '-y', '-i', resolve(app, 'public/assets/football_ai_studio.mp4'),
    '-t', '4', '-vf', 'scale=640:360', '-pix_fmt', 'yuv420p', y4m]);
}
const server = await createServer({root: app, logLevel: 'error', server: {port: 5183}});
await server.listen();
const browser = await chromium.launch({channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu',
  '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', `--use-file-for-fake-video-capture=${y4m}`]});
let failed = false;
const check = (ok, msg) => {
  console.log(`${ok ? '  ok ' : '  FAIL'} ${msg}`);
  if (!ok) failed = true;
};
const objects = (page) => page.evaluate(() => window.__sam2.objects.map((o) => {
  const s = window.__sam2, m = s.results.get(s.frame)?.get(o.id);
  let n = 0, sx = 0, sy = 0;
  const N = m ? Math.sqrt(m.length) : 0;
  if (m) for (let i = 0; i < m.length; i++) if (m[i] > 0) { n++; sx += i % N; sy += Math.floor(i / N); }
  return {labels: o.point?.pts.map((p) => p.label).join('') ?? '', box: o.point?.pts.slice(0, 2).map((p) => [p.nx, p.ny]),
    area: m ? n / m.length : 0, cx: n ? (sx / n + 0.5) / N : 0, cy: n ? (sy / n + 0.5) / N : 0};
}));
const ask = async (page, what) => {
  await page.fill('#gemmaAsk', what);
  await page.click('#gemmaBtn');
  await page.waitForFunction(() => !window.__sam2.asking && !window.__sam2.busy, null, {timeout: 300000, polling: 200});
  return page.textContent('#objHint');
};
try {
  const page = await browser.newPage({viewport: {width: 1440, height: 900}});
  page.on('pageerror', (e) => console.log(`  [pageerror] ${e.message}`));
  await page.goto('http://localhost:5183/');
  await page.waitForFunction(() => window.__sam2?.engine && window.__sam2?.clip && !window.__sam2.loading,
      null, {timeout: 300000});
  await page.waitForFunction(() => window.__sam2.gemma !== null, null, {timeout: 10000}).catch(() => {});
  const models = await page.evaluate(() => window.__sam2.gemma);
  check(models?.includes(model), `LiteRT-LM server found with ${JSON.stringify(models)}; using ${model}`);
  await page.selectOption('#gemmaModel', model);

  // ---- 1. one object
  const info1 = await ask(page, 'the soccer ball');
  const [ball] = await objects(page);
  check(ball.labels === '23' && ball.area > 0.0005 && ball.area < 0.03 && ball.cx > 0.43 && ball.cx < 0.54 &&
      ball.cy > 0.7 && ball.cy < 0.88,
      `"the soccer ball": box prompt ${JSON.stringify(ball.box?.map((p) => p.map((v) => +v.toFixed(3))))}, ` +
      `SAM mask ${(100 * ball.area).toFixed(2)}% at (${ball.cx.toFixed(3)}, ${ball.cy.toFixed(3)}) — ${info1}`);
  await page.screenshot({path: `${shots}/gemma_ball.png`});

  // ---- 2. a new request replaces the selection (no Reset in between)
  const info2 = await ask(page, 'all players');
  const all = await objects(page);
  const ballGone = all.every((o) => !(o.cx > 0.43 && o.cx < 0.54 && o.cy > 0.7 && o.cy < 0.88 && o.area < 0.03));
  check(all.length >= 3 && all.length <= 5 && all.every((o) => o.labels === '23' && o.area > 0.002) && ballGone,
      `"all players" replaces the ball: ${all.length} objects, all box prompts, mask areas ` +
      `${all.map((o) => (100 * o.area).toFixed(1) + '%').join(', ')}, ball object gone: ${ballGone} — ${info2}`);
  await page.screenshot({path: `${shots}/gemma_players.png`});
  await page.click('#trackBtn');
  await page.waitForFunction(() => !window.__sam2.tracking && /Done|Stopped|failed/.test(
      document.getElementById('trackInfo').textContent), null, {timeout: 20 * 60 * 1000, polling: 1000});
  const tracked = await page.evaluate(() => {
    const s = window.__sam2;
    return s.objects.map((o) => [...s.results.values()].filter((r) => r.get(o.id)?.some((v) => v > 0)).length);
  });
  // Background players can leave the view or be occluded; each must be tracked on most of the clip.
  check(tracked.every((n) => n >= 0.75 * 192), `tracking the Gemma-selected players: masks on ${tracked.join(' / ')} of 192 frames`);

  // ---- 2b. back to one object: a request with one match leaves exactly one; no match changes nothing
  await page.evaluate(() => {
    const el = document.getElementById('scrubber');
    el.value = '0';
    el.dispatchEvent(new Event('input'));
  });
  const info2b = await ask(page, 'the soccer ball');
  const one = await objects(page);
  const tracked2 = await page.evaluate(() => [...window.__sam2.results.values()].filter((r) => r.size).length);
  check(one.length === 1 && one[0].labels === '23' && tracked2 <= 1,
      `"the soccer ball" after tracking ${all.length} players: ${one.length} object, old tracks cleared (${tracked2} frame with results) — ${info2b}`);
  const info2c = await ask(page, 'a purple elephant');
  const still = await objects(page);
  check(still.length === 1 && still[0].labels === '23' && /unchanged|found 0|no /.test(info2c),
      `a request Gemma finds nothing for keeps the objects: ${still.length} object — ${info2c}`);

  // ---- 3. camera
  await page.click('#camBtn');
  await page.waitForFunction(() => window.__sam2.live, null, {timeout: 30000});
  await page.waitForTimeout(1500);
  const info3 = await ask(page, 'the player in the blue shirt');
  const cam = await page.evaluate(() => window.__sam2.objects.filter((o) => o.point).map((o) => o.point.pts.map((p) => p.label).join('')));
  check(cam.length >= 1 && cam.every((l) => l.startsWith('23')), `camera: ${cam.length} object(s) from Gemma boxes — ${info3}`);
  await page.waitForTimeout(3000);
  await page.screenshot({path: `${shots}/gemma_camera.png`});
  await page.click('#camBtn');
  await page.close();

  // ---- 4. no server
  const page2 = await browser.newPage({viewport: {width: 1440, height: 900}});
  await page2.goto('http://localhost:5183/?llm=http://127.0.0.1:9');
  await page2.waitForFunction(() => window.__sam2?.engine && window.__sam2?.clip && !window.__sam2.loading,
      null, {timeout: 300000});
  await page2.waitForFunction(() => window.__sam2.gemma !== undefined, null, {timeout: 10000}).catch(() => {});
  // No server: only the in-browser models remain (Find stays usable with WebGPU).
  const off = await page2.evaluate(() => ({models: window.__sam2.gemma ?? [],
    options: [...document.getElementById('gemmaModel').options].filter((o) => !o.disabled && !o.hidden).map((o) => o.value)}));
  check(off.models.length > 0 && off.models.every((id) => id.startsWith('web:')),
      `without a server: only browser models ${JSON.stringify(off.models)}`);
} finally {
  await browser.close();
  await server.close();
}
console.log(failed ? 'GEMMA CHECK FAIL' : 'GEMMA CHECK PASS');
process.exit(failed ? 1 : 0);
