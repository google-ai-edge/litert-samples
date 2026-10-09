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

/** Lists candidate images (rect, selector-ish path, src) on a page. */
import { resolve } from 'node:path';
import { Cdp, attachTo, evalIn, findTarget, launchChrome, sleep, waitForEndpoint } from '../tools/cdp.mjs';

const [, , chromeBin, url] = process.argv;
const port = 9228;
launchChrome(chromeBin, { dist: resolve(import.meta.dirname, '..', 'dist3d'), port, url });
const cdp = await Cdp.connect(await waitForEndpoint(port));
const page = await findTarget(cdp, (t) => t.type === 'page' && t.url.startsWith(url.slice(0, 30)));
const session = await attachTo(cdp, page);
await sleep(4000);
console.log(await evalIn(cdp, session, `(() => {
  const rows = [...document.images]
    .filter((i) => i.naturalWidth > 80)
    .slice(0, 40)
    .map((i) => {
      const r = i.getBoundingClientRect();
      const path = [];
      for (let el = i; el && path.length < 4; el = el.parentElement) {
        path.push(el.tagName.toLowerCase() +
          (el.className ? '.' + String(el.className).split(' ').slice(0, 2).join('.') : ''));
      }
      return { x: Math.round(r.left), y: Math.round(r.top), w: Math.round(r.width),
        h: Math.round(r.height), path: path.join(' < '), src: (i.currentSrc || i.src).slice(-60) };
    })
    .filter((e) => e.w >= 120);
  return JSON.stringify(rows, null, 1);
})()`));
await cdp.send('Browser.close').catch(() => {});
process.exit(0);
