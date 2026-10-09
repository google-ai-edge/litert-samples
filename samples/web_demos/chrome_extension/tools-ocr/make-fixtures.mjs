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
 * Renders every test image the OCR tools read into out-ocr/, using a
 * throwaway Chromium page:
 *   fixture-*.png       screenshot-style posts at phone-Retina geometry
 *                       (1179px wide, ~34px body text): the quality bar.
 *   fixture-half-*.png  two posts at about half that size: a stress case for
 *                       smoke.mjs, and the det → rec inputs of verify.mjs.
 * Run it once before the other tools-ocr/ scripts.
 *
 *   node tools-ocr/make-fixtures.mjs <chrome-binary>
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { Cdp, attachTo, evalIn, findTarget, launchChrome, waitForEndpoint } from '../tools/cdp.mjs';

const chromeBin = process.argv[2];
if (!chromeBin) {
  console.error('usage: node tools-ocr/make-fixtures.mjs <chrome-binary>');
  process.exit(2);
}
const outDir = resolve(import.meta.dirname, '..', 'out-ocr');
mkdirSync(outDir, { recursive: true });

const RENDER = String(function render() {
  const FIXTURES = [
    {
      name: 'fixture-en-intro',
      dark: true,
      font: '-apple-system,"Helvetica Neue",Arial,sans-serif',
      author: 'Page Text',
      handle: '@demo_pagetext · 2h',
      lines: [
        'When long text is posted as a screenshot,',
        'nobody can search it or copy it later.',
        '',
        'This extension runs PP-OCRv5 in the',
        'browser and puts selectable text over',
        'the image. Nothing is sent to a server.',
        // A word longer than one rec window: it is cut between two letters
        // and must come back as one word (e2e.mjs).
        'Long words like internationalization stay whole.',
      ],
    },
    {
      // "Friday" recurs across three fixtures on purpose: the search demo
      // needs one query that lights up several images at once.
      name: 'fixture-en-ship',
      dark: true,
      font: '-apple-system,"Helvetica Neue",Arial,sans-serif',
      author: 'Release notes',
      handle: '@demo_ship · 3h',
      lines: [
        'Shipping on Friday, for real this time:',
        '',
        '- offline mode, including the editor',
        '- the download is 40% smaller',
        '- dark mode stops flashing white on launch',
        '',
        'Anything not on this list moves to next month.',
      ],
    },
    {
      name: 'fixture-en-meetup',
      dark: false,
      font: '-apple-system,"Helvetica Neue",Arial,sans-serif',
      author: 'Local meetup',
      handle: '@demo_meetup · 5h',
      lines: [
        'Doors open 18:30 on Friday.',
        'Three talks, 20 minutes each, starting at 19:00.',
        '',
        'The room holds 40 people and we are at 31,',
        'so there is space if you decide late.',
        '',
        'No food. There is coffee and a whiteboard.',
      ],
    },
    {
      // A generic wall of text, the kind people actually screenshot —
      // times, numbers and punctuation, nothing about OCR.
      name: 'fixture-en-dark',
      dark: true,
      font: '-apple-system,"Helvetica Neue",Arial,sans-serif',
      author: 'Workshop notes',
      handle: '@demo_workshop · 1h',
      lines: [
        'Notes for Saturday, since a few people asked:',
        '',
        'Bring a laptop with at least 16 GB of RAM.',
        'We start at 10:00 and break for lunch at 12:30.',
        'The room has 8 power outlets for 20 people, so',
        'bring a strip if you have one.',
        '',
        'Handouts go out on Friday — as a PDF, not a',
        'screenshot. You will be able to search them.',
      ],
    },
    {
      name: 'fixture-en-light',
      dark: false,
      font: '-apple-system,"Helvetica Neue",Arial,sans-serif',
      author: 'Page Text',
      handle: '@demo_pagetext · 2h',
      lines: [
        'Screenshots are where text goes to die.',
        '',
        'Someone posts a wall of text as an image,',
        'and every word inside it is stuck: you cannot',
        'copy it, search it, or translate it.',
        '',
        'So I built a browser extension that runs',
        'PP-OCRv5 with LiteRT.js, fully on-device.',
        'Right-click any image and the text inside',
        'becomes selectable, and copyable.',
      ],
    },
  ];
  // Half-size posts: 720px wide, 26px text, grey bars where the author row
  // would be, so every line of text in the image is a body line.
  const HALF = [
    {
      name: 'fixture-half-en-light',
      dark: false,
      font: '-apple-system,"Helvetica Neue",Arial,sans-serif',
      lines: [
        'Screenshots are where text goes to die.',
        'Every day someone posts a wall of text as an',
        'image, and every day the words inside it are',
        'stuck: no copy, no search, no translate.',
        'This extension runs PP-OCRv5 in your browser',
        'with LiteRT.js on WebGPU. Right-click any',
        'image and the text becomes selectable.',
      ],
    },
    {
      name: 'fixture-half-en-dark',
      dark: true,
      font: '-apple-system,"Helvetica Neue",Arial,sans-serif',
      lines: [
        'Long text shared as a screenshot cannot',
        'be copied or searched as it is.',
        'This extension runs PP-OCRv5 in your',
        'browser and puts selectable text on',
        'the image. Nothing goes to a server.',
      ],
    },
  ];
  const toDataUrl = (c) => c.convertToBlob({ type: 'image/png' }).then((b) => new Promise((res) => {
    const r = new FileReader();
    r.onload = () => res(r.result);
    r.readAsDataURL(b);
  }));
  const W = 1179;
  const PAD = 54;
  const BODY = 34;
  const LEAD = 1.5;
  const out = {};
  for (const f of FIXTURES) {
    const lineH = Math.round(BODY * LEAD);
    const h = PAD * 2 + 96 + f.lines.length * lineH;
    const c = new OffscreenCanvas(W, h);
    const x = c.getContext('2d');
    x.fillStyle = f.dark ? '#000000' : '#ffffff';
    x.fillRect(0, 0, W, h);
    // author row: avatar + name + handle
    x.fillStyle = f.dark ? '#2f3336' : '#cfd9de';
    x.beginPath();
    x.arc(PAD + 32, PAD + 32, 32, 0, Math.PI * 2);
    x.fill();
    x.fillStyle = f.dark ? '#e7e9ea' : '#0f1419';
    x.font = `bold 30px ${f.font}`;
    x.textBaseline = 'top';
    x.fillText(f.author, PAD + 84, PAD + 6);
    x.fillStyle = f.dark ? '#71767b' : '#536471';
    x.font = `28px ${f.font}`;
    x.fillText(f.handle, PAD + 84, PAD + 42);
    // body
    x.fillStyle = f.dark ? '#e7e9ea' : '#0f1419';
    x.font = `${BODY}px ${f.font}`;
    f.lines.forEach((line, i) => {
      if (line) x.fillText(line, PAD, PAD + 96 + i * lineH);
    });
    out[f.name] = toDataUrl(c);
  }
  for (const f of HALF) {
    const px = 26;
    const pad = 36;
    const lineH = Math.round(px * 1.55);
    const h = pad * 2 + 54 + f.lines.length * lineH;
    const c = new OffscreenCanvas(720, h);
    const x = c.getContext('2d', { willReadFrequently: true });
    x.fillStyle = f.dark ? '#15202b' : '#ffffff';
    x.fillRect(0, 0, 720, h);
    x.fillStyle = f.dark ? '#3d5466' : '#cfd9de';
    x.beginPath();
    x.arc(pad + 20, pad + 18, 20, 0, Math.PI * 2);
    x.fill();
    x.fillRect(pad + 52, pad + 4, 130, 12);
    x.fillRect(pad + 52, pad + 22, 90, 10);
    x.fillStyle = f.dark ? '#f7f9f9' : '#0f1419';
    x.font = `${px}px ${f.font}`;
    x.textBaseline = 'top';
    f.lines.forEach((line, i) => x.fillText(line, pad, pad + 54 + i * lineH));
    out[f.name] = toDataUrl(c);
  }
  return Promise.all(Object.entries(out).map(async ([k, v]) => [k, await v]))
    .then((entries) => JSON.stringify(Object.fromEntries(entries)));
});

const port = 9234;
launchChrome(chromeBin, { dist: resolve(import.meta.dirname, '..', 'dist-ocr'), port });
const cdp = await Cdp.connect(await waitForEndpoint(port));
const target = await findTarget(cdp, (t) => t.type === 'page');
const session = await attachTo(cdp, target);
const raw = await evalIn(cdp, session, `(${RENDER})()`, true, 60000);
const urls = JSON.parse(raw);
for (const [name, dataUrl] of Object.entries(urls)) {
  writeFileSync(resolve(outDir, `${name}.png`),
    Buffer.from(dataUrl.slice(dataUrl.indexOf(',') + 1), 'base64'));
  console.log(`${name}.png`);
}
await cdp.send('Browser.close').catch(() => {});
