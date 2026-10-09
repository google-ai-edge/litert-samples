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
 * Find-in-images: the search half of Page Text.
 *
 * The browser's own Cmd+F cannot see inside a screenshot, so this indexes the
 * images on the page as you scroll — OCR runs in the extension's offscreen
 * document, on the user's own GPU — and then matches queries against that
 * index, lighting up the lines that hit.
 *
 * Indexing is deliberately background work: requests are queued behind any
 * interactive right-click read, run one at a time, and stop the moment the
 * user leaves the page. Results are cached by image URL in the engine, so
 * scrolling back over an image is free.
 */

import { groupLines } from './ocr-pipeline.js';

const MIN_W = 120;
const MIN_H = 80;
const SCAN_MARGIN = 600; // index a screen ahead of the viewport
const SCAN_DEBOUNCE_MS = 350;
const MAX_IMAGES = 60; // safety cap per page

// url → { lines, natural } from the engine; null while in flight
const index = new Map();
const inFlight = new Set();
let queue = [];
let draining = false;
let ui = null; // { host, input, count } when the search bar is open
let matches = []; // [{ img, url, line, text }]
let current = -1;
let deps = null; // { send, sourceFor, fitRect }

export function initFind(dependencies) {
  deps = dependencies;
  window.addEventListener('scroll', scheduleScan, { passive: true });
  window.addEventListener('resize', () => { scheduleScan(); repaint(); }, { passive: true });
}

// --- indexing ------------------------------------------------------------------

function eligible(img) {
  if (!(img instanceof HTMLImageElement) || !img.complete || !img.naturalWidth) return false;
  const r = img.getBoundingClientRect();
  if (r.width < MIN_W || r.height < MIN_H) return false;
  return r.bottom > -SCAN_MARGIN && r.top < window.innerHeight + SCAN_MARGIN;
}

let scanTimer = null;
function scheduleScan() {
  clearTimeout(scanTimer);
  scanTimer = setTimeout(scan, SCAN_DEBOUNCE_MS);
}

/** Queue every not-yet-indexed image near the viewport. */
export function scan() {
  if (index.size >= MAX_IMAGES) return;
  for (const img of document.images) {
    if (!eligible(img)) continue;
    const url = deps.sourceFor(img);
    if (!url || index.has(url) || inFlight.has(url) || queue.includes(url)) continue;
    queue.push(url);
  }
  drain();
}

async function drain() {
  if (draining) return;
  draining = true;
  while (queue.length) {
    const url = queue.shift();
    if (index.has(url) || inFlight.has(url)) continue;
    inFlight.add(url);
    updateProgress();
    const payload = await deps.send({ type: 'ocr', url, background: true });
    inFlight.delete(url);
    if (payload?.ok) {
      // Search the LOGICAL lines: a phrase can straddle two rec windows
      // ("has 8 power" + "outlets for 20 people"), so match on the merged
      // text and highlight the pieces it came from.
      index.set(url, { groups: groupLines(payload.lines), natural: payload.natural });
      if (ui) run(ui.input.value); // a new image may add matches to a live query
    }
    updateProgress();
    // yield to the page between images: indexing must never feel like a freeze
    await new Promise((r) => setTimeout(r, 0));
  }
  draining = false;
}

function indexedCount() {
  let n = 0;
  for (const { groups } of index.values()) if (groups.length) n++;
  return n;
}

// --- search --------------------------------------------------------------------

/** Images currently on the page, keyed by the URL they were indexed under. */
function liveImages() {
  const byUrl = new Map();
  for (const img of document.images) {
    if (!img.complete || !img.naturalWidth) continue;
    const url = deps.sourceFor(img);
    if (url && !byUrl.has(url)) byUrl.set(url, img);
  }
  return byUrl;
}

function run(query) {
  matches = [];
  const q = query.trim().toLowerCase();
  if (q) {
    const byUrl = liveImages();
    for (const [url, entry] of index) {
      const img = byUrl.get(url);
      if (!img) continue;
      for (const group of entry.groups) {
        if (group.text.toLowerCase().includes(q)) {
          matches.push({ img, url, group, text: group.text });
        }
      }
    }
    matches.sort((a, b) => {
      const ra = a.img.getBoundingClientRect();
      const rb = b.img.getBoundingClientRect();
      return (ra.top + window.scrollY) - (rb.top + window.scrollY)
        || a.group.pieces[0].y - b.group.pieces[0].y;
    });
  }
  current = matches.length ? 0 : -1;
  repaint();
  updateCount();
}

/** Draw one highlight per matching line, in page coordinates. A logical line
 * is several rec windows, so merge the pieces that sit on the same row into a
 * single box — three abutting rectangles read as three matches. */
function repaint() {
  document.querySelectorAll('[data-pagetext-hit]').forEach((el) => el.remove());
  if (!ui) return;
  matches.forEach((m, i) => {
    const { left, top, width, height, uv } = deps.fitRect(m.img);
    const on = i === current;
    const rects = [];
    for (const piece of m.group.pieces) {
      const r = {
        x: ((piece.x - uv.x) / uv.w) * width,
        y: ((piece.y - uv.y) / uv.h) * height,
        w: (piece.w / uv.w) * width,
        h: (piece.h / uv.h) * height,
      };
      const row = rects.find((o) => Math.abs(o.y - r.y) < o.h * 0.6);
      if (row) {
        const right = Math.max(row.x + row.w, r.x + r.w);
        const bottom = Math.max(row.y + row.h, r.y + r.h);
        row.x = Math.min(row.x, r.x);
        row.y = Math.min(row.y, r.y);
        row.w = right - row.x;
        row.h = bottom - row.y;
      } else {
        rects.push(r);
      }
    }
    for (const r of rects) {
      if (r.x + r.w < 0 || r.y + r.h < 0 || r.x > width || r.y > height) continue;
      const box = document.createElement('div');
      box.setAttribute('data-pagetext-hit', '');
      box.style.cssText =
        `position:absolute;left:${left + window.scrollX + r.x}px;top:${top + window.scrollY + r.y}px;` +
        `width:${r.w}px;height:${r.h}px;z-index:2147483645;pointer-events:none;border-radius:3px;` +
        `background:${on ? 'rgba(255,196,0,.42)' : 'rgba(77,163,255,.28)'};` +
        `outline:1.5px solid ${on ? 'rgba(255,196,0,.95)' : 'rgba(77,163,255,.6)'};` +
        'transition:background .15s,outline-color .15s;';
      document.body.appendChild(box);
    }
  });
}

function step(delta) {
  if (!matches.length) return;
  current = (current + delta + matches.length) % matches.length;
  const m = matches[current];
  const r = m.img.getBoundingClientRect();
  if (r.top < 80 || r.bottom > window.innerHeight - 80) {
    m.img.scrollIntoView({ block: 'center', behavior: 'smooth' });
    setTimeout(repaint, 400);
  }
  repaint();
  updateCount();
}

// --- UI ------------------------------------------------------------------------

function updateCount() {
  if (!ui) return;
  const imgs = new Set(matches.map((m) => m.url)).size;
  ui.count.textContent = matches.length
    ? `${current + 1}/${matches.length} · ${imgs} image${imgs > 1 ? 's' : ''}`
    : (ui.input.value.trim() ? 'no matches' : '');
}

function updateProgress() {
  if (!ui) return;
  const pending = queue.length + inFlight.size;
  ui.progress.textContent = pending
    ? `reading ${pending} more…`
    : `${indexedCount()} image${indexedCount() === 1 ? '' : 's'} indexed`;
}

export function toggleFind() {
  if (ui) { closeFind(); return; }
  const host = document.createElement('div');
  host.setAttribute('data-pagetext-find', '');
  host.style.cssText =
    'position:fixed;top:16px;right:16px;z-index:2147483647;display:flex;align-items:center;' +
    'gap:10px;padding:9px 12px;border-radius:12px;background:rgba(18,18,22,.95);' +
    'box-shadow:0 6px 24px rgba(0,0,0,.4);font:13px/1.3 -apple-system,system-ui,sans-serif;' +
    'color:#e8e8ec;';

  const input = document.createElement('input');
  input.type = 'text';
  input.placeholder = 'Find in images…';
  input.style.cssText =
    'border:0;outline:0;background:transparent;color:#e8e8ec;font:inherit;width:190px;';

  const count = document.createElement('span');
  count.style.cssText = 'color:#9a9aa5;font-size:12px;min-width:96px;text-align:right;';

  const progress = document.createElement('span');
  progress.style.cssText = 'color:#6f6f7a;font-size:11px;border-left:1px solid #33333c;padding-left:10px;';

  const close = document.createElement('button');
  close.textContent = '✕';
  close.style.cssText =
    'border:0;background:transparent;color:#9a9aa5;font:inherit;cursor:pointer;padding:0 2px;';
  close.addEventListener('click', closeFind);

  host.append(input, count, progress, close);
  document.documentElement.appendChild(host);
  ui = { host, input, count, progress };

  input.addEventListener('input', () => run(input.value));
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); step(e.shiftKey ? -1 : 1); }
    if (e.key === 'Escape') { e.preventDefault(); closeFind(); }
  });
  input.focus();
  updateProgress();
  scan();
}

export function closeFind() {
  document.querySelectorAll('[data-pagetext-hit]').forEach((el) => el.remove());
  ui?.host.remove();
  ui = null;
  matches = [];
  current = -1;
}

export const __findDebug = {
  get indexed() { return indexedCount(); },
  get pending() { return queue.length + inFlight.size; },
  get matches() { return matches.map((m) => m.text); },
  get open() { return Boolean(ui); },
  search(q) { if (!ui) toggleFind(); ui.input.value = q; run(q); return matches.length; },
  next() { step(1); return current; },
  scan,
};
