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
 * Content script (all pages). Two features over the same OCR engine:
 *   right-click an image → a transparent, selectable text layer over it
 *     (PDF.js-style: invisible glyphs, visible ::selection), and
 *   Alt+Shift+F → find-in-images, which indexes the page's images in the
 *     background and highlights the lines that match (see find.js).
 *
 * The overlay is absolutely positioned in page coordinates so it scrolls
 * with the content. Line boxes flash briefly on build so the user sees what
 * was found, then fade to invisible; a small toolbar offers Copy all / close.
 */

import { groupLines, windowSeparator } from './ocr-pipeline.js';
import { closeFind, initFind, scan, toggleFind, __findDebug } from './find.js';

const FLASH_MS = 900;
const CACHE_ENTRIES = 12;

let session = null; // the single active overlay
let activateToken = 0;
const resultCache = new Map(); // source url → engine payload (LRU)

function send(msg) {
  return chrome.runtime.sendMessage({ target: 'bg', ...msg }).catch(() => null);
}

// --- source resolution (same rules as Page 3D) --------------------------------

/** URL the engine should fetch. blob: URLs are page-scoped (the engine can't
 * fetch them) but same-origin to this document, so re-encode locally.
 * X serves feed media downscaled (name=small, 680px) — request the large
 * variant instead; OCR quality tracks input resolution directly. */
function sourceFor(img) {
  let src = img.currentSrc || img.src;
  if (!src) return null;
  try {
    const u = new URL(src);
    if (u.hostname === 'pbs.twimg.com' && u.pathname.startsWith('/media/')
      && ['small', 'medium', '900x900', '360x360'].includes(u.searchParams.get('name'))) {
      u.searchParams.set('name', 'large');
      src = u.toString();
    }
  } catch { /* keep original */ }
  if (!src.startsWith('blob:')) return src;
  try {
    const c = document.createElement('canvas');
    c.width = img.naturalWidth;
    c.height = img.naturalHeight;
    c.getContext('2d').drawImage(img, 0, 0);
    return c.toDataURL('image/png');
  } catch {
    return null; // tainted
  }
}

/** The on-screen rect the overlay should cover, honoring object-fit; uv is
 * the visible crop of the natural image (object-fit: cover). */
function fitRect(img) {
  const rect = img.getBoundingClientRect();
  const fit = getComputedStyle(img).objectFit || 'fill';
  const nw = img.naturalWidth;
  const nh = img.naturalHeight;
  let { left, top, width, height } = rect;
  let uv = { x: 0, y: 0, w: 1, h: 1 };
  if (fit === 'contain' || fit === 'scale-down') {
    const s = Math.min(width / nw, height / nh);
    const w = nw * s;
    const h = nh * s;
    left += (width - w) / 2;
    top += (height - h) / 2;
    width = w;
    height = h;
  } else if (fit === 'cover') {
    const s = Math.max(width / nw, height / nh);
    uv = {
      w: width / s / nw,
      h: height / s / nh,
      x: (1 - width / s / nw) / 2,
      y: (1 - height / s / nh) / 2,
    };
  }
  return { left, top, width, height, uv };
}

// --- activation ----------------------------------------------------------------

chrome.runtime.onMessage.addListener((msg) => {
  if (msg?.type === 'toggle-find') { toggleFind(); return; }
  if (msg?.type === 'activate-ocr' && msg.srcUrl) {
    const img = [...document.images].find(
      (i) => i.currentSrc === msg.srcUrl || i.src === msg.srcUrl,
    );
    if (img) activate(img);
  }
});

async function activate(img) {
  const source = sourceFor(img);
  if (!source) return;
  const token = ++activateToken;
  teardown();

  const badge = showBadge(img, 'reading…');
  const poll = setInterval(async () => {
    const s = await send({ type: 'status' });
    if (!s || token !== activateToken) return;
    if (s.state === 'downloading') badge.set(`OCR models ${s.downloadedMB ?? 0} MB…`);
    else if (s.state === 'compiling') badge.set('compiling…');
    else if (s.state === 'ready') badge.set('reading…');
  }, 900);

  let payload = resultCache.get(source);
  if (!payload) {
    payload = await send({ type: 'ocr', url: source });
    if (payload?.ok) {
      resultCache.set(source, payload);
      if (resultCache.size > CACHE_ENTRIES) {
        resultCache.delete(resultCache.keys().next().value);
      }
    }
  }
  clearInterval(poll);
  badge.remove();
  if (token !== activateToken) return; // superseded
  if (!img.isConnected) return;
  if (!payload?.ok) {
    flashMessage(img, payload?.error ? `OCR failed: ${payload.error}` : 'OCR failed');
    return;
  }
  if (!payload.lines.length) {
    flashMessage(img, 'no text found in this image');
    return;
  }

  try {
    buildOverlay(img, payload);
  } catch (err) {
    console.warn('[pagetext] overlay failed:', err);
  }
}

function teardown() {
  const s = session;
  if (!s) return;
  session = null;
  clearTimeout(s.resizeTimer);
  s.host.remove();
  window.removeEventListener('resize', s.onResize);
  document.removeEventListener('pointerdown', s.onPointerDown, true);
}

// --- badges ------------------------------------------------------------------------

function pill(textContent) {
  const el = document.createElement('div');
  el.style.cssText =
    'position:fixed;z-index:2147483647;pointer-events:none;' +
    'font:11px/1.2 -apple-system,system-ui,sans-serif;color:#e8e8ec;' +
    'background:rgba(20,20,24,.88);padding:5px 10px;border-radius:999px;' +
    'box-shadow:0 2px 10px rgba(0,0,0,.35);transition:opacity .3s;';
  el.textContent = textContent;
  return el;
}

function showBadge(img, text) {
  const el = pill(text);
  const r = img.getBoundingClientRect();
  el.style.left = `${Math.max(8, r.left + 8)}px`;
  el.style.top = `${Math.max(8, r.top + 8)}px`;
  document.documentElement.appendChild(el);
  return {
    set: (t) => { el.textContent = t; },
    remove: () => el.remove(),
  };
}

function flashMessage(img, text) {
  const badge = showBadge(img, text);
  setTimeout(() => badge.remove(), 1800);
}

// --- overlay -------------------------------------------------------------------------

const JA_RE = /[぀-ヿ㐀-䶿一-鿿]/;

/** Copy joins the rec windows back into logical lines (see groupLines). */
function joinLines(lines) {
  return groupLines(lines).map((g) => g.text).join('\n');
}

function buildOverlay(img, payload, { flash = true } = {}) {
  const { left, top, width, height, uv } = fitRect(img);
  const host = document.createElement('div');
  host.setAttribute('data-pagetext', '');
  host.style.cssText =
    `position:absolute;left:${left + window.scrollX}px;top:${top + window.scrollY}px;` +
    `width:${width}px;height:${height}px;z-index:2147483646;overflow:hidden;` +
    'cursor:text;user-select:text;-webkit-user-select:text;';

  const style = document.createElement('style');
  style.textContent =
    '[data-pagetext] span::selection { background: rgba(77,163,255,.45); color: transparent; }' +
    '[data-pagetext] span { color: transparent; position: absolute; white-space: pre; ' +
    'transform-origin: 0 50%; pointer-events: auto; caret-color: transparent; }' +
    '[data-pagetext] .pt-box { position: absolute; border-radius: 3px; pointer-events: none; ' +
    'background: rgba(77,163,255,.18); outline: 1px solid rgba(77,163,255,.55); ' +
    `transition: opacity .5s ease ${FLASH_MS}ms; }` +
    // One in-flow block per logical line, zero height so it moves nothing:
    // a copied selection gets a line break at each block boundary.
    '[data-pagetext] .pt-line { display: block; position: static; font-size: 0; ' +
    'line-height: 0; white-space: pre; }';
  host.appendChild(style);
  // Insert BEFORE building the spans: an element outside the document has no
  // layout, so measuring text in a detached host returns width 0 and every
  // span gets scaled by ~its own box width (spans ran ~50,000 px wide —
  // invisible, since the text is transparent, but selection lands anywhere).
  document.body.appendChild(host);

  // Map natural-relative coords → overlay px, honoring the uv crop (cover).
  const px = (line) => ({
    left: ((line.x - uv.x) / uv.w) * width,
    top: ((line.y - uv.y) / uv.h) * height,
    w: (line.w / uv.w) * width,
    h: (line.h / uv.h) * height,
  });

  const measurer = document.createElement('span');
  measurer.style.cssText =
    'position:absolute;visibility:hidden;white-space:pre;left:-9999px;top:0;';
  host.appendChild(measurer);

  // Each logical line (a groupLines group) gets a .pt-line block that holds
  // its rec windows, with windowSeparator() text between them. A copied
  // selection then reads like Copy all. Boxes and spans stay absolutely
  // positioned against the host.
  for (const group of groupLines(payload.lines)) {
    let row = null;
    let prev = null;
    for (const line of group.pieces) {
      const r = px(line);
      if (r.left + r.w < 0 || r.top + r.h < 0 || r.left > width || r.top > height) continue;
      if (!row) {
        row = document.createElement('div');
        row.className = 'pt-line';
        host.appendChild(row);
      } else {
        const sep = windowSeparator(prev.text, line.text);
        if (sep) row.append(sep);
      }
      prev = line;

      const box = document.createElement('div');
      box.className = 'pt-box';
      box.style.cssText += `left:${r.left}px;top:${r.top}px;width:${r.w}px;height:${r.h}px;`;
      row.appendChild(box);

      const span = document.createElement('span');
      const fontPx = Math.max(6, r.h * 0.82);
      const family = JA_RE.test(line.text)
        ? '"Hiragino Sans","Hiragino Kaku Gothic ProN",sans-serif'
        : '-apple-system,"Helvetica Neue",Arial,sans-serif';
      span.textContent = line.text;
      span.style.font = `${fontPx}px/1 ${family}`;
      span.style.left = `${r.left}px`;
      span.style.top = `${r.top + r.h / 2 - fontPx / 2}px`;
      measurer.style.font = span.style.font;
      measurer.textContent = line.text;
      const natW = measurer.getBoundingClientRect().width || 1;
      span.style.transform = `scaleX(${r.w / natW})`;
      row.appendChild(span);
    }
  }
  measurer.remove();

  // toolbar
  const bar = document.createElement('div');
  bar.style.cssText =
    'position:absolute;right:6px;top:6px;display:flex;gap:6px;z-index:1;' +
    'font:11px/1 -apple-system,system-ui,sans-serif;user-select:none;-webkit-user-select:none;';
  const mkBtn = (label) => {
    const b = document.createElement('button');
    b.textContent = label;
    b.style.cssText =
      'border:0;border-radius:999px;padding:6px 10px;cursor:pointer;font:inherit;' +
      'background:rgba(20,20,24,.88);color:#e8e8ec;box-shadow:0 2px 10px rgba(0,0,0,.35);';
    return b;
  };
  const copyBtn = mkBtn('Copy all');
  copyBtn.addEventListener('click', async (e) => {
    e.stopPropagation();
    const text = joinLines(payload.lines);
    try {
      await navigator.clipboard.writeText(text);
      copyBtn.textContent = 'Copied ✓';
      setTimeout(() => { copyBtn.textContent = 'Copy all'; }, 1200);
    } catch {
      copyBtn.textContent = 'Copy failed';
    }
  });
  const closeBtn = mkBtn('✕');
  closeBtn.addEventListener('click', (e) => { e.stopPropagation(); teardown(); });
  bar.append(copyBtn, closeBtn);
  host.appendChild(bar);

  // flash the boxes, then fade them out (text stays selectable). A rebuild
  // after a resize skips the flash — it already played once.
  if (flash) {
    requestAnimationFrame(() => {
      for (const el of host.querySelectorAll('.pt-box')) el.style.opacity = '0';
    });
  } else {
    for (const el of host.querySelectorAll('.pt-box')) el.style.opacity = '0';
  }

  // Reposition on resize rather than dropping the overlay: the text layer is
  // pure geometry over the image, so a rebuild from the same payload follows
  // the new layout. (Tearing down here also made the overlay vanish whenever
  // anything resized the viewport.)
  let resizeTimer = null;
  const onResize = () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (!session || session.host !== host) return;
      if (!img.isConnected) { teardown(); return; }
      host.remove();
      window.removeEventListener('resize', onResize);
      document.removeEventListener('pointerdown', onPointerDown, true);
      session = null;
      buildOverlay(img, payload, { flash: false });
    }, 150);
  };
  const onPointerDown = (e) => {
    if (!host.contains(e.target)) teardown();
  };
  window.addEventListener('resize', onResize);
  document.addEventListener('pointerdown', onPointerDown, true);
  session = { host, img, payload, onResize, onPointerDown };
}

window.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') teardown();
}, true);

initFind({ send, sourceFor, fitRect });
// The commands API shortcut is the main entry point; this fallback catches
// pages where the browser-level shortcut is taken by the site.
window.addEventListener('keydown', (event) => {
  if (event.altKey && event.shiftKey && (event.key === 'F' || event.key === 'f')) {
    event.preventDefault();
    toggleFind();
  }
}, true);

// debug hook for automated tests
if (__DEV__) {
  window.__ptContent = {
    activateBySrc(src) {
      const img = [...document.images].find((i) => i.currentSrc === src || i.src === src);
      if (img) activate(img);
      return Boolean(img);
    },
    get session() {
      return session
        ? { lines: session.host.querySelectorAll('span').length - 0 }
        : null;
    },
    find: __findDebug,
    toggleFind,
    closeFind,
    scan,
  };
}
