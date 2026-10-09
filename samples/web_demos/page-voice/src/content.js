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
 * Content script for AI chat sites (ChatGPT / Claude / Gemini): watches the
 * streaming assistant reply and reads it aloud sentence-by-sentence as it
 * arrives, plus a small status HUD.
 *
 * Watcher: MutationObserver over the page; per assistant-message element we
 * track how much text has been consumed, emit up to the last complete
 * sentence boundary on each (debounced) scan, and flush the tail once the
 * stream goes quiet.
 */

const ADAPTERS = [
  { host: /(^|\.)chatgpt\.com$/, selector: '[data-message-author-role="assistant"]' },
  // claude.ai and gemini DOM shift often — selectors are best-effort, verify live.
  { host: /(^|\.)claude\.ai$/, selector: '[data-testid="assistant-message"], .font-claude-message' },
  { host: /(^|\.)gemini\.google\.com$/, selector: 'message-content' },
  // dev: synthetic stream pages (tools/stream-test.html) use the ChatGPT shape
  { host: /^(127\.0\.0\.1|localhost)$/, selector: '[data-message-author-role="assistant"]' },
];
const adapter = ADAPTERS.find((a) => a.host.test(location.hostname));
const AUTOREAD_KEY = `autoread:${location.hostname}`;

// sentence end + closing quotes/brackets, followed by whitespace; the
// trailing-whitespace requirement keeps decimals ("3.5") intact and the
// quiet-flush below picks up the final unterminated tail.
const BOUNDARY = /[.!?…。!?]["')\]]*\s/g;
const SCAN_DEBOUNCE_MS = 300;
const QUIET_FLUSH_MS = 2000;

let autoRead = false;
let observer = null;
const tracked = new WeakMap(); // element → {consumed, timer}
let lastStatus = null;

// --- messaging ---------------------------------------------------------------

function send(msg) {
  return chrome.runtime.sendMessage({ target: 'bg', ...msg }).catch(() => null);
}

let port = null;
function connectPort() {
  try {
    port = chrome.runtime.connect({ name: 'ui' });
  } catch {
    return; // extension reloaded/uninstalled
  }
  port.onMessage.addListener((msg) => {
    if (msg?.type === 'status') {
      lastStatus = msg;
      updateHud();
    }
  });
  port.onDisconnect.addListener(() => {
    port = null;
    setTimeout(connectPort, 1000); // service worker restart
  });
}
connectPort();

// --- streaming watcher ---------------------------------------------------------

function setAutoRead(on) {
  autoRead = on;
  chrome.storage.local.set({ [AUTOREAD_KEY]: on }).catch(() => {});
  if (on) {
    startWatch();
    send({ type: 'preload' }); // start the one-time model download early
  }
  updateHud();
}

function startWatch() {
  if (observer || !adapter) return;
  // everything already on the page counts as consumed — only new text is read
  for (const el of document.querySelectorAll(adapter.selector)) {
    tracked.set(el, { consumed: (el.innerText || '').length, timer: null });
  }
  observer = new MutationObserver(scheduleScan);
  observer.observe(document.body, { childList: true, subtree: true, characterData: true });
}

let scanScheduled = false;
function scheduleScan() {
  if (scanScheduled) return;
  scanScheduled = true;
  setTimeout(() => {
    scanScheduled = false;
    scan();
  }, SCAN_DEBOUNCE_MS);
}

function scan() {
  if (!autoRead || !adapter) return;
  const els = document.querySelectorAll(adapter.selector);
  els.forEach((el, i) => {
    let st = tracked.get(el);
    if (!st) {
      // Unseen element: read it from the start only if it's the newest
      // message; older ones (e.g. after a React re-mount) stay silent.
      const isLast = i === els.length - 1;
      st = { consumed: isLast ? 0 : (el.innerText || '').length, timer: null };
      tracked.set(el, st);
    }
    const text = el.innerText || '';
    if (text.length <= st.consumed) return;

    const fresh = text.slice(st.consumed);
    BOUNDARY.lastIndex = 0;
    let lastEnd = -1;
    for (let m; (m = BOUNDARY.exec(fresh)) !== null;) lastEnd = m.index + m[0].length;
    if (lastEnd > 0) {
      const out = fresh.slice(0, lastEnd).trim();
      st.consumed += lastEnd;
      if (out) send({ type: 'speak', text: out });
    }
    // flush whatever remains once the stream stops growing
    clearTimeout(st.timer);
    st.timer = setTimeout(() => {
      const t = el.innerText || '';
      if (t.length > st.consumed) {
        const out = t.slice(st.consumed).trim();
        st.consumed = t.length;
        if (out) send({ type: 'speak', text: out });
      }
    }, QUIET_FLUSH_MS);
  });
}

// --- HUD -----------------------------------------------------------------------

let hud = null;
function ensureHud() {
  if (hud) return hud;
  const hostEl = document.createElement('div');
  hostEl.style.cssText = 'position:fixed;bottom:16px;right:16px;z-index:2147483647;';
  const root = hostEl.attachShadow({ mode: 'open' });
  root.innerHTML = `
    <style>
      .pill { display:flex; align-items:center; gap:8px; font:12px/1.4 -apple-system, system-ui, sans-serif;
        background:rgba(20,20,24,.92); color:#e8e8ec; padding:8px 12px; border-radius:999px;
        box-shadow:0 4px 16px rgba(0,0,0,.35); }
      .dot { width:8px; height:8px; border-radius:50%; background:#7a7a85; flex:none; }
      .dot.ready { background:#3ecf6e; }
      .dot.busy { background:#f0b62f; }
      .dot.speaking { background:#4da3ff; animation:pulse 1s infinite; }
      .dot.error { background:#ff5d5d; }
      @keyframes pulse { 50% { opacity:.35; } }
      button { font:inherit; border:0; border-radius:999px; padding:2px 10px; cursor:pointer;
        background:#33333d; color:#e8e8ec; }
      button.on { background:#4da3ff; color:#0b0b10; }
      #stop { display:none; }
    </style>
    <div class="pill">
      <span class="dot" id="dot"></span><span id="label">voice</span>
      <button id="auto">Auto-read</button><button id="stop">Stop</button>
    </div>`;
  root.getElementById('auto').addEventListener('click', () => setAutoRead(!autoRead));
  root.getElementById('stop').addEventListener('click', () => send({ type: 'stop' }));
  document.documentElement.appendChild(hostEl);
  hud = {
    dot: root.getElementById('dot'),
    label: root.getElementById('label'),
    auto: root.getElementById('auto'),
    stop: root.getElementById('stop'),
  };
  return hud;
}

function updateHud() {
  const h = ensureHud();
  h.auto.classList.toggle('on', autoRead);
  const s = lastStatus;
  let dot = 'idle';
  let label = autoRead ? 'voice armed' : 'voice off';
  if (s && s.state !== 'unloaded') {
    if (s.state === 'error') {
      dot = 'error';
      label = `error — ${s.error ?? 'see console'}`;
    } else if (s.speaking) {
      dot = 'speaking';
      label = s.stats ? `speaking · RTF ${s.stats.rtf} · ${s.env}` : 'speaking…';
    } else if (s.state === 'ready') {
      dot = 'ready';
      label = `ready · ${s.env}`;
    } else if (s.state === 'downloading') {
      dot = 'busy';
      label = `downloading model (one-time)… ${s.downloadedMB ?? 0} MB`;
    } else {
      dot = 'busy';
      label = `${s.state}…`;
    }
  }
  h.dot.className = `dot ${dot}`;
  h.label.textContent = label;
  h.stop.style.display = s?.speaking ? 'inline-block' : 'none';
}

// --- init ------------------------------------------------------------------------

chrome.storage.local.get(AUTOREAD_KEY).then((v) => {
  if (v?.[AUTOREAD_KEY]) setAutoRead(true);
  else updateHud();
}).catch(() => updateHud());
