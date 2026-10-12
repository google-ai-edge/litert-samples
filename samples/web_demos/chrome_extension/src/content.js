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
 * Content script for ChatGPT, Claude and Gemini: Auto-read speaks an
 * assistant reply sentence by sentence while it streams in. A small HUD at
 * the bottom right holds the switch (kept per site), the engine status and
 * Stop.
 *
 * A MutationObserver scans the newest assistant message. A message counts as
 * read when it is first seen, so history and other conversations stay
 * silent. Once it grows, it is a reply being streamed and is read from its
 * start: each complete sentence as it arrives, the rest when the stream has
 * been quiet for QUIET_FLUSH_MS.
 */

// The three sites' assistant-message selectors; a site redesign may need an
// update here.
const SELECTORS = new Map([
  ['chatgpt.com', '[data-message-author-role="assistant"]'],
  ['claude.ai', '[data-testid="assistant-message"], .font-claude-message'],
  ['gemini.google.com', 'message-content'],
]);
// A build made with --test-host also runs on that host (tools/autoread-e2e.mjs),
// where each test page names the site whose markup it copies: ?site=claude.ai.
const site = __TEST_HOST__ && location.hostname === __TEST_HOST__
  ? new URLSearchParams(location.search).get('site')
  : location.hostname;
const selector = SELECTORS.get(site);
const AUTOREAD_KEY = `autoread:${location.hostname}`;

// A sentence ends at . ! ? or … (and closing quotes or brackets) followed by
// whitespace, so "3.5" stays whole.
const BOUNDARY = /[.!?…]["')\]]*\s/g;
const SCAN_DEBOUNCE_MS = 300;
const QUIET_FLUSH_MS = 2000;

let autoRead = false;
let observer = null;
let tracked = new WeakMap(); // message element → { said, live, timer }
let newest = null; // { el, st } of the message scanned last
let lastStatus = null;
let port = null;

// --- messaging ---------------------------------------------------------------

function send(msg) {
  connect();
  try {
    return chrome.runtime.sendMessage({ target: 'bg', ...msg }).catch(() => null);
  } catch {
    return Promise.resolve(null); // the extension was reloaded or removed
  }
}

// Content scripts get no runtime broadcasts, so the engine's status comes over
// a port that the service worker passes it on to. The port closes when the
// worker stops and is opened again with the next message.
function connect() {
  if (port) return;
  try {
    port = chrome.runtime.connect({ name: 'hud' });
  } catch {
    return;
  }
  port.onMessage.addListener(showStatus);
  port.onDisconnect.addListener(() => { port = null; });
}

function showStatus(s) {
  if (!s?.state) return;
  lastStatus = s;
  updateHud();
}

// --- Auto-read ---------------------------------------------------------------

function setAutoRead(on) {
  if (on === autoRead) return;
  autoRead = on;
  if (on) {
    // Start over: every message on the page counts as read until it grows.
    tracked = new WeakMap();
    newest = null;
    observer = new MutationObserver(scheduleScan);
    observer.observe(document.body, { childList: true, subtree: true, characterData: true });
    send({ type: 'preload' }).then(showStatus); // starts a first-time model download
  } else {
    observer?.disconnect();
    observer = null;
  }
  updateHud();
}

// The first change after a quiet spell is scanned at once, so a new reply is
// first seen while it is still short; later ones at most every
// SCAN_DEBOUNCE_MS.
let scanTimer = 0;
function scheduleScan() {
  if (scanTimer) return;
  scan();
  scanTimer = setTimeout(() => {
    scanTimer = 0;
    scan();
  }, SCAN_DEBOUNCE_MS);
}

// Whitespace is collapsed, so a re-render that only changes the markup (a
// paragraph break for a space) leaves the text, and the read position, as is.
const textOf = (el) => (el.innerText ?? '').replace(/\s+/g, ' ').trimStart();

function scan() {
  const all = document.querySelectorAll(selector);
  const el = all[all.length - 1];
  if (!el) return;
  const text = textOf(el);
  let st = tracked.get(el);
  if (!st) {
    // A new node for the message scanned last (a re-render) takes over its
    // state; any other message counts as read so far.
    const prev = newest?.st.said && !newest.el.isConnected && text.startsWith(newest.st.said)
      ? newest.st : null;
    st = { said: prev?.said ?? text, live: prev?.live ?? false, timer: 0 };
    tracked.set(el, st);
  }
  newest = { el, st };
  if (!text.startsWith(st.said)) {
    st.said = text; // rewritten rather than extended: skip it, never read twice
    return;
  }
  if (text.length === st.said.length) return;
  if (!st.live) {
    st.live = true; // it grows, so it is a reply being streamed: read it all
    st.said = '';
  }
  const fresh = text.slice(st.said.length);
  let end = 0;
  for (const m of fresh.matchAll(BOUNDARY)) {
    speak(fresh.slice(end, m.index + m[0].length));
    end = m.index + m[0].length;
  }
  st.said += fresh.slice(0, end);
  clearTimeout(st.timer);
  st.timer = setTimeout(() => flush(el, st), QUIET_FLUSH_MS);
}

/** Reads what is left once the stream has been quiet for QUIET_FLUSH_MS. */
function flush(el, st) {
  if (!autoRead || !el.isConnected) return; // switched off, or re-rendered
  const text = textOf(el);
  if (!text.startsWith(st.said) || text.length === st.said.length) return;
  speak(text.slice(st.said.length));
  st.said = text;
}

function speak(text) {
  text = text.trim();
  if (text) send({ type: 'speak', text });
}

// --- HUD ---------------------------------------------------------------------

let hud = null;

function createHud() {
  const host = document.createElement('div');
  host.setAttribute('data-page-voice', '');
  host.style.cssText = 'position:fixed;right:16px;bottom:16px;z-index:2147483647;';
  const root = host.attachShadow({ mode: 'open' });
  root.innerHTML = `
    <style>
      .pill { display:flex; align-items:center; gap:8px; padding:8px 12px; border-radius:999px;
        font:12px/1.4 -apple-system, system-ui, sans-serif; color:#e8e8ec;
        background:rgba(20,20,24,.92); box-shadow:0 4px 16px rgba(0,0,0,.35); }
      .dot { width:8px; height:8px; border-radius:50%; background:#7a7a85; flex:none; }
      .dot.ready { background:#3ecf6e; }
      .dot.busy { background:#f0b62f; }
      .dot.speaking { background:#4da3ff; }
      .dot.error { background:#ff5d5d; }
      button { font:inherit; border:0; border-radius:999px; padding:2px 10px; cursor:pointer;
        background:#33333d; color:#e8e8ec; }
      button.on { background:#4da3ff; color:#0b0b10; }
    </style>
    <div class="pill">
      <span class="dot" id="dot"></span><span id="label"></span>
      <button id="auto">Auto-read</button><button id="stop" hidden>Stop</button>
    </div>`;
  root.getElementById('auto').addEventListener('click', () => {
    setAutoRead(!autoRead);
    chrome.storage.local.set({ [AUTOREAD_KEY]: autoRead }).catch(() => {});
  });
  root.getElementById('stop').addEventListener('click', () => send({ type: 'stop' }));
  document.documentElement.append(host);
  hud = {
    dot: root.getElementById('dot'),
    label: root.getElementById('label'),
    auto: root.getElementById('auto'),
    stop: root.getElementById('stop'),
  };
}

function updateHud() {
  const s = lastStatus;
  let cls = 'idle';
  let text = 'Page Voice';
  if (s?.state === 'error') {
    cls = 'error';
    text = 'error — see the popup';
  } else if (s?.speaking) {
    cls = 'speaking';
    text = 'speaking';
  } else if (s?.state === 'ready') {
    cls = 'ready';
    text = 'ready';
  } else if (s?.state === 'downloading') {
    cls = 'busy';
    text = `downloading model (one-time)… ${s.downloadedMB ?? 0} MB`;
  } else if (s && s.state !== 'unloaded') {
    cls = 'busy';
    text = `${s.state}…`;
  }
  hud.dot.className = `dot ${cls}`;
  hud.label.textContent = text;
  hud.auto.classList.toggle('on', autoRead);
  hud.stop.hidden = !s?.speaking;
}

// --- init --------------------------------------------------------------------

if (selector) {
  createHud();
  updateHud();
  chrome.storage.local.get(AUTOREAD_KEY)
    .then((v) => setAutoRead(v[AUTOREAD_KEY] === true))
    .catch(() => {});
  // The switch is per site: a click in one tab applies to the site's other tabs.
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area === 'local' && AUTOREAD_KEY in changes) {
      setAutoRead(changes[AUTOREAD_KEY].newValue === true);
    }
  });
}
