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
 * Page 3D service worker: offscreen-engine lifecycle, message routing, and
 * the privileged image-fetch relay.
 *
 * Routing map:
 *   content → {target:'bg', type:'depth', url}   → offscreen engine (MoGe)
 *   content/popup → {target:'bg', type:'status'} → engine status
 *   offscreen → {target:'bg', type:'fetch-image'} → fetched here (the service
 *     worker has no COEP, so it can fetch images from hosts that send neither
 *     CORS nor CORP headers) and returned as base64.
 *   context menu "See in 3D" on an image → content script of that tab.
 */

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'see-in-3d',
    title: 'See in 3D',
    contexts: ['image'],
  });
});

// Dev build only: keep the engine warm from browser start (no cold start in
// demos; lets automated smoke tests reach the engine without UI).
if (__DEV__) ensureOffscreen().catch(() => {});

// --- offscreen lifecycle -----------------------------------------------------

async function ensureOffscreen() {
  if (await chrome.offscreen.hasDocument()) return;
  try {
    await chrome.offscreen.createDocument({
      url: 'offscreen.html',
      reasons: ['BLOBS'],
      justification:
        'Runs the local monocular-depth model on image bytes; model files ' +
        'are held as blobs in the Cache API.',
    });
  } catch (err) {
    // Concurrent createDocument calls race; "only a single offscreen
    // document" just means someone else won.
    if (!String(err).toLowerCase().includes('single offscreen')) throw err;
  }
}

async function sendToOffscreen(msg) {
  await ensureOffscreen();
  let lastErr = null;
  for (let i = 0; i < 4; i++) {
    try {
      // An open popup receives runtime messages too but never answers them,
      // so no answer means the offscreen document is not listening yet.
      const response = await chrome.runtime.sendMessage({ target: 'offscreen', ...msg });
      if (response !== undefined) return response;
    } catch (err) {
      lastErr = err; // document still loading — its listener registers first thing
    }
    await new Promise((r) => setTimeout(r, 150 * (i + 1)));
  }
  console.warn('offscreen unreachable:', lastErr);
  return null;
}

// --- privileged fetch relay ----------------------------------------------------

function bytesToBase64(bytes) {
  let out = '';
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    out += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
  }
  return btoa(out);
}

async function fetchImageB64(url) {
  const response = await fetch(url, { credentials: 'omit' });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  return {
    ok: true,
    b64: bytesToBase64(bytes),
    contentType: response.headers.get('content-type') ?? '',
  };
}

// --- context menu ---------------------------------------------------------------

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === 'see-in-3d' && tab?.id && info.srcUrl) {
    chrome.tabs.sendMessage(tab.id, { type: 'activate-3d', srcUrl: info.srcUrl })
      .catch(() => {});
  }
});

// --- message router --------------------------------------------------------------

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (!msg || msg.target !== 'bg') return;
  switch (msg.type) {
    case 'depth':
      sendToOffscreen({ type: 'depth', url: msg.url })
        .then((r) => sendResponse(r ?? { ok: false, error: 'engine unreachable' }));
      return true;
    case 'fetch-image':
      fetchImageB64(msg.url)
        .then(sendResponse)
        .catch((err) => sendResponse({ ok: false, error: String(err?.message ?? err) }));
      return true;
    case 'preload': // create the engine + kick the model download
      sendToOffscreen({ type: 'status' }).then((s) => sendResponse(s ?? { state: 'unloaded' }));
      return true;
    case 'status':
      (async () => {
        if (!(await chrome.offscreen.hasDocument())) {
          sendResponse({ state: 'unloaded' });
          return;
        }
        try {
          sendResponse(await chrome.runtime.sendMessage({ target: 'offscreen', type: 'status' }));
        } catch {
          sendResponse({ state: 'unloaded' });
        }
      })();
      return true;
  }
});
