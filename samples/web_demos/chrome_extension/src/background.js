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
 * Service worker: context menu + keyboard commands, offscreen-document
 * lifecycle, and message routing.
 *
 * Routing map:
 *   popup → {target:'bg', ...}                → handled here
 *   here → {target:'offscreen', ...}          → TTS engine (created on demand)
 *   offscreen → {target:'ui', type:'status'}  → popup directly.
 *
 * No content script: the selection is read with scripting.executeScript
 * under the activeTab grant that Alt+R, the menu and the popup give.
 */

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'read-selection',
    title: 'Read aloud',
    contexts: ['selection'],
  });
});

// Dev build only: keep the engine warm from browser start — no cold start in
// demos, and it lets automated smoke tests reach the engine without UI.
// Without __DEV__ the offscreen document is created on first use instead.
if (__DEV__) ensureOffscreen().catch(() => {});

// --- offscreen lifecycle -----------------------------------------------------

async function ensureOffscreen() {
  if (await chrome.offscreen.hasDocument()) return;
  try {
    await chrome.offscreen.createDocument({
      url: 'offscreen.html',
      reasons: ['AUDIO_PLAYBACK', 'BLOBS'],
      justification:
        'Runs the local TTS model and plays the synthesized speech; ' +
        'model files are held as blobs in the Cache API.',
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

async function stopSpeaking() {
  if (!(await chrome.offscreen.hasDocument())) return;
  try {
    await chrome.runtime.sendMessage({ target: 'offscreen', type: 'stop' });
  } catch { /* nothing to stop */ }
}

// --- selection reading -------------------------------------------------------

/** Full selection via scripting (activeTab is granted by the menu/command/
 *  popup invocation); falls back to the possibly-truncated menu text. */
async function getSelectionText(tabId, fallback) {
  try {
    const [res] = await chrome.scripting.executeScript({
      target: { tabId },
      func: () => String(getSelection()),
    });
    const text = res?.result?.trim();
    if (text) return text;
  } catch { /* restricted page (chrome://, store, …) */ }
  return (fallback ?? '').trim();
}

async function readSelection(tabId, fallback) {
  const text = await getSelectionText(tabId, fallback);
  if (text) await sendToOffscreen({ type: 'speak', text });
}

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === 'read-selection' && tab?.id) {
    readSelection(tab.id, info.selectionText);
  }
});

chrome.commands.onCommand.addListener((command, tab) => {
  if (command === 'read-selection' && tab?.id) readSelection(tab.id);
  else if (command === 'stop-speaking') stopSpeaking();
});

// --- message router ------------------------------------------------------------

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (!msg || msg.target !== 'bg') return;
  switch (msg.type) {
    case 'speak':
      sendToOffscreen({ type: 'speak', text: msg.text }).then(() => sendResponse({ ok: true }));
      return true;
    case 'stop':
      stopSpeaking().then(() => sendResponse({ ok: true }));
      return true;
    case 'read-active':
      chrome.tabs.query({ active: true, currentWindow: true }).then(([tab]) => {
        if (tab?.id) readSelection(tab.id);
        sendResponse({ ok: true });
      });
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
