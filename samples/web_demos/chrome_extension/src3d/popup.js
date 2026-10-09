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

/** Popup: enable toggle + engine status (state, env, last latency). */

const dot = document.getElementById('dot');
const detail = document.getElementById('detail');
const env = document.getElementById('env');
const toggleBtn = document.getElementById('toggle');
const loadBtn = document.getElementById('load');

let enabled = true;

function send(msg) {
  return chrome.runtime.sendMessage({ target: 'bg', ...msg }).catch(() => null);
}

function render(s) {
  toggleBtn.classList.toggle('on', enabled);
  toggleBtn.textContent = enabled ? 'Enabled' : 'Disabled';
  if (!s || s.state === 'unloaded') {
    dot.className = 'dot';
    detail.textContent = 'Engine not loaded. Hover an image on any page, or press Load model.';
    env.textContent = '';
    return;
  }
  env.textContent = s.env ?? '';
  if (s.state === 'error') {
    dot.className = 'dot error';
    detail.textContent = `Error: ${s.error}`;
  } else if (s.state === 'ready') {
    dot.className = 'dot ready';
    const st = s.stats;
    detail.textContent = st
      ? `Ready · last depth ${st.inferMs} ms (fetch ${st.fetchMs} ms via ${st.via}) · ${s.runs} runs`
      : 'Ready — hover an image on any page.';
  } else if (s.state === 'downloading') {
    dot.className = 'dot busy';
    detail.textContent = `Downloading MoGe-2 (one-time)… ${s.downloadedMB ?? 0} MB`;
  } else {
    dot.className = 'dot busy';
    detail.textContent = `${s.state}…`;
  }
}

toggleBtn.addEventListener('click', async () => {
  enabled = !enabled;
  await chrome.storage.local.set({ 'p3-enabled': enabled }).catch(() => {});
  render(await send({ type: 'status' }));
});

loadBtn.addEventListener('click', async () => {
  render(await send({ type: 'preload' }));
});

chrome.runtime.onMessage.addListener((msg) => {
  if (msg?.target === 'ui' && msg.type === 'status') render(msg);
});

(async () => {
  enabled = (await chrome.storage.local.get('p3-enabled').catch(() => null))?.['p3-enabled'] ?? true;
  render(await send({ type: 'status' }));
})();
