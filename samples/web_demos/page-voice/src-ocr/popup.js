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

/** Popup: engine status (state, env, last OCR latency). */

const dot = document.getElementById('dot');
const detail = document.getElementById('detail');
const env = document.getElementById('env');
const loadBtn = document.getElementById('load');

function send(msg) {
  return chrome.runtime.sendMessage({ target: 'bg', ...msg }).catch(() => null);
}

function render(s) {
  if (!s || s.state === 'unloaded') {
    dot.className = 'dot';
    detail.textContent =
      'Engine not loaded. Right-click an image → "Select text in this image", or press Load models.';
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
      ? `Ready · last read ${st.lineCount} lines in det ${st.detMs} + rec ${st.recMs} ms · ${s.runs} runs`
      : 'Ready — right-click an image → "Select text in this image".';
  } else if (s.state === 'downloading') {
    dot.className = 'dot busy';
    detail.textContent = `Downloading PP-OCRv5 (one-time)… ${s.downloadedMB ?? 0} MB`;
  } else {
    dot.className = 'dot busy';
    detail.textContent = `${s.state}…`;
  }
}

loadBtn.addEventListener('click', async () => {
  render(await send({ type: 'preload' }));
});

chrome.runtime.onMessage.addListener((msg) => {
  if (msg?.target === 'ui' && msg.type === 'status') render(msg);
});

(async () => {
  render(await send({ type: 'status' }));
})();
