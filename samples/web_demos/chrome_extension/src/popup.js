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
 * Popup: engine status + manual controls. Doubles as the smoke-test panel —
 * the detail line surfaces the WebGPU / threads / JSPI probe results from the
 * offscreen document.
 */

const TEST_SENTENCE =
  'This page can talk now. Everything you hear is generated locally, ' +
  'on this computer, and the text never leaves your device.';

const dot = document.getElementById('dot');
const label = document.getElementById('label');
const detail = document.getElementById('detail');

function send(msg) {
  return chrome.runtime.sendMessage({ target: 'bg', ...msg }).catch(() => null);
}

function render(s) {
  if (!s) return;
  let cls = 'idle';
  let text;
  if (s.state === 'unloaded') {
    text = 'idle — model not loaded yet';
  } else if (s.state === 'error') {
    cls = 'error';
    text = `error — ${s.error ?? 'see offscreen console'}`;
  } else if (s.speaking) {
    cls = 'speaking';
    text = `speaking${s.queued ? ` (+${s.queued} queued)` : ''}`;
  } else if (s.state === 'ready') {
    cls = 'ready';
    text = 'ready';
  } else if (s.state === 'downloading') {
    cls = 'busy';
    text = `downloading model (one-time)… ${s.downloadedMB ?? 0} MB`;
  } else {
    cls = 'busy';
    text = `${s.state}…`;
  }
  dot.className = `dot ${cls}`;
  label.textContent = text;

  const bits = [];
  if (s.env) bits.push(s.env);
  if (s.flags) {
    bits.push(`webgpu ${s.flags.webgpu ? 'yes' : 'no'}`);
    bits.push(`threads ${s.flags.threads ? 'yes' : 'no'}`);
    bits.push(`jspi ${s.flags.jspi ? 'yes' : 'no'}`);
  }
  if (s.stats) bits.push(`RTF ${s.stats.rtf}`);
  detail.textContent = bits.join(' · ');
}

async function refresh() {
  render(await send({ type: 'status' }));
}

// live updates while the popup is open (offscreen broadcasts reach us directly)
chrome.runtime.onMessage.addListener((msg) => {
  if (msg?.target === 'ui' && msg.type === 'status') render(msg);
});

document.getElementById('read').addEventListener('click', () => send({ type: 'read-active' }));
document.getElementById('stop').addEventListener('click', () => send({ type: 'stop' }));
document.getElementById('test').addEventListener('click', () => send({ type: 'speak', text: TEST_SENTENCE }));
document.getElementById('load').addEventListener('click', async () => {
  render(await send({ type: 'preload' }));
});

refresh();
setInterval(refresh, 1000);
