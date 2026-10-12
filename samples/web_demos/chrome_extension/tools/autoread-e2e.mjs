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
 * Auto-read E2E over CDP: serves local chat pages with each site's
 * assistant-message markup (ChatGPT, Claude, Gemini), turns Auto-read on with
 * the HUD's switch, streams a three-sentence reply into the page a few
 * characters at a time for ~3 s, and checks that the engine got the three
 * sentences once each, in order. A listener added to the offscreen document
 * records every speak message the engine gets.
 *
 *   node build.mjs --test-host=127.0.0.1
 *   node tools/autoread-e2e.mjs <chrome-binary> [--profile=<dir>] [--case=<name>]
 *
 * --case runs one failure path instead:
 *   off          Auto-read is off: nothing is read.
 *   no-selector  the page lacks the site's message markup: nothing is read.
 *   not-ready    the reply streams in while the model is still downloading
 *                (a throwaway profile; the engine is restarted with its
 *                download throttled until the reply is in): it is read once
 *                the model is ready.
 *   reload       the page reloads mid-reply: nothing more is read until a new
 *                reply streams in, and that one is read once.
 *   redraw       while it streams, the reply is re-rendered as a new node
 *                (mid-sentence) and, at its end, as paragraphs and as a new
 *                node again; once done, three more ways (same markup,
 *                paragraphs, a new node): each sentence is read once.
 * Every run also fails on an exception in the page, its content script, the
 * service worker or the offscreen document. Exit code 0 = PASS.
 */
import { readFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { resolve } from 'node:path';
import {
  Cdp, attachTo, evalIn, findTarget, launchChrome, sleep, waitForEndpoint, waitForEngine,
} from './cdp.mjs';

const CASES = ['off', 'no-selector', 'not-ready', 'reload', 'redraw'];
const chromeBin = process.argv[2];
const caseName = process.argv.find((a) => a.startsWith('--case='))?.slice(7);
let profile = process.argv.find((a) => a.startsWith('--profile='))?.slice(10);
if (!chromeBin || (caseName !== undefined && !CASES.includes(caseName))) {
  console.error('usage: node tools/autoread-e2e.mjs <chrome-binary> [--profile=<dir>] ' +
    `[--case=${CASES.join('|')}]`);
  process.exit(2);
}
const dist = resolve(import.meta.dirname, '..', 'dist');
const manifest = JSON.parse(readFileSync(resolve(dist, 'manifest.json'), 'utf8'));
if (!manifest.content_scripts?.[0]?.matches.includes('http://127.0.0.1/*')) {
  console.error('dist/ does not run on the test pages: build it with ' +
    '`node build.mjs --test-host=127.0.0.1`');
  process.exit(2);
}
if (caseName === 'not-ready' && profile) {
  console.log('not-ready: ignoring --profile (the model must still be downloading)');
  profile = undefined;
}

const SENTENCES = [
  'The model runs in your browser, so the reply never leaves this computer.',
  'Can you hear each sentence as soon as it is complete?',
  'Numbers like 3.5 stay whole, and the last sentence is read when the stream goes quiet.',
];
const HISTORY = 'This reply was already here. It is not read aloud.';
const STREAM_MS = 3000;
// src/content.js reads the tail 2 s after the last change; scans run every 0.3 s.
const QUIET_MS = 3500;

// Site and reply markup per page: what that site's selector in src/content.js
// finds. "plain" has none of it.
const PAGES = {
  chatgpt: ['chatgpt.com', '<div data-message-author-role="assistant" data-e2e>' +
    '<div class="markdown"><p></p></div></div>'],
  claude: ['claude.ai', '<div data-testid="assistant-message" data-e2e>' +
    '<div class="font-claude-message"><p></p></div></div>'],
  gemini: ['gemini.google.com', '<model-response><message-content data-e2e>' +
    '<div class="markdown"><p></p></div></message-content></model-response>'],
  plain: ['chatgpt.com', '<div class="reply" data-e2e><p></p></div>'],
};

const pageHtml = (reply) => `<!doctype html><meta charset="utf-8"><title>Auto-read e2e</title>
<body style="margin:40px;max-width:640px;font:16px/1.5 sans-serif">
<main id="chat"><p>An earlier question</p></main>
<template id="reply">${reply}</template>
<script>
  const chat = document.getElementById('chat');
  function addReply() {
    chat.append(document.getElementById('reply').content.cloneNode(true));
    return chat.lastElementChild.querySelector('p');
  }
  // The earlier reply arrives after load, as on a site that fetches the conversation.
  setTimeout(() => {
    addReply().textContent = ${JSON.stringify(HISTORY)};
    window.historyShown = true;
  }, 500);
  const newest = () => [...document.querySelectorAll('[data-e2e]')].at(-1);
  // Re-renders the newest reply: the same markup again, a paragraph per
  // sentence (of the whole reply), or a copy as a new node. Returns the text
  // node that now holds the end of the reply.
  function rerender(how, sentences) {
    let msg = newest();
    if (how === 'same') msg.innerHTML = msg.innerHTML;
    if (how === 'paragraphs') {
      msg.querySelector('p').parentElement.innerHTML = sentences.map((s) => '<p>' + s + '</p>').join('');
    }
    if (how === 'node') {
      const copy = msg.cloneNode(true);
      msg.replaceWith(copy);
      msg = copy;
    }
    return [...msg.querySelectorAll('p')].at(-1).firstChild;
  }
  // Streams the sentences into a new reply, one to three characters at a time;
  // rerenders lists [fraction of the text, how] to re-render it on the way.
  window.stream = (sentences, ms, rerenders = []) => new Promise((done) => {
    const text = sentences.join(' ');
    const question = document.createElement('p');
    question.textContent = 'Another question';
    chat.append(question);
    let node = document.createTextNode('');
    addReply().append(node);
    const marks = rerenders.map(([f, how]) => [Math.round(f * text.length), how]);
    let i = 0;
    const timer = setInterval(() => {
      i = Math.min(text.length, i + 1 + (i % 3));
      node.data = text.slice(0, i);
      while (marks.length && i >= marks[0][0]) node = rerender(marks.shift()[1], sentences);
      if (i === text.length) {
        clearInterval(timer);
        done();
      }
    }, (2 * ms) / text.length);
  });
  window.redraw = (how, sentences) => {
    rerender(how, sentences);
    return newest().innerText.replace(/\\s+/g, ' ').trim();
  };
</script>`;

const server = createServer((req, res) => {
  const name = new URL(req.url, 'http://localhost').pathname.slice(1);
  if (!PAGES[name]) {
    res.writeHead(404).end();
    return;
  }
  res.writeHead(200, { 'Content-Type': 'text/html' }).end(pageHtml(PAGES[name][1]));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;
const urlOf = (name) => `${base}/${name}?site=${PAGES[name][0]}`;

const port = 9226;
launchChrome(chromeBin, {
  dist,
  port,
  profile,
  url: urlOf(caseName === 'no-selector' ? 'plain' : 'chatgpt'),
});

const cdp = await Cdp.connect(await waitForEndpoint(port));
const errors = [];
const sessionNames = new Map();
cdp.ws.addEventListener('message', (ev) => {
  const { method, params, sessionId } = JSON.parse(ev.data);
  const where = sessionNames.get(sessionId);
  if (method === 'Runtime.exceptionThrown') {
    const d = params.exceptionDetails;
    errors.push(`${where}: ${d.exception?.description ?? d.text}`);
  } else if (method === 'Runtime.consoleAPICalled' && params.type === 'error' && where === 'page') {
    errors.push(`page console: ${params.args.map((a) => a.value ?? a.description).join(' ')}`);
  }
});
async function watch(session, name) {
  sessionNames.set(session, name);
  await cdp.send('Runtime.enable', {}, session);
}

async function waitFor(fn, what, ms = 15000) {
  const deadline = Date.now() + ms;
  for (;;) {
    let value = null;
    try {
      value = await fn();
    } catch { /* the page is navigating */ }
    if (value) return value;
    if (Date.now() > deadline) throw new Error(`timed out waiting for ${what}`);
    await sleep(200);
  }
}

let engine = null;
let page = null;
const HUD = "document.querySelector('[data-page-voice]')?.shadowRoot";
const hud = () => evalIn(cdp, page, `(() => {
  const root = ${HUD};
  return root && {
    on: root.getElementById('auto').classList.contains('on'),
    label: root.getElementById('label').textContent,
  };
})()`);
const heard = async () => JSON.parse(await evalIn(cdp, engine, 'JSON.stringify(__heard)'));
const status = async () => JSON.parse(await evalIn(cdp, engine, 'JSON.stringify(__pv.status)'));
const throttle = (session, bytesPerSecond) => cdp.send('Network.emulateNetworkConditions', {
  offline: false, latency: 0, downloadThroughput: bytesPerSecond, uploadThroughput: -1,
}, session);

/** Uses this offscreen document as the engine, recording every speak message it gets. */
async function useEngine(session) {
  engine = session;
  await waitFor(() => evalIn(cdp, engine, "typeof __pv !== 'undefined'"), 'the engine');
  await watch(engine, 'offscreen');
  await evalIn(cdp, engine, `(() => {
    globalThis.__heard = [];
    chrome.runtime.onMessage.addListener((msg) => {
      if (msg?.target === 'offscreen' && msg.type === 'speak') {
        __heard.push({ text: msg.text, state: __pv.status.state });
      }
    });
  })()`);
}

/** Closes the engine; the next one has its model download (~94 MB) throttled
 *  to 256 KB/s from its first request: new targets wait for the debugger until
 *  the throttle is on. `next` resolves to the new engine's session. */
async function restartEngineThrottled(worker) {
  const next = new Promise((resolve) => {
    cdp.ws.addEventListener('message', async (ev) => {
      const { method, params } = JSON.parse(ev.data);
      if (method !== 'Target.attachedToTarget' || !params.waitingForDebugger) return;
      // A new offscreen document has no URL yet when it waits: throttle every
      // new target, then keep the one that turns out to be the engine.
      const session = params.sessionId;
      await cdp.send('Network.enable', {}, session).catch(() => {});
      await throttle(session, 256 * 1024).catch(() => {});
      await cdp.send('Runtime.runIfWaitingForDebugger', {}, session).catch(() => {});
      const isEngine = await waitFor(() => evalIn(cdp, session, "location.pathname === '/offscreen.html'"),
        'a new target to load', 5000).catch(() => false);
      if (isEngine) resolve(session);
    });
  });
  await cdp.send('Target.setAutoAttach', { autoAttach: true, waitForDebuggerOnStart: true, flatten: true });
  await evalIn(cdp, worker, 'chrome.offscreen.closeDocument()', true);
  return { next };
}

async function setAutoRead(on) {
  const state = await waitFor(hud, 'the HUD');
  if (state.on !== on) await evalIn(cdp, page, `${HUD}.getElementById('auto').click()`);
  await waitFor(async () => (await hud())?.on === on, `Auto-read ${on ? 'on' : 'off'}`);
}

/** Loads a test page (or waits for the current one) until its content script
 *  and its earlier reply are up. */
async function open(name) {
  if (name) await cdp.send('Page.navigate', { url: urlOf(name) }, page);
  await waitFor(() => evalIn(cdp, page, `location.pathname === '/${name ?? 'chatgpt'}' &&
    window.historyShown === true`), 'the test page');
  await waitFor(hud, 'the HUD');
  await sleep(300); // the switch is read from storage as the content script starts
}

const streamReply = (rerenders = []) => evalIn(cdp, page,
  `stream(${JSON.stringify(SENTENCES)}, ${STREAM_MS}, ${JSON.stringify(rerenders)})`, true, 30000);

function expectSentences(got, what) {
  const texts = got.map((h) => h.text);
  if (JSON.stringify(texts) !== JSON.stringify(SENTENCES)) {
    throw new Error(`${what}: the engine got ${JSON.stringify(texts)}`);
  }
}

/** Waits until the engine has spoken everything it was given. */
const drained = (ms = 60000) => waitFor(async () => {
  const s = await status();
  return s.state === 'ready' && !s.speaking && s.queued === 0 && s;
}, 'the engine to finish speaking', ms);

async function expectNothingRead(what) {
  const before = (await heard()).length;
  const spoken = (await status()).spoken;
  await streamReply();
  await sleep(QUIET_MS);
  const s = await status();
  const got = (await heard()).slice(before);
  if (got.length || s.spoken !== spoken || s.error) {
    throw new Error(`${what}: the engine got ${JSON.stringify(got)}, error ${s.error}`);
  }
  console.log(`${what}: nothing read (spoken ${s.spoken}, error null)`);
}

let pass = false;
try {
  const offscreen = await findTarget(cdp, (t) => t.url.includes('/offscreen.html'));
  const extensionId = new URL(offscreen.url).host;
  const worker = await attachTo(cdp, await findTarget(cdp, (t) =>
    t.type === 'service_worker' && t.url === `chrome-extension://${extensionId}/background.js`));
  await watch(worker, 'service worker');
  if (caseName !== 'not-ready') {
    const { status: boot, session } = await waitForEngine(cdp);
    if (boot?.state !== 'ready') throw new Error(`engine: ${JSON.stringify(boot)}`);
    await useEngine(session);
  }

  page = await attachTo(cdp, await findTarget(cdp, (t) => t.type === 'page' && t.url.startsWith(base)));
  await watch(page, 'page');
  await cdp.send('Page.enable', {}, page);
  // The page opened with the browser can load before the extension has
  // registered its content script, and then never gets one: reload it.
  await cdp.send('Page.reload', {}, page);

  if (caseName === undefined) {
    // Each site's markup in turn; the switch is kept per site (here, per test host).
    const labels = new Set();
    const poll = setInterval(() => hud().then((h) => labels.add(h?.label)).catch(() => {}), 150);
    for (const name of ['chatgpt', 'claude', 'gemini']) {
      await open(name === 'chatgpt' ? undefined : name);
      await setAutoRead(true);
      const before = (await heard()).length;
      const spoken = (await status()).spoken;
      await streamReply();
      await sleep(QUIET_MS);
      const s = await drained();
      const got = (await heard()).slice(before);
      expectSentences(got, name);
      if (s.spoken - spoken !== 3 || s.error) throw new Error(`${name}: ${JSON.stringify(s)}`);
      console.log(`${name}: read ${JSON.stringify(got.map((h) => h.text))}`);
      console.log(`${name}: spoken +3, error null, RTF ${s.stats?.rtf}`);
    }
    clearInterval(poll);
    // The engine's status reaches the HUD through the service worker.
    if (!labels.has('speaking')) throw new Error(`the HUD never showed "speaking": ${[...labels]}`);
    console.log(`HUD labels seen: ${JSON.stringify([...labels])}`);
  } else if (caseName === 'off') {
    await open();
    await setAutoRead(false);
    await expectNothingRead('off');
  } else if (caseName === 'no-selector') {
    await open('plain');
    await setAutoRead(true);
    await expectNothingRead('no-selector');
  } else if (caseName === 'not-ready') {
    await open();
    const { next } = await restartEngineThrottled(worker);
    await setAutoRead(true); // its preload message starts the new engine
    await useEngine(await Promise.race([next, sleep(20000).then(() => {
      throw new Error('no new engine under the debugger');
    })]));
    console.log(`HUD: ${(await hud()).label}`);
    await streamReply();
    await sleep(QUIET_MS);
    const early = await heard();
    console.log(`while loading: ${JSON.stringify(early)}`);
    if (early.some((h) => h.state === 'ready')) {
      throw new Error('inconclusive: the model was ready before the reply was in');
    }
    expectSentences(early, 'not-ready, while the model downloads');
    await throttle(engine, -1);
    const { status: boot } = await waitForEngine(cdp);
    if (boot?.state !== 'ready') throw new Error(`engine: ${JSON.stringify(boot)}`);
    const s = await drained();
    expectSentences(await heard(), 'not-ready');
    if (s.spoken !== 3 || s.error) throw new Error(`not-ready: ${JSON.stringify(s)}`);
    console.log(`not-ready: read once ready (spoken ${s.spoken}, error null, RTF ${s.stats?.rtf})`);
  } else if (caseName === 'reload') {
    await open();
    await setAutoRead(true);
    const before = (await heard()).length;
    await evalIn(cdp, page, `stream(${JSON.stringify(SENTENCES)}, ${STREAM_MS}), 0`);
    await waitFor(async () => (await heard()).length > before, 'the first sentence', 10000);
    await evalIn(cdp, page, 'window.oldPage = true');
    await cdp.send('Page.reload', {}, page);
    // Count from the moment the old page is gone (it may finish one more sentence first).
    await waitFor(() => evalIn(cdp, page, 'window.oldPage === undefined'), 'the reload');
    const atReload = (await heard()).length;
    console.log(`reload: ${atReload - before} sentence(s) read before the reload`);
    await open();
    await sleep(QUIET_MS);
    const after = (await heard()).slice(atReload);
    if (after.length) throw new Error(`reload: read after the reload: ${JSON.stringify(after)}`);
    const { on } = await hud();
    console.log(`reload: nothing read after the reload; Auto-read ${on ? 'on' : 'off'} (kept per site)`);
    if (!on) throw new Error('reload: the switch was not kept');
    await streamReply();
    await sleep(QUIET_MS);
    await drained();
    expectSentences((await heard()).slice(atReload), 'reload, the next reply');
    console.log('reload: the next reply read once');
  } else if (caseName === 'redraw') {
    for (const name of ['chatgpt', 'claude', 'gemini']) {
      await open(name === 'chatgpt' ? undefined : name);
      await setAutoRead(true);
      const before = (await heard()).length;
      // The last two land before the tail (the third sentence) is read.
      await streamReply([[0.5, 'node'], [1, 'paragraphs'], [1, 'node']]);
      await sleep(QUIET_MS);
      expectSentences((await heard()).slice(before), `${name}, re-rendered while streaming`);
      for (const how of ['same', 'paragraphs', 'node']) {
        const text = await evalIn(cdp, page, `redraw('${how}', ${JSON.stringify(SENTENCES)})`);
        if (text !== SENTENCES.join(' ')) throw new Error(`${name} ${how}: the page shows ${text}`);
        await sleep(QUIET_MS);
        const extra = (await heard()).slice(before + 3);
        if (extra.length) throw new Error(`${name} ${how}: read again: ${JSON.stringify(extra)}`);
      }
      await drained();
      console.log(`${name}: read once through 3 re-renders while streaming; ` +
        're-rendered 3 ways when done, nothing read again');
    }
  }
  // Nothing else was read: no earlier reply, no piece of a sentence.
  const stray = (await heard()).filter((h) => !SENTENCES.includes(h.text));
  if (stray.length) throw new Error(`also read: ${JSON.stringify(stray)}`);
  if (errors.length) throw new Error(`${errors.length} exception(s)`);
  const s = await status();
  console.log('AUTOREAD_RESULT ' + JSON.stringify({
    case: caseName ?? 'stream', spoken: s.spoken, error: s.error, exceptions: 0,
  }));
  pass = true;
} catch (err) {
  console.error(`autoread e2e failed${caseName ? ` (${caseName})` : ''}: ${err.message}`);
  for (const e of errors.slice(0, 5)) console.error(`  ${e}`);
  if (errors.length > 5) console.error(`  … and ${errors.length - 5} more`);
}
console.log(pass ? 'PASS' : 'FAIL');
await cdp.send('Browser.close').catch(() => {});
server.close();
process.exit(pass ? 0 : 1);
