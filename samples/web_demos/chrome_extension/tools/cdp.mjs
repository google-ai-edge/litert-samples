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
 * Minimal CDP helpers shared by the smoke scripts. Uses only node built-ins
 * (global WebSocket needs node ≥22).
 */
import { spawn, spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Test page per DevTools port, opened by waitForEndpoint (see launchChrome).
const startPages = new Map();

/** Launch a Chromium build that honors --load-extension, with a throwaway
 *  profile unless one is given (reuse a profile to keep the model cache).
 *  The throwaway profile is deleted when this process exits. */
export function launchChrome(bin, { dist, port, profile, url = 'about:blank' }) {
  const tempProfile = profile ? null : mkdtempSync(join(tmpdir(), 'pv-smoke-'));
  profile = tempProfile ?? profile;
  console.log(`profile: ${profile}`);
  const child = spawn(bin, [
    `--user-data-dir=${profile}`,
    `--load-extension=${dist}`,
    `--remote-debugging-port=${port}`,
    '--no-first-run',
    '--no-default-browser-check',
    '--disable-sync',
    // macOS: keep the test profile's cookie key out of the login Keychain.
    // Otherwise every page load waits for Keychain access first — seconds to
    // minutes — and the tests time out before the test page has loaded.
    '--use-mock-keychain',
    // The tests run on a desktop someone is actively using. A window shown
    // at startup takes the keyboard focus, so Chrome starts without one and
    // waitForEndpoint opens the test page in a background window that is
    // almost entirely off screen. No sound either.
    '--no-startup-window',
    '--mute-audio',
    // Keep rAF/timers running even when the window counts as covered.
    '--disable-backgrounding-occluded-windows',
    '--disable-renderer-backgrounding',
    '--disable-background-timer-throttling',
    // Reused profiles from Browser.close'd runs otherwise show a "restore
    // pages?" bubble over the page under test.
    '--hide-crash-restore-bubble',
  ], { stdio: 'ignore' });
  startPages.set(port, url);
  // A binary that cannot start fails the caller at waitForEndpoint, and the
  // exit hook below still removes the throwaway profile.
  child.on('error', (err) => console.error(`cannot start ${bin}: ${err.message}`));
  process.on('exit', () => {
    try { child.kill(); } catch { /* gone */ }
    if (tempProfile) removeProfile(tempProfile);
  });
  return child;
}

/** Deletes a throwaway profile once no Chrome process uses it (they write to
 *  it until they exit), waiting at most 10 s. This runs in the 'exit' hook,
 *  where nothing can be awaited and the exited browser is not reaped yet, so
 *  it polls pgrep for processes started with this --user-data-dir: Chrome
 *  passes the flag to every helper process. */
function removeProfile(dir) {
  const pattern = `--user-data-dir=${dir}`.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const nap = new Int32Array(new SharedArrayBuffer(4));
  const deadline = Date.now() + 10 * 1000;
  while (Date.now() < deadline && spawnSync('pgrep', ['-f', '--', pattern]).status === 0) {
    Atomics.wait(nap, 0, 0, 100);
  }
  try {
    rmSync(dir, { recursive: true, force: true });
  } catch (err) {
    console.error(`could not remove ${dir}: ${err.message}`);
  }
}

export async function waitForEndpoint(port) {
  for (let i = 0; i < 50; i++) {
    let wsUrl;
    try {
      const res = await fetch(`http://127.0.0.1:${port}/json/version`);
      wsUrl = (await res.json()).webSocketDebuggerUrl;
    } catch {
      await sleep(200);
      continue;
    }
    if (startPages.has(port)) {
      await openStartPage(wsUrl, startPages.get(port));
      startPages.delete(port);
    }
    return wsUrl;
  }
  throw new Error('DevTools endpoint never came up');
}

/** Opens a page launchChrome was given in a new window without focusing it.
 *  Asked for far off screen, macOS still keeps 40 px of it in view at the
 *  left edge; the page renders as usual. */
async function openStartPage(wsUrl, url) {
  const cdp = await Cdp.connect(wsUrl);
  try {
    await cdp.send('Target.createTarget', {
      url, newWindow: true, background: true, left: -32000, top: -32000, width: 1200, height: 926,
    });
  } finally {
    cdp.ws.close();
  }
}

/** Flat-session CDP client. */
export class Cdp {
  constructor(ws) {
    this.ws = ws;
    this.id = 0;
    this.pending = new Map();
    ws.addEventListener('message', (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id !== undefined && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        if (msg.error) reject(new Error(msg.error.message));
        else resolve(msg.result);
      }
    });
  }

  static async connect(url) {
    const ws = new WebSocket(url);
    await new Promise((res, rej) => {
      ws.addEventListener('open', res, { once: true });
      ws.addEventListener('error', rej, { once: true });
    });
    return new Cdp(ws);
  }

  send(method, params = {}, sessionId = undefined, timeoutMs = 20000) {
    const id = ++this.id;
    return new Promise((resolve, reject) => {
      // Reject instead of hanging forever: a session whose renderer crashed
      // or detached mid-navigation simply never answers.
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`CDP ${method} timed out (${timeoutMs}ms)`));
      }, timeoutMs);
      this.pending.set(id, {
        resolve: (v) => { clearTimeout(timer); resolve(v); },
        reject: (e) => { clearTimeout(timer); reject(e); },
      });
      this.ws.send(JSON.stringify({ id, method, params, sessionId }));
    });
  }
}

export async function attachTo(cdp, target) {
  const { sessionId } = await cdp.send('Target.attachToTarget', {
    targetId: target.targetId,
    flatten: true,
  });
  return sessionId;
}

export async function evalIn(cdp, sessionId, expression, awaitPromise = false, timeoutMs = 20000) {
  const { result, exceptionDetails } = await cdp.send(
    'Runtime.evaluate',
    { expression, awaitPromise, returnByValue: true },
    sessionId,
    timeoutMs,
  );
  if (exceptionDetails) {
    throw new Error(exceptionDetails.exception?.description ?? 'evaluate failed');
  }
  return result?.value;
}

export async function findTarget(cdp, predicate, timeoutMs = 15000) {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const { targetInfos } = await cdp.send('Target.getTargets');
    const hit = targetInfos.find(predicate);
    if (hit) return hit;
    if (Date.now() > deadline) {
      throw new Error(`target not found; saw: ${targetInfos.map((t) => `${t.type}:${t.url}`).join(', ')}`);
    }
    await sleep(300);
  }
}

/** Wait for the offscreen engine to reach ready/error; returns final status.
 *  `hook` is the engine's globalThis debug object (__pv = voice, __p3 = 3d). */
export async function waitForEngine(cdp, { timeoutMs = 10 * 60 * 1000, hook = '__pv' } = {}) {
  const off = await findTarget(cdp, (t) => t.url.includes('/offscreen.html'));
  const session = await attachTo(cdp, off);
  const deadline = Date.now() + timeoutMs;
  let status = null;
  let lastLine = '';
  while (Date.now() < deadline) {
    // the document may still be executing its script — wait for the hook
    let raw = null;
    try {
      raw = await evalIn(cdp, session,
        `typeof ${hook} === 'undefined' ? null : JSON.stringify(${hook}.status)`);
    } catch { /* context not ready yet */ }
    if (raw === null) {
      await sleep(300);
      continue;
    }
    status = JSON.parse(raw);
    const line = `state=${status.state} downloaded=${status.downloadedMB}MB`;
    if (line !== lastLine) console.log(line);
    lastLine = line;
    if (status.state === 'ready' || status.state === 'error') break;
    await sleep(1000);
  }
  return { status, session };
}
