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

// Shows the pipeline's composite on screen. The composite ([1,H,W,3] fp32,
import {describe, expect, it} from 'vitest';

import {runTurn, type AgentEngine, type ToolEvent, type ToolImpl} from '../src/agent';
import {makeTools, type AppActions} from '../src/tools';
import {formatCall, parseToolCalls, replyPrompt, stripToolCalls, toolPrompt, TOOLS} from '../src/toolcalls';

const KNOWN = new Set(TOOLS.map((t) => t.name));

describe('parseToolCalls', () => {
  it('reads tagged calls, in order', () => {
    const text = '<tool_call>{"name": "find_objects", "arguments": {"what": "all the players"}}</tool_call>\n' +
        '<tool_call>{"name": "set_effect", "arguments": {"effect": "cutout"}}</tool_call>';
    expect(parseToolCalls(text, KNOWN)).toEqual([
      {name: 'find_objects', arguments: {what: 'all the players'}},
      {name: 'set_effect', arguments: {effect: 'cutout'}},
    ]);
  });

  it('waits for a call that is still being written', () => {
    const partial = '<tool_call>{"name": "find_objects", "arguments": {"what": "the ba';
    expect(parseToolCalls(partial, KNOWN)).toEqual([]);
    expect(parseToolCalls(partial + 'll"}}</tool_call>', KNOWN)).toHaveLength(1);
  });

  it('accepts bare objects and code fences', () => {
    const text = 'Sure.\n```json\n{"name": "measure", "arguments": {}}\n```\n{"name":"playback","arguments":{"action":"track"}}';
    expect(parseToolCalls(text, KNOWN).map((c) => c.name)).toEqual(['measure', 'playback']);
  });

  it('accepts "parameters" for "arguments" and nested braces in strings', () => {
    const text = '<tool_call>{"name": "find_objects", "parameters": {"what": "the {red} ball"}}</tool_call>';
    expect(parseToolCalls(text, KNOWN)[0].arguments).toEqual({what: 'the {red} ball'});
  });

  it('skips invalid JSON and reports unknown names', () => {
    const text = '<tool_call>{"name": "set_effect", "arguments": {effect: cutout}}</tool_call>' +
        '<tool_call>{"name": "launch_rockets", "arguments": {"n": 3}}</tool_call>';
    expect(parseToolCalls(text, KNOWN)).toEqual([{name: 'launch_rockets', arguments: {}}]);
  });

  it('strips calls from the words around them', () => {
    expect(stripToolCalls('Done. <tool_call>{"name":"measure","arguments":{}}</tool_call> ')).toBe('Done.');
  });
});

describe('prompts', () => {
  it('describes every tool and the format', () => {
    const p = toolPrompt(TOOLS, 'camera is on');
    for (const t of TOOLS) expect(p).toContain(`- ${t.name}(`);
    expect(p).toContain('<tool_call>{"name": NAME');
    expect(p).toContain('Current state: camera is on');
    expect(p).toContain('set_effect(effect: "overlay" | "spotlight" | "cutout", outline?: integer)');
  });

  it('round-trips an example through the parser', () => {
    const call = {name: 'set_quality', arguments: {size: 1024}};
    expect(parseToolCalls(formatCall(call), KNOWN)).toEqual([call]);
  });

  it('asks for a worded reply from results only', () => {
    const p = replyPrompt('how fast?', [{name: 'measure', result: {fps: 30}}]);
    expect(p).toContain('Result of measure: {"fps":30}');
    expect(p).toContain('No tool calls');
  });
});

/** An engine that streams `reply` in chunks, then `second` for the reply turn. */
function fakeEngine(reply: string, second = 'About 30 frames per second.', chunk = 7): AgentEngine & {prompts: string[]} {
  const prompts: string[] = [];
  return {
    prompts,
    async generate(prompt, onPartial) {
      prompts.push(prompt);
      const text = prompts.length === 1 ? reply : second;
      let out = '';
      for (let i = 0; i < text.length; i += chunk) {
        out = text.slice(0, i + chunk);
        await Promise.resolve();
        onPartial(out);
      }
      return text;
    },
    cancel() {},
  };
}

describe('runTurn', () => {
  it('executes calls in order as they stream, and defers the vision tool until the stream ends', async () => {
    const log: string[] = [];
    let streamDone = false;
    const tools: Record<string, ToolImpl> = {
      async find_objects(args, ctx) {
        log.push(`find:start:${args.what}`);
        await ctx.afterGeneration;
        streamDone = true;
        log.push('find:done');
        return {found: 2};
      },
      set_effect(args) {
        log.push(`effect:${args.effect}:${streamDone ? 'after' : 'before'}`);
      },
    };
    const engine = fakeEngine(
        '<tool_call>{"name":"find_objects","arguments":{"what":"players"}}</tool_call>\n' +
        '<tool_call>{"name":"set_effect","arguments":{"effect":"cutout"}}</tool_call>');
    const events: ToolEvent[] = [];
    const r = await runTurn('find the players and cut them out', {engine, tools, onEvent: (e) => events.push({...e})});
    expect(log).toEqual(['find:start:players', 'find:done', 'effect:cutout:after']);
    expect(r.events.map((e) => [e.name, e.status])).toEqual([['find_objects', 'done'], ['set_effect', 'done']]);
    expect(events.filter((e) => e.status === 'started')).toHaveLength(2);
    expect(r.text).toBe('');
    expect(engine.prompts).toHaveLength(1);  // no reply turn without a reply tool
    expect(engine.prompts[0]).toMatch(/User: find the players and cut them out$/);
  });

  it('answers in words from a reply tool, and reports unknown tools', async () => {
    const tools: Record<string, ToolImpl> = {measure: () => ({fps: 30, ms_per_frame: 27})};
    const engine = fakeEngine('<tool_call>{"name":"measure","arguments":{}}</tool_call>' +
        '<tool_call>{"name":"explode","arguments":{}}</tool_call>');
    const said: string[] = [];
    const r = await runTurn('how fast is this?', {engine, tools, onText: (t) => said.push(t)});
    expect(r.events.map((e) => e.status)).toEqual(['done', 'unknown']);
    expect(said).toEqual(['About 30 frames per second.']);
    expect(engine.prompts[1]).toContain('"fps":30');
  });

  it('passes plain words through when the model calls nothing', async () => {
    const engine = fakeEngine('I can find objects, change the effect, or measure speed.');
    const r = await runTurn('what can you do?', {engine, tools: {}});
    expect(r.events).toEqual([]);
    expect(r.text).toBe('I can find objects, change the effect, or measure speed.');
  });

  it('does not run the same call twice and stops at maxCalls', async () => {
    let n = 0;
    const tools: Record<string, ToolImpl> = {set_effect: () => void n++};
    const dup = '<tool_call>{"name":"set_effect","arguments":{"effect":"cutout"}}</tool_call>';
    const engine = fakeEngine(dup + dup + dup.replace('cutout', 'spotlight') + dup.replace('cutout', 'overlay'));
    const r = await runTurn('x', {engine, tools, maxCalls: 2});
    expect(n).toBe(2);
    expect(r.events.map((e) => e.arguments.effect)).toEqual(['cutout', 'spotlight']);
  });

  it('records a failing tool and keeps going', async () => {
    const tools: Record<string, ToolImpl> = {
      playback: () => {
        throw new Error('no video loaded');
      },
      set_effect: () => undefined,
    };
    const engine = fakeEngine('<tool_call>{"name":"playback","arguments":{"action":"play"}}</tool_call>' +
        '<tool_call>{"name":"set_effect","arguments":{"effect":"overlay"}}</tool_call>');
    const r = await runTurn('play it', {engine, tools});
    expect(r.events.map((e) => e.status)).toEqual(['error', 'done']);
    expect(r.events[0].error).toBe('no video loaded');
  });
});

describe('makeTools', () => {
  function app(): AppActions & {calls: string[]} {
    const calls: string[] = [];
    const objects = [{id: 1, label: 'soccer ball', prompted: true}, {id: 2, label: 'player', prompted: true},
      {id: 3, label: 'goalkeeper', prompted: true}];
    return {
      calls,
      maxObjects: () => 5,
      async findObjects(what, max) {
        calls.push(`find ${what} ${max}`);
        return {found: 1, labels: [what], seconds: 1.234};
      },
      setEffect: (e, o) => void calls.push(`effect ${e} ${o}`),
      objects: () => objects,
      removeObjects: (ids) => void calls.push(`remove ${ids.join(',')}`),
      removeAll: () => void calls.push('removeAll'),
      playback: (a) => (calls.push(`playback ${a}`), a),
      useCamera: async (on) => (calls.push(`camera ${on}`), 'ok'),
      setQuality: async (s, m) => (calls.push(`quality ${s} ${m}`), 'ok'),
      describeScene: () => ({objects: 3}),
      measure: () => ({fps: 30}),
    };
  }
  const ctx = {afterGeneration: Promise.resolve(), signal: new AbortController().signal};

  it('validates and normalizes arguments', async () => {
    const a = app();
    const t = makeTools(a);
    expect(await t.find_objects({what: ' the ball ', max: 9}, ctx)).toEqual({found: 1, labels: ['the ball'], seconds: 1.2});
    await expect(t.find_objects({}, ctx)).rejects.toThrow('needs `what`');
    expect(t.set_effect({effect: 'green screen'}, ctx)).toEqual({effect: 'cutout'});
    t.set_effect({effect: 'Spotlight', outline: 9}, ctx);
    expect(() => t.set_effect({effect: 'sepia'}, ctx)).toThrow('unknown effect');
    expect(await t.playback({action: 'follow them'}, ctx)).toEqual({status: 'track'});
    expect(await t.use_camera({on: 'yes'}, ctx)).toEqual({status: 'ok'});
    expect(await t.set_quality({size: '1024'}, ctx)).toEqual({status: 'ok'});
    await expect(t.set_quality({size: 999}, ctx)).rejects.toThrow('set_quality needs');
    expect(a.calls).toEqual(['find the ball 5', 'effect cutout undefined', 'effect spotlight 6', 'playback track',
      'camera true', 'quality 1024 undefined']);
  });

  it('removes by label, keeps by label, or clears everything', () => {
    const a = app();
    const t = makeTools(a);
    expect(t.remove_objects({labels: ['the goalkeeper']}, ctx)).toEqual({removed: 1, kept: 2});
    expect(t.remove_objects({keep: ['ball']}, ctx)).toEqual({removed: 2, kept: 1});
    expect(t.remove_objects({labels: ['object 2']}, ctx)).toEqual({removed: 1, kept: 2});
    expect(t.remove_objects({all: true}, ctx)).toEqual({removed: 3, kept: 0});
    expect(a.calls).toEqual(['remove 3', 'remove 2,3', 'remove 2', 'removeAll']);
  });
});
