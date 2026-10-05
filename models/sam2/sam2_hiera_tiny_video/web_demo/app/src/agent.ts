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
// One agent turn: the user's request -> the model's tool calls -> the app's
// actions, executed as the calls stream in. Engine-agnostic: `generate` is
// whatever produces text for a prompt (MediaPipe GenAI in the page, the
// LiteRT-LM server, a fake in tests). LiteRT-LM.js can do this loop natively
// (AutoToolChat); until its web Gemma 4 builds take images, the vision tool
// runs through the same MediaPipe model, so one engine serves both.

import {type JsonValue, parseToolCalls, REPLY_TOOLS, replyPrompt, stripToolCalls, type ToolCall, toolPrompt, TOOLS} from './toolcalls';

/** Streams text for a prompt; resolves with the whole reply. */
export interface AgentEngine {
  generate(prompt: string, onPartial: (text: string) => void): Promise<string>;
  /** Stops the current generation (the pending generate() then resolves or rejects). */
  cancel(): void;
}

/** What a tool implementation gets besides its arguments. */
export interface ToolContext {
  /** Resolves when the model has finished writing this turn's calls (tools that need the engine wait for it). */
  afterGeneration: Promise<void>;
  signal: AbortSignal;
}

export type ToolImpl = (args: Record<string, JsonValue>, ctx: ToolContext) => JsonValue | void | Promise<JsonValue | void>;

export interface ToolEvent {
  id: number;
  name: string;
  arguments: Record<string, JsonValue>;
  status: 'started' | 'done' | 'error' | 'unknown';
  result?: JsonValue;
  error?: string;
  ms?: number;
}

export interface TurnOptions {
  engine: AgentEngine;
  tools: Record<string, ToolImpl>;
  /** One line about the current state, included in the prompt. */
  scene?: () => string;
  onEvent?: (e: ToolEvent) => void;
  /** The model's words: a plain answer when no tool fits, or the reply built from tool results. */
  onText?: (text: string) => void;
  signal?: AbortSignal;
  /** Safety valve against a model that loops on the examples. */
  maxCalls?: number;
}

export interface TurnResult {
  events: ToolEvent[];
  /** The model's reply in words, '' if it only called tools. */
  text: string;
  raw: string;
  seconds: number;
}

const same = (a: ToolCall, b: ToolCall) => a.name === b.name && JSON.stringify(a.arguments) === JSON.stringify(b.arguments);

/**
 * Runs one turn. Calls are executed in order as soon as each is complete in
 * the stream; a tool that needs the engine (find_objects) waits for the
 * stream to end, and the calls after it wait for it. Tools in REPLY_TOOLS
 * feed a second, short generation that answers the user in words.
 */
export async function runTurn(utterance: string, opts: TurnOptions): Promise<TurnResult> {
  const {engine, tools, onEvent, onText} = opts;
  const signal = opts.signal ?? new AbortController().signal;
  const maxCalls = opts.maxCalls ?? 6;
  const known = new Set(Object.keys(tools));
  let finishGeneration!: () => void;
  const afterGeneration = new Promise<void>((r) => (finishGeneration = r));
  const ctx: ToolContext = {afterGeneration, signal};
  const events: ToolEvent[] = [];
  const seen: ToolCall[] = [];
  let chain = Promise.resolve();
  let nextId = 1;

  const run = (call: ToolCall) => (chain = chain.then(async () => {
    if (signal.aborted) return;
    const e: ToolEvent = {id: nextId++, name: call.name, arguments: call.arguments, status: 'started'};
    events.push(e);
    const impl = tools[call.name];
    if (!impl) {
      e.status = 'unknown';
      e.error = `no tool named ${call.name}`;
      onEvent?.(e);
      return;
    }
    onEvent?.({...e});
    const t0 = performance.now();
    try {
      const r = await impl(call.arguments, ctx);
      e.status = 'done';
      if (r !== undefined) e.result = r;
    } catch (err) {
      e.status = 'error';
      e.error = (err as Error).message || String(err);
    }
    e.ms = performance.now() - t0;
    onEvent?.(e);
  }));

  const take = (text: string) => {
    const calls = parseToolCalls(text, known);
    for (let i = seen.length; i < calls.length && seen.length < maxCalls; i++) {
      // The model repeating itself (or an example) is not a second action.
      if (seen.some((s) => same(s, calls[i]))) continue;
      seen.push(calls[i]);
      run(calls[i]);
    }
  };

  const onAbort = () => engine.cancel();
  signal.addEventListener('abort', onAbort, {once: true});
  const prompt = `${toolPrompt(TOOLS, opts.scene?.() ?? '')}\nUser: ${utterance}`;
  const t0 = performance.now();
  let raw = '';
  try {
    raw = await engine.generate(prompt, (partial) => {
      raw = partial;
      take(raw);
    });
    take(raw);
  } catch (e) {
    if (!signal.aborted) {
      finishGeneration();
      throw e;
    }
  } finally {
    finishGeneration();
  }
  await chain;
  signal.removeEventListener('abort', onAbort);

  let text = '';
  if (!signal.aborted) {
    const replies = events.filter((e) => e.status === 'done' && REPLY_TOOLS.has(e.name) && e.result !== undefined)
        .map((e) => ({name: e.name, result: e.result as JsonValue}));
    if (replies.length) {
      text = stripToolCalls(await engine.generate(replyPrompt(utterance, replies), () => undefined));
    } else if (!events.length) {
      text = stripToolCalls(raw);
    }
    if (text) onText?.(text);
  }
  return {events, text, raw, seconds: (performance.now() - t0) / 1000};
}
