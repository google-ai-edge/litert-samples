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
// Tool calling for "Ask Gemma": the request is a plan, not only a query.
//
// The model is given a short list of tools (the app's own actions: find
// objects, change the effect, remove objects, playback, camera, model quality,
// describe the scene, measure speed) and answers with tool calls, one per line:
//
//   <tool_call>{"name": "set_effect", "arguments": {"effect": "cutout"}}</tool_call>
//
// This file is engine-agnostic: it holds the tool declarations, the prompt
// that describes them, and a tolerant streaming parser. agent.ts runs a turn
// on whichever engine is loaded (MediaPipe GenAI today; LiteRT-LM.js has the
// same declarations natively through AutoToolChat) and executes the calls as
// they land, in order, so the first visible result appears while the model is
// still writing the next call.

export type JsonValue = string | number | boolean | null | JsonValue[] | {[k: string]: JsonValue};

/** A JSON-schema-ish parameter description (what the model sees). */
export interface ToolParam {
  type: 'string' | 'number' | 'integer' | 'boolean' | 'array';
  description?: string;
  enum?: ReadonlyArray<string | number>;
  items?: {type: 'string' | 'number'};
  minimum?: number;
  maximum?: number;
}

/** A tool declaration in the OpenAI / LiteRT-LM function shape. */
export interface ToolSpec {
  name: string;
  description: string;
  parameters: {
    type: 'object';
    properties: Record<string, ToolParam>;
    required?: string[];
  };
}

/** One call predicted by the model. */
export interface ToolCall {
  name: string;
  arguments: Record<string, JsonValue>;
}

/** Tool declarations: short, concrete, and few — small models choose well from short lists. */
export const TOOLS: ToolSpec[] = [
  {
    name: 'find_objects',
    description: 'Find objects in the frame on screen and segment them with SAM 2. Replaces the current objects. ' +
        'Use the user\'s own words for `what`.',
    parameters: {
      type: 'object',
      properties: {
        what: {type: 'string', description: 'What to find, e.g. "the ball", "all the players", "the person on the left".'},
        max: {type: 'integer', description: 'At most this many objects; omit to find all of them.', minimum: 1},
      },
      required: ['what'],
    },
  },
  {
    name: 'set_effect',
    description: 'How the masks are drawn. overlay: coloured masks with an outline. spotlight: dim everything ' +
        'except the objects. cutout: green screen, only the objects remain (also "remove the background").',
    parameters: {
      type: 'object',
      properties: {
        effect: {type: 'string', enum: ['overlay', 'spotlight', 'cutout']},
        outline: {type: 'integer', description: 'Outline width in pixels (0-6), optional.', minimum: 0, maximum: 6},
      },
      required: ['effect'],
    },
  },
  {
    name: 'remove_objects',
    description: 'Remove tracked objects. `labels`: remove the objects whose label matches (e.g. ["goalkeeper"]). ' +
        '`keep`: remove every object except these. `all`: remove everything.',
    parameters: {
      type: 'object',
      properties: {
        labels: {type: 'array', items: {type: 'string'}},
        keep: {type: 'array', items: {type: 'string'}},
        all: {type: 'boolean'},
      },
    },
  },
  {
    name: 'playback',
    description: 'track: follow the current objects through the whole video. play / pause / restart the video. ' +
        'stop: stop tracking or playing.',
    parameters: {
      type: 'object',
      properties: {action: {type: 'string', enum: ['track', 'play', 'pause', 'restart', 'stop']}},
      required: ['action'],
    },
  },
  {
    name: 'use_camera',
    description: 'Switch to the live webcam (on: true) or back to the video (on: false).',
    parameters: {type: 'object', properties: {on: {type: 'boolean'}}, required: ['on']},
  },
  {
    name: 'set_quality',
    description: 'Model resolution and memory. size 384 is fastest, 512 finer, 1024 best but slow. ' +
        'memory 2 is fastest, 7 remembers more frames.',
    parameters: {
      type: 'object',
      properties: {size: {type: 'integer', enum: [384, 512, 1024]}, memory: {type: 'integer', enum: [2, 7]}},
    },
  },
  {
    name: 'describe_scene',
    description: 'What is tracked right now (objects, labels, sizes), the video or camera, and the current ' +
        'settings. Call it to answer questions about the app\'s state.',
    parameters: {type: 'object', properties: {}},
  },
  {
    name: 'measure',
    description: 'Measured speed: frames per second and GPU time per stage. Call it when asked how fast ' +
        'or how long anything takes.',
    parameters: {type: 'object', properties: {}},
  },
];

/** Tools whose result the model should see, to answer the user in words. */
export const REPLY_TOOLS = new Set(['describe_scene', 'measure']);

/** One line per tool: name(args) - description. Compact beats JSON schema for a 2-4B model. */
function describeTool(t: ToolSpec): string {
  const args = Object.entries(t.parameters.properties).map(([k, p]) => {
    const req = t.parameters.required?.includes(k);
    const type = p.enum ? p.enum.map((v) => JSON.stringify(v)).join(' | ')
      : p.type === 'array' ? `${p.items?.type ?? 'string'}[]` : p.type;
    return `${k}${req ? '' : '?'}: ${type}`;
  });
  return `- ${t.name}(${args.join(', ')}): ${t.description}`;
}

const EXAMPLES: Array<[string, ToolCall[]]> = [
  ['find all the players and cut them out',
    [{name: 'find_objects', arguments: {what: 'all the players'}}, {name: 'set_effect', arguments: {effect: 'cutout'}}]],
  ['the ball', [{name: 'find_objects', arguments: {what: 'the ball'}}]],
  ['keep only the ball and spotlight it',
    [{name: 'remove_objects', arguments: {keep: ['ball']}}, {name: 'set_effect', arguments: {effect: 'spotlight'}}]],
  ['turn the camera on and track me',
    [{name: 'use_camera', arguments: {on: true}}, {name: 'find_objects', arguments: {what: 'the person', max: 1}}]],
  ['how fast is this running?', [{name: 'measure', arguments: {}}]],
  ['track them through the video', [{name: 'playback', arguments: {action: 'track'}}]],
];

export const formatCall = (c: ToolCall) => `<tool_call>${JSON.stringify({name: c.name, arguments: c.arguments})}</tool_call>`;

/**
 * The instructions: what the app is, the tools, the output format, examples.
 * `scene` is a one-line summary of the current state so the model can choose
 * well (e.g. not call use_camera when the camera is already on).
 */
export function toolPrompt(tools: ToolSpec[] = TOOLS, scene = ''): string {
  return [
    'You control a video app that segments and tracks objects with SAM 2 on the GPU, in the browser. ' +
        'The user speaks or types a request. Carry it out by calling tools.',
    'Tools:',
    ...tools.map(describeTool),
    'Rules:',
    '- Answer with tool calls only, one per line, each exactly: <tool_call>{"name": NAME, "arguments": {...}}</tool_call>',
    '- Call the tools in the order they should run. Do not repeat a call. Do not explain.',
    '- A bare description of things ("the ball", "two dogs") means find_objects.',
    '- If no tool fits, answer in one short sentence instead.',
    ...(scene ? [`Current state: ${scene}`] : []),
    'Examples:',
    ...EXAMPLES.flatMap(([u, calls]) => [`User: ${u}`, ...calls.map(formatCall)]),
  ].join('\n');
}

/**
 * The prompt that asks the model to answer the user from tool results, in
 * words (describe_scene, measure).
 */
export function replyPrompt(utterance: string, results: Array<{name: string; result: JsonValue}>): string {
  return 'You are the assistant of a browser video app that segments objects with SAM 2 on the GPU. ' +
      'Answer the user in one or two short sentences, using only the numbers and facts in the tool results. ' +
      'No tool calls, no markdown.\n' +
      `User: ${utterance}\n` +
      results.map((r) => `Result of ${r.name}: ${JSON.stringify(r.result)}`).join('\n');
}

const CALL_RE = /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g;
// A bare object with "name" and "arguments" (models sometimes drop the tags or fence them).
const BARE_RE = /\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{/g;

/** Reads one JSON object starting at `text[start]` ('{'); null if it isn't complete yet. */
function readObject(text: string, start: number): {json: string; end: number} | null {
  let depth = 0, inStr = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inStr) {
      if (ch === '\\') i++;
      else if (ch === '"') inStr = false;
    } else if (ch === '"') inStr = true;
    else if (ch === '{') depth++;
    else if (ch === '}' && --depth === 0) return {json: text.slice(start, i + 1), end: i + 1};
  }
  return null;
}

function asCall(json: string, known: ReadonlySet<string> | null): ToolCall | null {
  let v: unknown;
  try {
    v = JSON.parse(json);
  } catch {
    return null;
  }
  if (!v || typeof v !== 'object') return null;
  const o = v as {name?: unknown; arguments?: unknown; parameters?: unknown};
  if (typeof o.name !== 'string') return null;
  if (known && !known.has(o.name)) return {name: o.name, arguments: {}};  // reported as unknown by the executor
  const args = (o.arguments ?? o.parameters ?? {}) as Record<string, JsonValue>;
  return {name: o.name, arguments: args && typeof args === 'object' && !Array.isArray(args) ? args : {}};
}

/**
 * Complete tool calls in a (possibly partial) reply, in order. Accepts calls
 * in <tool_call> tags, bare {"name", "arguments"} objects, and either inside
 * code fences. A call whose JSON is still being written is not returned yet;
 * calls that are not valid JSON are skipped. `known` restricts names (unknown
 * names come back with empty arguments so the UI can show them).
 */
export function parseToolCalls(text: string, known: ReadonlySet<string> | null = null): ToolCall[] {
  const found: Array<{at: number; call: ToolCall}> = [];
  const taken: Array<[number, number]> = [];
  for (const m of text.matchAll(CALL_RE)) {
    const inner = m[1].replace(/^```\w*\s*|\s*```$/g, '');
    const brace = inner.indexOf('{');
    const obj = brace >= 0 ? readObject(inner, brace) : null;
    const call = obj ? asCall(obj.json, known) : null;
    if (call) {
      found.push({at: m.index!, call});
      taken.push([m.index!, m.index! + m[0].length]);
    }
  }
  for (const m of text.matchAll(BARE_RE)) {
    const at = m.index!;
    if (taken.some(([a, b]) => at >= a && at < b)) continue;
    const obj = readObject(text, at);
    const call = obj ? asCall(obj.json, known) : null;
    if (call) found.push({at, call});
  }
  return found.sort((a, b) => a.at - b.at).map((f) => f.call);
}

/** The reply without its tool calls (what the model said in words, if anything). */
export function stripToolCalls(text: string): string {
  return text.replace(CALL_RE, '').replace(/```\w*\s*```/g, '').trim();
}
