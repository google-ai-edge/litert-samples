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

import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import {detect, detectionPrompt, parseBoxes} from '../src/gemma';

describe('parseBoxes', () => {
  it('reads Gemma\'s fenced JSON list, [ymin, xmin, ymax, xmax] / 1000', () => {
    const raw = '```json\n[\n  {"box_2d": [717, 474, 864, 531], "label": "soccer ball"}\n]\n```';
    const [b] = parseBoxes(raw);
    expect(b.label).toBe('soccer ball');
    expect(b.x0).toBeCloseTo(0.474);
    expect(b.y0).toBeCloseTo(0.717);
    expect(b.x1).toBeCloseTo(0.531);
    expect(b.y1).toBeCloseTo(0.864);
  });

  it('reads several objects, in order', () => {
    const raw = '[{"box_2d": [45, 325, 832, 565], "label": "person"}, ' +
        '{"box_2d": [342, 194, 632, 253], "label": "person"}, {"label": "person", "box_2d": [352, 594, 632, 658]}]';
    const boxes = parseBoxes(raw);
    expect(boxes.map((b) => b.x0)).toEqual([0.325, 0.194, 0.594]);
    expect(boxes.every((b) => b.label === 'person')).toBe(true);
  });

  it('keeps complete entries of a reply cut off mid-list', () => {
    const raw = '[{"box_2d": [100, 200, 300, 400], "label": "a"}, {"box_2d": [500, 600, 7';
    expect(parseBoxes(raw)).toHaveLength(1);
  });

  it('orders swapped corners, clamps to the frame, drops degenerate boxes', () => {
    const [b, ...rest] = parseBoxes(
        '[{"box_2d": [900, 800, 100, -50], "label": "x"}, {"box_2d": [10, 10, 11, 11], "label": "dot"}]');
    expect([b.x0, b.y0, b.x1, b.y1]).toEqual([0, 0.1, 0.8, 0.9]);
    expect(rest).toHaveLength(0);
  });

  it('returns nothing for text without boxes', () => {
    expect(parseBoxes('I could not find that object.')).toEqual([]);
  });

  it('asks in Gemma\'s native detection format', () => {
    expect(detectionPrompt('the ball')).toContain('Detect the ball');
    expect(detectionPrompt('the ball')).toContain('"box_2d"');
  });
});

describe('detect (streamed)', () => {
  const sse = (pieces: string[]) => new ReadableStream<Uint8Array>({
    start(c) {
      const enc = new TextEncoder();
      for (const p of pieces) c.enqueue(enc.encode(`data: ${JSON.stringify({choices: [{delta: {content: p}}]})}\n\n`));
      c.enqueue(enc.encode('data: [DONE]\n\n'));
      c.close();
    },
  });
  const frame = {width: 4, height: 4} as unknown as CanvasImageSource;
  // The frame -> JPEG step needs a canvas; the stream parsing is what is under test.
  beforeEach(() => vi.stubGlobal('OffscreenCanvas', class {
    constructor(public width: number, public height: number) {}
    getContext() { return {drawImage() {}}; }
    convertToBlob() { return Promise.resolve(new Blob([new Uint8Array([1, 2, 3])])); }
  }));
  afterEach(() => vi.unstubAllGlobals());

  // Tokens as Gemma streams them: entries split mid-number and mid-label.
  const pieces = ['```json\n[\n  {"box_2d": [100, 2', '00, 300, 400], "lab', 'el": "a"},\n  {"box_2d": [500, 600, 700,',
    ' 800], "label": "b"},\n  {"box_2d": [10, 20, 30, 40], "label": "c"}', '\n]\n```'];

  it('hands over each box once its entry is complete, in order', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(sse(pieces))));
    const seen: string[] = [];
    const r = await detect(frame, 'x', 'm', 'http://s', (b, i) => void seen.push(`${i}:${b.label}`));
    expect(seen).toEqual(['0:a', '1:b', '2:c']);
    expect(r.boxes.map((b) => b.label)).toEqual(['a', 'b', 'c']);
  });

  it('stops the request after maxBoxes', async () => {
    let signal: AbortSignal | undefined;
    vi.stubGlobal('fetch', vi.fn(async (_u: string, init: RequestInit) => {
      signal = init.signal!;
      return new Response(sse(pieces));
    }));
    const seen: string[] = [];
    await detect(frame, 'x', 'm', 'http://s', (b) => void seen.push(b.label), 2);
    expect(seen).toEqual(['a', 'b']);
    expect(signal?.aborted).toBe(true);
  });
});
