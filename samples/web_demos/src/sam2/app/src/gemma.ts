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

// "Ask Gemma": Gemma 4 (E2B / E4B) running on LiteRT-LM finds objects in the
// frame on screen and returns their bounding boxes, which become SAM 2 box
// prompts.
//
// Gemma runs in a local OpenAI-compatible server on the LiteRT-LM Python API
// (tools/gemma_server.py; `litert-lm serve` also works, only slower): the LiteRT-LM web runtime's Gemma 4 models are text-only
// today, while the full .litertlm models include the vision encoder. The page
// sends the frame as a JPEG data URL with Gemma's native detection prompt; the
// reply is a JSON list of {"box_2d": [ymin, xmin, ymax, xmax], "label"} with
// coordinates normalized to 0..1000 (the Gemini / PaliGemma convention Gemma
// is trained on).
//
//   tools/gemma_server.sh http://localhost:5175
//
// gemma_web.ts runs the same prompt in the page instead (MediaPipe on WebGPU).

export interface GemmaBox {
  label: string;
  /** Normalized frame coords, top-left / bottom-right. */
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface GemmaResult {
  boxes: GemmaBox[];
  seconds: number;
  model: string;
  raw: string;
}

export const DEFAULT_SERVER = 'http://127.0.0.1:9379';

export function detectionPrompt(what: string): string {
  return `Detect ${what}. Output a json list where each entry contains the 2D bounding box in "box_2d" ` +
      'and a text label in "label".';
}

const BOX_RE = /"box_2d"\s*:\s*\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]/;
const LABEL_RE = /"label"\s*:\s*"([^"]*)"/;

/**
 * Boxes in a Gemma reply: every entry with a box_2d, in any surrounding text
 * (code fences, prose, a cut-off list). [ymin, xmin, ymax, xmax] / 1000 ->
 * normalized {x0, y0, x1, y1}, clamped and ordered; degenerate boxes dropped.
 */
export function parseBoxes(text: string): GemmaBox[] {
  const out: GemmaBox[] = [];
  for (const part of text.split(/\}\s*,?\s*\{/)) {
    const m = BOX_RE.exec(part);
    if (!m) continue;
    const [ya, xa, yb, xb] = m.slice(1, 5).map((v) => Math.min(Math.max(Number(v) / 1000, 0), 1));
    const box = {label: LABEL_RE.exec(part)?.[1] ?? '', x0: Math.min(xa, xb), y0: Math.min(ya, yb),
      x1: Math.max(xa, xb), y1: Math.max(ya, yb)};
    if (box.x1 - box.x0 > 0.002 && box.y1 - box.y0 > 0.002) out.push(box);
  }
  return out;
}

/** Model ids the server has imported (e.g. gemma-4-e2b), or null if unreachable. */
export async function listModels(server = DEFAULT_SERVER, timeoutMs = 1500): Promise<string[] | null> {
  try {
    const res = await fetch(`${server}/v1/models`, {signal: AbortSignal.timeout(timeoutMs)});
    if (!res.ok) return null;
    const j = await res.json() as {data?: Array<{id: string}>};
    return (j.data ?? []).map((m) => m.id);
  } catch {
    return null;
  }
}

export type FrameSource = CanvasImageSource & {width?: number; height?: number;
    videoWidth?: number; videoHeight?: number};

/** Frame -> canvas, longest side at most `maxSide` (Gemma's vision budget is ~280 tokens). */
export function frameToCanvas(frame: FrameSource, maxSide = 1024): OffscreenCanvas {
  const w = Number(frame.videoWidth || frame.width || 0), h = Number(frame.videoHeight || frame.height || 0);
  const s = Math.min(1, maxSide / Math.max(w, h));
  const c = new OffscreenCanvas(Math.round(w * s), Math.round(h * s));
  c.getContext('2d')!.drawImage(frame, 0, 0, c.width, c.height);
  return c;
}

/** Frame -> JPEG data URL, longest side at most `maxSide`. */
async function frameToDataUrl(frame: FrameSource, maxSide = 1024): Promise<string> {
  const c = frameToCanvas(frame, maxSide);
  const blob = await c.convertToBlob({type: 'image/jpeg', quality: 0.9});
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let bin = '';
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return `data:image/jpeg;base64,${btoa(bin)}`;
}

/**
 * Asks Gemma for boxes of `what` in `frame`. `model` is a server model id, e.g.
 * "gemma-4-e2b,gpu". The reply is streamed: `onBox` gets each box as soon as
 * its entry is complete (Gemma writes ~30 tokens per box, so the first object
 * can be segmented while the rest are still being written). After `maxBoxes`
 * boxes the request is cancelled.
 */
export async function detect(frame: CanvasImageSource, what: string, model: string,
                             server = DEFAULT_SERVER, onBox?: (box: GemmaBox, index: number) => void | Promise<void>,
                             maxBoxes = Infinity): Promise<GemmaResult> {
  const url = await frameToDataUrl(frame as never);
  const t0 = performance.now();
  const abort = new AbortController();
  const res = await fetch(`${server}/v1/chat/completions`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    signal: abort.signal,
    body: JSON.stringify({
      model,
      temperature: 0,
      stream: true,
      messages: [{role: 'user', content: [
        {type: 'image_url', image_url: {url}},
        {type: 'text', text: detectionPrompt(what)},
      ]}],
    }),
  });
  if (!res.ok) throw new Error(`LiteRT-LM server: HTTP ${res.status} ${(await res.text()).slice(0, 200)}`);
  let raw = '';
  let emitted = 0;
  const pending: Array<void | Promise<void>> = [];
  // Complete entries only: up to the last closing brace.
  const complete = () => parseBoxes(raw.slice(0, raw.lastIndexOf('}') + 1));
  const reader = res.body!.pipeThrough(new TextDecoderStream()).getReader();
  let buf = '';
  try {
    read: for (;;) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += value;
      let nl;
      while ((nl = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line.startsWith('data:')) continue;
        const data = line.slice(5).trim();
        if (data === '[DONE]') break read;
        const j = JSON.parse(data) as {choices?: Array<{delta?: {content?: string}}>};
        raw += j.choices?.[0]?.delta?.content ?? '';
      }
      const boxes = complete();
      while (emitted < Math.min(boxes.length, maxBoxes)) {
        pending.push(onBox?.(boxes[emitted], emitted));
        emitted++;
      }
      if (emitted >= maxBoxes) {
        abort.abort();  // enough objects: stop Gemma writing more
        break;
      }
    }
  } catch (e) {
    if (!abort.signal.aborted) throw e;
  }
  const seconds = (performance.now() - t0) / 1000;
  const boxes = parseBoxes(raw);
  while (emitted < Math.min(boxes.length, maxBoxes)) {  // a last entry without a closing brace
    pending.push(onBox?.(boxes[emitted], emitted));
    emitted++;
  }
  await Promise.all(pending);
  return {boxes, seconds, model, raw};
}
