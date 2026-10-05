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
// The tools' implementations: thin, validated adapters from the model's
// arguments to the app's actions (main.ts provides `AppActions`). Nothing
// here touches the DOM or the pipeline directly, so the same registry can be
// handed to LiteRT-LM.js's AutoToolChat or registered as WebMCP tools.

import {type ToolContext, type ToolImpl} from './agent';
import {type JsonValue} from './toolcalls';

export type EffectName = 'overlay' | 'spotlight' | 'cutout';
export type PlaybackAction = 'track' | 'play' | 'pause' | 'restart' | 'stop';

/** What the app lets the model do. Every method returns something the model can be told. */
export interface AppActions {
  /** Gemma finds `what` in the frame on screen; each box becomes an object (replaces the current ones). */
  findObjects(what: string, max: number, ctx: ToolContext): Promise<{found: number; labels: string[]; seconds: number}>;
  /** Object slots of the loaded pipeline (the most `find_objects` may add). */
  maxObjects(): number;
  setEffect(effect: EffectName, outline?: number): void;
  /** Objects on screen: id, label (from Gemma, if any), and whether it has a prompt. */
  objects(): Array<{id: number; label: string; prompted: boolean}>;
  removeObjects(ids: number[]): void;
  removeAll(): void;
  playback(action: PlaybackAction): Promise<string> | string;
  useCamera(on: boolean): Promise<string>;
  setQuality(size?: 384 | 512 | 1024, memory?: 2 | 7): Promise<string>;
  describeScene(): JsonValue;
  measure(): JsonValue;
}

const str = (v: JsonValue | undefined) => (typeof v === 'string' ? v.trim() : '');
const num = (v: JsonValue | undefined) => (typeof v === 'number' ? v : typeof v === 'string' && v.trim() !== '' ? Number(v) : NaN);
const bool = (v: JsonValue | undefined) => (typeof v === 'boolean' ? v : typeof v === 'string' ? /^(true|yes|on|1)$/i.test(v) : undefined);
const strs = (v: JsonValue | undefined): string[] =>
    Array.isArray(v) ? v.map(str).filter(Boolean) : str(v) ? [str(v)] : [];

/** "goalkeeper" matches "the goalkeeper", "player 2" matches "player"; ids ("object 2", "2") match too. */
function matches(label: string, id: number, want: string): boolean {
  const w = want.toLowerCase().replace(/^(the|an?)\s+/, '');
  const l = label.toLowerCase();
  if (!w) return false;
  if (w === String(id) || w === `object ${id}` || w === `#${id}`) return true;
  return !!l && (l.includes(w) || w.includes(l));
}

export function makeTools(app: AppActions): Record<string, ToolImpl> {
  return {
    async find_objects(args, ctx) {
      const what = str(args.what) || str(args.query) || str(args.object);
      if (!what) throw new Error('find_objects needs `what`');
      const slots = app.maxObjects();
      const max = Math.min(slots, Math.max(1, Math.round(num(args.max)) || slots));
      const r = await app.findObjects(what, max, ctx);
      return {found: r.found, labels: r.labels, seconds: Number(r.seconds.toFixed(1))};
    },

    set_effect(args) {
      const e = str(args.effect).toLowerCase();
      const effect: EffectName | undefined = /cut|green|remove.*back|background/.test(e) ? 'cutout'
        : /spot|dim|focus|highlight/.test(e) ? 'spotlight' : /over|mask|outline|normal|default/.test(e) ? 'overlay' : undefined;
      if (!effect) throw new Error(`unknown effect "${e}" (overlay, spotlight, cutout)`);
      const outline = num(args.outline);
      app.setEffect(effect, Number.isFinite(outline) ? Math.min(6, Math.max(0, Math.round(outline))) : undefined);
      return {effect};
    },

    remove_objects(args) {
      const objs = app.objects();
      const keep = strs(args.keep), labels = strs(args.labels ?? args.label ?? args.objects);
      if (bool(args.all) || (!keep.length && !labels.length)) {
        app.removeAll();
        return {removed: objs.length, kept: 0};
      }
      const gone = objs.filter((o) => keep.length ? !keep.some((k) => matches(o.label, o.id, k))
        : labels.some((k) => matches(o.label, o.id, k)));
      app.removeObjects(gone.map((o) => o.id));
      return {removed: gone.length, kept: objs.length - gone.length};
    },

    async playback(args) {
      const a = str(args.action).toLowerCase();
      const action: PlaybackAction | undefined = /track|follow/.test(a) ? 'track' : /restart|beginning|start over|rewind/.test(a) ? 'restart'
        : /pause/.test(a) ? 'pause' : /stop|halt/.test(a) ? 'stop' : /play|resume|go/.test(a) ? 'play' : undefined;
      if (!action) throw new Error(`unknown playback action "${a}"`);
      return {status: await app.playback(action)};
    },

    async use_camera(args) {
      const on = bool(args.on) ?? bool(args.enabled) ?? true;
      return {status: await app.useCamera(on)};
    },

    async set_quality(args) {
      const size = num(args.size), memory = num(args.memory);
      const s = [384, 512, 1024].includes(size) ? size as 384 | 512 | 1024 : undefined;
      const m = [2, 7].includes(memory) ? memory as 2 | 7 : undefined;
      if (!s && !m) throw new Error('set_quality needs size (384, 512, 1024) or memory (2, 7)');
      return {status: await app.setQuality(s, m)};
    },

    describe_scene: () => app.describeScene(),
    measure: () => app.measure(),
  };
}
