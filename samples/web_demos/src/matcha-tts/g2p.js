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
 * G2P: 275k-word espeak-IPA dictionary (OpenPhonemizer, primary) with a
 * DeepPhonemizer transformer on LiteRT WASM as the out-of-dictionary fallback.
 * Port of the validated kotlin_replica.py host logic:
 *   word → [<en_us>] + each char ×3 + [<end>] → dp_g2p [1,96]f32 →
 *   logits [1,96,64] → per-position argmax → collapse consecutive repeats →
 *   drop specials → IPA string (acronym hyphens stripped).
 */
import { Tensor } from '@litertjs/core';

const TOKEN = /[a-z']+|\d+(?:\.\d+)?|[.,!?;:—…"]/g;
const WORD = /^[a-z']+$/;

const ONES = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
  'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen',
  'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen'];
const TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty',
  'seventy', 'eighty', 'ninety'];

function intToWords(n) {
  if (n < 20) return ONES[n];
  if (n < 100) return TENS[(n / 10) | 0] + (n % 10 ? ' ' + ONES[n % 10] : '');
  if (n < 1000) return ONES[(n / 100) | 0] + ' hundred' + (n % 100 ? ' ' + intToWords(n % 100) : '');
  for (const [div, name] of [[1e9, 'billion'], [1e6, 'million'], [1e3, 'thousand']]) {
    if (n >= div) {
      const head = intToWords(Math.floor(n / div)) + ' ' + name;
      return head + (n % div ? ' ' + intToWords(n % div) : '');
    }
  }
  return ONES[0];
}

function numberToWords(tok) {
  const [int, frac] = tok.split('.');
  let out = int.length > 12
    ? [...int].map((d) => ONES[+d]).join(' ')
    : intToWords(parseInt(int, 10));
  if (frac !== undefined) out += ' point ' + [...frac].map((d) => ONES[+d]).join(' ');
  return out;
}

export class G2P {
  /**
   * @param dict Map<string,string> word → espeak-IPA
   * @param meta g2p_meta.json contents
   * @param model dp_g2p compiled model (WASM) or null (dictionary-only)
   */
  constructor(dict, meta, model) {
    this.dict = dict;
    this.meta = meta;
    this.model = model;
    this.special = new Set(meta.special);
    this.cache = new Map();
  }

  async wordToIpa(word) {
    const hit = this.dict.get(word);
    if (hit !== undefined) return hit;
    let ipa = this.cache.get(word);
    if (ipa === undefined) {
      ipa = this.model ? await this.neural(word) : '';
      this.cache.set(word, ipa);
    }
    return ipa;
  }

  async neural(word) {
    const { char2idx, idx2ph, char_repeats: rep, start, end, MAXT, n_phonemes: nph } = this.meta;
    const ids = [start];
    for (const ch of word) {
      const id = char2idx[ch];
      if (id !== undefined) for (let r = 0; r < rep; r++) ids.push(id);
    }
    ids.push(end);
    const L = Math.min(ids.length, MAXT);
    const input = new Float32Array(MAXT);
    for (let i = 0; i < L; i++) input[i] = ids[i];
    const inT = Tensor.fromTypedArray(input, [1, MAXT]);
    const outs = await this.model.run([inT]);
    const logits = await outs[0].data(); // [1, MAXT, nph] flattened
    for (const o of outs) o.delete();
    inT.delete();

    let prev = -1;
    let out = '';
    for (let pos = 0; pos < L; pos++) {
      let best = 0;
      let bestV = -Infinity;
      for (let k = 0; k < nph; k++) {
        const v = logits[pos * nph + k];
        if (v > bestV) { bestV = v; best = k; }
      }
      if (best === prev) continue;
      prev = best;
      const ph = idx2ph[best];
      if (best === 0 || ph === undefined || this.special.has(ph)) continue;
      out += ph.replaceAll('-', '');
    }
    return out;
  }
}

/**
 * Text → chunks of Matcha symbol ids (each ≤ maxPids, the 256-slot budget after
 * blank interspersing). Words join with a single space; punctuation attaches
 * directly; a sentence-final mark closes the chunk (Matcha is trained on
 * single sentences).
 * @returns [{ids: number[], ipa: string}]
 */
export async function phonemize(g2p, symToId, text, maxPids = 127) {
  const spaceId = symToId.get(' ');
  const pieces = [];
  for (const m of text.toLowerCase().matchAll(TOKEN)) {
    const tok = m[0];
    let words = null;
    if (/^\d/.test(tok)) words = numberToWords(tok).split(' ');
    else if (WORD.test(tok)) words = [tok];
    if (words) {
      for (const w of words) {
        const ipa = await g2p.wordToIpa(w);
        if (!ipa) continue;
        const ids = [];
        for (const ch of ipa) {
          const id = symToId.get(ch);
          if (id !== undefined) ids.push(id);
        }
        if (ids.length) pieces.push({ ids, ipa, isWord: true, sentenceEnd: false });
      }
    } else {
      // '!' gets an awkward pause from the duration model (rare in LJSpeech) —
      // speak it as a period.
      const norm = tok === '!' ? '.' : tok;
      const id = symToId.get(norm);
      if (id !== undefined) {
        pieces.push({ ids: [id], ipa: norm, isWord: false, sentenceEnd: /[.!?…]/.test(norm) });
      }
    }
  }

  const chunks = [];
  let cur = { ids: [], ipa: '' };
  const flush = () => {
    if (cur.ids.length) chunks.push(cur);
    cur = { ids: [], ipa: '' };
  };
  for (const p of pieces) {
    const sep = p.isWord && cur.ids.length ? 1 : 0;
    if (cur.ids.length + sep + p.ids.length > maxPids) flush();
    if (p.isWord && cur.ids.length) { cur.ids.push(spaceId); cur.ipa += ' '; }
    cur.ids.push(...p.ids);
    cur.ipa += p.ipa;
    if (p.sentenceEnd) flush();
  }
  flush();
  return chunks;
}
