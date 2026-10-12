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
 * CLIP's byte-level BPE tokenizer (vocab.json + merges.txt), ported from the
 * Hugging Face CLIPTokenizer that CLIPSeg ships with, so a typed prompt
 * becomes the same 77 token ids as in Python:
 *   NFC → every whitespace run to one space → lowercase, character by
 *   character → the special tokens <|startoftext|> / <|endoftext|> taken
 *   out as they are → CLIP's pre-token pattern (contractions, letter runs,
 *   single digits, other non-space runs) → each piece's UTF-8 bytes in
 *   GPT-2's printable byte alphabet → BPE by merge rank, with "</w>" on the
 *   last symbol of the piece
 *   → [BOS, up to 75 tokens, EOT, EOT padding…] (77 ids).
 * The EOT position is the first 49407, i.e. argmax(ids): the row CLIPSeg
 * pools the text features from.
 */

export const BOS = 49406; // <|startoftext|>
export const EOT = 49407; // <|endoftext|>, also the padding id
export const CONTEXT_LENGTH = 77;
export const MAX_TOKENS = CONTEXT_LENGTH - 2; // between BOS and EOT; the rest is cut

// Whitespace is Unicode White_Space, as in the Rust regex the Hugging Face
// tokenizer runs; JavaScript's \s differs (it includes U+FEFF and misses
// U+0085).
const WHITESPACE = /\p{White_Space}+/gu;
const SPECIAL = /(<\|startoftext\|>|<\|endoftext\|>)/u;
const PRETOKEN = /'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}|[^\p{White_Space}\p{L}\p{N}]+/gu;

/** GPT-2's byte → printable character table: bytes that are already
 * printable map to themselves, the rest to code points from U+0100 up. */
function byteToUnicode() {
  const bytes = [];
  for (let b = 0x21; b <= 0x7e; b++) bytes.push(b);
  for (let b = 0xa1; b <= 0xac; b++) bytes.push(b);
  for (let b = 0xae; b <= 0xff; b++) bytes.push(b);
  const table = new Array(256);
  for (const b of bytes) table[b] = String.fromCharCode(b);
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (table[b] === undefined) table[b] = String.fromCharCode(256 + n++);
  }
  return table;
}

const BYTE_TO_UNICODE = byteToUnicode();
const utf8 = new TextEncoder();

export class ClipTokenizer {
  /**
   * @param {Record<string, number>} vocab parsed vocab.json (token → id)
   * @param {string} merges merges.txt (a version line, then one merge per line)
   */
  constructor(vocab, merges) {
    this.vocab = new Map(Object.entries(vocab));
    // Merge rank by pair; symbols never contain a space (the byte alphabet
    // maps it to "Ġ"), so "first second" is an unambiguous key.
    this.ranks = new Map();
    const lines = merges.split('\n');
    for (let i = 1; i < lines.length; i++) {
      const pair = lines[i].trim().split(/\s+/);
      if (pair.length === 2) this.ranks.set(`${pair[0]} ${pair[1]}`, this.ranks.size);
    }
    this.cache = new Map();
  }

  /** One pre-token (already in the byte alphabet) → its BPE symbols. */
  bpe(piece) {
    const hit = this.cache.get(piece);
    if (hit) return hit;
    const chars = Array.from(piece);
    let word = chars.slice(0, -1);
    word.push(chars[chars.length - 1] + '</w>');
    while (word.length > 1) {
      // The lowest-ranked adjacent pair is merged everywhere it occurs,
      // left to right, as in the reference implementation.
      let best = -1;
      let bestRank = Infinity;
      for (let i = 0; i < word.length - 1; i++) {
        const rank = this.ranks.get(`${word[i]} ${word[i + 1]}`);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          best = i;
        }
      }
      if (best < 0) break;
      const first = word[best];
      const second = word[best + 1];
      const merged = [];
      for (let i = 0; i < word.length;) {
        if (i < word.length - 1 && word[i] === first && word[i + 1] === second) {
          merged.push(first + second);
          i += 2;
        } else {
          merged.push(word[i]);
          i += 1;
        }
      }
      word = merged;
    }
    this.cache.set(piece, word);
    return word;
  }

  /**
   * @param {string} text the prompt as typed
   * @returns {{ids: Int32Array, eot: number, count: number}} the 77 ids, the
   *   EOT position, and the token count before truncation
   */
  encode(text) {
    const clean = Array.from(
      text.normalize('NFC').replace(WHITESPACE, ' '),
      (c) => c.toLowerCase(),
    ).join('');
    const tokens = [];
    for (const part of clean.split(SPECIAL)) {
      if (part === '<|startoftext|>') {
        tokens.push(BOS);
      } else if (part === '<|endoftext|>') {
        tokens.push(EOT);
      } else {
        for (const [piece] of part.matchAll(PRETOKEN)) {
          let mapped = '';
          for (const b of utf8.encode(piece)) mapped += BYTE_TO_UNICODE[b];
          for (const symbol of this.bpe(mapped)) {
            const id = this.vocab.get(symbol);
            // Every byte and every merge result is in CLIP's vocabulary, so a
            // miss means vocab.json and merges.txt do not belong together.
            // Mapping it to EOT would end the prompt there without a word:
            // the text features are read at the first EOT.
            if (id === undefined) throw new Error(`could not tokenize “${piece}” (a symbol missing from vocab.json)`);
            tokens.push(id);
          }
        }
      }
    }
    const ids = new Int32Array(CONTEXT_LENGTH).fill(EOT);
    ids[0] = BOS;
    const n = Math.min(tokens.length, MAX_TOKENS);
    for (let i = 0; i < n; i++) ids[i + 1] = tokens[i];
    return { ids, eot: ids.indexOf(EOT), count: tokens.length };
  }
}
