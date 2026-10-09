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
 * The host-side steps around CLIPSeg's three graphs: image preprocessing,
 * the token-embedding lookup that feeds the text graph, and the projection
 * of its EOT row into the decoder's conditional vector. Pure functions on
 * typed arrays (no DOM, no LiteRT), so each step can be checked against the
 * Python reference on the same numbers.
 */

export const SIZE = 352; // vision input [1, 3, 352, 352], logits [1, 352, 352]
export const PATCHES = 485; // 22 x 22 patches + the class token
export const VISION_DIM = 768; // t3 / t6 / t9 are [1, 485, 768]
export const TEXT_DIM = 512; // token embeddings and text hidden states
export const COND_DIM = 512; // conditional vector [1, 512]
// A pixel is in the mask when sigmoid(logit) is above this (logit > 0).
export const MASK_THRESHOLD = 0.5;
const MASK_LOGIT = Math.log(MASK_THRESHOLD / (1 - MASK_THRESHOLD));

// --- image ---------------------------------------------------------------

// Pillow's BILINEAR resize, ported exactly: a triangle filter whose support
// widens with the downscale factor (antialiased), fixed-point weights, a
// horizontal pass then a vertical pass, each rounded to uint8. This is the
// resize inside the Hugging Face image processor for CLIPSeg (352 x 352,
// no crop), so the page feeds the vision graph the same pixels as Python.
const PRECISION_BITS = 22; // Pillow: 32 - 8 - 2
const coeffCache = new Map();

function coefficients(inSize, outSize) {
  const key = `${inSize}>${outSize}`;
  const hit = coeffCache.get(key);
  if (hit) return hit;
  const scale = inSize / outSize;
  const filterScale = Math.max(scale, 1);
  const support = filterScale; // bilinear support is 1
  const ksize = Math.ceil(support) * 2 + 1;
  const bounds = new Int32Array(outSize * 2);
  const weights = new Int32Array(outSize * ksize);
  const w = new Float64Array(ksize);
  const ss = 1 / filterScale; // multiply by the reciprocal, as Pillow does
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const xmin = Math.max(Math.trunc(center - support + 0.5), 0);
    const count = Math.min(Math.trunc(center + support + 0.5), inSize) - xmin;
    let sum = 0;
    for (let x = 0; x < count; x++) {
      const t = Math.abs((x + xmin - center + 0.5) * ss);
      w[x] = t < 1 ? 1 - t : 0;
      sum += w[x];
    }
    for (let x = 0; x < count; x++) {
      const v = sum !== 0 ? w[x] / sum : w[x];
      weights[xx * ksize + x] = Math.trunc(0.5 + v * (1 << PRECISION_BITS));
    }
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = count;
  }
  const result = { ksize, bounds, weights };
  coeffCache.set(key, result);
  return result;
}

const clip8 = (v) => {
  const x = v >> PRECISION_BITS;
  return x < 0 ? 0 : x > 255 ? 255 : x;
};

/**
 * RGBA pixels (width x height) → SIZE x SIZE RGB, HWC uint8.
 * @param {Uint8ClampedArray|Uint8Array} rgba
 */
export function resizeToInput(rgba, width, height) {
  const h = coefficients(width, SIZE);
  const v = coefficients(height, SIZE);
  // Only the source rows the vertical pass reads.
  const yFirst = v.bounds[0];
  const yLast = v.bounds[(SIZE - 1) * 2] + v.bounds[(SIZE - 1) * 2 + 1];
  const rows = yLast - yFirst;
  const tmp = new Uint8Array(rows * SIZE * 3);
  for (let y = 0; y < rows; y++) {
    const rowBase = (y + yFirst) * width * 4;
    for (let xx = 0; xx < SIZE; xx++) {
      const xmin = h.bounds[xx * 2];
      const count = h.bounds[xx * 2 + 1];
      const k = xx * h.ksize;
      let r = 1 << (PRECISION_BITS - 1);
      let g = r;
      let b = r;
      for (let x = 0; x < count; x++) {
        const wgt = h.weights[k + x];
        const p = rowBase + (x + xmin) * 4;
        r += rgba[p] * wgt;
        g += rgba[p + 1] * wgt;
        b += rgba[p + 2] * wgt;
      }
      const o = (y * SIZE + xx) * 3;
      tmp[o] = clip8(r);
      tmp[o + 1] = clip8(g);
      tmp[o + 2] = clip8(b);
    }
  }
  const out = new Uint8Array(SIZE * SIZE * 3);
  for (let yy = 0; yy < SIZE; yy++) {
    const ymin = v.bounds[yy * 2] - yFirst;
    const count = v.bounds[yy * 2 + 1];
    const k = yy * v.ksize;
    for (let xx = 0; xx < SIZE; xx++) {
      let r = 1 << (PRECISION_BITS - 1);
      let g = r;
      let b = r;
      for (let y = 0; y < count; y++) {
        const wgt = v.weights[k + y];
        const p = ((y + ymin) * SIZE + xx) * 3;
        r += tmp[p] * wgt;
        g += tmp[p + 1] * wgt;
        b += tmp[p + 2] * wgt;
      }
      const o = (yy * SIZE + xx) * 3;
      out[o] = clip8(r);
      out[o + 1] = clip8(g);
      out[o + 2] = clip8(b);
    }
  }
  return out;
}

// The processor's rescale and normalize as float32 lookup tables, rounded at
// every step like Python: float32(v * rescale_factor in float64), then
// float32 (x - mean) and float32 (... / std). The input tensor therefore
// matches the Python reference bit for bit.
const RESCALE = 0.00392156862745098; // rescale_factor in preprocessor_config.json
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const NORM_LUT = MEAN.map((mean, c) => {
  const lut = new Float32Array(256);
  const m = Math.fround(mean);
  const s = Math.fround(STD[c]);
  for (let v = 0; v < 256; v++) {
    lut[v] = Math.fround(Math.fround(Math.fround(v * RESCALE) - m) / s);
  }
  return lut;
});

/** SIZE x SIZE RGB uint8 (HWC) → normalized NCHW float32 [1, 3, SIZE, SIZE]. */
export function toInputTensor(rgb) {
  const plane = SIZE * SIZE;
  const nchw = new Float32Array(3 * plane);
  const [lr, lg, lb] = NORM_LUT;
  for (let i = 0; i < plane; i++) {
    nchw[i] = lr[rgb[i * 3]];
    nchw[plane + i] = lg[rgb[i * 3 + 1]];
    nchw[2 * plane + i] = lb[rgb[i * 3 + 2]];
  }
  return nchw;
}

// --- text ----------------------------------------------------------------

/** IEEE half → float, exact for every bit pattern (no Float16Array). */
function halfToFloat(h) {
  const sign = h & 0x8000 ? -1 : 1;
  const exponent = (h >> 10) & 0x1f;
  const mantissa = h & 0x3ff;
  if (exponent === 0) return sign * mantissa * 2 ** -24; // zero, subnormal
  if (exponent === 31) return mantissa ? NaN : sign * Infinity;
  return sign * (1024 + mantissa) * 2 ** (exponent - 25);
}

let halfTable = null;
function halves() {
  if (!halfTable) {
    halfTable = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) halfTable[h] = halfToFloat(h);
  }
  return halfTable;
}

/**
 * Raw little-endian float16 bytes → a Uint16Array view of them. The
 * embedding table (49408 x 512) stays in half precision; rows are widened
 * on lookup.
 * @param {Uint8Array} bytes
 */
export function halfArray(bytes) {
  if (bytes.byteLength % 2) throw new Error('float16 data with an odd byte count');
  // Copy if the view is unaligned for 16-bit access.
  const aligned = bytes.byteOffset % 2 ? bytes.slice() : bytes;
  return new Uint16Array(aligned.buffer, aligned.byteOffset, aligned.byteLength / 2);
}

/** float16 values → float32. */
export function widen(half) {
  const table = halves();
  const out = new Float32Array(half.length);
  for (let i = 0; i < half.length; i++) out[i] = table[half[i]];
  return out;
}

/**
 * Token ids → the text graph's input [1, 77, 512]: one row of the float16
 * embedding table per token, widened to float32.
 * @param {Int32Array} ids
 * @param {Uint16Array} table token_embedding_f16.bin
 */
export function embedTokens(ids, table) {
  const lut = halves();
  const out = new Float32Array(ids.length * TEXT_DIM);
  for (let t = 0; t < ids.length; t++) {
    const src = ids[t] * TEXT_DIM;
    if (src + TEXT_DIM > table.length) throw new Error(`token id ${ids[t]} is outside the embedding table`);
    const dst = t * TEXT_DIM;
    for (let i = 0; i < TEXT_DIM; i++) out[dst + i] = lut[table[src + i]];
  }
  return out;
}

/**
 * The text graph's hidden state at the EOT position, projected into the
 * decoder's conditional vector: cond[o] = Σ_i h[eot][i] · W[i·512 + o]
 * (text_projection_f16.bin is stored input-major, as ClipSeg.kt reads it).
 * @param {Float32Array} hidden [77 x 512]
 * @param {number} eot
 * @param {Float32Array} projection [512 x 512], widened
 */
export function project(hidden, eot, projection) {
  const acc = new Float64Array(COND_DIM);
  const row = eot * TEXT_DIM;
  for (let i = 0; i < TEXT_DIM; i++) {
    const h = hidden[row + i];
    const base = i * COND_DIM;
    for (let o = 0; o < COND_DIM; o++) acc[o] += h * projection[base + o];
  }
  return Float32Array.from(acc);
}

// --- result --------------------------------------------------------------

/** How many pixels are in the mask (sigmoid above MASK_THRESHOLD). */
export function maskPixels(logits) {
  let n = 0;
  for (let i = 0; i < logits.length; i++) if (logits[i] > MASK_LOGIT) n++;
  return n;
}

/** FNV-1a over bytes — a cheap fingerprint of the exact model input. */
export function fnv1a(bytes) {
  let h = 0x811c9dc5;
  for (let i = 0; i < bytes.length; i++) {
    h ^= bytes[i];
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, '0');
}
