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
 * SAM 2.1 host-side math: image preprocessing, the point prompt encoder,
 * tensor binding by shape, and mask upsampling. No DOM, no LiteRT.js.
 *
 * Everything here mirrors the Python usage on the model cards
 * (litert-community/SAM2.1-Hiera-Tiny-Image-Encoder and -Mask-Decoder), so a
 * Python LiteRT run gets the same tensors from the same decoded pixels.
 */

/** Model input side: the photo is squashed to SIZE×SIZE (no letterbox). */
export const SIZE = 1024;
/** Side of the low-res mask logits the decoder returns. */
export const MASK_SIZE = 256;

const MEAN = [0.485, 0.456, 0.406].map(Math.fround);
const STD = [0.229, 0.224, 0.225].map(Math.fround);

// Decoder-ready encoder outputs / decoder inputs, by shape. Two of them hold
// the same number of floats (256·64·64 = 64·128·128 = 1,048,576), so tensors
// are matched by their full shape, never by element count.
export const SHAPES = {
  image: [1, 3, SIZE, SIZE],
  imageEmbeddings: [1, 256, 64, 64],
  featS1: [1, 64, 128, 128],
  featS0: [1, 32, 256, 256],
  sparsePrompt: [1, 2, 256],
  predMasks: [1, 3, MASK_SIZE, MASK_SIZE],
  iouScores: [1, 3],
};

// Decoder input order of the published .tflite. LiteRT.js 2.5.3 names the
// inputs args_0..args_3 in this order (?debug=1 logs them as SAM2_IO); the
// binding goes by shape, and the names only add a position check.
export const DECODER_INPUT_ORDER = ['imageEmbeddings', 'sparsePrompt', 'featS1', 'featS0'];

const sameShape = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

/**
 * Map each wanted key to the tensor detail with that exact shape.
 * `details` = model.getInputDetails() / getOutputDetails(). When `order` is
 * given and the names follow the args_N pattern, also check that args_N sits
 * at position N of `order` (catches a re-exported graph with shuffled inputs).
 */
export function bindByShape(details, keys, order = null) {
  const bound = {};
  for (const key of keys) {
    const want = SHAPES[key];
    const hits = details.filter((d) => sameShape([...d.shape], want));
    if (hits.length !== 1) {
      const have = details.map((d) => `${d.name}[${[...d.shape].join(',')}]`).join(' ');
      throw new Error(`expected one tensor of shape [${want.join(',')}] for ${key}, model has ${have}`);
    }
    const detail = hits[0];
    const arg = /args_(\d+)$/.exec(detail.name);
    if (order && arg && order[Number(arg[1])] !== key) {
      throw new Error(`tensor ${detail.name} has the shape of ${key} but the position of ${order[Number(arg[1])]}`);
    }
    bound[key] = detail;
  }
  return bound;
}

// --- preprocessing ---------------------------------------------------------

// Pillow's resampler (libImaging/Resample.c), bilinear filter: a triangle
// filter whose support widens with the downscale factor (antialiased), with
// 22-bit fixed-point coefficients, a horizontal pass, then a vertical pass,
// each rounded to 8 bits. Reproduced exactly so the page and
// PIL.Image.resize(..., Image.BILINEAR) give byte-identical inputs from the
// same decoded pixels.
const PRECISION_BITS = 32 - 8 - 2;

function precomputeCoeffs(inSize, outSize) {
  const scale = inSize / outSize;
  const filterScale = Math.max(scale, 1);
  const support = 1.0 * filterScale; // bilinear support = 1
  const ksize = Math.ceil(support) * 2 + 1;
  const bounds = new Int32Array(outSize * 2);
  const coeffs = new Int32Array(outSize * ksize);
  const weights = new Float64Array(ksize);
  const invScale = 1.0 / filterScale;
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    let total = 0;
    for (let x = 0; x < xmax; x++) {
      const t = Math.abs((x + xmin - center + 0.5) * invScale);
      const w = t < 1 ? 1 - t : 0;
      weights[x] = w;
      total += w;
    }
    for (let x = 0; x < xmax; x++) {
      const w = total !== 0 ? weights[x] / total : weights[x];
      coeffs[xx * ksize + x] = w < 0
        ? Math.trunc(-0.5 + w * (1 << PRECISION_BITS))
        : Math.trunc(0.5 + w * (1 << PRECISION_BITS));
    }
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  return { ksize, bounds, coeffs };
}

const clip8 = (v) => (v <= 0 ? 0 : v >= 256 << PRECISION_BITS ? 255 : v >> PRECISION_BITS);

/**
 * Resize RGBA pixels (canvas ImageData layout) to `dw`×`dh` RGB, exactly as
 * Pillow's bilinear resize does. Returns a Uint8Array of dw·dh·3.
 */
export function resizeBilinear(rgba, sw, sh, dw, dh) {
  // Planar RGB working copy of the source.
  let width = sw;
  let height = sh;
  let rgb = new Uint8Array(sw * sh * 3);
  for (let i = 0, j = 0; i < sw * sh; i++, j += 4) {
    rgb[i * 3] = rgba[j];
    rgb[i * 3 + 1] = rgba[j + 1];
    rgb[i * 3 + 2] = rgba[j + 2];
  }
  const horiz = precomputeCoeffs(sw, dw);
  const vert = precomputeCoeffs(sh, dh);
  const half = 1 << (PRECISION_BITS - 1);

  // Horizontal pass over the source rows the vertical pass will read.
  if (sw !== dw) {
    const yFirst = vert.bounds[0];
    const yLast = vert.bounds[(dh - 1) * 2] + vert.bounds[(dh - 1) * 2 + 1];
    const rows = yLast - yFirst;
    const out = new Uint8Array(dw * rows * 3);
    const { ksize, bounds, coeffs } = horiz;
    for (let y = 0; y < rows; y++) {
      const rowIn = (y + yFirst) * sw * 3;
      const rowOut = y * dw * 3;
      for (let xx = 0; xx < dw; xx++) {
        const xmin = bounds[xx * 2];
        const xmax = bounds[xx * 2 + 1];
        const k = xx * ksize;
        let s0 = half;
        let s1 = half;
        let s2 = half;
        for (let x = 0; x < xmax; x++) {
          const p = rowIn + (x + xmin) * 3;
          const c = coeffs[k + x];
          s0 += rgb[p] * c;
          s1 += rgb[p + 1] * c;
          s2 += rgb[p + 2] * c;
        }
        out[rowOut + xx * 3] = clip8(s0);
        out[rowOut + xx * 3 + 1] = clip8(s1);
        out[rowOut + xx * 3 + 2] = clip8(s2);
      }
    }
    for (let yy = 0; yy < dh; yy++) vert.bounds[yy * 2] -= yFirst;
    rgb = out;
    width = dw;
    height = rows;
  }

  // Vertical pass.
  if (sh !== dh) {
    const out = new Uint8Array(width * dh * 3);
    const { ksize, bounds, coeffs } = vert;
    for (let yy = 0; yy < dh; yy++) {
      const ymin = bounds[yy * 2];
      const ymax = bounds[yy * 2 + 1];
      const k = yy * ksize;
      const rowOut = yy * width * 3;
      for (let xx = 0; xx < width; xx++) {
        let s0 = half;
        let s1 = half;
        let s2 = half;
        for (let y = 0; y < ymax; y++) {
          const p = ((y + ymin) * width + xx) * 3;
          const c = coeffs[k + y];
          s0 += rgb[p] * c;
          s1 += rgb[p + 1] * c;
          s2 += rgb[p + 2] * c;
        }
        out[rowOut + xx * 3] = clip8(s0);
        out[rowOut + xx * 3 + 1] = clip8(s1);
        out[rowOut + xx * 3 + 2] = clip8(s2);
      }
    }
    rgb = out;
    height = dh;
  }
  if (width !== dw || height !== dh) throw new Error('resize produced the wrong size');
  return rgb;
}

/**
 * SIZE×SIZE RGB bytes → [1,3,SIZE,SIZE] float32, ImageNet-normalized. Each
 * step is rounded to float32 like numpy's `(x / 255 - mean) / std` on a
 * float32 array, so the tensor matches the Python reference bit for bit.
 */
export function toNchw(rgb) {
  const plane = SIZE * SIZE;
  const nchw = new Float32Array(3 * plane);
  for (let c = 0; c < 3; c++) {
    const mean = MEAN[c];
    const std = STD[c];
    const base = c * plane;
    for (let i = 0; i < plane; i++) {
      nchw[base + i] = Math.fround(Math.fround(rgb[i * 3 + c] / 255) - mean) / std;
    }
  }
  return nchw;
}

// --- prompt encoding -------------------------------------------------------

/**
 * prompt_encode_const.bin (3,072 bytes, float32 little-endian):
 * posmat [2,128] (row 0 = x projection, row 1 = y) | point_embed[1] [256]
 * (positive point) | not_a_point [256] (padding point).
 */
export function parsePromptConstants(bytes) {
  if (bytes.byteLength !== 3072) {
    throw new Error(`prompt_encode_const.bin is ${bytes.byteLength} bytes, expected 3072`);
  }
  const all = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + 3072));
  return {
    posmat: all.subarray(0, 256),
    pointEmbed: all.subarray(256, 512),
    notAPoint: all.subarray(512, 768),
  };
}

/**
 * One positive click at (x, y) in SIZE×SIZE model space → sparse_prompt
 * [1,2,256]: token 0 = random-Fourier positional encoding of the point +
 * the positive-point embedding, token 1 = the padding point (no box).
 */
export function encodePoint(consts, x, y) {
  const { posmat, pointEmbed, notAPoint } = consts;
  const cx = 2 * ((x + 0.5) / SIZE) - 1;
  const cy = 2 * ((y + 0.5) / SIZE) - 1;
  const out = new Float32Array(512);
  for (let k = 0; k < 128; k++) {
    const proj = 2 * Math.PI * (cx * posmat[k] + cy * posmat[128 + k]);
    out[k] = Math.sin(proj) + pointEmbed[k];
    out[128 + k] = Math.cos(proj) + pointEmbed[128 + k];
  }
  out.set(notAPoint, 256);
  return out;
}

// --- masks -----------------------------------------------------------------

/** Index of the largest predicted IoU (the mask the page shows). */
export function argmax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

/** Pixels with logit > 0 in one 256×256 mask of the [1,3,256,256] output. */
export function countPositive(logits, index) {
  const plane = MASK_SIZE * MASK_SIZE;
  let n = 0;
  for (let i = index * plane; i < (index + 1) * plane; i++) if (logits[i] > 0) n++;
  return n;
}

/** Whether (x, y) in SIZE×SIZE model space falls inside one mask (logit > 0
 * in its 256×256 cell). */
export function maskContains(logits, index, x, y) {
  const cell = (v) => Math.min(MASK_SIZE - 1, Math.floor((v * MASK_SIZE) / SIZE));
  return logits[index * MASK_SIZE * MASK_SIZE + cell(y) * MASK_SIZE + cell(x)] > 0;
}

/**
 * Upsample one 256×256 logit map to `w`×`h` (bilinear, half-pixel centers =
 * torch interpolate align_corners=False, as SAM 2 postprocesses) and
 * threshold at 0. Returns a Uint8Array of w·h with 1 inside the mask.
 */
export function upsampleMask(logits, index, w, h) {
  const n = MASK_SIZE;
  const base = index * n * n;
  const xs0 = new Int32Array(w);
  const xs1 = new Int32Array(w);
  const xw = new Float32Array(w);
  for (let x = 0; x < w; x++) {
    let s = ((x + 0.5) * n) / w - 0.5;
    if (s < 0) s = 0;
    const x0 = Math.min(Math.floor(s), n - 1);
    xs0[x] = x0;
    xs1[x] = Math.min(x0 + 1, n - 1);
    xw[x] = s - x0;
  }
  const mask = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    let s = ((y + 0.5) * n) / h - 0.5;
    if (s < 0) s = 0;
    const y0 = Math.min(Math.floor(s), n - 1);
    const y1 = Math.min(y0 + 1, n - 1);
    const wy = s - y0;
    const r0 = base + y0 * n;
    const r1 = base + y1 * n;
    for (let x = 0; x < w; x++) {
      const a = logits[r0 + xs0[x]];
      const b = logits[r0 + xs1[x]];
      const c = logits[r1 + xs0[x]];
      const d = logits[r1 + xs1[x]];
      const top = a + (b - a) * xw[x];
      const bottom = c + (d - c) * xw[x];
      if (top + (bottom - top) * wy > 0) mask[y * w + x] = 1;
    }
  }
  return mask;
}
