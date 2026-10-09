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
 * The host-side steps around RF-DETR's two graphs: preprocessing, the
 * two-stage query selection between Graph A and Graph B, and decoding.
 * Pure functions on typed arrays (no DOM, no LiteRT), so each step can be
 * checked against the Python reference on the same numbers.
 */

export const SIZE = 384; // model input is [1, 3, 384, 384]
export const NPROP = 576; // 24x24 encoder proposals
export const NQ = 300; // decoder queries
export const NCLS = 91; // COCO id space (index == COCO category id)
export const HID = 256;
// The model card's post-processing values (the Android sample RfDetr.kt uses
// the same): keep score > 0.45, then a light per-class NMS at IoU 0.6 that
// cleans near-duplicate queries.
export const SCORE_THRESH = 0.45;
export const IOU_THRESH = 0.6;

// --- preprocessing ---------------------------------------------------------

// Pillow's BILINEAR resize, ported exactly: a triangle filter whose support
// widens with the downscale factor (antialiased, like the torchvision bilinear
// resize RF-DETR uses), 8-bit fixed-point weights, a horizontal pass then a
// vertical pass, each rounded to uint8. The page therefore feeds Graph A the
// same pixels as `Image.resize((384, 384), Image.BILINEAR)` in Python.
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

// ImageNet mean/std as float32 lookup tables, rounded at every step like
// numpy's float32 `(x / 255 - mean) / std`, so the input tensor matches the
// Python reference bit for bit.
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const NORM_LUT = MEAN.map((mean, c) => {
  const lut = new Float32Array(256);
  const m = Math.fround(mean);
  const s = Math.fround(STD[c]);
  for (let v = 0; v < 256; v++) {
    lut[v] = Math.fround(Math.fround(Math.fround(v / 255) - m) / s);
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

// --- between the graphs ----------------------------------------------------

/**
 * Two-stage query selection: the top-300 encoder proposals by their max class
 * logit, in descending order (stable, so ties keep the lower index), and
 * their enc_coord boxes gathered into refpoint_ts [1, 300, 4]. The order
 * matters — Graph B pairs query i with its learned i-th query embedding.
 */
export function selectQueries(encClass, encCoord) {
  const maxLogit = new Float32Array(NPROP);
  for (let p = 0; p < NPROP; p++) {
    let m = -Infinity;
    const base = p * NCLS;
    for (let c = 0; c < NCLS; c++) {
      if (encClass[base + c] > m) m = encClass[base + c];
    }
    maxLogit[p] = m;
  }
  const order = Array.from({ length: NPROP }, (_, i) => i);
  order.sort((a, b) => maxLogit[b] - maxLogit[a]); // Array.sort is stable
  const top = Int32Array.from(order.slice(0, NQ));
  const refpoints = new Float32Array(NQ * 4);
  for (let i = 0; i < NQ; i++) {
    refpoints.set(encCoord.subarray(top[i] * 4, top[i] * 4 + 4), i * 4);
  }
  return { top, refpoints };
}

// --- decoding --------------------------------------------------------------

function iou(a, b) {
  const iw = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const ih = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = iw * ih;
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return union > 0 ? inter / union : 0;
}

/**
 * Graph B outputs → detections, highest score first:
 * score = sigmoid(max logit), cls = argmax (index = COCO id, 0 = none),
 * keep score > 0.45 and cls > 0, cxcywh → xyxy (normalized to the image),
 * then per-class NMS at IoU 0.6.
 * @returns {{cls: number, score: number, xyxy: number[]}[]}
 */
export function decode(boxes, logits) {
  const candidates = [];
  for (let q = 0; q < NQ; q++) {
    let best = -Infinity;
    let cls = -1;
    const base = q * NCLS;
    for (let c = 0; c < NCLS; c++) {
      if (logits[base + c] > best) {
        best = logits[base + c];
        cls = c;
      }
    }
    const score = 1 / (1 + Math.exp(-best));
    if (!(score > SCORE_THRESH) || cls <= 0) continue;
    const [cx, cy, w, h] = boxes.subarray(q * 4, q * 4 + 4);
    candidates.push({ cls, score, xyxy: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2] });
  }
  const kept = [];
  const classes = [...new Set(candidates.map((d) => d.cls))];
  for (const cls of classes) {
    const group = candidates.filter((d) => d.cls === cls).sort((a, b) => b.score - a.score);
    const taken = new Array(group.length).fill(false);
    for (let i = 0; i < group.length; i++) {
      if (taken[i]) continue;
      kept.push(group[i]);
      for (let j = i + 1; j < group.length; j++) {
        if (!taken[j] && iou(group[i].xyxy, group[j].xyxy) > IOU_THRESH) taken[j] = true;
      }
    }
  }
  return kept.sort((a, b) => b.score - a.score);
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
