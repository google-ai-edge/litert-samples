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
 * PP-OCRv5 pipeline pieces shared by the offscreen engine and the
 * verification/e2e harnesses. Pure functions — no chrome.*, no model I/O.
 *
 * Spec source: litert-community/PP-OCRv5-LiteRT model card.
 *   det: image [1,3,640,640] NCHW, /255 then ImageNet mean/std
 *        → prob map [1,1,640,640]
 *   rec: line [1,3,48,320] NCHW, (x/255 − 0.5)/0.5, keep-aspect h=48,
 *        zero-pad to width 320 → CTC logits [1,T,18385]
 *   dict: 18383 lines; CTC layout = blank(0) + dict + space(18384)
 */

export const DET_SIZE = 640;
export const REC_H = 48;
export const REC_W = 320;
// A rec window reads at most REC_W/REC_H ≈ 6.7 of width:height. Lines longer
// than that are split at prob-map valleys (see splitLongBox).
export const REC_MAX_RATIO = REC_W / REC_H;

const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

/** RGBA ImageData (any size) → det input: stretch to 640×640, ImageNet
 * mean/std, NCHW. Returns {nchw, scaleX, scaleY} — box coords map back to
 * source pixels via the two scales. */
export function detPreprocess(rgba, srcW, srcH) {
  // Caller draws the source onto a 640×640 canvas; this just normalizes.
  const plane = DET_SIZE * DET_SIZE;
  const nchw = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) {
      nchw[c * plane + i] = (rgba[i * 4 + c] / 255 - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
    }
  }
  return { nchw, scaleX: srcW / DET_SIZE, scaleY: srcH / DET_SIZE };
}

/**
 * DB prob map → text-line boxes in det (640²) space.
 * Approximation of DB postprocess: binarize at `thresh`, 4-connected
 * components (BFS), drop tiny/low-confidence blobs, pad each box
 * (~unclip), then merge horizontally-adjacent boxes into lines — DB's
 * shrunk map often splits one visual line at wide word gaps.
 * Returns [{x0, y0, x1, y1, score}] sorted top-to-bottom, left-to-right.
 */
export function probToBoxes(prob, { thresh = 0.3, minScore = 0.5, minSize = 3 } = {}) {
  const W = DET_SIZE;
  const labels = new Int32Array(W * W); // 0 = unvisited/below-threshold
  const comps = [];
  const stack = new Int32Array(W * W);
  for (let i = 0; i < W * W; i++) {
    if (labels[i] !== 0 || prob[i] <= thresh) continue;
    const label = comps.length + 1;
    let top = 0;
    stack[top++] = i;
    labels[i] = label;
    let minX = W, minY = W, maxX = 0, maxY = 0, sum = 0, count = 0;
    while (top > 0) {
      const p = stack[--top];
      const x = p % W;
      const y = (p / W) | 0;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      sum += prob[p];
      count++;
      if (x > 0 && labels[p - 1] === 0 && prob[p - 1] > thresh) { labels[p - 1] = label; stack[top++] = p - 1; }
      if (x < W - 1 && labels[p + 1] === 0 && prob[p + 1] > thresh) { labels[p + 1] = label; stack[top++] = p + 1; }
      if (y > 0 && labels[p - W] === 0 && prob[p - W] > thresh) { labels[p - W] = label; stack[top++] = p - W; }
      if (y < W - 1 && labels[p + W] === 0 && prob[p + W] > thresh) { labels[p + W] = label; stack[top++] = p + W; }
    }
    comps.push({ minX, minY, maxX, maxY, score: sum / count });
  }

  let boxes = [];
  for (const c of comps) {
    const w = c.maxX - c.minX + 1;
    const h = c.maxY - c.minY + 1;
    if (w < minSize || h < minSize || c.score < minScore) continue;
    // DB's training shrinks text masks — the raw component is SMALLER than
    // the glyphs (measured: 26px text → 16px box). Grow it back with the
    // real DB unclip formula, d = ratio·area/(2·perimeter): ≈0.75·h for
    // long thin lines. Too little pad crops ascenders/descenders and
    // over-stretches the rec strip.
    const pad = Math.min(40, Math.max(2, Math.round((1.5 * w * h) / (2 * (w + h)))));
    boxes.push({
      x0: Math.max(0, c.minX - pad),
      y0: Math.max(0, c.minY - pad),
      x1: Math.min(W - 1, c.maxX + pad),
      y1: Math.min(W - 1, c.maxY + pad),
      score: c.score,
    });
  }

  // Merge boxes on the same text line: strong vertical overlap and a
  // horizontal gap smaller than the line height.
  boxes.sort((a, b) => a.x0 - b.x0);
  let merged = true;
  while (merged) {
    merged = false;
    for (let i = 0; i < boxes.length && !merged; i++) {
      for (let j = i + 1; j < boxes.length; j++) {
        const a = boxes[i];
        const b = boxes[j];
        const ha = a.y1 - a.y0;
        const hb = b.y1 - b.y0;
        const overlap = Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0);
        if (overlap < 0.5 * Math.min(ha, hb)) continue;
        const gap = Math.max(a.x0, b.x0) - Math.min(a.x1, b.x1);
        if (gap > 1.2 * Math.min(ha, hb)) continue;
        boxes[i] = {
          x0: Math.min(a.x0, b.x0),
          y0: Math.min(a.y0, b.y0),
          x1: Math.max(a.x1, b.x1),
          y1: Math.max(a.y1, b.y1),
          score: (a.score + b.score) / 2,
        };
        boxes.splice(j, 1);
        merged = true;
        break;
      }
    }
  }

  boxes.sort((a, b) => (a.y0 - b.y0) || (a.x0 - b.x0));
  return boxes;
}

/**
 * A det box wider than REC_MAX_RATIO × height must be read in several rec
 * windows. Cut at gaps in the prob map's column profile — never through a
 * glyph. The raw profile dips between CHARACTERS too (and between a kana
 * base and its dakuten), so smooth it over ~h/5 first, then prefer the
 * WIDEST low run (a real word/phrase gap) inside the search window over the
 * single lowest column. Returns x-ranges in det space.
 */
export function splitLongBox(prob, box, maxRatio = REC_MAX_RATIO * 0.92) {
  const W = DET_SIZE;
  const h = box.y1 - box.y0 + 1;
  const w = box.x1 - box.x0 + 1;
  const maxW = Math.max(8, Math.round(h * maxRatio));
  if (w <= maxW) return [[box.x0, box.x1]];

  // Column ink profile from the prob map, box-blurred over radius ~h/5.
  const raw = new Float32Array(w);
  for (let x = 0; x < w; x++) {
    let s = 0;
    for (let y = box.y0; y <= box.y1; y++) s += prob[y * W + box.x0 + x];
    raw[x] = s / h;
  }
  const r = Math.max(1, Math.round(h / 5));
  const profile = new Float32Array(w);
  for (let x = 0; x < w; x++) {
    let s = 0;
    let n = 0;
    for (let k = Math.max(0, x - r); k <= Math.min(w - 1, x + r); k++) { s += raw[k]; n++; }
    profile[x] = s / n;
  }
  let peak = 0;
  for (let x = 0; x < w; x++) if (profile[x] > peak) peak = profile[x];
  const low = peak * 0.12;

  const pieces = [];
  let start = 0;
  while (w - start > maxW) {
    const idealEnd = start + maxW;
    const from = Math.max(start + Math.round(maxW * 0.5), start + 4);
    // Widest low run inside [from, idealEnd] → cut at its center.
    let bestRunStart = -1;
    let bestRunLen = 0;
    let runStart = -1;
    for (let x = from; x <= idealEnd + 1; x++) {
      const isLow = x <= idealEnd && profile[x] <= low;
      if (isLow && runStart < 0) runStart = x;
      if (!isLow && runStart >= 0) {
        if (x - runStart > bestRunLen) { bestRunLen = x - runStart; bestRunStart = runStart; }
        runStart = -1;
      }
    }
    let cut;
    if (bestRunLen > 0) {
      cut = bestRunStart + (bestRunLen >> 1);
    } else {
      // No real gap — fall back to the lowest smoothed column.
      cut = idealEnd;
      let bestV = Infinity;
      for (let x = idealEnd; x >= from; x--) {
        if (profile[x] < bestV) { bestV = profile[x]; cut = x; }
      }
    }
    pieces.push([box.x0 + start, box.x0 + cut]);
    start = cut + 1;
  }
  pieces.push([box.x0 + start, box.x1]);
  return pieces;
}

/**
 * Column ink profile of a rendered line strip (RGBA, w×h): per column, the
 * max absolute deviation from the background color (estimated from the
 * strip's corners), 0..765. Sharp enough to see true inter-word gaps —
 * unlike the det prob map, whose minima fall inside glyphs.
 * Returns {profile, bg} — bg so callers can pad rec windows with real
 * background (content flush against a window edge makes the recognizer
 * hallucinate phantom edge characters, on every backend).
 */
export function columnInkProfile(rgba, w, h) {
  let br = 0, bg_ = 0, bb = 0, n = 0;
  for (const [cx, cy] of [[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]]) {
    for (let d = 0; d < 3; d++) {
      const x = Math.min(w - 1, Math.max(0, cx + (cx === 0 ? d : -d)));
      const i = (cy * w + x) * 4;
      br += rgba[i]; bg_ += rgba[i + 1]; bb += rgba[i + 2]; n++;
    }
  }
  br /= n; bg_ /= n; bb /= n;
  const profile = new Float32Array(w);
  for (let x = 0; x < w; x++) {
    let m = 0;
    for (let y = 0; y < h; y++) {
      const i = (y * w + x) * 4;
      const d = Math.abs(rgba[i] - br) + Math.abs(rgba[i + 1] - bg_) + Math.abs(rgba[i + 2] - bb);
      if (d > m) m = d;
    }
    profile[x] = m;
  }
  return { profile, bg: [Math.round(br), Math.round(bg_), Math.round(bb)] };
}

/**
 * Split a 48-high line strip of width lw into rec windows using the ink
 * profile. Cuts only at true background gaps (runs of near-background
 * columns); when a stretch has no gap, the window is allowed to grow to
 * squashLimit and squashed into REC_W at rec time (mild squash is in the
 * model's training distribution; cutting through a glyph never is).
 * Returns [{from, to}] in strip px; (to − from) may exceed maxW.
 */
export function splitByInk(profile, lw, { maxW = REC_W, squashLimit = REC_W * 2.2, gapFrac = 0.06 } = {}) {
  if (lw <= maxW) return [{ from: 0, to: lw }];
  let peak = 0;
  for (let x = 0; x < lw; x++) if (profile[x] > peak) peak = profile[x];
  const low = Math.max(12, peak * gapFrac);

  const pieces = [];
  let start = 0;
  for (;;) {
    const remaining = lw - start;
    if (remaining <= squashLimit) {
      // One (possibly squashed) window beats cutting: every boundary is a
      // chance for a duplicated or phantom edge character.
      pieces.push({ from: start, to: lw });
      break;
    }
    const from = start + Math.round(maxW * 0.55);
    const until = start + maxW;
    let bestRunStart = -1;
    let bestRunLen = 0;
    let runStart = -1;
    for (let x = from; x <= until + 1; x++) {
      const isLow = x <= until && x < lw && profile[x] <= low;
      if (isLow && runStart < 0) runStart = x;
      if (!isLow && runStart >= 0) {
        if (x - runStart > bestRunLen) { bestRunLen = x - runStart; bestRunStart = runStart; }
        runStart = -1;
      }
    }
    if (bestRunLen >= 2) {
      const cut = bestRunStart + (bestRunLen >> 1);
      pieces.push({ from: start, to: cut });
      start = cut;
    } else if (remaining <= squashLimit) {
      pieces.push({ from: start, to: lw });
      break;
    } else {
      // No gap in the window — take a squashed oversized window and search
      // for the next gap beyond it.
      let end = Math.min(lw, start + squashLimit);
      for (let x = end; x >= start + maxW; x--) {
        if (profile[x] <= low) { end = x; break; }
      }
      pieces.push({ from: start, to: end });
      start = end;
    }
    if (start >= lw - 2) break;
  }
  return pieces;
}

/** Widest background run strictly inside [from, to) (15% edge exclusion),
 * or null. Used to re-split a window whose decode scored poorly. */
export function widestInteriorGap(profile, from, to) {
  const w = to - from;
  const a = from + Math.round(w * 0.15);
  const b = to - Math.round(w * 0.15);
  let peak = 0;
  for (let x = from; x < to; x++) if (profile[x] > peak) peak = profile[x];
  const low = Math.max(12, peak * 0.06);
  let bestStart = -1;
  let bestLen = 0;
  let runStart = -1;
  for (let x = a; x <= b; x++) {
    const isLow = x < b && profile[x] <= low;
    if (isLow && runStart < 0) runStart = x;
    if (!isLow && runStart >= 0) {
      if (x - runStart > bestLen) { bestLen = x - runStart; bestStart = runStart; }
      runStart = -1;
    }
  }
  if (bestLen < 2) return null;
  return bestStart + (bestLen >> 1);
}

/** Tighten [from, to) to the actual ink columns (profile above the same
 * relative threshold splitByInk uses), with a small margin. Returns null
 * when the range holds no ink at all. */
export function inkBounds(profile, from, to, margin = 4) {
  let peak = 0;
  for (let x = from; x < to; x++) if (profile[x] > peak) peak = profile[x];
  const low = Math.max(12, peak * 0.06);
  let a = -1;
  let b = -1;
  for (let x = from; x < to; x++) {
    if (profile[x] > low) { if (a < 0) a = x; b = x; }
  }
  if (a < 0) return null;
  return { from: Math.max(from, a - margin), to: Math.min(to, b + 1 + margin) };
}

/** RGBA ImageData of a line crop already resized to 48×contentW (≤320) →
 * rec input tensor data, zero-padded ((0/255−0.5)/0.5 = −1) to 48×320. */
export function recPreprocess(rgba, contentW) {
  const plane = REC_H * REC_W;
  const nchw = new Float32Array(3 * plane).fill(-1);
  for (let y = 0; y < REC_H; y++) {
    for (let x = 0; x < contentW; x++) {
      const src = (y * contentW + x) * 4;
      const dst = y * REC_W + x;
      nchw[dst] = rgba[src] / 255 / 0.5 - 1;
      nchw[plane + dst] = rgba[src + 1] / 255 / 0.5 - 1;
      nchw[2 * plane + dst] = rgba[src + 2] / 255 / 0.5 - 1;
    }
  }
  return nchw;
}

/** CTC greedy decode. logits [T, C] flat → {text, ids, score} where ids are
 * the per-timestep argmaxes and score is the mean top1−top2 margin over
 * non-blank timesteps. Real text scores high (≳0.5); hallucinations on
 * decorative blobs (avatars, UI bars) hover near zero — filter on it. */
export function ctcDecode(logits, T, C, chars) {
  const ids = new Int32Array(T);
  let marginSum = 0;
  let marginN = 0;
  for (let t = 0; t < T; t++) {
    const off = t * C;
    let best = 0;
    let bestV = logits[off];
    let second = -Infinity;
    for (let c = 1; c < C; c++) {
      const v = logits[off + c];
      if (v > bestV) { second = bestV; bestV = v; best = c; }
      else if (v > second) second = v;
    }
    ids[t] = best;
    if (best !== 0) { marginSum += bestV - second; marginN++; }
  }
  let text = '';
  for (let t = 0; t < T; t++) {
    const c = ids[t];
    if (c !== 0 && (t === 0 || c !== ids[t - 1])) text += chars[c] ?? '';
  }
  return { text, ids: Array.from(ids), score: marginN ? marginSum / marginN : 0 };
}

/** dict file text → CTC char table (blank + 18383 chars + space). */
export function buildCharTable(dictText) {
  const lines = dictText.split('\n');
  if (lines[lines.length - 1] === '') lines.pop();
  return ['', ...lines, ' '];
}

const CJK_EDGE = /[぀-ヿ㐀-䶿一-鿿。、!?」』)]$|^[぀-ヿ㐀-䶿一-鿿「『(]/;

/** Text between two rec windows of one line: none at a CJK edge, else one
 * space. Shared by groupLines and the overlay's selectable text, so a copied
 * selection reads the same as Copy all. */
export function windowSeparator(before, after) {
  return CJK_EDGE.test(before.slice(-1)) || CJK_EDGE.test(after[0]) ? '' : ' ';
}

/**
 * Rec windows that split one detected line share a group id. Merge them back
 * into logical lines: search and copy both need the whole line, because a
 * phrase can straddle a window boundary ("power outlets" split as
 * "has 8 power" + "outlets for 20 people"). Returns [{text, pieces}] where
 * pieces keep their own rects for highlighting.
 */
export function groupLines(lines) {
  const out = [];
  let prev = null;
  for (const line of lines) {
    const same = line.group != null && line.group === prev && out.length;
    if (same) {
      const g = out[out.length - 1];
      g.text += windowSeparator(g.text, line.text) + line.text;
      g.pieces.push(line);
    } else {
      out.push({ text: line.text, pieces: [line] });
    }
    prev = line.group ?? null;
  }
  return out;
}
