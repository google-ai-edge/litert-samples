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
 * verification/e2e harnesses. No chrome.*, no model I/O, no imports: the
 * recognition loop (recognizeLines) gets the recognizer and its canvases
 * from the caller. Also bundled into the web demos' ppocr page
 * (samples/web_demos/src/ppocr/): after a change, rebuild samples/web_demos/
 * dist/ and run `npm run check -- ppocr` there.
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
 * A gap narrower than wordGap is the space between two letters, not two
 * words (on the fixtures: letters 1–8 px apart, words 9–19 px, at 48 px ≈
 * 1 em). Rather than split a word, a window reaches up to 10% past maxW
 * for a word gap; when there is none, the window after the cut gets
 * midWord, and windowSeparator joins it to the one before without a space.
 * Returns [{from, to, midWord}] in strip px; (to − from) may exceed maxW.
 */
export function splitByInk(profile, lw, {
  maxW = REC_W, squashLimit = REC_W * 2.2, gapFrac = 0.06, wordGap = REC_H * 0.15,
} = {}) {
  if (lw <= maxW) return [{ from: 0, to: lw, midWord: false }];
  let peak = 0;
  for (let x = 0; x < lw; x++) if (profile[x] > peak) peak = profile[x];
  const low = Math.max(12, peak * gapFrac);
  // Whole width of the background run around column x, also the part
  // outside the search window; 0 inside ink.
  const runWidth = (x) => {
    let a = Math.round(x);
    if (!(profile[a] <= low)) return 0;
    let b = a;
    while (a > 0 && profile[a - 1] <= low) a--;
    while (b < lw - 1 && profile[b + 1] <= low) b++;
    return b - a + 1;
  };
  // Center of the widest background run (≥ 2 px) inside [a, b], or null.
  const widestRun = (a, b) => {
    let bestRunStart = -1;
    let bestRunLen = 0;
    let runStart = -1;
    for (let x = a; x <= b + 1; x++) {
      const isLow = x <= b && x < lw && profile[x] <= low;
      if (isLow && runStart < 0) runStart = x;
      if (!isLow && runStart >= 0) {
        if (x - runStart > bestRunLen) { bestRunLen = x - runStart; bestRunStart = runStart; }
        runStart = -1;
      }
    }
    return bestRunLen >= 2 ? bestRunStart + (bestRunLen >> 1) : null;
  };

  const pieces = [];
  let start = 0;
  let midWord = false;
  for (;;) {
    const remaining = lw - start;
    if (remaining <= squashLimit) {
      // One (possibly squashed) window beats cutting: every boundary is a
      // chance for a duplicated or phantom edge character.
      pieces.push({ from: start, to: lw, midWord });
      break;
    }
    let cut = widestRun(start + Math.round(maxW * 0.55), start + maxW);
    if (cut == null || runWidth(cut) < wordGap) {
      // Only letter gaps here: a word gap a little past maxW (mildly
      // squashed at rec time) beats cutting the word.
      const past = widestRun(start + maxW + 1, Math.min(lw - 1, start + Math.round(maxW * 1.1)));
      if (past != null && runWidth(past) >= wordGap) cut = past;
    }
    if (cut != null) {
      pieces.push({ from: start, to: cut, midWord });
      midWord = runWidth(cut) < wordGap;
      start = cut;
    } else if (remaining <= squashLimit) {
      pieces.push({ from: start, to: lw, midWord });
      break;
    } else {
      // No gap in the window — take a squashed oversized window and search
      // for the next gap beyond it.
      // Integer column: a fractional end would index the profile at x.8
      // and leave every later window without ink bounds.
      let end = Math.min(lw, start + Math.floor(squashLimit));
      for (let x = end; x >= start + maxW; x--) {
        if (profile[x] <= low) { end = x; break; }
      }
      pieces.push({ from: start, to: end, midWord });
      // no gap at all → end cuts through a glyph (runWidth 0)
      midWord = runWidth(end) < wordGap;
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

/** Text between two adjacent rec windows of one line (line objects): none
 * where the cut went through a word (after.midWord, see splitByInk) or at a
 * CJK edge, else one space. Shared by groupLines and the overlay's
 * selectable text, so a copied selection reads the same as Copy all. */
export function windowSeparator(before, after) {
  if (after.midWord) return '';
  return CJK_EDGE.test(before.text.slice(-1)) || CJK_EDGE.test(after.text[0]) ? '' : ' ';
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
      g.text += windowSeparator(g.pieces[g.pieces.length - 1], line) + line.text;
      g.pieces.push(line);
    } else {
      out.push({ text: line.text, pieces: [line] });
    }
    prev = line.group ?? null;
  }
  return out;
}

// --- recognition loop ----------------------------------------------------------

// Background frame around every rec window, px at rec scale.
export const EDGE_PAD = 8;
// Mean CTC top1−top2 margin below which a "line" is treated as a
// hallucination on a decorative blob (avatar, UI bar) and dropped. Real
// text on the mock posts scores ≳0.5; the fake-header bars scored ≈0.05.
const SCORE_MIN = 0.2;
const CJK_RE = /[぀-ヿ㐀-䶿一-鿿]/;

/**
 * Recognize one window of a line strip, robustly.
 *
 * The recognizer has deterministic failure pockets: a crop that renders
 * clean text can decode to confident-looking garbage ("Every day" →
 * "YveerydaYyw"), and a tiny geometry change (slight rescale/shift) moves
 * it back out of the pocket. Failed pockets score low (≤0.62 mean CTC
 * margin) while healthy decodes score ≥0.85, so: decode geometry variants
 * until one is healthy and keep the highest-scoring one. Content is always
 * framed with real background — flush edges hallucinate phantom edge
 * characters.
 */
async function recognizeWindow(rec, strip, from, pw, bg, maxVariants = 5) {
  const C = rec.chars.length;
  const variants = [
    { pad: EDGE_PAD, scale: 1, grow: 0 },
    // Short crops left at native scale sit in a pocket: a clean "Notes for"
    // (203 px of a 304 px window) decoded as "Yotesow" at margin 0.15, and
    // stretching the same pixels to fill the window read it correctly at
    // 0.95. Capped at 2.5× so a one-word crop is not smeared.
    { pad: EDGE_PAD, scale: 1, grow: 0, fill: true },
    { pad: EDGE_PAD + 10, scale: 0.92, grow: 0 },
    { pad: EDGE_PAD, scale: 0.96, grow: 10 }, // widened bounds: new context
    { pad: EDGE_PAD + 4, scale: 0.85, grow: 0 },
    { pad: EDGE_PAD, scale: 1, grow: 22 },
  ].slice(0, maxVariants);
  let best = null;
  let ms = 0;
  for (const v of variants) {
    const f = Math.max(0, from - v.grow);
    const w = Math.min(strip.width - f, pw + v.grow + (from - f));
    const availW = REC_W - 2 * v.pad;
    const natW = Math.round(w * v.scale);
    const drawnW = v.fill
      ? Math.min(availW, Math.round(natW * 2.5))
      : Math.min(availW, natW);
    const drawnH = Math.round(REC_H * v.scale);
    const contentW = Math.min(REC_W, drawnW + 2 * v.pad);
    const win = rec.makeCanvas(contentW, REC_H);
    const ctx = win.getContext('2d', { willReadFrequently: true });
    ctx.fillStyle = `rgb(${bg[0]},${bg[1]},${bg[2]})`;
    ctx.fillRect(0, 0, contentW, REC_H);
    ctx.drawImage(strip, f, 0, w, REC_H,
      v.pad, Math.floor((REC_H - drawnH) / 2), drawnW, drawnH);
    const rgba = ctx.getImageData(0, 0, contentW, REC_H).data;
    const out = await rec.recognize(recPreprocess(rgba, contentW));
    ms += out.ms;
    const d = ctcDecode(out.data, out.data.length / C, C, rec.chars);
    if (!best || d.score > best.score) best = d;
    // Healthy decode — no need to pay for more variants.
    if (best.score >= 0.85) break;
  }
  return { text: best.text, score: best.score, ms };
}

/** Recognize [from, to) of a strip; when the decode scores poorly, re-split
 * at the widest interior background gap and keep the halves if they read
 * better. Squashed Latin windows lose thin glyphs — two unsquashed halves
 * usually recover them. */
async function recognizePiece(rec, strip, profile, from, to, bg, depth = 0) {
  const pw = to - from;
  // Depth-0 windows get the full variant sweep; re-split halves get a
  // cheaper one so a stubborn window can't multiply into dozens of runs.
  const r = await recognizeWindow(rec, strip, from, pw, bg, depth === 0 ? 6 : 2);
  if (r.score >= 0.75 || depth >= 2 || pw < 60) return r;
  const cut = widestInteriorGap(profile, from, to);
  if (cut == null) return r;
  const left = await recognizePiece(rec, strip, profile, from, cut, bg, depth + 1);
  const right = await recognizePiece(rec, strip, profile, cut, to, bg, depth + 1);
  const combinedScore = Math.min(left.score, right.score);
  if (combinedScore <= r.score) return { ...r, ms: r.ms + left.ms + right.ms };
  const sep = !left.text || !right.text
    || (CJK_RE.test(left.text.slice(-1)) && CJK_RE.test(right.text[0])) ? '' : ' ';
  return {
    text: left.text + sep + right.text,
    score: combinedScore,
    ms: r.ms + left.ms + right.ms,
  };
}

/**
 * Detected line boxes → recognized text: per box, the whole line is drawn
 * once at rec height from the FULL-RES source and split on the strip's own
 * ink profile, and every window is read by recognizePiece. The caller
 * brings the model and the canvases:
 *   recognize(nchw) → Promise<{data, ms}>: the recognizer's CTC logits
 *     [T, C] for one [1, 3, REC_H, REC_W] input, and the time it took
 *   makeCanvas(w, h) → a canvas with a 2D context (an OffscreenCanvas)
 *   onLine(i, n), optional: called before box i of n is read
 * source = the image the boxes were found on (an ImageBitmap); boxes =
 * probToBoxes output; chars = buildCharTable output.
 * Returns {lines, windows, recMs}: lines = [{x, y, w, h, text, score, group,
 * midWord}] with rects as fractions of the source size; windows = windows
 * read (retries not counted); recMs = the time spent in recognize().
 */
export async function recognizeLines(source, boxes, { chars, recognize, makeCanvas, onLine = null }) {
  const rec = { chars, recognize, makeCanvas };
  const nw = source.width;
  const nh = source.height;
  const scaleX = nw / DET_SIZE;
  const scaleY = nh / DET_SIZE;
  const lines = [];
  let recMs = 0;
  let windows = 0;
  for (const [group, box] of boxes.entries()) {
    onLine?.(group, boxes.length);
    // Render the whole detected line once at rec height; split on the strip's
    // own ink profile (source pixels — the det map is too blurry to find true
    // gaps and its minima fall inside glyphs).
    const sx = box.x0 * scaleX;
    const sy = box.y0 * scaleY;
    const sw = (box.x1 - box.x0 + 1) * scaleX;
    const sh = (box.y1 - box.y0 + 1) * scaleY;
    const lw = Math.min(4096, Math.max(1, Math.round(REC_H * (sw / sh))));
    const strip = makeCanvas(lw, REC_H);
    const stripCtx = strip.getContext('2d', { willReadFrequently: true });
    stripCtx.drawImage(source, sx, sy, sw, sh, 0, 0, lw, REC_H);
    const stripRgba = stripCtx.getImageData(0, 0, lw, REC_H).data;
    const { profile, bg } = columnInkProfile(stripRgba, lw, REC_H);
    // Prefer one mildly squashed window over many cuts: every extra window
    // boundary is a chance for a duplicated/phantom edge character.
    const pieces = splitByInk(profile, lw, {
      maxW: REC_W - 2 * EDGE_PAD,
      squashLimit: (REC_W - 2 * EDGE_PAD) * 2.2,
    });

    let lastKept = -1; // index of the last piece that became a line
    for (const [i, piece] of pieces.entries()) {
      // Tighten to ink: unclip margins leave large variable bg runs at the
      // window edges, and rec quality is sensitive to them.
      const tight = inkBounds(profile, piece.from, piece.to);
      if (!tight) continue;
      const { from, to } = tight;
      const pw = to - from;
      if (pw < 3) continue;
      const best = await recognizePiece(rec, strip, profile, from, to, bg);
      recMs += best.ms;
      windows++;
      const { text, score } = best;
      if (!text.trim()) continue;
      if (score < SCORE_MIN) continue;
      const toSrc = sh / REC_H; // strip px → source px
      lines.push({
        x: (sx + from * toSrc) / nw,
        y: sy / nh,
        w: (pw * toSrc) / nw,
        h: sh / nh,
        text,
        score: +score.toFixed(3),
        group, // pieces of one detected line share a group → joined on copy
        // The cut before this window went through a word: joined without a
        // space, unless the window before it was dropped.
        midWord: piece.midWord && lastKept === i - 1,
      });
      lastKept = i;
    }
  }
  return { lines, windows, recMs };
}
