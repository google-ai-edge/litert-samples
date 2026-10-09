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
 * Browser side of the recognizer backend comparison (tools-ocr/verify.mjs).
 * OCR outputs CTC-decoded text, so the comparison is on decoded strings:
 * do WASM and WebGPU decode the same crop to the same text, with the fp16
 * and with the fp32 recognizer?
 *
 * Stage A: render single text lines (EN/JA/mixed, several font sizes)
 * straight into rec crops — no det involved — run both backends on
 * identical tensors, compare decoded text + per-timestep argmax ids.
 * Stage B: the two half-size posts from make-fixtures.mjs through
 * det(webgpu) → boxes → split → rec on both backends.
 *
 * Everything is exposed on globalThis.__ocrv for the CDP runner.
 */
import { Tensor, isWebGPUSupported, loadAndCompile, loadLiteRt } from '@litertjs/core';
import {
  DET_SIZE, REC_H, REC_W, buildCharTable, ctcDecode, detPreprocess,
  probToBoxes, recPreprocess, splitLongBox,
} from '../src-ocr/ocr-pipeline.js';

const status = { state: 'boot', error: null, progress: '' };
let result = null;
globalThis.__ocrv = {
  get status() { return { ...status }; },
  get result() { return result; },
};

// ---- test content (all self-authored → license-clean) -----------------------

const LINES = [
  ['en', 'Hello OCR 2026'],
  ['en', 'LiteRT.js on WebGPU'],
  ['en', 'runs in your browser'],
  ['en', 'no server, no upload'],
  ['mix', 'det 12ms / rec 15ms'],
  ['mix', 'GPU: 66-72ms (M4)'],
  ['mix', '#OnDeviceAI 100%'],
  ['ja', '画像の文字がコピーできる'],
  ['ja', '日本語対応です'],
  ['ja', '深度推定はブラウザ内'],
  ['ja', 'スクショの長文もOK'],
];
const SIZES = [18, 28, 44]; // native render px → all upscaled/downscaled to h=48
const POSTS = ['fixture-half-en-light', 'fixture-half-ja-dark']; // make-fixtures.mjs

// ---- rendering helpers -------------------------------------------------------

const FONT = {
  en: '-apple-system, "Helvetica Neue", Arial, sans-serif',
  mix: '-apple-system, "Helvetica Neue", Arial, sans-serif',
  ja: '"Hiragino Sans", "Hiragino Kaku Gothic ProN", sans-serif',
};

/** Render one line at `px` height, then rescale to a 48-high rec crop.
 * Returns {nchw, contentW, dataUrl}. */
function renderLineCrop(text, kind, px, { fg = '#111', bg = '#fff' } = {}, leftPad = 0) {
  const c = new OffscreenCanvas(10, 10);
  const ctx = c.getContext('2d');
  ctx.font = `${px}px ${FONT[kind]}`;
  const m = ctx.measureText(text);
  const w = Math.ceil(m.width) + Math.round(px * 0.6);
  const h = Math.round(px * 1.5);
  c.width = w;
  c.height = h;
  const cx = c.getContext('2d');
  cx.fillStyle = bg;
  cx.fillRect(0, 0, w, h);
  cx.fillStyle = fg;
  cx.font = `${px}px ${FONT[kind]}`;
  cx.textBaseline = 'middle';
  cx.fillText(text, Math.round(px * 0.3), Math.round(h / 2));
  return canvasToRecCrop(c, 0, 0, w, h, { leftPad, bg });
}

/** Crop a region of any canvas → rec tensor (h=48, keep aspect, cap 320).
 * leftPad inserts that many bg-colored columns before the content. */
function canvasToRecCrop(srcCanvas, sx, sy, sw, sh, { leftPad = 0, bg = null } = {}) {
  const contentW = Math.min(REC_W - leftPad, Math.max(1, Math.round(REC_H * (sw / sh))));
  const c = new OffscreenCanvas(leftPad + contentW, REC_H);
  const ctx = c.getContext('2d', { willReadFrequently: true });
  if (leftPad > 0) {
    ctx.fillStyle = bg ?? '#fff';
    ctx.fillRect(0, 0, leftPad, REC_H);
  }
  ctx.drawImage(srcCanvas, sx, sy, sw, sh, leftPad, 0, contentW, REC_H);
  const rgba = ctx.getImageData(0, 0, leftPad + contentW, REC_H).data;
  return { nchw: recPreprocess(rgba, leftPad + contentW), contentW: leftPad + contentW, canvas: c };
}

/** A fixture image (served next to the page) drawn onto a canvas. */
async function loadCanvas(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  const bitmap = await createImageBitmap(await res.blob());
  const c = new OffscreenCanvas(bitmap.width, bitmap.height);
  c.getContext('2d').drawImage(bitmap, 0, 0);
  bitmap.close();
  return c;
}

async function toDataUrl(canvas) {
  const blob = await canvas.convertToBlob({ type: 'image/png' });
  return await new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(r.result);
    r.onerror = rej;
    r.readAsDataURL(blob);
  });
}

// ---- model plumbing ----------------------------------------------------------

async function fetchBytes(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

async function runModel(model, nchw, shape) {
  const input = Tensor.fromTypedArray(nchw, shape);
  const t0 = performance.now();
  const outputs = await model.run([input]);
  const data = await outputs[0].data();
  const ms = performance.now() - t0;
  const dims = outputs[0].type?.layout?.dimensions ?? null;
  for (const o of outputs) o.delete();
  input.delete();
  return { data, ms, dims };
}

function diffIds(a, b) {
  let flips = 0;
  for (let t = 0; t < a.length; t++) if (a[t] !== b[t]) flips++;
  return flips;
}

/** For each timestep where the two backends' argmax differ, report the
 * top-2 margin on each side — how near the tie was. Also the global min
 * margin (any timestep), which a confidence gate would key on. */
function flipDetails(gpuOut, cpuOut, gpuIds, cpuIds, C, chars) {
  const T = gpuIds.length;
  const margin = (data, t) => {
    const off = t * C;
    let b1 = -Infinity;
    let b2 = -Infinity;
    for (let c = 0; c < C; c++) {
      const v = data[off + c];
      if (v > b1) { b2 = b1; b1 = v; } else if (v > b2) b2 = v;
    }
    return b1 - b2;
  };
  const flips = [];
  let minMarginGpu = Infinity;
  for (let t = 0; t < T; t++) {
    const m = margin(gpuOut, t);
    if (m < minMarginGpu) minMarginGpu = m;
    if (gpuIds[t] !== cpuIds[t]) {
      const show = (id) => (id === 0 ? '∅' : id === C - 1 ? '␣' : chars[id] ?? '?');
      flips.push({
        t,
        gpu: show(gpuIds[t]),
        cpu: show(cpuIds[t]),
        gpuMargin: +m.toFixed(4),
        cpuMargin: +margin(cpuOut, t).toFixed(4),
      });
    }
  }
  return { flips, minMarginGpu: +minMarginGpu.toFixed(4) };
}

// ---- main --------------------------------------------------------------------

async function main() {
  status.state = 'loading-runtime';
  const coi = globalThis.crossOriginIsolated;
  await loadLiteRt('litert-wasm/', { threads: coi, jspi: false });
  if (!isWebGPUSupported()) throw new Error('WebGPU unavailable — this check needs both backends');

  status.state = 'loading-models';
  const [detBytes, recBytes, rec32Bytes, dictText] = await Promise.all([
    fetchBytes('models/ppocr_det_fp16.tflite'),
    fetchBytes('models/ppocr_rec_fp16.tflite'),
    fetchBytes('models/ppocr_rec_fp32.tflite'),
    fetch('models/ppocrv5_dict.txt').then((r) => r.text()),
  ]);
  const chars = buildCharTable(dictText);

  status.state = 'compiling';
  const numThreads = Math.min(8, navigator.hardwareConcurrency || 4);
  // fp16 and fp32 recognizer weights, each on WebGPU and on WASM.
  const recGpu = await loadAndCompile(recBytes, { accelerator: 'webgpu' });
  const recGpu32 = await loadAndCompile(rec32Bytes, { accelerator: 'webgpu' });
  const recCpu32 = await loadAndCompile(rec32Bytes, { accelerator: 'wasm', cpuOptions: { numThreads } });
  const recCpu = await loadAndCompile(recBytes, { accelerator: 'wasm', cpuOptions: { numThreads } });
  const detGpu = await loadAndCompile(detBytes, { accelerator: 'webgpu' });

  const decode = (out) => {
    const C = chars.length;
    const T = out.data.length / C;
    if (!Number.isInteger(T)) throw new Error(`bad logits length ${out.data.length} for C=${C}`);
    return ctcDecode(out.data, T, C, chars);
  };

  // ---- stage A: rec on identical crops --------------------------------------
  // Each line × sizes (× left pads, if PADS lists more than 0).
  status.state = 'stage-a';
  const PADS = [0];
  const stageA = [];
  const crops = [];
  for (const [kind, text] of LINES) {
    for (const px of SIZES) {
      for (const pad of PADS) {
        crops.push({ label: `${kind}-${px}px-p${pad}`, truth: text, kind, px, pad,
          crop: renderLineCrop(text, kind, px, {}, pad) });
      }
    }
  }
  // plus dark-mode variants at one size
  for (const [kind, text] of [['en', 'no server, no upload'], ['ja', '日本語対応です']]) {
    for (const pad of PADS) {
      crops.push({ label: `${kind}-28px-dark-p${pad}`, truth: text, kind, px: 28, pad,
        crop: renderLineCrop(text, kind, 28, { fg: '#f7f9f9', bg: '#15202b' }, pad) });
    }
  }

  for (const [idx, item] of crops.entries()) {
    const gpu = await runModel(recGpu, item.crop.nchw, [1, 3, REC_H, REC_W]);
    const gpu2 = await runModel(recGpu, item.crop.nchw, [1, 3, REC_H, REC_W]);
    const gpu32 = await runModel(recGpu32, item.crop.nchw, [1, 3, REC_H, REC_W]);
    const cpu32 = await runModel(recCpu32, item.crop.nchw, [1, 3, REC_H, REC_W]);
    const cpu = await runModel(recCpu, item.crop.nchw, [1, 3, REC_H, REC_W]);
    const dGpu = decode(gpu);
    const dGpu2 = decode(gpu2);
    const dGpu32 = decode(gpu32);
    const dCpu32 = decode(cpu32);
    const dCpu = decode(cpu);
    const C = chars.length;
    const detail = flipDetails(gpu.data, cpu.data, dGpu.ids, dCpu.ids, C, chars);
    stageA.push({
      idx,
      label: item.label,
      truth: item.truth,
      gpuText: dGpu.text,
      gpu32Text: dGpu32.text,
      cpu32Text: dCpu32.text,
      cpuText: dCpu.text,
      equal: dGpu.text === dCpu.text,
      equal32: dGpu32.text === dCpu32.text,
      gpuDeterministic: dGpu.text === dGpu2.text && diffIds(dGpu.ids, dGpu2.ids) === 0,
      idFlips: diffIds(dGpu.ids, dCpu.ids),
      idFlips32: diffIds(dGpu32.ids, dCpu32.ids),
      T: dGpu.ids.length,
      minMarginGpu: detail.minMarginGpu,
      flipDetail: detail.flips,
      gpuMs: +gpu.ms.toFixed(1),
      gpu32Ms: +gpu32.ms.toFixed(1),
      cpu32Ms: +cpu32.ms.toFixed(1),
      cpuMs: +cpu.ms.toFixed(1),
      cropUrl: await toDataUrl(item.crop.canvas),
    });
    status.progress = `stageA ${stageA.length}/${crops.length}`;
  }

  // ---- stage B: det → split → rec, both backends -----------------------------
  status.state = 'stage-b';
  const stageB = [];
  for (const name of POSTS) {
    const canvas = await loadCanvas(`fixtures/${name}.png`);
    const det640 = new OffscreenCanvas(DET_SIZE, DET_SIZE);
    const dctx = det640.getContext('2d', { willReadFrequently: true });
    dctx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, DET_SIZE, DET_SIZE);
    const rgba = dctx.getImageData(0, 0, DET_SIZE, DET_SIZE).data;
    const { nchw, scaleX, scaleY } = detPreprocess(rgba, canvas.width, canvas.height);
    const det = await runModel(detGpu, nchw, [1, 3, DET_SIZE, DET_SIZE]);
    const boxes = probToBoxes(det.data);
    const lines = [];
    for (const box of boxes) {
      for (const [x0, x1] of splitLongBox(det.data, box)) {
        // det space → source space
        const sx = x0 * scaleX;
        const sw = (x1 - x0 + 1) * scaleX;
        const sy = box.y0 * scaleY;
        const sh = (box.y1 - box.y0 + 1) * scaleY;
        const crop = canvasToRecCrop(canvas, sx, sy, sw, sh);
        const gpu = await runModel(recGpu, crop.nchw, [1, 3, REC_H, REC_W]);
        const gpu32 = await runModel(recGpu32, crop.nchw, [1, 3, REC_H, REC_W]);
        const cpu32 = await runModel(recCpu32, crop.nchw, [1, 3, REC_H, REC_W]);
        const cpu = await runModel(recCpu, crop.nchw, [1, 3, REC_H, REC_W]);
        const dGpu = decode(gpu);
        const dGpu32 = decode(gpu32);
        const dCpu32 = decode(cpu32);
        const dCpu = decode(cpu);
        lines.push({
          box: [Math.round(sx), Math.round(sy), Math.round(sw), Math.round(sh)],
          gpuText: dGpu.text,
          gpu32Text: dGpu32.text,
          cpu32Text: dCpu32.text,
          cpuText: dCpu.text,
          equal: dGpu.text === dCpu.text,
          equal32: dGpu32.text === dCpu32.text,
          idFlips: diffIds(dGpu.ids, dCpu.ids),
          gpuMs: +gpu.ms.toFixed(1),
          gpu32Ms: +gpu32.ms.toFixed(1),
          cpu32Ms: +cpu32.ms.toFixed(1),
          cpuMs: +cpu.ms.toFixed(1),
          cropUrl: await toDataUrl(crop.canvas),
        });
        status.progress = `stageB ${name} ${lines.length} lines`;
      }
    }
    stageB.push({
      name,
      detMs: +det.ms.toFixed(1),
      boxCount: boxes.length,
      lines,
    });
  }

  const aEqual = stageA.filter((r) => r.equal).length;
  const byPad = {};
  for (const pad of PADS) {
    const rows = stageA.filter((r, i) => crops[i].pad === pad);
    byPad[`p${pad}`] = {
      total: rows.length,
      equal: rows.filter((r) => r.equal).length,
      gpuExactTruth: rows.filter((r) => r.gpuText === r.truth).length,
      cpuExactTruth: rows.filter((r) => r.cpuText === r.truth).length,
    };
  }
  const bLines = stageB.flatMap((p) => p.lines);
  result = {
    ok: true,
    env: { coi, numThreads },
    summary: {
      stageAByPad: byPad,
      stageA: { total: stageA.length, equal: aEqual,
        equal32: stageA.filter((r) => r.equal32).length,
        gpuExactTruth: stageA.filter((r) => r.gpuText === r.truth).length,
        gpu32ExactTruth: stageA.filter((r) => r.gpu32Text === r.truth).length,
        cpu32ExactTruth: stageA.filter((r) => r.cpu32Text === r.truth).length,
        cpuExactTruth: stageA.filter((r) => r.cpuText === r.truth).length,
        gpuDeterministic: stageA.filter((r) => r.gpuDeterministic).length,
        maxIdFlips: Math.max(...stageA.map((r) => r.idFlips)),
        maxIdFlips32: Math.max(...stageA.map((r) => r.idFlips32)),
        gpuMsP50: median(stageA.map((r) => r.gpuMs)),
        gpu32MsP50: median(stageA.map((r) => r.gpu32Ms)),
        cpu32MsP50: median(stageA.map((r) => r.cpu32Ms)),
        cpuMsP50: median(stageA.map((r) => r.cpuMs)) },
      stageB: { lines: bLines.length, equal: bLines.filter((l) => l.equal).length,
        equal32: bLines.filter((l) => l.equal32).length,
        maxIdFlips: bLines.length ? Math.max(...bLines.map((l) => l.idFlips)) : 0 },
    },
    stageA,
    stageB,
  };
  status.state = 'done';
}

function median(xs) {
  const s = [...xs].sort((a, b) => a - b);
  return s[s.length >> 1];
}

main().catch((err) => {
  status.state = 'error';
  status.error = String(err?.stack ?? err);
  result = { ok: false, error: status.error };
});
