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
 * Matcha-TTS host orchestration, port of the validated kotlin_replica.py:
 *   symbol ids → blank-interspersed [256] → emb.bin lookup [1,256,192]
 *   → textenc → mu[1,80,256], logw[1,1,256]
 *   → durations ceil(exp(logw))·length_scale → integer length-regulator
 *   → N Euler ODE steps of decoder(x, mu_y, t_sin[1,160], ymask)
 *   → mel = x·std + mean → vocoder → wav, trimmed to ylen·hop samples.
 */
import { Tensor } from '@litertjs/core';

/** Deterministic standard-normal source: mulberry32 + Box–Muller. */
export function makeRandn(seed) {
  let s = seed >>> 0;
  const uniform = () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let spare = null;
  return () => {
    if (spare !== null) {
      const v = spare;
      spare = null;
      return v;
    }
    let a = 0;
    while (a === 0) a = uniform();
    const r = Math.sqrt(-2 * Math.log(a));
    const th = 2 * Math.PI * uniform();
    spare = r * Math.sin(th);
    return r * Math.cos(th);
  };
}

/** Sinusoidal ODE-time embedding (Matcha SinusoidalPosEmb, scale=1000). */
export function sinPosEmb(t, dim) {
  const half = dim / 2;
  const out = new Float32Array(dim);
  const k = -Math.log(10000) / (half - 1);
  for (let i = 0; i < half; i++) {
    const e = 1000.0 * t * Math.exp(i * k);
    out[i] = Math.sin(e);
    out[half + i] = Math.cos(e);
  }
  return out;
}

export class Synthesizer {
  /**
   * @param models {textenc, decoder, vocoder} compiled LiteRT models
   * @param emb Float32Array 178×192 phoneme embedding table (emb.bin)
   * @param cfg config.json contents
   */
  constructor(models, emb, cfg) {
    this.m = models;
    this.emb = emb;
    this.cfg = cfg;
  }

  /**
   * @param pids symbol ids (≤ (MAX_TEXT-1)/2)
   * @returns {wav, ylen, mel, timings} — wav trimmed+clipped, mel [80×MAX_MEL]
   */
  async run(pids, { steps = 10, seed = 0 } = {}) {
    const { MAX_TEXT, MAX_MEL, n_feats: F, n_channels: CH, hop,
      mel_mean, mel_std, length_scale, in_channels: TDIM } = this.cfg;
    const timings = { textenc: 0, decoder: 0, vocoder: 0 };

    // ids[2k+1] = pids[k]; blanks (id 0) elsewhere → embedding lookup
    const embx = new Float32Array(MAX_TEXT * CH);
    for (let t = 0; t < MAX_TEXT; t++) {
      const k = (t - 1) / 2;
      const row = t % 2 === 1 && k < pids.length ? pids[k] : 0;
      embx.set(this.emb.subarray(row * CH, row * CH + CH), t * CH);
    }
    const tx = Math.min(pids.length * 2 + 1, MAX_TEXT);
    const tmask = new Float32Array(MAX_TEXT);
    for (let t = 0; t < tx; t++) tmask[t] = 1;

    // text encoder → mu[1,80,256], logw[1,1,256] (output order not guaranteed)
    let t0 = performance.now();
    const embT = Tensor.fromTypedArray(embx, [1, MAX_TEXT, CH]);
    const tmT = Tensor.fromTypedArray(tmask, [1, 1, MAX_TEXT]);
    const teOuts = await this.m.textenc.run([embT, tmT]);
    const teBufs = [];
    for (const o of teOuts) teBufs.push(await o.data());
    for (const o of teOuts) o.delete();
    embT.delete();
    tmT.delete();
    const [mu, logw] = teBufs[0].length === F * MAX_TEXT
      ? [teBufs[0], teBufs[1]] : [teBufs[1], teBufs[0]];
    timings.textenc += performance.now() - t0;

    // durations → integer length-regulator (walk the cumulative sum)
    const cum = new Float64Array(MAX_TEXT);
    let acc = 0;
    for (let t = 0; t < MAX_TEXT; t++) {
      acc += Math.ceil(Math.exp(logw[t]) * tmask[t]) * length_scale;
      cum[t] = acc;
    }
    const ylen = Math.min(Math.max(Math.trunc(acc), 1), MAX_MEL);
    const muY = new Float32Array(F * MAX_MEL);
    let p = 0;
    for (let f = 0; f < ylen; f++) {
      while (p < MAX_TEXT - 1 && cum[p] <= f) p++;
      for (let c = 0; c < F; c++) muY[c * MAX_MEL + f] = mu[c * MAX_TEXT + p];
    }
    const ymask = new Float32Array(MAX_MEL);
    for (let f = 0; f < ylen; f++) ymask[f] = 1;

    // Euler ODE: x ~ N(0,1) in the valid region, x += v/steps per step
    const randn = makeRandn(seed);
    const x = new Float32Array(F * MAX_MEL);
    for (let c = 0; c < F; c++) {
      for (let f = 0; f < ylen; f++) x[c * MAX_MEL + f] = randn();
    }
    const dt = 1 / steps;
    t0 = performance.now();
    for (let s = 0; s < steps; s++) {
      const xT = Tensor.fromTypedArray(x, [1, F, MAX_MEL]);
      const muT = Tensor.fromTypedArray(muY, [1, F, MAX_MEL]);
      const tsT = Tensor.fromTypedArray(sinPosEmb(s / steps, TDIM), [1, TDIM]);
      const ymT = Tensor.fromTypedArray(ymask, [1, 1, MAX_MEL]);
      const outs = await this.m.decoder.run([xT, muT, tsT, ymT]);
      const v = await outs[0].data();
      for (const o of outs) o.delete();
      xT.delete(); muT.delete(); tsT.delete(); ymT.delete();
      for (let i = 0; i < x.length; i++) x[i] += dt * v[i];
    }
    timings.decoder += performance.now() - t0;

    // denormalize (valid region only) → vocoder → trim to real length
    const mel = new Float32Array(F * MAX_MEL);
    for (let c = 0; c < F; c++) {
      for (let f = 0; f < ylen; f++) {
        mel[c * MAX_MEL + f] = x[c * MAX_MEL + f] * mel_std + mel_mean;
      }
    }
    t0 = performance.now();
    const wav = await this.vocode(mel, ylen);
    timings.vocoder += performance.now() - t0;
    return { wav, ylen, mel, timings };
  }

  /** mel [1,80,MAX_MEL] → trimmed, clipped waveform (ylen·hop samples). */
  async vocode(mel, ylen, model = this.m.vocoder) {
    const { MAX_MEL, n_feats: F, hop } = this.cfg;
    const melT = Tensor.fromTypedArray(mel, [1, F, MAX_MEL]);
    const outs = await model.run([melT]);
    const wavFull = await outs[0].data();
    for (const o of outs) o.delete();
    melT.delete();
    const n = ylen * hop;
    const wav = new Float32Array(n);
    for (let i = 0; i < n; i++) wav[i] = Math.max(-1, Math.min(1, wavFull[i]));
    return wav;
  }
}

/** Float32 mono → 16-bit PCM WAV blob. */
export function encodeWav(samples, sampleRate) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const dv = new DataView(buf);
  const str = (off, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(off + i, s.charCodeAt(i)); };
  str(0, 'RIFF');
  dv.setUint32(4, 36 + samples.length * 2, true);
  str(8, 'WAVE');
  str(12, 'fmt ');
  dv.setUint32(16, 16, true);
  dv.setUint16(20, 1, true);
  dv.setUint16(22, 1, true);
  dv.setUint32(24, sampleRate, true);
  dv.setUint32(28, sampleRate * 2, true);
  dv.setUint16(32, 2, true);
  dv.setUint16(34, 16, true);
  str(36, 'data');
  dv.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const v = Math.max(-1, Math.min(1, samples[i]));
    dv.setInt16(44 + i * 2, v < 0 ? v * 32768 : v * 32767, true);
  }
  return new Blob([buf], { type: 'audio/wav' });
}
