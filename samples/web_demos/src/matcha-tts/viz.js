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
 * Full-screen mel-spectrogram + waveform rendering with a playback cursor.
 * The spectrogram is drawn from the decoder's own mel output (the actual
 * tensor sent to the vocoder), not re-analyzed from audio.
 */

const STOPS = [
  [0.0, [0, 0, 4]],
  [0.25, [80, 18, 123]],
  [0.5, [182, 54, 121]],
  [0.75, [251, 136, 97]],
  [1.0, [252, 253, 191]],
];

function magma(t) {
  const v = Math.max(0, Math.min(1, t));
  for (let i = 1; i < STOPS.length; i++) {
    if (v <= STOPS[i][0]) {
      const [t0, c0] = STOPS[i - 1];
      const [t1, c1] = STOPS[i];
      const f = (v - t0) / (t1 - t0);
      return [
        c0[0] + (c1[0] - c0[0]) * f,
        c0[1] + (c1[1] - c0[1]) * f,
        c0[2] + (c1[2] - c0[2]) * f,
      ];
    }
  }
  return STOPS[STOPS.length - 1][1];
}

export class Viz {
  constructor(canvas, { nFeats, melMean, melStd, sampleRate }) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.F = nFeats;
    this.melLo = melMean - 2.5 * melStd;
    this.melHi = melMean + 3.0 * melStd;
    this.sr = sampleRate;
    this.frames = 0;
    this.melCanvas = null; // offscreen, 1px per mel frame × F px tall
    this.wavChunks = [];
    this.wavLen = 0;
    this.audioCtx = null;
    this.playStart = 0;
    this.playDur = 0;
    this.dirty = true;
    window.addEventListener('resize', () => { this.dirty = true; });
    const loop = () => {
      this.draw();
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  reset() {
    this.frames = 0;
    this.melCanvas = null;
    this.wavChunks = [];
    this.wavLen = 0;
    this.playDur = 0;
    this.dirty = true;
  }

  /** Append one synthesized chunk: mel [F×maxMel] with ylen valid frames + its wav. */
  append(mel, ylen, maxMel, wav) {
    const old = this.melCanvas;
    const next = document.createElement('canvas');
    next.width = this.frames + ylen;
    next.height = this.F;
    const nctx = next.getContext('2d');
    if (old) nctx.drawImage(old, 0, 0);
    const img = nctx.createImageData(ylen, this.F);
    for (let f = 0; f < ylen; f++) {
      for (let c = 0; c < this.F; c++) {
        const v = (mel[c * maxMel + f] - this.melLo) / (this.melHi - this.melLo);
        const [r, g, b] = magma(v);
        // low mel bins at the bottom
        const o = ((this.F - 1 - c) * ylen + f) * 4;
        img.data[o] = r;
        img.data[o + 1] = g;
        img.data[o + 2] = b;
        img.data[o + 3] = 255;
      }
    }
    nctx.putImageData(img, this.frames, 0);
    this.melCanvas = next;
    this.frames += ylen;
    this.wavChunks.push(wav);
    this.wavLen += wav.length;
    this.dirty = true;
  }

  /** Attach the playback clock: cursor runs [when, when+durSeconds]. */
  playFrom(audioCtx, when, durSeconds) {
    this.audioCtx = audioCtx;
    this.playStart = when;
    this.playDur = durSeconds;
  }

  playFraction() {
    if (!this.audioCtx || this.playDur <= 0) return -1;
    const f = (this.audioCtx.currentTime - this.playStart) / this.playDur;
    return f >= 0 && f <= 1 ? f : -1;
  }

  draw() {
    const playing = this.playFraction() >= 0;
    if (!this.dirty && !playing && !this.wasPlaying) return;
    this.wasPlaying = playing;
    this.dirty = false;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W = Math.round(this.canvas.clientWidth * dpr);
    const H = Math.round(this.canvas.clientHeight * dpr);
    if (this.canvas.width !== W || this.canvas.height !== H) {
      this.canvas.width = W;
      this.canvas.height = H;
    }
    const ctx = this.ctx;
    ctx.fillStyle = '#0b0d10';
    ctx.fillRect(0, 0, W, H);
    if (!this.frames) return;

    const specY = H * 0.08;
    const specH = H * 0.52;
    const wavY = H * 0.66;
    const wavH = H * 0.26;

    ctx.imageSmoothingEnabled = true;
    ctx.globalAlpha = 0.9;
    ctx.drawImage(this.melCanvas, 0, 0, this.frames, this.F, 0, specY, W, specH);
    ctx.globalAlpha = 1;

    // waveform: min/max per pixel column over the concatenated samples
    if (this.wavLen) {
      ctx.strokeStyle = 'rgba(124, 196, 255, 0.85)';
      ctx.lineWidth = Math.max(1, dpr * 0.8);
      ctx.beginPath();
      const mid = wavY + wavH / 2;
      const perPx = this.wavLen / W;
      let chunkIdx = 0;
      let chunkOff = 0;
      for (let px = 0; px < W; px++) {
        const s0 = Math.floor(px * perPx);
        const s1 = Math.max(s0 + 1, Math.floor((px + 1) * perPx));
        let lo = 1;
        let hi = -1;
        for (let s = s0; s < s1; s++) {
          while (chunkIdx < this.wavChunks.length && s - chunkOff >= this.wavChunks[chunkIdx].length) {
            chunkOff += this.wavChunks[chunkIdx].length;
            chunkIdx++;
          }
          if (chunkIdx >= this.wavChunks.length) break;
          const v = this.wavChunks[chunkIdx][s - chunkOff];
          if (v < lo) lo = v;
          if (v > hi) hi = v;
        }
        if (hi < lo) continue;
        ctx.moveTo(px + 0.5, mid - hi * (wavH / 2));
        ctx.lineTo(px + 0.5, mid - lo * (wavH / 2) + 1);
      }
      ctx.stroke();
    }

    const frac = this.playFraction();
    if (frac >= 0) {
      const x = frac * W;
      ctx.strokeStyle = 'rgba(232, 234, 237, 0.9)';
      ctx.lineWidth = dpr;
      ctx.beginPath();
      ctx.moveTo(x, specY);
      ctx.lineTo(x, wavY + wavH);
      ctx.stroke();
    }
  }
}
