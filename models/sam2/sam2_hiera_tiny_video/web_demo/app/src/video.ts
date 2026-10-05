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

// Frame sources: decode an uploaded video at a fixed sampling rate, or render
// the built-in procedural demo clip. Frames are kept as display-resolution
// ImageBitmaps; the C++ pipeline's preprocess graph makes the model input.

export interface Clip {
  name: string;
  frames: ImageBitmap[];
  width: number;
  height: number;
  fps: number;
}

function once(el: EventTarget, ev: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const ok = () => {
      el.removeEventListener('error', bad);
      resolve();
    };
    const bad = () => {
      el.removeEventListener(ev, ok);
      reject(new Error(`video ${ev} failed`));
    };
    el.addEventListener(ev, ok, {once: true});
    el.addEventListener('error', bad, {once: true});
  });
}

/** Decoded frames are kept as RGBA bitmaps; cap their total size (bytes). */
const FRAME_BUDGET = 400 * 1024 * 1024;

/** An uploaded file, or a video served by URL (e.g. the bundled sample). */
export type VideoSource = File | {url: string; name: string};

export async function decodeVideo(src: VideoSource, fps: number, maxSeconds: number, maxSide: number,
                                  onProgress: (f: number) => void): Promise<Clip> {
  const owned = src instanceof File;
  const url = owned ? URL.createObjectURL(src) : src.url;
  const v = document.createElement('video');
  v.muted = true;
  v.playsInline = true;
  v.preload = 'auto';
  v.src = url;
  try {
    await once(v, 'loadeddata');
    const w = v.videoWidth, h = v.videoHeight;
    if (!w || !h) throw new Error('could not read video dimensions');
    const duration = Number.isFinite(v.duration) ? v.duration : maxSeconds;
    const n = Math.max(1, Math.floor(Math.min(duration, maxSeconds) * fps));
    // Display resolution: at most maxSide, and small enough that n frames fit
    // the memory budget (the model input is resized separately anyway).
    const budgetScale = Math.sqrt(FRAME_BUDGET / (n * 4 * w * h));
    const scale = Math.min(1, maxSide / Math.max(w, h), budgetScale);
    const rw = Math.round(w * scale), rh = Math.round(h * scale);
    const frames: ImageBitmap[] = [];
    for (let i = 0; i < n; i++) {
      // Seek to the middle of each sampling interval; never to exactly the
      // current time (no 'seeked' event would fire).
      const target = Math.min((i + 0.5) / fps, Math.max(0, duration - 1e-3));
      v.currentTime = target;
      await once(v, 'seeked');
      frames.push(await createImageBitmap(v, {resizeWidth: rw, resizeHeight: rh, resizeQuality: 'high'}));
      onProgress((i + 1) / n);
    }
    return {name: src.name, frames, width: rw, height: rh, fps};
  } finally {
    v.removeAttribute('src');
    v.load();
    if (owned) URL.revokeObjectURL(url);
  }
}

/** Procedural demo: a bouncing ball, a car and a kite crossing a park scene. */
export async function demoClip(n = 60, width = 960, height = 540, fps = 12): Promise<Clip> {
  const c = new OffscreenCanvas(width, height);
  const g = c.getContext('2d')!;
  const frames: ImageBitmap[] = [];
  const groundY = height * 0.68;
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    // Sky + ground.
    const sky = g.createLinearGradient(0, 0, 0, groundY);
    sky.addColorStop(0, '#7fb6e8');
    sky.addColorStop(1, '#d9ecf7');
    g.fillStyle = sky;
    g.fillRect(0, 0, width, groundY);
    const grass = g.createLinearGradient(0, groundY, 0, height);
    grass.addColorStop(0, '#6aa84f');
    grass.addColorStop(1, '#38761d');
    g.fillStyle = grass;
    g.fillRect(0, groundY, width, height - groundY);
    // Static scenery (distractors).
    g.fillStyle = '#f6d55c';
    g.beginPath();
    g.arc(width * 0.86, height * 0.14, 38, 0, Math.PI * 2);
    g.fill();
    for (const [tx, s] of [[0.12, 1], [0.3, 0.8], [0.72, 1.1], [0.93, 0.7]] as const) {
      g.fillStyle = '#6b4f2a';
      g.fillRect(width * tx - 8 * s, groundY - 70 * s, 16 * s, 80 * s);
      g.fillStyle = '#2e7d32';
      g.beginPath();
      g.arc(width * tx, groundY - 95 * s, 48 * s, 0, Math.PI * 2);
      g.fill();
    }
    // Kite: drifts across the sky, rotating.
    const kx = width * (0.15 + 0.6 * t), ky = height * (0.22 + 0.06 * Math.sin(t * 9));
    g.save();
    g.translate(kx, ky);
    g.rotate(Math.sin(t * 7) * 0.4);
    g.fillStyle = '#e53935';
    g.beginPath();
    g.moveTo(0, -46);
    g.lineTo(32, 0);
    g.lineTo(0, 56);
    g.lineTo(-32, 0);
    g.closePath();
    g.fill();
    g.strokeStyle = '#fff';
    g.lineWidth = 3;
    g.beginPath();
    g.moveTo(0, -46);
    g.lineTo(0, 56);
    g.moveTo(-32, 0);
    g.lineTo(32, 0);
    g.stroke();
    g.restore();
    // Car: drives right-to-left along the ground.
    const cx = width * (1.05 - 1.0 * t), cy = groundY + 40;
    g.fillStyle = '#1e5bd6';
    g.beginPath();
    g.roundRect(cx - 90, cy - 38, 180, 46, 12);
    g.fill();
    g.beginPath();
    g.roundRect(cx - 50, cy - 70, 100, 36, 14);
    g.fill();
    g.fillStyle = '#bfe3ff';
    g.fillRect(cx - 40, cy - 62, 36, 24);
    g.fillRect(cx + 4, cy - 62, 36, 24);
    g.fillStyle = '#222';
    for (const wx of [-52, 52]) {
      g.beginPath();
      g.arc(cx + wx, cy + 10, 20, 0, Math.PI * 2);
      g.fill();
    }
    // Ball: bounces left-to-right in front of the car.
    const bx = width * (0.08 + 0.84 * t);
    const by = groundY + 60 - Math.abs(Math.sin(t * Math.PI * 3)) * 190;
    const ball = g.createRadialGradient(bx - 12, by - 12, 4, bx, by, 34);
    ball.addColorStop(0, '#ffd180');
    ball.addColorStop(1, '#ef6c00');
    g.fillStyle = ball;
    g.beginPath();
    g.arc(bx, by, 34, 0, Math.PI * 2);
    g.fill();
    frames.push(await createImageBitmap(c));
  }
  return {name: 'Demo clip (procedural)', frames, width, height, fps};
}
