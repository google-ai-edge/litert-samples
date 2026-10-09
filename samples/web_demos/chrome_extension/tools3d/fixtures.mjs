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
 * Test photos for the Page 3D tests, drawn in Node and encoded as PNG — no
 * network, no files on disk. Three scenes with clear depth cues, in three
 * aspect ratios so the letterbox runs both ways:
 *   landscape  960×640  sky, hill ridges fading with distance, a meadow
 *   spheres    800×800  three spheres on a checkered floor (ray-cast)
 *   figure     640×800  a head-and-shoulders shape in front of a wall
 *
 *   import { fixturePng } from './fixtures.mjs';
 *   fixturePng('spheres')  // → Buffer, cached per process
 */
import { deflateSync } from 'node:zlib';

export const FIXTURE_NAMES = ['landscape', 'spheres', 'figure'];

// --- PNG encoding (8-bit RGB, no filtering) ----------------------------------

const CRC_TABLE = Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});

function crc32(bytes) {
  let c = 0xffffffff;
  for (const b of bytes) c = CRC_TABLE[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const out = Buffer.alloc(12 + data.length);
  out.writeUInt32BE(data.length, 0);
  out.write(type, 4, 'latin1');
  data.copy(out, 8);
  out.writeUInt32BE(crc32(out.subarray(4, 8 + data.length)), 8 + data.length);
  return out;
}

function encodePng(width, height, shade) {
  const stride = width * 3 + 1;
  const raw = Buffer.alloc(stride * height); // each row starts with filter byte 0
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const [r, g, b] = shade(x, y);
      const i = y * stride + 1 + x * 3;
      raw[i] = clamp255(r);
      raw[i + 1] = clamp255(g);
      raw[i + 2] = clamp255(b);
    }
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 2; // RGB
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw)),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

const clamp255 = (v) => Math.max(0, Math.min(255, Math.round(v)));
const mix = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);

// --- landscape: layered ridges with aerial perspective -----------------------

function landscape(width, height) {
  const ridges = [ // far → near: base height (0 = top), amplitude, color
    { base: 0.50, amp: 0.05, f: [5.0, 11.0], color: [150, 172, 196] },
    { base: 0.58, amp: 0.06, f: [3.5, 8.0], color: [96, 132, 120] },
    { base: 0.70, amp: 0.05, f: [2.2, 6.5], color: [70, 112, 62] },
  ];
  const ridgeAt = (r, u, k) =>
    r.base + r.amp * (0.6 * Math.sin(u * r.f[0] + k) + 0.4 * Math.sin(u * r.f[1] + 2 * k));
  return (x, y) => {
    const u = x / width;
    const v = y / height;
    for (let k = ridges.length - 1; k >= 0; k--) {
      const top = ridgeAt(ridges[k], u, k + 1);
      if (v < top) continue;
      if (k === ridges.length - 1) {
        // meadow: stripes that get finer toward the horizon
        const depth = (v - top) / (1 - top);
        const stripe = 0.5 + 0.5 * Math.sin(1 / (0.03 + 0.25 * depth) + u * 6);
        return mix(ridges[k].color, [120, 160, 70], 0.35 * depth + 0.15 * stripe);
      }
      return mix(ridges[k].color, [205, 220, 236], 0.15 * (ridges.length - 1 - k));
    }
    const sun = Math.hypot(u - 0.76, (v - 0.18) * (height / width));
    const sky = mix([92, 146, 214], [214, 228, 242], Math.min(1, v / 0.55));
    return sun < 0.045 ? [255, 246, 220] : mix(sky, [255, 240, 205], Math.max(0, 0.25 - sun) * 2);
  };
}

// --- ray-cast scenes: ellipsoids, a floor plane, an optional back wall -----

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a) => {
  const l = Math.hypot(a[0], a[1], a[2]);
  return [a[0] / l, a[1] / l, a[2] / l];
};

/** Nearest hit of a ray (origin o, unit direction d) or null. */
function intersect(scene, o, d) {
  let best = null;
  for (const e of scene.ellipsoids) {
    const oc = sub(o, e.c).map((v, i) => v / e.r[i]);
    const dd = d.map((v, i) => v / e.r[i]);
    const a = dot(dd, dd);
    const b = 2 * dot(oc, dd);
    const disc = b * b - 4 * a * (dot(oc, oc) - 1);
    if (disc < 0) continue;
    const t = (-b - Math.sqrt(disc)) / (2 * a);
    if (t > 1e-4 && (!best || t < best.t)) {
      const p = o.map((v, i) => v + t * d[i]);
      const n = norm(sub(p, e.c).map((v, i) => v / (e.r[i] * e.r[i])));
      best = { t, p, n, color: e.color };
    }
  }
  if (d[1] < 0) { // floor y = scene.floor
    const t = (scene.floor - o[1]) / d[1];
    if (t > 1e-4 && (!best || t < best.t)) {
      const p = o.map((v, i) => v + t * d[i]);
      const check = (Math.floor(p[0] * 1.5) + Math.floor(p[2] * 1.5)) & 1;
      best = { t, p, n: [0, 1, 0], color: check ? [215, 212, 204] : [92, 88, 84] };
    }
  }
  if (scene.wall !== undefined && d[2] < 0) { // back wall z = scene.wall
    const t = (scene.wall - o[2]) / d[2];
    if (t > 1e-4 && (!best || t < best.t)) {
      const p = o.map((v, i) => v + t * d[i]);
      best = { t, p, n: [0, 0, 1], color: mix([178, 186, 198], [120, 128, 142], (p[1] + 2) / 6) };
    }
  }
  return best;
}

function raycast(scene, width, height) {
  const light = norm([-0.55, 0.8, 0.45]);
  const tanHalf = Math.tan((50 * Math.PI) / 360);
  const sample = (sx, sy) => {
    const d = norm([
      (2 * sx / width - 1) * tanHalf * (width / height),
      (1 - 2 * sy / height) * tanHalf,
      -1,
    ]);
    const hit = intersect(scene, [0, 0, 0], d);
    if (!hit) return mix([200, 214, 232], [110, 150, 210], Math.max(0, d[1]) * 2.5);
    const lit = intersect(scene, hit.p, light) ? 0 : Math.max(0, dot(hit.n, light));
    const fog = Math.min(0.5, hit.t / 40);
    return mix(hit.color.map((c) => c * (0.3 + 0.7 * lit)), [200, 214, 232], fog);
  };
  return (x, y) => { // 2×2 supersampling
    const acc = [0, 0, 0];
    for (const [ox, oy] of [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]) {
      const c = sample(x + ox, y + oy);
      for (let i = 0; i < 3; i++) acc[i] += c[i] / 4;
    }
    return acc;
  };
}

const SPHERES = {
  floor: -1,
  ellipsoids: [
    { c: [-0.9, -0.45, -3.2], r: [0.55, 0.55, 0.55], color: [205, 70, 60] },
    { c: [0.55, -0.35, -4.6], r: [0.65, 0.65, 0.65], color: [70, 160, 95] },
    { c: [1.6, -0.2, -7.5], r: [0.8, 0.8, 0.8], color: [70, 105, 200] },
  ],
};

const FIGURE = {
  floor: -2.2,
  wall: -6,
  ellipsoids: [
    { c: [0, 0.35, -3.4], r: [0.42, 0.55, 0.45], color: [214, 170, 140] }, // head
    { c: [0, -0.3, -3.45], r: [0.21, 0.32, 0.21], color: [200, 156, 128] }, // neck
    { c: [0, -1.3, -3.6], r: [1.1, 0.72, 0.5], color: [60, 82, 120] }, // shoulders
  ],
};

// --- public API -------------------------------------------------------------

const SIZES = { landscape: [960, 640], spheres: [800, 800], figure: [640, 800] };
const cache = new Map();

export function fixturePng(name) {
  if (!SIZES[name]) throw new Error(`unknown fixture ${name}`);
  if (!cache.has(name)) {
    const [w, h] = SIZES[name];
    const shade = name === 'landscape' ? landscape(w, h)
      : raycast(name === 'spheres' ? SPHERES : FIGURE, w, h);
    cache.set(name, encodePng(w, h, shade));
  }
  return cache.get(name);
}
