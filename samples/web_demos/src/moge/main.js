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
 * Photo → 3D point cloud, fully client-side.
 *
 * Pipeline: image → contain-fit → 448×448 NCHW [0,1] float32 → MoGe-2
 * (LiteRT.js, WebGPU with WASM fallback) → per-pixel affine point map +
 * confidence mask → three.js point cloud colored from the photo.
 *
 * Model I/O (see the conversion notes on the model card):
 *   input : [1, 3, 448, 448] float32, range [0, 1] (no ImageNet norm)
 *   outputs (order not guaranteed in the .tflite — resolved by shape/range):
 *     points [1,448,448,3] · normal [1,448,448,3] · mask(sigmoid) [1,448,448,1]
 *     · metric scale [1,1,1,1]
 *
 * Debug URL params: ?img=<url> (run on that image at boot) · ?backend=wasm
 *   · ?models=<base url> (fetch the weights from somewhere other than HF)
 */
import {
  Tensor,
  isWebGPUSupported,
  loadAndCompile,
  loadLiteRt,
} from '@litertjs/core';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const params = new URLSearchParams(location.search);

// Weights stream from the model card's repo on Hugging Face and are cached
// with the Cache API after the first visit. ?models=<base url> points the
// page at another copy (a local directory, a mirror) — same file names.
const DEFAULT_MODEL_BASE = 'https://huggingface.co/litert-community/MoGe-2-LiteRT/resolve/main/';
const MODEL_BASE = (params.get('models') ?? DEFAULT_MODEL_BASE).replace(/\/?$/, '/');
// fp16 weights halve the download and run slightly faster on WebGPU
// (55.6 vs 58.5 ms on an M4 Max), with outputs equal to fp32 up to the
// weight cast (max |Δ| 0.003). XNNPACK declines the fp16 graph on WASM
// (31x slower, reference kernels), so the WASM path keeps fp32.
const MODEL_URLS = {
  webgpu: MODEL_BASE + 'moge_fp16.tflite',
  wasm: MODEL_BASE + 'moge.tflite',
};
const MODEL_NAME = 'MoGe-2';
// The LiteRT.js WASM runtime is served from litert-wasm/ at the site root
// (vite.config.js copies it there from node_modules). Resolve against this
// module's own URL: in dev it is <root>/moge/main.js, in the build
// <root>/assets/<hash>.js — one level below the runtime dir either way.
const WASM_DIR = new URL(/* @vite-ignore */ '../litert-wasm/', import.meta.url).href;
const SIZE = 448;
const MASK_THRESHOLD = 0.5;
// World-space depth (in units) the photo's median depth is mapped to.
const DEPTH_ANCHOR = 1.6;

const statusEl = document.getElementById('status');
const latencyEl = document.getElementById('latency');
const latencyValueEl = latencyEl.querySelector('b');
const backendEl = document.getElementById('backend');
const envEl = document.getElementById('env');
const backendButtons = [...document.querySelectorAll('#backend-switch button')];
const hintEl = document.getElementById('hint');
const fileEl = document.getElementById('file');
const camBtn = document.getElementById('cam');
const shutterBtn = document.getElementById('shutter');
const videoEl = document.getElementById('video');
const dropOverlay = document.getElementById('drop-overlay');

let model = null;
let accelerator = 'wasm';
let wasmOpts = null; // which loadLiteRt attempt succeeded
const modelBytesByAcc = {}; // webgpu -> fp16 bytes, wasm -> fp32 bytes
let lastSource = null; // last inferred bitmap, re-run on backend switch
let cloud = null;

function status(text, pct = null) {
  statusEl.textContent = text;
  if (pct !== null) {
    const bar = document.createElement('span');
    bar.className = 'bar';
    const fill = document.createElement('i');
    fill.style.width = `${Math.round(pct * 100)}%`;
    bar.appendChild(fill);
    statusEl.appendChild(bar);
  }
}

// --- three.js scene -------------------------------------------------------

const stage = document.getElementById('stage');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
stage.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0d10);
const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
camera.position.set(0, 0, 2.6);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

// Instead of a full auto-rotate (which eventually shows the hollow back and
// blows up the far-background points as the camera swings close), gently
// oscillate the viewpoint left↔right around the capture direction — enough
// parallax to read as 3D, always looking at the subject's good side.
let autoOrbit = true;
let orbitClock = 0;
const ORBIT_YAW_AMP = 0.32; // radians each way
const ORBIT_PERIOD = 7; // seconds per full left-right-left cycle
// Set per-cloud so the whole subject stays framed as the camera swings.
let subjectCenter = new THREE.Vector3(0, 0, -DEPTH_ANCHOR);
let viewDistance = DEPTH_ANCHOR;
renderer.domElement.addEventListener('pointerdown', () => {
  autoOrbit = false;
});

function resize() {
  const { innerWidth: w, innerHeight: h } = window;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

let lastFrame = performance.now();
function animate() {
  const now = performance.now();
  const dt = Math.min((now - lastFrame) / 1000, 0.05);
  lastFrame = now;
  if (autoOrbit && cloud) {
    orbitClock += dt;
    const yaw = ORBIT_YAW_AMP * Math.sin((orbitClock / ORBIT_PERIOD) * Math.PI * 2);
    // Orbit around the subject's own centroid at the framing distance, so the
    // whole subject stays centered and in-frame while the view moves.
    camera.position.set(
      subjectCenter.x + Math.sin(yaw) * viewDistance,
      subjectCenter.y,
      subjectCenter.z + Math.cos(yaw) * viewDistance,
    );
    camera.lookAt(subjectCenter);
  } else {
    controls.update();
  }
  renderer.render(scene, camera);
}
renderer.setAnimationLoop(animate);

// --- model loading --------------------------------------------------------

/** Runtime failures are not always Error objects (a failed <script> load
 * rejects with an Event; WebKit sometimes throws bare strings). */
function errText(err) {
  if (err instanceof Error) return err.message;
  if (typeof err === 'string') return err;
  if (err && typeof err.type === 'string') return `${err.type} event`;
  return String(err);
}

async function fetchModelBytes(url) {
  const cache = 'caches' in window ? await caches.open('moge-demo-v1') : null;
  if (cache) {
    const hit = await cache.match(url);
    if (hit) {
      status('Loading model from cache…');
      return new Uint8Array(await hit.arrayBuffer());
    }
  }
  let response;
  try {
    response = await fetch(url);
  } catch (err) {
    // The browser reports a blocked or unreachable host as a bare
    // "Failed to fetch" — name the URL so the failure is diagnosable.
    const offline = navigator.onLine === false ? ', browser is offline' : '';
    throw new Error(`could not fetch ${url} (${errText(err)}${offline})`);
  }
  if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`);
  const total = Number(response.headers.get('Content-Length')) || 0;
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    const mb = (received / 1048576).toFixed(0);
    status(
      total
        ? `Downloading model (one-time)… ${mb} MB`
        : `Downloading model… ${mb} MB`,
      total ? received / total : null,
    );
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  if (cache) {
    await cache.put(url, new Response(bytes.slice().buffer));
  }
  return bytes;
}

/** wasm / wasm·1-thread / webgpu — mirrors what actually loaded, so a page
 * that silently lost threads (or WebGPU) is visible at a glance. */
function envLabel() {
  if (accelerator === 'webgpu') return 'webgpu';
  return wasmOpts?.threads ? 'wasm' : 'wasm·1-thread';
}

// Which boot stage is in flight, so a failure names the culprit
// (runtime / download / compile webgpu / warm-up wasm / …).
let bootStage = 'runtime';

/** Compile for `acc`, then burn one throwaway inference: the first run after
 * compile carries shader/kernel warm-up (~5 s on WebGPU) and must never land
 * on a user photo or in the latency display. On failure the previous
 * model/backend stay in place. */
async function compileAndWarm(acc) {
  const prevAccelerator = accelerator;
  const prevModel = model;
  accelerator = acc;
  for (const b of backendButtons) b.classList.toggle('active', b.dataset.backend === acc);
  try {
    bootStage = `download ${acc}`;
    modelBytesByAcc[acc] ??= await fetchModelBytes(MODEL_URLS[acc]);
    bootStage = `compile ${acc}`;
    status(`Compiling for ${acc === 'webgpu' ? 'WebGPU' : 'WASM'}…`);
    model = await loadAndCompile(modelBytesByAcc[acc], { accelerator: acc });
    bootStage = `warm-up ${acc}`;
    status('Warming up (one throwaway run)…');
    const gray = new Float32Array(3 * SIZE * SIZE).fill(0.5);
    const start = performance.now();
    await infer(gray);
    const warmSeconds = (performance.now() - start) / 1000;
    envEl.textContent = `${MODEL_NAME} ${acc === 'webgpu' ? 'fp16' : 'fp32'} · ${envLabel()} · warm-up ${warmSeconds.toFixed(1)} s`;
    envEl.style.display = 'block';
    backendEl.textContent = `${envLabel()} · your device`;
  } catch (err) {
    accelerator = prevAccelerator;
    model = prevModel;
    for (const b of backendButtons) {
      b.classList.toggle('active', b.dataset.backend === accelerator);
    }
    throw err;
  }
}

async function boot() {
  try {
    status('Loading runtime…');
    // `threads` and `jspi` are mutually exclusive in LiteRT.js, and threads
    // only work on a cross-origin-isolated page — ask for what can succeed,
    // then fall back to plain.
    const rungs = window.crossOriginIsolated
      ? [{ threads: true }, { threads: false }]
      : [{ threads: false }];
    for (const [index, opts] of rungs.entries()) {
      try {
        await loadLiteRt(WASM_DIR, opts);
        wasmOpts = opts;
        break;
      } catch (err) {
        if (index === rungs.length - 1) {
          throw new Error(`LiteRT.js runtime did not load from ${WASM_DIR} (${errText(err)})`);
        }
      }
    }

    const requested = params.get('backend'); // ?backend=wasm|webgpu
    let acc = isWebGPUSupported() ? 'webgpu' : 'wasm';
    if (requested === 'wasm') acc = 'wasm';
    if (!isWebGPUSupported()) {
      for (const b of backendButtons) {
        if (b.dataset.backend === 'webgpu') {
          b.disabled = true;
          b.title = 'WebGPU is not available in this browser';
        }
      }
    }

    try {
      await compileAndWarm(acc);
    } catch (err) {
      // WebGPU exists on paper in more browsers than it works in (mobile
      // WebKit in particular) — fall back to WASM instead of dying.
      if (acc !== 'webgpu') throw err;
      status(`WebGPU failed (${errText(err)}) — retrying on WASM…`);
      for (const b of backendButtons) {
        if (b.dataset.backend === 'webgpu') {
          b.disabled = true;
          b.title = 'WebGPU failed on this device';
        }
      }
      await compileAndWarm('wasm');
    }
    status('Ready — choose a photo, use the camera, or drop an image.');

    const testUrl = params.get('img');
    if (testUrl) {
      const blob = await (await fetch(testUrl)).blob();
      await runOnImage(await createImageBitmap(blob));
    }
  } catch (err) {
    const hint = bootStage.startsWith('download')
      ? ' The weights stream from Hugging Face: check that huggingface.co is reachable, or pass ?models=<url> to load them from elsewhere.'
      : '';
    status(`Failed to start (${bootStage}): ${errText(err)}.${hint}`);
    console.error(`[moge] boot failed at stage "${bootStage}":`, err);
  }
}

// --- inference ------------------------------------------------------------

const cropCanvas = document.createElement('canvas');
cropCanvas.width = SIZE;
cropCanvas.height = SIZE;
const cropCtx = cropCanvas.getContext('2d', { willReadFrequently: true });

function preprocess(source, sourceWidth, sourceHeight) {
  // Contain-fit (letterbox), NOT center-crop: the whole photo must survive so
  // the subject is never cut off. The padded margins are recorded and later
  // excluded from the point cloud so they don't become a flat backdrop.
  const scale = Math.min(SIZE / sourceWidth, SIZE / sourceHeight);
  const drawW = Math.round(sourceWidth * scale);
  const drawH = Math.round(sourceHeight * scale);
  const offX = Math.floor((SIZE - drawW) / 2);
  const offY = Math.floor((SIZE - drawH) / 2);
  // Pad with edge-ish neutral so MoGe sees a plausible background rather than a
  // hard black frame; the pad is dropped from the cloud regardless.
  cropCtx.fillStyle = '#7f7f7f';
  cropCtx.fillRect(0, 0, SIZE, SIZE);
  cropCtx.drawImage(source, 0, 0, sourceWidth, sourceHeight, offX, offY, drawW, drawH);
  const { data } = cropCtx.getImageData(0, 0, SIZE, SIZE);
  const plane = SIZE * SIZE;
  const nchw = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    nchw[i] = data[i * 4] / 255;
    nchw[plane + i] = data[i * 4 + 1] / 255;
    nchw[2 * plane + i] = data[i * 4 + 2] / 255;
  }
  // Valid = inside the drawn content rect (1 = photo pixel, 0 = padding).
  const valid = new Uint8Array(plane);
  for (let y = offY; y < offY + drawH; y++) {
    for (let x = offX; x < offX + drawW; x++) valid[y * SIZE + x] = 1;
  }
  return { nchw, rgba: data, valid };
}

function sampleAbsMax(array) {
  let max = 0;
  const step = Math.max(1, Math.floor(array.length / 5000));
  for (let i = 0; i < array.length; i += step) {
    const v = Math.abs(array[i]);
    if (v > max) max = v;
  }
  return max;
}

/** The .tflite output order is not guaranteed; identify by shape and range
 * (same strategy as the reference Android app): points is the [h,w,3] map
 * with values beyond [-1,1]; normals are unit vectors. */
function resolveOutputs(buffers) {
  const plane = SIZE * SIZE;
  const big = buffers.filter((b) => b.length === plane * 3);
  const mask = buffers.find((b) => b.length === plane);
  const scale = buffers.find((b) => b.length === 1);
  if (big.length !== 2 || !mask || !scale) {
    throw new Error('unexpected model outputs');
  }
  const points = sampleAbsMax(big[0]) > 2 ? big[0] : big[1];
  return { points, mask, scale: scale[0] };
}

async function infer(nchw) {
  const input = Tensor.fromTypedArray(nchw, [1, 3, SIZE, SIZE]);
  const start = performance.now();
  const outputs = await model.run([input]);
  const buffers = [];
  for (const output of outputs) {
    buffers.push(await output.data());
  }
  const elapsed = performance.now() - start;
  for (const output of outputs) output.delete();
  input.delete();
  return { ...resolveOutputs(buffers), elapsed };
}

// --- point cloud ----------------------------------------------------------

function buildCloud(points, mask, rgba, valid) {
  const plane = SIZE * SIZE;
  const positions = [];
  const colors = [];
  const border = 2; // trim 1-2px inside the content edge (ambiguous-depth rim)
  for (let i = 0; i < plane; i++) {
    if (!valid[i]) continue; // letterbox padding — not part of the photo
    const px = i % SIZE;
    const py = (i / SIZE) | 0;
    // Skip pixels adjacent to a padding pixel (the content-edge rim smears).
    if (
      !valid[i - 1] || !valid[i + 1] ||
      !valid[i - SIZE] || !valid[i + SIZE] ||
      px < border || px >= SIZE - border || py < border || py >= SIZE - border
    ) continue;
    if (mask[i] <= MASK_THRESHOLD) continue;
    const x = points[i * 3];
    const y = points[i * 3 + 1];
    const z = points[i * 3 + 2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    // MoGe camera frame (x right, y down, z forward) → three.js (y up, -z forward)
    positions.push(x, -y, -z);
    colors.push(rgba[i * 4] / 255, rgba[i * 4 + 1] / 255, rgba[i * 4 + 2] / 255);
  }
  if (positions.length === 0) throw new Error('no confident points in this photo');

  // Keep the cloud in camera coordinates and view it FROM the capture
  // viewpoint (three camera at the origin): the cloud starts out looking
  // exactly like the photo, and orbiting reveals the parallax. Scale so the
  // median depth sits at DEPTH_ANCHOR world units, and trim the far tail
  // (deep background shells) which otherwise dwarfs the subject.
  const count = positions.length / 3;
  const depthSample = [];
  const step = Math.max(1, Math.floor(count / 20000));
  for (let i = 0; i < count; i += step) depthSample.push(-positions[i * 3 + 2]);
  depthSample.sort((a, b) => a - b);
  const medianDepth = Math.max(depthSample[Math.floor(depthSample.length / 2)], 1e-6);
  const maxDepth = medianDepth * 3.5;
  const minDepth = medianDepth * 0.45; // trim the near lip (front sand edge streaks)
  const fit = DEPTH_ANCHOR / medianDepth;

  const keptPositions = [];
  const keptColors = [];
  const xs = [];
  const ys = [];
  const zs = [];
  for (let i = 0; i < count; i++) {
    const depth = -positions[i * 3 + 2];
    if (depth > maxDepth || depth < minDepth) continue;
    const x = positions[i * 3] * fit;
    const y = positions[i * 3 + 1] * fit;
    const z = positions[i * 3 + 2] * fit;
    keptPositions.push(x, y, z);
    keptColors.push(colors[i * 3], colors[i * 3 + 1], colors[i * 3 + 2]);
    xs.push(x);
    ys.push(y);
    zs.push(z);
  }

  // Robust screen-plane framing from medians (ignores the smeared
  // silhouette rim and stray points). The orbit pivots on this center so the
  // subject stays put.
  const median = (arr) => {
    const s = arr.slice().sort((a, b) => a - b);
    return s[s.length >> 1];
  };
  const center = new THREE.Vector3(median(xs), median(ys), median(zs));

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(keptPositions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(keptColors, 3));

  const material = new THREE.PointsMaterial({
    size: 0.0075,
    vertexColors: true,
    sizeAttenuation: true,
  });
  const cloudPoints = new THREE.Points(geometry, material);
  cloudPoints.userData.center = center;
  return cloudPoints;
}

async function runOnImage(source) {
  if (!model) return;
  lastSource = source;
  const width = source.videoWidth ?? source.width;
  const height = source.videoHeight ?? source.height;
  status('Running…');
  try {
    const { nchw, rgba, valid } = preprocess(source, width, height);
    const { points, mask, elapsed } = await infer(nchw);
    if (cloud) {
      scene.remove(cloud);
      cloud.geometry.dispose();
      cloud.material.dispose();
    }
    cloud = buildCloud(points, mask, rgba, valid);
    scene.add(cloud);
    // Pivot the orbit on the subject's robust centroid so it stays centered
    // (never translates off-frame); keep the tuned viewing distance that
    // frames the subject large. Start on the capture axis so the cloud opens
    // looking exactly like the photo.
    subjectCenter = cloud.userData.center.clone();
    viewDistance = DEPTH_ANCHOR; // tuned framing: subject large, whole photo shown
    autoOrbit = true;
    orbitClock = 0;
    camera.position.set(subjectCenter.x, subjectCenter.y, subjectCenter.z + viewDistance);
    camera.lookAt(subjectCenter);
    controls.target.copy(subjectCenter);

    latencyValueEl.textContent = `${elapsed.toFixed(0)} ms`;
    latencyEl.style.display = 'block';
    hintEl.style.display = 'block';
    status('Done.');
  } catch (err) {
    status(`Failed: ${errText(err)}`);
  }
}

// --- inputs: file, drop, paste, webcam ------------------------------------

fileEl.addEventListener('change', async () => {
  const file = fileEl.files?.[0];
  if (file) await runOnImage(await createImageBitmap(file));
  fileEl.value = '';
});

window.addEventListener('dragover', (event) => {
  event.preventDefault();
  dropOverlay.style.display = 'flex';
});
window.addEventListener('dragleave', (event) => {
  if (event.relatedTarget === null) dropOverlay.style.display = 'none';
});
window.addEventListener('drop', async (event) => {
  event.preventDefault();
  dropOverlay.style.display = 'none';
  const file = event.dataTransfer?.files?.[0];
  if (file && file.type.startsWith('image/')) {
    await runOnImage(await createImageBitmap(file));
  }
});

window.addEventListener('paste', async (event) => {
  const item = [...(event.clipboardData?.items ?? [])].find((entry) =>
    entry.type.startsWith('image/'),
  );
  const file = item?.getAsFile();
  if (file) await runOnImage(await createImageBitmap(file));
});

let stream = null;
camBtn.addEventListener('click', async () => {
  if (stream) {
    stopCamera();
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment' },
    });
    videoEl.srcObject = stream;
    videoEl.style.display = 'block';
    shutterBtn.style.display = 'inline-block';
    camBtn.textContent = 'Stop camera';
  } catch (err) {
    status(`Camera: ${errText(err)}`);
  }
});

function stopCamera() {
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  videoEl.srcObject = null;
  videoEl.style.display = 'none';
  shutterBtn.style.display = 'none';
  camBtn.textContent = 'Use camera';
}

shutterBtn.addEventListener('click', async () => {
  if (videoEl.videoWidth) {
    // Snapshot to a bitmap so the frame survives stopCamera() and backend
    // switches can re-run it.
    const frame = await createImageBitmap(videoEl);
    stopCamera();
    await runOnImage(frame);
  }
});

// --- backend switch (webgpu / wasm) ---------------------------------------

for (const button of backendButtons) {
  button.addEventListener('click', async () => {
    const acc = button.dataset.backend;
    if (acc === accelerator || !model || button.disabled) return;
    const webgpuMissing = !isWebGPUSupported();
    for (const b of backendButtons) b.disabled = true;
    try {
      await compileAndWarm(acc);
      if (lastSource) {
        await runOnImage(lastSource);
      } else {
        status('Ready — choose a photo, use the camera, or drop an image.');
      }
    } catch (err) {
      status(`Backend switch failed: ${errText(err)}`);
    } finally {
      for (const b of backendButtons) {
        b.disabled = webgpuMissing && b.dataset.backend === 'webgpu';
      }
    }
  });
}

boot();
