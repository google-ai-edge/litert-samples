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
 * Builds one of the three unpacked extensions:
 *   node build.mjs           → dist/        (Page Voice, from src/ + public/)
 *   node build.mjs --3d      → dist3d/      (Page 3D, from src3d/ + public3d/)
 *   node build.mjs --ocr     → dist-ocr/    (Page Text, from src-ocr/ +
 *                                            public-ocr/)
 *
 * All bundle each entry point with esbuild (classic scripts — content
 * scripts and the MV3 service worker don't take ESM here), copy static files
 * from the public dir, and copy the LiteRT.js wasm runtime next to
 * offscreen.html.
 *
 * __DEV__ turns on what the tests under tools/, tools3d/ and tools-ocr/
 * drive: the service worker starts the engine with the browser, and the
 * offscreen document exposes its status on __pv / __p3 / __pt.
 */
import { build } from 'esbuild';
import { cpSync, mkdirSync, rmSync } from 'node:fs';

const three = process.argv.includes('--3d');
const ocr = process.argv.includes('--ocr');
const src = three ? 'src3d' : ocr ? 'src-ocr' : 'src';
const pub = three ? 'public3d' : ocr ? 'public-ocr' : 'public';
const outdir = three ? 'dist3d' : ocr ? 'dist-ocr' : 'dist';

rmSync(outdir, { recursive: true, force: true });

await build({
  entryPoints: {
    background: `${src}/background.js`,
    content: `${src}/content.js`,
    popup: `${src}/popup.js`,
    offscreen: `${src}/offscreen/main.js`,
  },
  bundle: true,
  format: 'iife',
  target: 'chrome128',
  outdir,
  logLevel: 'info',
  define: { __DEV__: 'true' },
});

cpSync(pub, outdir, { recursive: true });
// LiteRT.js picks a wasm variant as: !relaxedSimd → compat, threads →
// threaded, jspi → jspi, else → plain. Chrome 128+ (our minimum) always has
// relaxed SIMD, and `threads` and `jspi` are mutually exclusive — we always
// ask for threads — so compat and jspi can never be selected. Copying all
// four would put 37 MB in the extension to use 18 MB of it.
const WASM_VARIANTS = ['litert_wasm_internal', 'litert_wasm_threaded_internal'];
mkdirSync(`${outdir}/litert-wasm`, { recursive: true });
for (const v of WASM_VARIANTS) {
  for (const ext of ['js', 'wasm']) {
    cpSync(`node_modules/@litertjs/core/wasm/${v}.${ext}`, `${outdir}/litert-wasm/${v}.${ext}`);
  }
}

console.log(`${outdir}/ ready — load it as an unpacked extension.`);
