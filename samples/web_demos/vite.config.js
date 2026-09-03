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

import { copyFileSync, createReadStream, mkdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

const HERE = dirname(fileURLToPath(import.meta.url));

// Files the pages need at the site root, taken straight from node_modules:
// the LiteRT.js WASM runtime (plain / threaded / compat variants, which
// loadLiteRt() fetches by name from one directory) and coi-serviceworker,
// which injects COOP/COEP on hosts that cannot send them (GitHub Pages).
// Served from node_modules in dev and copied into dist/ by the build, so no
// generated file lives in src/.
const RUNTIME_FILES = [
  ...['litert_wasm_internal', 'litert_wasm_threaded_internal', 'litert_wasm_compat_internal']
    .flatMap((variant) => ['js', 'wasm'].map((ext) => ({
      from: `node_modules/@litertjs/core/wasm/${variant}.${ext}`,
      to: `litert-wasm/${variant}.${ext}`,
    }))),
  { from: 'node_modules/coi-serviceworker/coi-serviceworker.min.js', to: 'coi-serviceworker.min.js' },
];

// COOP/COEP make the page cross-origin isolated so LiteRT.js can use the
// threaded WASM build (SharedArrayBuffer); without them the CPU backend is
// single-threaded and the Matcha decoder is ~5-50x slower. The deployed site
// gets the same headers from coi-serviceworker. 'require-corp' rather than
// 'credentialless' because Safari only isolates under require-corp; the
// Hugging Face model fetches are CORS requests against ACAO:*, which
// require-corp accepts.
const coi = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

function runtimeFiles() {
  let outDir;
  return {
    name: 'litert-runtime-files',
    configResolved(config) {
      outDir = resolve(config.root, config.build.outDir);
    },
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const path = req.url.split('?')[0].slice(1);
        const file = RUNTIME_FILES.find((f) => f.to === path);
        if (!file) return next();
        // Same COOP/COEP as the pages: a dedicated worker's script must carry
        // the owner's COEP or the browser refuses to start it (the threaded
        // runtime spawns its thread pool from this file).
        for (const [k, v] of Object.entries(coi)) res.setHeader(k, v);
        res.setHeader('Content-Type', path.endsWith('.wasm') ? 'application/wasm' : 'text/javascript');
        createReadStream(join(HERE, file.from)).pipe(res);
      });
    },
    writeBundle() {
      for (const { from, to } of RUNTIME_FILES) {
        mkdirSync(join(outDir, dirname(to)), { recursive: true });
        copyFileSync(join(HERE, from), join(outDir, to));
      }
    },
  };
}

export default defineConfig({
  root: 'src',
  // Relative asset URLs, so the built site works from any sub-path
  // (this repo's GitHub Pages serves it under /litert-samples/...).
  base: './',
  publicDir: false,
  server: { headers: coi },
  preview: { headers: coi },
  build: {
    outDir: '../dist',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        index: resolve(HERE, 'src/index.html'),
        moge: resolve(HERE, 'src/moge/index.html'),
        'matcha-tts': resolve(HERE, 'src/matcha-tts/index.html'),
      },
    },
  },
  plugins: [runtimeFiles()],
});
