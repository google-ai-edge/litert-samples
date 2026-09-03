# Model web demos

Real models running fully client-side on
[LiteRT.js](https://www.npmjs.com/package/@litertjs/core) (`@litertjs/core`
2.5.3) — WebGPU when available, WASM otherwise. Nothing the user loads or
types leaves the page.

| Demo | Model | What it does |
| --- | --- | --- |
| [`moge/`](dist/moge/) | [MoGe-2-LiteRT](https://huggingface.co/litert-community/MoGe-2-LiteRT) | Photo → orbitable 3D point cloud (monocular geometry, three.js) |
| [`matcha-tts/`](dist/matcha-tts/) | [Matcha-TTS](https://huggingface.co/litert-community/Matcha-TTS) | Text → speech (flow-matching acoustic model + HiFi-GAN vocoder, 22 kHz) |

[`dist/index.html`](dist/) is the index page linking every demo.

## Layout

```
src/                 source — this is what you edit
  index.html         demos index
  moge/              index.html + main.js
  matcha-tts/        index.html + main.js, g2p.js, synth.js, viz.js
dist/                build output — this is what GitHub Pages serves (committed)
  index.html, moge/, matcha-tts/, assets/   built by `npm run build`
  litert-wasm/       LiteRT.js WASM runtime, copied from node_modules/@litertjs/core
  coi-serviceworker.min.js                  copied from node_modules/coi-serviceworker
tools/check.mjs      deploy-shaped end-to-end check (headless browser)
vite.config.js       one Vite project, three pages
```

## Build

```sh
cd samples/web_demos
npm ci
npm run dev      # http://localhost:5173/  (moge/, matcha-tts/)
npm run build    # rebuilds dist/ from src/ — commit dist/ together with src/
```

`dist/` is committed because this repo's GitHub Pages site is served
straight from `main`; there is no build step on deploy. Every file under
`dist/` is produced by `npm run build` from `src/` and `node_modules/` —
the WASM runtime and the service worker are copied from their npm packages
(see `RUNTIME_FILES` in `vite.config.js`), nothing is hand-edited.

### End-to-end check

```sh
npx playwright install chromium   # once
npm run check                     # both demos; add `moge` / `matcha` for one
npm run check -- matcha --block-hf   # what a user sees when huggingface.co is unreachable
```

The check serves `dist/` under `/litert-samples/samples/web_demos/dist/`
with no COOP/COEP headers (as GitHub Pages does), then requires the service
worker to turn on cross-origin isolation, the threaded WASM runtime to load,
the models to download, and one inference / one synthesis to complete.
Headless browsers have no usable WebGPU, so this exercises the WASM path.

## How it works

- **No weights in this repo.** Each page streams its model from
  [litert-community](https://huggingface.co/litert-community) on Hugging Face
  and caches it with the Cache API (one-time download):
  MoGe-2 **71 MB** (fp16, used on WebGPU) or 136 MB (fp32, used on WASM —
  XNNPACK declines the fp16 graph); Matcha-TTS ~92 MB (all fp16).
  `?models=<base url>` on either page loads the same file names from
  another location (a local copy, a mirror).
- **`litert-wasm/`** is the stock `@litertjs/core` WASM runtime (plain,
  threaded, and compat variants), shared by all demos. `loadLiteRt()` picks
  the variant; the pages ask for threads when the page is cross-origin
  isolated and fall back to the plain build otherwise.
- **`coi-serviceworker.min.js`**
  ([coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker) v0.1.7,
  MIT) injects COOP/COEP, which GitHub Pages cannot send, so the pages run
  cross-origin isolated and the WASM backend can use threads — the Matcha
  decoder is ~5–50× slower single-threaded. One automatic reload on the
  first visit. It is registered from `dist/` so its scope covers the shared
  `litert-wasm/` directory (thread workers are matched to a service worker
  by the worker script URL); no other page on the site is affected.
- **Boot errors name their stage** (`runtime` / `download` / `compile` /
  `warm-up`) and the URL that failed, so "Failed to start (download): could
  not fetch https://huggingface.co/…" means the browser could not reach
  Hugging Face, not that the page is broken.

### Debug URL parameters

- MoGe: `?img=<url>` runs on that image at boot; `?backend=wasm`.
- Matcha: `?text=…` speaks at boot; `&steps=4&seed=0&voc=wasm&enc=wasm`;
  `&threads=0` forces the single-thread runtime; `&nosound=1` synthesizes
  without playback and logs a `MATCHA_STATS` JSON line to the console.
