# Model web demos

Real models running fully client-side on
[LiteRT.js](https://www.npmjs.com/package/@litertjs/core) (`@litertjs/core`
2.5.3) — WebGPU when available, WASM otherwise. Nothing the user loads or
types leaves the page.

| Demo | Model | What it does |
| --- | --- | --- |
| [`moge/`](dist/moge/) | [MoGe-2-LiteRT](https://huggingface.co/litert-community/MoGe-2-LiteRT) | Photo → orbitable 3D point cloud (monocular geometry, three.js) |
| [`matcha-tts/`](dist/matcha-tts/) | [Matcha-TTS](https://huggingface.co/litert-community/Matcha-TTS) | Text → speech (flow-matching acoustic model + HiFi-GAN vocoder, 22 kHz) |
| [`ppocr/`](dist/ppocr/) | [PP-OCRv5-LiteRT](https://huggingface.co/litert-community/PP-OCRv5-LiteRT) | Photo or screenshot → selectable lines of text (text detection + recognition) |

[`dist/index.html`](dist/) is the index page linking every demo.

## Chrome extension

[`chrome_extension/`](chrome_extension/) runs three of the same models
(MoGe-2, Matcha-TTS and PP-OCRv5) inside a Chrome extension, on the pages
you browse. It reads selected text aloud, shows photos in 3D on hover, and
finds text inside images. It is a separate npm project with its own build and
tests, not part of the Vite build or the GitHub Pages site — except
`src-ocr/ocr-pipeline.js` (no imports): the `ppocr/` page imports the OCR
pipeline and its recognition loop from it instead of keeping a copy, so the
page and the extension read text the same way. After a change to that file,
rebuild `dist/` and run `npm run check -- ppocr`.

## Layout

```
src/                 source — this is what you edit
  index.html         demos index
  moge/              index.html + main.js
  matcha-tts/        index.html + main.js, g2p.js, synth.js, viz.js
  ppocr/             index.html + main.js, sample.png (pipeline: chrome_extension/src-ocr/ocr-pipeline.js)
dist/                build output — this is what GitHub Pages serves (committed)
  index.html, moge/, matcha-tts/, ppocr/, assets/   built by `npm run build`
  litert-wasm/       LiteRT.js WASM runtime, copied from node_modules/@litertjs/core
  coi-serviceworker.min.js                  copied from node_modules/coi-serviceworker
tools/check.mjs      deploy-shaped end-to-end check (headless browser)
vite.config.js       one Vite project, four pages
```

## Build

```sh
cd samples/web_demos
npm ci
npm run dev      # http://localhost:5173/  (moge/, matcha-tts/, ppocr/)
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
npm run check                     # every demo; add `moge` / `matcha` / `ppocr` for one
npm run check -- matcha --block-hf   # what a user sees when huggingface.co is unreachable
```

The check serves `dist/` under `/litert-samples/samples/web_demos/dist/`
with no COOP/COEP headers (as GitHub Pages does), then requires the service
worker to turn on cross-origin isolation, the threaded WASM runtime to load,
the models to download, and one inference / one synthesis / one OCR read
(of the PP-OCRv5 page's sample, `src/ppocr/sample.png`, which must come back
as its six lines, with the recognizer on threaded WASM) to complete. The
sample is a menu card drawn on a canvas (Chromium on macOS, the system
sans-serif) and saved as a PNG; to replace it, set `OCR_LINES` in
`tools/check.mjs` to the lines the new image must read.
Headless Chromium launched without flags has no usable WebGPU, so this
exercises the WASM path; `--webgpu` runs the full Chromium build in its new
headless mode instead, where WebGPU works (`npm run check -- ppocr --webgpu`).

## How it works

- **No weights in this repo.** Each page streams its model from
  [litert-community](https://huggingface.co/litert-community) on Hugging Face
  and caches it with the Cache API (one-time download):
  MoGe-2 **71 MB** (fp16, used on WebGPU) or 136 MB (fp32, used on WASM —
  XNNPACK declines the fp16 graph); Matcha-TTS ~92 MB (all fp16);
  PP-OCRv5 43 MB (fp16 detector; fp32 recognizer, which runs on WASM
  because the WebGPU delegate mis-decodes it —
  [LiteRT #9661](https://github.com/google-ai-edge/LiteRT/issues/9661) —
  and XNNPACK declines the fp16 recognizer graph).
  `?models=<base url>` on any page loads the same file names from
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
- PP-OCRv5: `?img=<url>` (or `?img=sample`) reads that image at boot;
  `?backend=wasm` puts the detector on WASM too. With or without them, every
  read logs one `OCR_STATS` JSON line to the console (what the check reads).
- Matcha: `?text=…` speaks at boot; `&steps=4&seed=0&voc=wasm&enc=wasm`;
  `&threads=0` forces the single-thread runtime; `&nosound=1` synthesizes
  without playback and logs a `MATCHA_STATS` JSON line to the console.
