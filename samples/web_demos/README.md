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
| [`clipseg/`](dist/clipseg/) | [CLIPSeg-rd64-LiteRT](https://huggingface.co/litert-community/CLIPSeg-rd64-LiteRT) | Photo + typed words → a mask of what the words describe (open-vocabulary segmentation, CLIP tokenizer in JavaScript) |
| [`sam2/`](dist/sam2/) | SAM 2.1 Hiera-Tiny [image encoder](https://huggingface.co/litert-community/SAM2.1-Hiera-Tiny-Image-Encoder) + [mask decoder](https://huggingface.co/litert-community/SAM2.1-Hiera-Tiny-Mask-Decoder) | Photo + click → mask of the clicked object (encoder once per photo, decoder once per click) |
| [`rfdetr/`](dist/rfdetr/) | [RF-DETR-Nano-LiteRT](https://huggingface.co/litert-community/RF-DETR-Nano-LiteRT) | Photo or live camera → boxes and labels for the 80 COCO object classes (two-graph transformer detector) |

[`dist/index.html`](dist/) is the index page linking every demo.

The CLIPSeg example photo (`src/clipseg/example.jpg`) is
[“Lily the Golden Retriever in the grass”](https://commons.wikimedia.org/wiki/File:Lily_the_Golden_Retriever_in_the_grass.jpg)
by Ltshears (a Wikipedia user), released into the public domain by its author
(`{{PD-self}}` on its Wikimedia Commons file page).

The SAM 2.1 example photo (`src/sam2/example.jpg`) is
[“Young tabby cat keeping watch”](https://commons.wikimedia.org/wiki/File:Young_tabby_cat_keeping_watch.jpg)
by W.carter, CC0 1.0 (`{{self|cc-zero}}` on its Wikimedia Commons file page).

The RF-DETR example photo (`src/rfdetr/example.jpg`) is
[“The tram is coming, at the streetcorner of the Overtoom & 1e Constantijn Huygensstraat in Amsterdam city”](https://commons.wikimedia.org/wiki/File:The_tram_is_coming,_at_the_streetcorner_of_the_Overtoom_%26_1e_Constantijn_Huygensstraat_in_Amsterdam_city._Free_photo_by_Fons_Heijnsbroek,_Summer_2022.tif)
by Fons Heijnsbroek, CC0 1.0 (`{{self|cc-zero}}` on its Wikimedia Commons
file page).

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
  clipseg/           index.html + main.js, host.js (pre/post-processing), tokenizer.js (CLIP BPE), example.jpg
  sam2/              index.html + main.js, sam2.js, example.jpg
  rfdetr/            index.html + main.js, host.js (pre/post-processing), coco.js, example.jpg
dist/                build output — this is what GitHub Pages serves (committed)
  index.html, moge/, matcha-tts/, ppocr/, clipseg/, sam2/, rfdetr/, assets/   built by `npm run build`
  litert-wasm/       LiteRT.js WASM runtime, copied from node_modules/@litertjs/core
  coi-serviceworker.min.js                  copied from node_modules/coi-serviceworker
tools/check.mjs      deploy-shaped end-to-end check (headless browser)
vite.config.js       one Vite project, seven pages
```

## Build

```sh
cd samples/web_demos
npm ci
npm run dev      # http://localhost:5173/  (moge/, matcha-tts/, ppocr/, clipseg/, sam2/, rfdetr/)
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
npm run check                     # every demo; add `moge` / `matcha` / `ppocr` / `clipseg` / `sam2` / `rfdetr` for one
npm run check -- matcha --block-hf   # what a user sees when huggingface.co is unreachable
```

The check serves `dist/` under `/litert-samples/samples/web_demos/dist/`
with no COOP/COEP headers (as GitHub Pages does), then requires the service
worker to turn on cross-origin isolation, the threaded WASM runtime to load,
the models to download, and one inference / one synthesis / one OCR read
(of the PP-OCRv5 page's sample, `src/ppocr/sample.png`, which must come back
as its six lines, with the recognizer on threaded WASM) / two
segmentations / one click on the bundled SAM 2.1 example / one detection (on
the bundled example photo) to complete. The
sample is a menu card drawn on a canvas (Chromium on macOS, the system
sans-serif) and saved as a PNG; to replace it, set `OCR_LINES` in
`tools/check.mjs` to the lines the new image must read.
CLIPSeg segments the bundled example photo with
"a dog", then with "the grass" typed into the page; the check requires each
prompt's token ids, each mask's share of the output (15–25% and 45–55%; the
Python LiteRT reference gives 17% and 50%), and the second prompt to reuse
the photo's image features.
Headless Chromium launched without flags has no usable WebGPU, so this
exercises the WASM path; `--webgpu` runs the full Chromium build in its new
headless mode instead, where WebGPU works (`npm run check -- ppocr --webgpu`,
`npm run check -- rfdetr --webgpu`;
`npm run check -- clipseg --webgpu` also requires all three CLIPSeg graphs on
WebGPU).

## How it works

- **No weights in this repo.** Each page streams its model from
  [litert-community](https://huggingface.co/litert-community) on Hugging Face
  and caches it with the Cache API (one-time download):
  MoGe-2 **71 MB** (fp16, used on WebGPU) or 136 MB (fp32, used on WASM —
  XNNPACK declines the fp16 graph); Matcha-TTS ~92 MB (all fp16);
  PP-OCRv5 43 MB (fp16 detector; fp32 recognizer, which runs on WASM
  because the WebGPU delegate mis-decodes it —
  [LiteRT #9661](https://github.com/google-ai-edge/LiteRT/issues/9661) —
  and XNNPACK declines the fp16 recognizer graph);
  CLIPSeg ~278 MB (fp16 image and text encoders, the fp32 decoder, a
  float16 token-embedding table and the tokenizer files);
  SAM 2.1 **97 MB** (fp16 image encoder 80 MB + mask decoder 17 MB). On an
  M4 Max, WebGPU (fp16 compute) takes ~60 ms per photo and ~4 ms per click;
  the WASM fallback runs the same files at ~13 s per photo and 0.4 s per
  click with 16 threads.
  RF-DETR-Nano 56 MB (two fp16 graphs).
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
- **CLIPSeg runs three graphs and a tokenizer.** The CLIP BPE tokenizer is
  ported to JavaScript (`tokenizer.js`: the same token ids as the Hugging Face
  `CLIPTokenizer` on 670 test strings), and the resize is Pillow's bilinear,
  so on the example the image encoder gets exactly the pixels of the Hugging
  Face image processor. The image encoder runs once per photo; new words on
  the same photo rerun only the text encoder and the decoder. All three
  graphs run on WebGPU and match a Python LiteRT CPU run of the same files on
  the example: mask IoU 0.9996 for "a dog" and 0.9998 for "the grass" (on
  WASM the masks are identical). The two encoders compute in fp16. The
  decoder computes in fp32: with fp16 compute on WebGPU, its logits for
  "a dog" on the example were all 0. Without WebGPU every graph runs on WASM
  (XNNPACK does not fully take the fp16 encoders).
  In headless Chromium 151 on an M4 Max with WebGPU enabled by flag (median
  of 10 runs on the example), the image encoder takes 18 ms (25 ms with fp32
  compute), the text encoder 8 ms and the decoder 2 ms, so new words take
  about 10 ms. With every graph on WASM, the image encoder takes 4.1 s and
  the text encoder 0.37 s (median of 6).
- **RF-DETR runs as two graphs with a host step between them**: graph A
  (backbone + encoder) on WebGPU, then the top-300 proposals picked in
  JavaScript, then graph B (decoder) on WASM. On five photos, with the same
  input pixels, graph A on WebGPU matches the Python LiteRT reference (same
  boxes); graph B on WebGPU does not (logits max |Δ| 6.4–8.5; extra boxes,
  moved boxes or scores off by more than 0.05 on every photo), so the
  decoder stays on WASM. The resize is Pillow's bilinear ported to
  JavaScript, so the page feeds the model exactly the pixels
  `Image.resize((384, 384), Image.BILINEAR)` gives in Python.
- **Boot errors name their stage** (`runtime` / `download` / `compile` /
  `warm-up`) and the URL that failed, so "Failed to start (download): could
  not fetch https://huggingface.co/…" means the browser could not reach
  Hugging Face, not that the page is broken.

### Debug URL parameters

- MoGe: `?img=<url>` runs on that image at boot; `?backend=wasm`.
- PP-OCRv5: `?img=<url>` (or `?img=sample`) reads that image at boot;
  `?backend=wasm` puts the detector on WASM too. With or without them, every
  read logs one `OCR_STATS` JSON line to the console (what the check reads).
- CLIPSeg: `?img=<url>` (or `?img=example`) segments at boot with
  `&prompt=<words>` (default "a dog"); every run logs a `CLIPSEG_STATS` JSON
  line to the console. `?backend=wasm` puts every graph on WASM;
  `?precision=fp32` runs the two encoders in fp32 on WebGPU; `?raw=1` keeps
  the model input, the image features and the logits in
  `window.__lastResult`.
- RF-DETR: `?img=<url>` (or `?img=example`) detects at boot and `&repeat=N`
  runs it N times; every photo run logs an `RFDETR_STATS` JSON line to the
  console, and the camera logs one line per second. `?backend=wasm` puts
  graph A on WASM. `?a=webgpu|wasm&b=webgpu|wasm` sets each graph's backend,
  to reproduce the comparison above. `&raw=1` keeps the raw model outputs in
  `window.__lastRaw`.
- Matcha: `?text=…` speaks at boot; `&steps=4&seed=0&voc=wasm&enc=wasm`;
  `&threads=0` forces the single-thread runtime; `&nosound=1` synthesizes
  without playback and logs a `MATCHA_STATS` JSON line to the console.
- SAM 2.1: `?img=<url>` (or `?img=example` for the bundled photo) encodes
  that photo at boot and `&point=x,y` clicks it, with x, y as fractions of
  the photo's width and height (0..1); every click logs a `SAM2_STATS` JSON
  line. `?backend=wasm` runs both graphs on WASM; `?precision=fp32` makes
  WebGPU compute in fp32. With `?img=` or `?debug=1` the last click's
  logits and IoU scores stay in `window.__lastResult`. `?debug=1` also takes
  `&bench=N`, `&enc=` / `&dec=` and `&threads=0`, kept to reproduce the
  speed numbers above.
