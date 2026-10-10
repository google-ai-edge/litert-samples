# Model web demos

Real models running fully client-side on
[LiteRT.js](https://www.npmjs.com/package/@litertjs/core) (`@litertjs/core`
2.5.3) — WebGPU when available, WASM otherwise. Nothing the user loads or
types leaves the page.

| Demo | Model | What it does |
| --- | --- | --- |
| [`moge/`](dist/moge/) | [MoGe-2-LiteRT](https://huggingface.co/litert-community/MoGe-2-LiteRT) | Photo → orbitable 3D point cloud (monocular geometry, three.js) |
| [`matcha-tts/`](dist/matcha-tts/) | [Matcha-TTS](https://huggingface.co/litert-community/Matcha-TTS) | Text → speech (flow-matching acoustic model + HiFi-GAN vocoder, 22 kHz) |
| [`clipseg/`](dist/clipseg/) | [CLIPSeg-rd64-LiteRT](https://huggingface.co/litert-community/CLIPSeg-rd64-LiteRT) | Photo + typed words → a mask of what the words describe (open-vocabulary segmentation, CLIP tokenizer in JavaScript) |

[`dist/index.html`](dist/) is the index page linking every demo.

The CLIPSeg example photo (`src/clipseg/example.jpg`) is
[“Lily the Golden Retriever in the grass”](https://commons.wikimedia.org/wiki/File:Lily_the_Golden_Retriever_in_the_grass.jpg)
by Ltshears (a Wikipedia user), released into the public domain by its author
(`{{PD-self}}` on its Wikimedia Commons file page).

## Chrome extension

[`chrome_extension/`](chrome_extension/) runs two of the same models
(MoGe-2, Matcha-TTS) plus PP-OCRv5 inside a Chrome extension, on the pages
you browse. It reads selected text aloud, shows photos in 3D on hover, and
finds text inside images. It is a separate npm project with its own build and
tests, not part of the Vite build or the GitHub Pages site.

## Layout

```
src/                 source — this is what you edit
  index.html         demos index
  moge/              index.html + main.js
  matcha-tts/        index.html + main.js, g2p.js, synth.js, viz.js
  clipseg/           index.html + main.js, host.js (pre/post-processing), tokenizer.js (CLIP BPE), example.jpg
dist/                build output — this is what GitHub Pages serves (committed)
  index.html, moge/, matcha-tts/, clipseg/, assets/   built by `npm run build`
  litert-wasm/       LiteRT.js WASM runtime, copied from node_modules/@litertjs/core
  coi-serviceworker.min.js                  copied from node_modules/coi-serviceworker
tools/check.mjs      deploy-shaped end-to-end check (headless browser)
vite.config.js       one Vite project, four pages
```

## Build

```sh
cd samples/web_demos
npm ci
npm run dev      # http://localhost:5173/  (moge/, matcha-tts/, clipseg/)
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
npm run check                     # every demo; add `moge` / `matcha` / `clipseg` for one
npm run check -- matcha --block-hf   # what a user sees when huggingface.co is unreachable
```

The check serves `dist/` under `/litert-samples/samples/web_demos/dist/`
with no COOP/COEP headers (as GitHub Pages does), then requires the service
worker to turn on cross-origin isolation, the threaded WASM runtime to load,
the models to download, and one inference / one synthesis / two
segmentations to complete. CLIPSeg segments the bundled example photo with
"a dog", then with "the grass" typed into the page; the check requires each
prompt's token ids, each mask's share of the output (15–25% and 45–55%; the
Python LiteRT reference gives 17% and 50%), and the second prompt to reuse
the photo's image features.
Headless Chromium launched without flags has no usable WebGPU, so this
exercises the WASM path; `--webgpu` runs the full Chromium build in its new
headless mode instead, where WebGPU works (`npm run check -- clipseg
--webgpu` also requires all three CLIPSeg graphs on WebGPU).

## How it works

- **No weights in this repo.** Each page streams its model from
  [litert-community](https://huggingface.co/litert-community) on Hugging Face
  and caches it with the Cache API (one-time download):
  MoGe-2 **71 MB** (fp16, used on WebGPU) or 136 MB (fp32, used on WASM —
  XNNPACK declines the fp16 graph); Matcha-TTS ~92 MB (all fp16);
  CLIPSeg ~278 MB (fp16 image and text encoders, the fp32 decoder, a
  float16 token-embedding table and the tokenizer files).
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
- **Boot errors name their stage** (`runtime` / `download` / `compile` /
  `warm-up`) and the URL that failed, so "Failed to start (download): could
  not fetch https://huggingface.co/…" means the browser could not reach
  Hugging Face, not that the page is broken.

### Debug URL parameters

- MoGe: `?img=<url>` runs on that image at boot; `?backend=wasm`.
- CLIPSeg: `?img=<url>` (or `?img=example`) segments at boot with
  `&prompt=<words>` (default "a dog"); every run logs a `CLIPSEG_STATS` JSON
  line to the console. `?backend=wasm` puts every graph on WASM;
  `?precision=fp32` runs the two encoders in fp32 on WebGPU; `?raw=1` keeps
  the model input, the image features and the logits in
  `window.__lastResult`.
- Matcha: `?text=…` speaks at boot; `&steps=4&seed=0&voc=wasm&enc=wasm`;
  `&threads=0` forces the single-thread runtime; `&nosound=1` synthesizes
  without playback and logs a `MATCHA_STATS` JSON line to the console.
