# Page Voice (Chrome extension)

A Manifest V3 Chrome extension that runs LiteRT models on the pages you
browse, with [LiteRT.js](https://www.npmjs.com/package/@litertjs/core)
(`@litertjs/core` 2.5.3). It builds as three extensions, one effect each:

| Build | Effect | Model |
| --- | --- | --- |
| `dist/` Page Voice | Reads selected text aloud (Alt+R or the context menu); *Auto-read* speaks ChatGPT, Claude and Gemini replies as they stream | [Matcha-TTS](https://huggingface.co/litert-community/Matcha-TTS) |
| `dist3d/` Page 3D | Hover a photo to see it in 3D; hold Shift for a cursor light | [MoGe-2-LiteRT](https://huggingface.co/litert-community/MoGe-2-LiteRT) |
| `dist-ocr/` Page Text | Right-click an image to select its text; Alt+Shift+F searches the text inside images | [PP-OCRv5-LiteRT](https://huggingface.co/litert-community/PP-OCRv5-LiteRT) |

Inference runs in the extension's offscreen document; no text or image is
uploaded anywhere.

## Build and load

```sh
cd samples/web_demos/page-voice
npm ci
npm run build && node build.mjs --3d && node build.mjs --ocr   # dist/ dist3d/ dist-ocr/
```

In `chrome://extensions`, turn on Developer mode, click *Load unpacked* and
pick a `dist*/` folder. Builds are not committed: `build.mjs` bundles
`src*/` with esbuild, copies `public*/`, and copies the plain and threaded
WASM runtimes from `node_modules/@litertjs/core`.

## Where it runs

- **Page Voice:** text encoder and HiFi-GAN vocoder on WebGPU; G2P and the
  flow-matching decoder on threaded WASM, as in the Matcha-TTS web demo
  (same `g2p.js` and `synth.js`): the decoder's WebGPU output does not
  match WASM.
- **Page 3D:** MoGe-2 on WebGPU.
- **Page Text:** detector on WebGPU; recognizer on WASM with fp32 weights.
  On WebGPU the recognizer decodes real text crops to different strings
  (deterministic, fp16 and fp32 weights alike); on WASM the fp16 recognizer
  is ~20× slower than fp32 (456 vs 20 ms per line).

Every WebGPU part falls back to WASM when WebGPU is missing. On an Apple M4
Max (Chrome for Testing 155): MoGe-2 65–74 ms per image on WebGPU, ~0.6 s
on WASM; the OCR detector 8–16 ms, ~0.9 s on WASM; a sentence takes 0.4–0.65×
its audio length to synthesize.

## Models

No weights in this repo. Each engine streams its files from
[litert-community](https://huggingface.co/litert-community) on Hugging Face
and keeps them in the Cache API: Matcha-TTS **94 MB** (four fp16 models, G2P
dictionary, embeddings), MoGe-2 **136 MB** (`moge.tflite`, fp32), PP-OCRv5
**43 MB** (fp16 detector, fp32 recognizer, dictionary). These builds start
the engine, and so the download, as soon as the extension loads.

## Tests

The tests drive Chrome for Testing over the DevTools protocol: they need a
Chromium build that honors `--load-extension`, which branded Chrome ignores
since 137.

```sh
npx @puppeteer/browsers install chrome@stable   # prints the path to pass as <chrome>
node tools/smoke.mjs <chrome> --speak --profile=<dir>
node tools/stream-smoke.mjs <chrome> --profile=<dir>
node tools3d/smoke.mjs <chrome> --profile=<dir>
node tools3d/e2e.mjs <chrome> --profile=<dir>
node tools-ocr/verify.mjs <chrome>
node tools-ocr/make-fixtures.mjs <chrome>
node tools-ocr/smoke.mjs <chrome> --profile=<dir>
node tools-ocr/e2e.mjs <chrome> --profile=<dir>
node tools-ocr/find-e2e.mjs <chrome> --profile=<dir>
```

Reuse one `--profile` directory per build to keep the models. `smoke` boots
an engine and runs it; `stream-smoke` streams a reply into a ChatGPT-shaped
local page with Auto-read on; the `e2e` tests drive the page (hover, the
context-menu message, find in images). Run `verify.mjs` first: it compares the
OCR recognizer on WebGPU and WASM and saves mock screenshots that the OCR
smoke test reads.

## Known limitations

- `tools-ocr/verify.mjs` exits 1 while WebGPU and WASM decode text
  differently, as they do with LiteRT.js 2.5.3 on Chrome 155.
- Without WebGPU, Page Voice takes ~7× a sentence's length to synthesize
  it (25 s for 3.4 s of audio), and its engine is unresponsive meanwhile.
- A failed boot names its stage in the popup, e.g. `Failed to start
  (download): could not fetch https://huggingface.co/…`, and the engine
  stays in that state until the extension is reloaded. Page 3D shows no
  error on the page.
- Auto-read finds replies with DOM selectors (`ADAPTERS` in
  `src/content.js`); the claude.ai and gemini.google.com ones are best-effort.
- With Developer mode on, `chrome://extensions` lists LiteRT.js log lines
  under Errors; the runtime prints INFO lines with `console.error`.
