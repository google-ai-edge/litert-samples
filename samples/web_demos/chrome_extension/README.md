# LiteRT.js in a Chrome extension

A Manifest V3 Chrome extension that runs
[LiteRT](https://github.com/google-ai-edge/litert) models on the pages you
browse, with [LiteRT.js](https://www.npmjs.com/package/@litertjs/core)
(`@litertjs/core` 2.5.3). The project builds three extensions, one effect
each:

| Build | Effect | Model |
| --- | --- | --- |
| `dist/` Page Voice | Reads selected text aloud (Alt+R, the context menu or the popup) | [Matcha-TTS](https://huggingface.co/litert-community/Matcha-TTS) |
| `dist3d/` Page 3D | Hover a photo to see it in 3D; hold Shift for a cursor light | [MoGe-2-LiteRT](https://huggingface.co/litert-community/MoGe-2-LiteRT) |
| `dist-ocr/` Page Text | Right-click an image to select its text; Alt+Shift+F searches the text inside images | [PP-OCRv5-LiteRT](https://huggingface.co/litert-community/PP-OCRv5-LiteRT) |

Inference runs in the extension's offscreen document. No text or image is
uploaded anywhere. This is a desktop Chrome sample, tested with Chrome for
Testing 155 on macOS.

## Build and load

```sh
cd samples/web_demos/chrome_extension
npm ci
npm run build && node build.mjs --3d && node build.mjs --ocr   # dist/ dist3d/ dist-ocr/
```

`build.mjs` bundles `src*/` with esbuild and copies `public*/`. It also
copies the plain and threaded WASM runtimes from `node_modules/@litertjs/core`.
Builds are not committed.

In `chrome://extensions`, turn on Developer mode, click *Load unpacked* and
pick a `dist*/` folder. With Developer mode on, `chrome://extensions` → Errors
also lists the runtime's INFO/WARNING log lines.

Permissions: Page Voice reads the selection through `activeTab` and has no
host permission or content script; Page 3D and Page Text run a content script
on every http(s) page and use `<all_urls>` to fetch the images they read.

## Backends

- **Page Voice:** text encoder and HiFi-GAN vocoder on WebGPU. G2P and the
  flow-matching decoder run on threaded WASM, as in the Matcha-TTS web demo
  (same `g2p.js` and `synth.js`; backend comparison in
  [LiteRT #9662](https://github.com/google-ai-edge/LiteRT/issues/9662)).
- **Page 3D:** MoGe-2 on WebGPU.
- **Page Text:** detector on WebGPU; recognizer on WASM with fp32 weights
  (WebGPU comparison in
  [LiteRT #9661](https://github.com/google-ai-edge/LiteRT/issues/9661); on
  WASM the fp16 recognizer is ~20× slower, 430–460 vs 20 ms per line).

Every WebGPU part falls back to WASM when WebGPU is missing. On an Apple M4
Max (Chrome for Testing 155), MoGe-2 takes 64–74 ms per image on WebGPU and
~0.6 s on WASM. The OCR detector takes 8–16 ms, or ~0.9 s on WASM. A sentence
takes 0.4–0.7× its audio length to synthesize, or ~7× on WASM only (25 s for
3.4 s of audio). In that case the voice engine answers no messages until the
sentence is done.

## Models

No weights in this repo. Each engine streams its files from
[litert-community](https://huggingface.co/litert-community) on Hugging Face
and keeps them in the Cache API: Matcha-TTS **94 MB** (four fp16 models, G2P
dictionary, embeddings), MoGe-2 **136 MB** (`moge.tflite`, fp32), PP-OCRv5
**43 MB** (fp16 detector, fp32 recognizer, dictionary).

`build.mjs` sets `__DEV__`, so each engine, and its download, starts as soon
as the extension loads; with `__DEV__` false, the engine starts on first use.

Boot errors name their stage (`runtime`, `download` or `compile`), for
example `Failed to start (download): could not fetch <model file>`. Reload
the extension to retry. Page Text also shows the error on the image; Page
Voice and Page 3D show it in the popup only.

## Tests

The tests drive Chrome for Testing over the DevTools protocol. They load the
builds with `--load-extension`, which branded Chrome has ignored since version
137. `npx @puppeteer/browsers install chrome@stable` installs Chrome for
Testing and prints the path to pass as `<chrome>`.

Fixtures (run once): `node tools-ocr/make-fixtures.mjs <chrome>` draws the OCR test images.

Tests, in this order:

```sh
node tools/smoke.mjs <chrome> --speak --profile=<dir>
node tools3d/smoke.mjs <chrome> --profile=<dir>
node tools3d/e2e.mjs <chrome> --profile=<dir>
node tools-ocr/smoke.mjs <chrome> --profile=<dir>
node tools-ocr/e2e.mjs <chrome> --profile=<dir>
node tools-ocr/find-e2e.mjs <chrome> --profile=<dir>
```

- Each `smoke` boots one engine and runs it on test input.
- `tools3d/e2e` hovers a generated photo in a local gallery and waits for the 3D overlay.
- `tools-ocr/e2e` sends the context-menu message for an image and checks the selectable text.
- `tools-ocr/find-e2e` indexes a page of screenshots and finds text that is only inside them.
- Reuse one `--profile` directory per build to keep its models.
- After changing a `background.js`, use a new `--profile`: a reused profile keeps running the old service worker.

Tools: `node tools-ocr/verify.mjs <chrome>` compares the OCR recognizer's text on WebGPU and WASM.
