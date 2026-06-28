# Line Segment Detection with LiteRT — M-LSD-tiny (on-device, fully-GPU)

An Android sample that runs **M-LSD-tiny** ([navervision/mlsd](https://github.com/navervision/mlsd), AAAI
2022, Apache-2.0) end-to-end on device with the LiteRT `CompiledModel` API. M-LSD detects straight **line
segments** — building edges, document borders, wireframes, room layout — in real time. The tiny variant
(MobileNetV2 backbone, 0.62M params) is **1.4 MB** in fp16. The app detects lines in a bundled image and any
image picked from the gallery, drawing the segments.

## Model

| Graph | In → Out | Delegate (Pixel 8a) |
| :-- | :-- | :--: |
| M-LSD-tiny | image[1,4,512,512] → tpMap[1,9,256,256] | **GPU** |

fp16, converted with [litert-torch](https://github.com/google-ai-edge/litert) — tflite-vs-torch corr **1.0**,
device-vs-torch corr **0.997** (127 vs 128 lines decoded). On a Pixel 8a (Tensor G3): **99 / 99** nodes on
`LITERT_CL` (full GPU residency), **~2 ms**, **1.4 MB** fp16. The TP-map decode runs in the app.

## Re-authoring (litert-torch, single fix)

Pure CNN encoder-decoder. The only re-authoring: the decoder's `F.interpolate(mode='bilinear',
align_corners=True)` → **`align_corners=False`** (the Mali delegate bans `align_corners=True` + half-pixel).
MobileNetV2 has no max-pool (strided convs → no `PADV2`), and the upsample is `RESIZE_BILINEAR`, not a
transposed conv → fully GPU-clean. See [`conversion/`](conversion/).

## Output & decode

The output is a "TP map": channel 0 = line-center heatmap, channels 1–4 = start/end displacement. The host
decodes it: sigmoid the center map, 3×3 max NMS, threshold (0.10), displacement → endpoints, filter by length,
×2 to the 512 input space. See `MlsdDetector.kt`.

## Run

1. Build the tflite with `conversion/build_mlsd.py`, or get it from
   [litert-community/M-LSD-tiny-LiteRT](https://huggingface.co/litert-community/M-LSD-tiny-LiteRT).
2. Build/install the app and push the model:
   ```bash
   cd kotlin_cpu_gpu/android
   ./gradlew :app:installDebug
   ./install_to_device.sh <dir-with-mlsd_fp16.tflite>
   ```
3. Launch. (The first launch fails with "Model not found" until the model is pushed.)

**Preprocessing**: resize to 512×512, append a 4th channel of ones, scale `(x/127.5)-1`, NCHW.

Upstream: [navervision/mlsd](https://github.com/navervision/mlsd) (Apache-2.0); PyTorch port
[lhwcv/mlsd_pytorch](https://github.com/lhwcv/mlsd_pytorch).
