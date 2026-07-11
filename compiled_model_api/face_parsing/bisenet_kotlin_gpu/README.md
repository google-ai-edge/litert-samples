# BiSeNet — Face parsing (LiteRT CompiledModel GPU)

Real-time **face parsing** running **fully on the LiteRT `CompiledModel` GPU** delegate. [BiSeNet](https://arxiv.org/abs/1808.00897) ([zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)) segments a face into the **19 CelebAMask-HQ classes** (skin, brows, eyes, nose, lips, ears, hair, hat, glasses, neck, cloth, …) — for AR / beauty / makeup. ~22 ms/frame on a Pixel 8a.

- **Model:** [litert-community/BiSeNet-Face-Parsing-LiteRT](https://huggingface.co/litert-community/BiSeNet-Face-Parsing-LiteRT) · 53 MB
- **Weights:** [zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) · MIT
- **Input:** `[1, 3, 512, 512]` NCHW, RGB, ImageNet-normalized
- **Output:** `[1, 19, 512, 512]` class logits (argmax → face-part map)

## How it works

BiSeNet is a pure CNN (ResNet18 backbone + context path + feature fusion). Three re-authoring patches make it a fully GPU-compatible graph (**74/74 nodes on the delegate, 1 partition**; device corr 0.99999, argmax 99.96% vs PyTorch):

1. **`align_corners=True` → `False`** — the output upsamples use `align_corners=True`, which the GPU delegate rejects.
2. **global `avg_pool2d(x, x.size()[2:])` → `mean([2,3])`** — full-spatial-kernel pooling is rejected by the Mali delegate as `AVERAGE_POOL_2D`; a MEAN reduce is supported.
3. **zero-pad maxpool** — the ResNet stem `MaxPool2d(padding=1)` lowers to a `-inf` PADV2 (`PADV2: src has wrong size` on Mali); an explicit 0-pad + unpadded maxpool is exact (post-ReLU input ≥ 0).

These are on-device-only rejections (op inventory clean, CPU parity 1.0).

## App architecture

The Android app is **MVVM + Jetpack Compose** (Compose Material). `MainActivity` is a thin
Compose host: it collects a single `UiState` and wires the gallery picker. `MainViewModel`
owns the `FaceParser` and runs both model creation and every inference on one confined worker
(`Dispatchers.Default.limitedParallelism(1)`) so the reused native buffers are never touched
concurrently. On launch it parses a bundled face photo; the picker re-runs on a gallery image.

| File | Role |
| --- | --- |
| `MainActivity.kt` | Thin `ComponentActivity` host: gallery picker + `collectAsStateWithLifecycle` into Compose |
| `MainViewModel.kt` | Owns `FaceParser`; loads the model from `filesDir`, runs inference, blends the overlay, exposes `UiState` |
| `UiState.kt` | Immutable UI snapshot (result bitmap, model-ready / processing flags, inference time, error) |
| `FaceParser.kt` | LiteRT `CompiledModel` (GPU) wrapper: preprocess → run → argmax → CelebAMask label bitmap |
| `CelebAMaskPalette.kt` | 19 CelebAMask-HQ class names + overlay colors |
| `ImageUtils.kt` | Bitmap helpers (asset decode, EXIF-oriented gallery load) |
| `view/FaceParsingScreen.kt` | Compose screen: status header, image picker, blended result |
| `view/Theme.kt`, `view/Color.kt` | Compose Material theme |

## Run

```bash
# 1. Get the model (build with ../conversion or download from Hugging Face)
cd android
./install_to_device.sh <dir-with-faceparsing.tflite>

# 2. Build & run
./gradlew :app:installDebug
```

The sample parses a bundled face photo (and any image you pick from the gallery) and overlays
the 19-class CelebAMask parsing. Adapt it to feed live front-camera frames for a real-time AR demo.

## Convert

See [`conversion/`](conversion/) — `build_faceparsing.py` loads the trained BiSeNet weights and converts with litert-torch.
