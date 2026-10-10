# U²-Net Portrait — Photo → pencil line drawing (LiteRT CompiledModel GPU)

**Portrait sketch generation** running **fully on the LiteRT `CompiledModel` GPU** delegate.
The [U²-Net](https://github.com/xuebinqin/U-2-Net) portrait model turns a face photo into a
**hand-drawn pencil line portrait** — a fun creative / AR filter. ~12 ms/frame on a Pixel 8a.

- **Model:** [litert-community/U2Net-Portrait-Sketch-LiteRT](https://huggingface.co/litert-community/U2Net-Portrait-Sketch-LiteRT)
- **Weights:** [xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net) (`u2net_portrait`) · Apache-2.0
- **Input:** `[1, 3, 512, 512]` NCHW, RGB, `x/255` then ImageNet-normalized
- **Output:** `[1, 1, 512, 512]` in `[0,1]` → min-max normalize, invert (`1−x`)

## How it works

U²-Net is a pure CNN → fully GPU-compatible (**893/893 nodes on the delegate, 1 partition**;
device corr 0.998683, ~12 ms) with one defensive patch: `align_corners=False`. CPU-exact vs
PyTorch (corr 1.0). Decode: min-max normalize the output, then invert for dark strokes on white.

## App architecture

The Android app follows **MVVM + Jetpack Compose**:

- **`MainActivity`** is a thin Compose host — it collects `UiState` and wires the gallery picker;
  it holds no inference logic.
- **`MainViewModel`** owns the `PortraitSketcher` helper, loads the model from `filesDir`, runs
  the bundled demo image at launch, and processes gallery picks. All model creation and inference
  are **confined to a single worker** (`Dispatchers.Default.limitedParallelism(1)`) because the
  176 MB model reuses native buffers; the helper is closed in `onCleared()`.
- **`PortraitSketcher`** is the LiteRT `CompiledModel` (GPU) wrapper. `sketch()` returns the
  finished pencil-portrait `Bitmap` (the min-max-normalize + invert decode lives here).
- **`SketchScreen`** renders a status header, a "Pick image" button, and the result `Image`.

| File | Role |
| --- | --- |
| `MainActivity.kt` | Thin Compose host: `viewModels`, gallery picker, `SketchScreen` |
| `MainViewModel.kt` | Owns the helper + `UiState`; confines inference to one worker |
| `UiState.kt` | Immutable screen state (`resultImage`, flags, inference time) |
| `PortraitSketcher.kt` | LiteRT `CompiledModel` GPU wrapper; `sketch()` → result `Bitmap` |
| `ImageUtils.kt` | Asset/gallery bitmap decoding (EXIF-oriented), resize helpers |
| `view/SketchScreen.kt` | Compose screen: status header, picker button, result image |
| `view/Theme.kt`, `view/Color.kt` | Compose Material theme + palette |

## Run

```bash
cd android
./install_to_device.sh <dir-with-portrait.tflite>
./gradlew :app:installDebug
```

The sample renders a bundled face photo as a pencil portrait at launch and lets you pick any
gallery image. Adapt `MainViewModel`/`PortraitSketcher` to feed camera frames for a live sketch
filter (center the face).

## Convert

See [`conversion/`](conversion/) — `build_portrait.py` loads the Apache-2.0 weights and converts
with litert-torch.
