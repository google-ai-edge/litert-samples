# Cloth Segmentation (U²-Net) — LiteRT CompiledModel GPU

Real-time **clothing segmentation** running **fully on the LiteRT `CompiledModel` GPU**
delegate. [cloth-segmentation](https://github.com/levindabhi/cloth-segmentation) is a
U²-Net trained on iMaterialist-Fashion to segment **upper-body / lower-body / full-body
clothing** — the building block for virtual try-on and fashion apps. ~88 ms/frame on a
Pixel 8a.

- **Model:** [litert-community/Cloth-Segmentation-U2Net-LiteRT](https://huggingface.co/litert-community/Cloth-Segmentation-U2Net-LiteRT)
- **Weights:** [levindabhi/cloth-segmentation](https://github.com/levindabhi/cloth-segmentation) · MIT · U²-Net
- **Input:** `[1, 3, 768, 768]` NCHW, RGB, `(x/255 - 0.5)/0.5`
- **Output:** `[1, 4, 768, 768]` logits — argmax: 0 bg, 1 upper, 2 lower, 3 full body

## How it works

U²-Net is a pure CNN → fully GPU-compatible (**254/254 nodes on the delegate, 1
partition**; device corr 0.999798, ~88 ms) with one defensive patch: `align_corners=False`.
CPU-exact vs PyTorch (corr 1.0). Decode: `argmax` over the 4 classes.

## Run

```bash
# 1. Get the model (build with ../conversion or download from Hugging Face)
cd android
./install_to_device.sh <dir-with-clothseg.tflite>

# 2. Build & run
./gradlew :app:installDebug
```

The sample overlays the clothing classes on a bundled photo (upper = cyan, lower =
orange, full body = magenta) at launch, and a **Pick image** button runs the same
segmentation on any gallery photo.

## App architecture

The Android app is MVVM + Jetpack Compose (Material, `compose-bom`, version catalog).
`ClothSegmenter` runs U²-Net on the LiteRT `CompiledModel` GPU delegate and returns an
`OUT × OUT` class map; `MainViewModel` owns the segmenter on a single confined worker
(`Dispatchers.Default.limitedParallelism(1)`), loads the model from `filesDir`, runs the
bundled image at launch, and blends the per-class overlay into a result `Bitmap` exposed
through an immutable `UiState`. `MainActivity` is a thin Compose host; `ClothSegScreen`
renders the status header, the gallery picker, and the result image.

## Files

| File | Role |
| --- | --- |
| `android/app/src/main/java/.../clothseg/MainActivity.kt` | Thin Compose host: view-model wiring + gallery picker |
| `android/app/src/main/java/.../clothseg/MainViewModel.kt` | Owns `ClothSegmenter`, model load, class-map → overlay `Bitmap`, `UiState` |
| `android/app/src/main/java/.../clothseg/UiState.kt` | Immutable UI snapshot |
| `android/app/src/main/java/.../clothseg/ClothSegmenter.kt` | LiteRT `CompiledModel` GPU inference (U²-Net, argmax → class map) |
| `android/app/src/main/java/.../clothseg/ImageUtils.kt` | Asset/gallery bitmap decode helpers (EXIF orientation) |
| `android/app/src/main/java/.../clothseg/view/ClothSegScreen.kt` | Compose screen: status header, picker button, result image |
| `android/app/src/main/java/.../clothseg/view/Theme.kt`, `view/Color.kt` | App theme + colors |
| `conversion/build_clothseg.py` | Converts the MIT weights to LiteRT with litert-torch |

## Convert

See [`conversion/`](conversion/) — `build_clothseg.py` loads the MIT weights and converts
with litert-torch (strip the `module.` prefix).
