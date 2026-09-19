# PIDNet-S — Real-time semantic segmentation (LiteRT CompiledModel GPU)

Real-time **semantic segmentation** running **fully on the LiteRT `CompiledModel` GPU** delegate. [PIDNet-S](https://arxiv.org/abs/2206.02066) (CVPR 2023) segments a road scene into the **19 Cityscapes classes** (road, sidewalk, building, car, person, sky, …) at ~17 FPS on a Pixel 8a.

- **Model:** [litert-community/PIDNet-S-Cityscapes-LiteRT](https://huggingface.co/litert-community/PIDNet-S-Cityscapes-LiteRT) · 30 MB
- **Weights:** [XuJiacong/PIDNet](https://github.com/XuJiacong/PIDNet) · MIT · 78.8% mIoU (Cityscapes val)
- **Input:** `[1, 3, 1024, 1024]` NCHW, RGB, ImageNet-normalized
- **Output:** `[1, 19, 128, 128]` class logits (1/8 res; argmax + upscale)

## How it works

PIDNet is a three-branch CNN (P: detail, I: context, D: boundary) — no attention, no dynamic shapes at a fixed input size, and `align_corners=False` on every bilinear resize. So it converts to a **fully GPU-compatible graph with zero patches**: `CONV_2D` ×75, `RESIZE_BILINEAR` ×11 (align_corners=False), `AVERAGE_POOL_2D`, `ADD`/`MUL`/`SUB`/`SUM`, `LOGISTIC` — **0 tensors of rank > 4, 0 GPU-incompatible ops**. The converted graph matches the original PyTorch model bit-for-bit on CPU (corr 0.99999999999, 100% argmax); on the Mali GPU (fp16) it agrees with the fp32 reference at 97% of pixels with correct classes.

Preprocess: RGB → resize 1024×1024 → ImageNet normalize → NCHW. Postprocess: argmax over the 19 channels per pixel → Cityscapes-colored label map → upscale.

## Run

```bash
# 1. Get the model (build with ../conversion or download from Hugging Face)
#    then push it into the app's private storage:
cd android
./install_to_device.sh <dir-with-pidnet_s.tflite>

# 2. Build & run
./gradlew :app:installDebug
```

On launch the sample segments a bundled Cityscapes road scene and displays the input
with the 19-class colored segmentation overlaid. Tap **Pick image** to run the model on
any photo from the gallery. Adapt `MainViewModel.kt` to feed live camera frames for a
real-time demo.

## App architecture

The app is **MVVM + Jetpack Compose** (Compose Material, a `compose-bom`, and a Gradle
version catalog). `MainViewModel` owns the `PIDNet` segmenter, loads the model from
`filesDir`, runs the bundled image at launch, accepts gallery images, and confines every
inference to a single worker (`Dispatchers.Default.limitedParallelism(1)`); it produces the
blended overlay bitmap and exposes it through an immutable `UiState`. `MainActivity` is a
thin Compose host.

| File | Role |
| --- | --- |
| `MainActivity.kt` | Thin `ComponentActivity` host: wires the gallery picker, collects `UiState`. |
| `MainViewModel.kt` | Owns `PIDNet`, runs inference on a confined dispatcher, blends the label map over the input. |
| `UiState.kt` | Immutable UI snapshot: result bitmap, status flags, inference time. |
| `PIDNet.kt` | LiteRT `CompiledModel` (GPU) segmenter: preprocess → run → argmax → Cityscapes-colored label map. |
| `CityscapesPalette.kt` | Cityscapes 19-class names + overlay colors. |
| `ImageUtils.kt` | Asset / gallery bitmap decoding (EXIF-oriented) helpers. |
| `view/SegmentationScreen.kt` | Compose screen: status header, image picker, blended result image. |
| `view/Theme.kt`, `view/Color.kt` | Compose Material theme + colors. |

## Convert

See [`conversion/`](conversion/) — `build_pidnet.py` loads the trained PIDNet-S weights and converts with litert-torch.
