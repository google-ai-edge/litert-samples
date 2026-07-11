# Ultra-Fast-Lane-Detection (ResNet18, CULane) — LiteRT CompiledModel GPU

Real-time **lane detection** running **fully on the LiteRT `CompiledModel` GPU**
delegate. [Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection)
(ECCV 2020) reformulates lane detection as fast **row-wise classification**: the
ResNet18 network runs on the GPU, and a tiny host-side arg/expectation decode turns the
grid into lane points. ~20 ms/frame on a Pixel 8a.

- **Model:** [litert-community/Ultra-Fast-Lane-Detection-LiteRT](https://huggingface.co/litert-community/Ultra-Fast-Lane-Detection-LiteRT)
- **Weights:** [cfzd/Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection) (CULane, ResNet18) · MIT
- **Input:** `[1, 3, 288, 800]` NCHW, RGB, `x/255` then ImageNet-normalized
- **Output:** `[1, 201, 18, 4]` = `(griding+1, row_anchors, lanes)`

## How it works

UFLD is a pure CNN, so the graph converts fully GPU-compatible (**41/41 nodes on the
delegate, 1 partition**; device corr 0.999982, ~20 ms) with one patch: the ResNet18
stem `MaxPool2d(padding=1)` `-inf` PADV2 → 0-pad + unpadded maxpool (exact post-ReLU).
CPU-exact vs PyTorch (corr 0.9999999999996).

**Host-side decode** (`LaneDetector.kt`): per lane & row anchor, softmax over the 200
grid cells → expectation column (drop if argmax = "no lane" index 200); map column → x
via `linspace(0,799,200)`, row anchor → y.

## Run

```bash
# 1. Get the model (build with ../conversion or download from Hugging Face)
cd android
./install_to_device.sh <dir-with-ufld.tflite>

# 2. Build & run
./gradlew :app:installDebug
```

The sample runs on a bundled dashcam frame and draws the detected lane points, colored
per lane. Adapt `MainActivity.kt` to feed live camera frames for a real-time demo.
UFLD is CULane-trained, so it works best on forward dashcam highway views.

## Convert

See [`conversion/`](conversion/) — `build_ufld.py` fetches the MIT weights and converts
with litert-torch.
