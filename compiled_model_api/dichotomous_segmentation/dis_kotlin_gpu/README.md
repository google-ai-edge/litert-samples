# DIS (IS-Net, general-use) — High-precision object cutout (LiteRT CompiledModel GPU)

**Dichotomous image segmentation** running **fully on the LiteRT `CompiledModel` GPU**
delegate. [DIS](https://github.com/xuebinqin/DIS) (ECCV 2022) is a high-accuracy IS-Net that
cuts out the main object with **fine structure detail** (thin stems, petals, wires) — for
e-commerce product photos and graphics. ~11 ms/frame on a Pixel 8a.

- **Model:** [litert-community/DIS-ISNet-LiteRT](https://huggingface.co/litert-community/DIS-ISNet-LiteRT)
- **Weights:** [xuebinqin/DIS](https://github.com/xuebinqin/DIS) `isnet-general-use` · Apache-2.0 · IS-Net
- **Input:** `[1, 3, 1024, 1024]` NCHW, RGB, `x/255 - 0.5`
- **Output:** `[1, 1, 1024, 1024]` sigmoid alpha (0–1)

## How it works

DIS is a pure CNN (IS-Net RSU blocks) → fully GPU-compatible (**247/247 nodes on the
delegate, 1 partition**; device max|diff| 0.00034, ~11 ms) with one defensive patch:
`align_corners=False`. CPU-exact vs PyTorch (max|diff| 0.0). Resize the alpha to the image
and composite.

## Run

```bash
# 1. Get the model (build with ../conversion or download from Hugging Face)
cd android
./install_to_device.sh <dir-with-dis.tflite>

# 2. Build & run
./gradlew :app:installDebug
```

The sample cuts the object out of a bundled photo onto a transparency checkerboard. Adapt
`MainActivity.kt` to feed camera frames.

## Convert

See [`conversion/`](conversion/) — `build_dis.py` loads the Apache-2.0 weights and converts
with litert-torch.
