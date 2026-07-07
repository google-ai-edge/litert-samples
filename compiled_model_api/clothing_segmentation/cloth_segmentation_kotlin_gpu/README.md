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
orange, full body = magenta). Adapt `MainActivity.kt` to feed camera frames.

## Convert

See [`conversion/`](conversion/) — `build_clothseg.py` loads the MIT weights and converts
with litert-torch (strip the `module.` prefix).
