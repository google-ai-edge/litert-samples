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

## Run

```bash
cd android
./install_to_device.sh <dir-with-portrait.tflite>
./gradlew :app:installDebug
```

The sample renders a bundled face photo as a pencil portrait. Adapt `MainActivity.kt` to feed
camera frames for a live sketch filter (center the face).

## Convert

See [`conversion/`](conversion/) — `build_portrait.py` loads the Apache-2.0 weights and converts
with litert-torch.
