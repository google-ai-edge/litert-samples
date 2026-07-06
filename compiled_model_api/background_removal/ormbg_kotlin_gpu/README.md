# ormbg — Open background removal (LiteRT CompiledModel GPU)

**Background removal** running **fully on the LiteRT `CompiledModel` GPU** delegate.
[ormbg](https://huggingface.co/schirrmacher/ormbg) is a fully **open, Apache-2.0**
foreground/alpha matte model (an ISNet trained for photorealistic subject cut-out) —
the permissively-licensed alternative to the non-commercial RMBG-1.4. ~10 ms/frame on
a Pixel 8a.

- **Model:** [litert-community/ormbg-LiteRT](https://huggingface.co/litert-community/ormbg-LiteRT)
- **Weights:** [schirrmacher/ormbg](https://huggingface.co/schirrmacher/ormbg) · Apache-2.0 · ISNet
- **Input:** `[1, 3, 1024, 1024]` NCHW, RGB, `x / 255`
- **Output:** `[1, 1, 1024, 1024]` alpha matte in `[0,1]` (min-max normalize per frame)

## How it works

ormbg is a pure CNN (ISNet RSU blocks), so the graph converts fully GPU-compatible
(**246/246 nodes on the delegate, 1 partition**; device corr 0.999881, ~10 ms) with one
defensive patch: `align_corners=True` → `False` on the bilinear upsamples. CPU-exact vs
PyTorch (corr 0.9999999999). Post-process: min-max normalize the matte, resize, and use
as the foreground alpha.

## Run

```bash
# 1. Get the model (build with ../conversion or download from Hugging Face)
cd android
./install_to_device.sh <dir-with-ormbg.tflite>

# 2. Build & run
./gradlew :app:installDebug
```

The sample cuts the subject out of a bundled photo onto a transparency checkerboard.
Adapt `MainActivity.kt` to feed live camera frames (green-screen replacement) for a
real-time demo.

## Convert

See [`conversion/`](conversion/) — `build_ormbg.py` downloads the Apache-2.0 weights and
converts with litert-torch.
