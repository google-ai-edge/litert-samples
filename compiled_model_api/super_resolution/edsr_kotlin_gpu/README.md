# EDSR (×4) — Super-resolution (LiteRT CompiledModel GPU)

**×4 single-image super-resolution** running **fully on the LiteRT `CompiledModel` GPU**
delegate. [EDSR](https://arxiv.org/abs/1707.02921) (CVPR 2017 winner) upscales a low-res
image 4× with sharp detail. ~23 ms/frame on a Pixel 8a.

- **Model:** [litert-community/EDSR-x4-LiteRT](https://huggingface.co/litert-community/EDSR-x4-LiteRT)
- **Weights:** [eugenesiow/edsr-base](https://huggingface.co/eugenesiow/edsr-base) (super-image, DIV2K) · Apache-2.0
- **Input:** `[1, 3, 128, 128]` NCHW, RGB, `x/255`
- **Output:** `[1, 3, 512, 512]` NCHW, RGB 0–1 (clamp, ×255)

## How it works

EDSR is a pure CNN, but its **PixelShuffle** sub-pixel upsampler lowers to rank-5/6
reshapes the Mali delegate rejects (the classic super-resolution wall). Exact fix:
**PixelShuffle(r) ≡ a fixed-weight `ConvTranspose2d(stride=r)`** → ZeroStuffConvT2d.
Result: **68/68 nodes on the delegate, 1 partition**; device corr 0.999946, ~23 ms.
CPU-exact vs PyTorch (corr 1.0).

## Run

```bash
cd android
./gradlew :app:installDebug
```

The 7.7 MB `edsr.tflite` is bundled in `app/src/main/assets/`. The sample shows bicubic
×4 vs EDSR ×4 on a bundled low-res image. Adapt `MainActivity.kt` to feed camera frames
(tile larger images into 128 blocks).

## Convert

See [`conversion/`](conversion/) — `build_edsr.py` loads the Apache-2.0 weights and converts
with litert-torch (with the PixelShuffle patch).
