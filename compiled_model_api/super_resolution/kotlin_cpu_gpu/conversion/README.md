# Real-ESRGAN → GPU-native LiteRT (conversion)

Re-authors Real-ESRGAN **realesr-general-x4v3** (SRVGGNetCompact, BSD-3) into a GPU-native LiteRT `.tflite` with [`litert-torch`](https://github.com/google-ai-edge/ai-edge-torch). The pre-built model is at [`litert-community/real-esrgan-x4v3-litert`](https://huggingface.co/litert-community/real-esrgan-x4v3-litert) and is downloaded by the app at build time — you only need this to reproduce/modify it.

## The two re-authoring fixes (pure CNN, both gates pass)

Stock convert is NOT GPU-clean — PReLU and PixelShuffle lower to GPU-rejected ops:
- **PReLU → `relu(x) − a·relu(−x)`** (per-channel `a`): exact, RELU/MUL/SUB only (no GREATER/SELECT).
- **PixelShuffle(4) → one-hot `ConvTranspose2d(stride 4)` → ZeroStuffConvT** (zero-stuff nearest + Conv2d): exact, no `TRANSPOSE_CONV`, no >4-D tensors. (`PS_FIX=1`)

Result: zero GATHER/SELECT/TopK/Cast, no >4-D — full `LITERT_CL` residency (Pixel 8a 211/211, ~1 ms), GPU vs CPU corr ≈ 0.995, re-authored vs original corr 1.0.

## Setup & run

```bash
pip install torch litert-torch ai-edge-litert ai-edge-quantizer numpy
# official weights (BSD-3):
curl -L -o realesr-general-x4v3.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth
# point WEIGHTS in convert_realesrgan.py at the .pth, then:
PS_FIX=1 python convert_realesrgan.py --nhwc
```

## I/O
Input `[1, 128, 128, 3]` NHWC RGB 0–1; output `[1, 512, 512, 3]` NHWC RGB 0–1 (×4). Tile larger images.
