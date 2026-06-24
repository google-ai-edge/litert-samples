# Converting VDSR to LiteRT

Reproduces the super-resolution model used by this sample.

```bash
pip install litert-torch ai-edge-quantizer torch
python convert_vdsr_litert.py . 256
```

Produces:

- `vdsr_256.tflite` — fp32 (~2.68 MB)
- `vdsr_256_fp16.tflite` — fp16 (~1.35 MB, **used by the app**)

The official VDSR weights (`model_epoch_50.pth`, MIT) are downloaded automatically from
[`twtygqyy/pytorch-vdsr`](https://github.com/twtygqyy/pytorch-vdsr).

## Why this model converts cleanly

VDSR is a 20-layer CNN that refines the luminance of an image at display resolution —
there is **no upsampling layer inside the network** (it expects a bicubic-upscaled
input). So the graph is just `Conv2d + ReLU` repeated, plus a single global residual
`add`. It has no `PixelShuffle`, no `PReLU`, and no attention — none of the ops that the
PyTorch→LiteRT path turns into GPU-hostile primitives. It therefore lowers entirely to
GPU-clean builtins:

```
CONV_2D, DEPTHWISE_CONV_2D, ADD
```

(Contrast Real-ESRGAN, whose PReLU and PixelShuffle don't lower to native builtins.)

## Notes

- **Luminance only.** VDSR is trained on the Y channel. The Android app converts to
  YCbCr, super-resolves Y, and keeps Cb/Cr from the input.
- **Channel-last I/O** (`to_channel_last_io(..., args=[0], outputs=[0])`): the exported
  model takes and returns NHWC `1x256x256x1`.
- **fp16** (AI Edge Quantizer `FLOAT_CASTING`): GPU-native, corr 1.0 vs fp32.
- Verified on-device (Pixel 8a): the fp16 model compiles to **41/41 nodes on the LiteRT
  GPU delegate (LITERT_CL)** with no CPU fallback.
