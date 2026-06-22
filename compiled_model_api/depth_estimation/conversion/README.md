# Converting MiDaS_small to LiteRT

Reproduces the depth model used by this sample.

```bash
pip install litert-torch ai-edge-quantizer torch timm matplotlib pillow
python convert_midas_litert.py out 256
```

Produces:

- `midas_small_256.tflite` — fp32 (~66 MB)
- `midas_small_256_fp16.tflite` — fp16 (~33 MB, **used by the app**)

## Why this model converts cleanly

`MiDaS_small` is the CNN MiDaS (EfficientNet-Lite3 backbone), **not** the DPT/ViT
variants. It has no attention, no `PixelShuffle`, no `Focus`/strided-slice stems,
no `grid_sample`, and no `LayerScale` — i.e. none of the ops that force the
`litert-torch` converter into GPU-hostile primitives (`GATHER_ND`, `>4D`
reshapes, Flex). It therefore lowers entirely to GPU-clean builtins with no
patches:

```
CONV_2D, DEPTHWISE_CONV_2D, ADD, RELU, RESIZE_BILINEAR, RESHAPE
```

## Notes

- **Channel-last I/O** (`to_channel_last_io`): the exported model takes NHWC
  `1x256x256x3`, which removes the input transpose and matches the interleaved
  RGB the Android app writes.
- **fp16** (AI Edge Quantizer `FLOAT_CASTING`): half size, native on the GPU
  delegate, ~0.27 % difference vs fp32. Dynamic-range int8 is intentionally not
  used — it favors the CPU/XNNPACK path, not the ML Drift GPU delegate.
- Verified on-device (Pixel 8a): the fp16 model compiles to **234/234 nodes on
  the LiteRT GPU delegate (LITERT_CL)** with no CPU fallback, ~1–3 ms/inference.
  `RESIZE_BILINEAR align_corners=True` is GPU-supported as-is — no change needed.
