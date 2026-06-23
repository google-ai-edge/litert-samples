# Converting Zero-DCE to LiteRT

Reproduces the low-light enhancement model used by this sample.

```bash
pip install litert-torch ai-edge-quantizer torch
python convert_zerodce_litert.py . 512
```

Produces:

- `zerodce_512.tflite` — fp32 (~0.34 MB)
- `zerodce_512_fp16.tflite` — fp16 (~0.18 MB, **used by the app**)

The official Zero-DCE weights (`Epoch99.pth`) are downloaded automatically from
[`Li-Chongyi/Zero-DCE`](https://github.com/Li-Chongyi/Zero-DCE).

## Why this model converts cleanly

Zero-DCE's DCE-Net is a tiny pure CNN — 7 conv layers + ReLU + tanh + concat, followed
by an elementwise iterative curve application. It has no attention, no `GATHER_ND`, no
Flex/Custom ops, and all tensors stay 4-D, so it lowers entirely to GPU-clean builtins:

```
CONV_2D, MUL, SUB, ADD, SLICE, CONCATENATION, TANH
```

## Notes

- **Baked-in enhancement.** The forward returns only the final enhanced image — the 8
  curve iterations `x = x + r·(x² − x)` are part of the graph. `x*x` is used instead of
  `torch.pow(x, 2)` so the step lowers to `MUL` (GPU-native) rather than `POW`.
- **Channel-last I/O** (`to_channel_last_io(..., args=[0], outputs=[0])`): the exported
  model takes and returns NHWC `1x512x512x3` interleaved RGB in `[0,1]`, matching the
  interleaved RGB the Android app reads/writes (no transpose on either side).
- **fp16** (AI Edge Quantizer `FLOAT_CASTING`): GPU-native, corr 1.0 vs fp32.
- Verified on-device (Pixel 8a): the fp16 model compiles to **58/58 nodes on the LiteRT
  GPU delegate (LITERT_CL)** with no CPU fallback.
