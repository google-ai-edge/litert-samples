# LR-ASPP MobileNetV3 → LiteRT conversion

Produces the `lraspp_fp16.tflite` graph used by the Android sample, from torchvision's
`lraspp_mobilenet_v3_large` (COCO-VOC, 21 classes), with [litert-torch](https://github.com/google-ai-edge/litert).

## Environment

```bash
pip install litert-torch ai-edge-litert ai-edge-quantizer torch torchvision numpy pillow
# Weights are auto-downloaded by torchvision (LRASPP_MobileNet_V3_Large_Weights.DEFAULT).
```

## Run

```bash
python build_lraspp.py all          # re-author + op-check (banned NONE / >4D 0) + fp16 + tflite-vs-torch parity
python device_gate_seg.py           # real-image torch-vs-tflite parity fixture (optional)
```

`build_lraspp.py all` emits `lraspp.tflite` (fp32) and `lraspp_fp16.tflite`; push the fp16 file with
`../kotlin_cpu_gpu/android/install_to_device.sh`.

## Files

| File | What |
|---|---|
| `build_lraspp.py` | loads torchvision LR-ASPP, applies the one re-authoring, op-check, fp16, parity. |
| `device_gate_seg.py` | builds a real-image fixture for an on-device parity check. |

## Recipe (numerically exact)

Pure CNN (MobileNetV3 backbone + Lite R-ASPP head). The only GPU re-authoring: the 9 `AdaptiveAvgPool2d(1)`
global pools (8 MobileNetV3 Squeeze-Excite blocks + the R-ASPP scale branch) → **`mean(3).mean(2)`** (two
single-axis means; a single multi-axis global pool is mis-computed / can overflow on the Mali delegate).
Everything else is already GPU-clean: `Hardswish`/`Hardsigmoid` lower to the native `HARD_SWISH` builtin, and
the R-ASPP `F.interpolate` already uses `align_corners=False`. The output is NHWC 21-class logits; the
per-pixel argmax runs in the app.

Result: banned ops NONE, all tensors ≤4D, fp16 6.7 MB, tflite-vs-torch corr 1.0, device-vs-torch corr 0.99998.
