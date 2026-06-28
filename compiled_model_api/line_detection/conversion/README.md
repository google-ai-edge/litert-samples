# M-LSD-tiny → LiteRT conversion

Produces the `mlsd_fp16.tflite` graph used by the Line Detection sample, from M-LSD-tiny (MobileNetV2 backbone),
with [litert-torch](https://github.com/google-ai-edge/litert).

## Environment

```bash
pip install litert-torch ai-edge-litert ai-edge-quantizer torch numpy pillow scipy
git clone https://github.com/lhwcv/mlsd_pytorch     # ships mlsd_tiny_512_fp32.pth + the model code
```

`build_mlsd.py` expects `mlsd_pytorch/` next to it (it imports `models/mbv2_mlsd_tiny.py` and loads
`models/mlsd_tiny_512_fp32.pth`). Before converting, set `align_corners=False` in the decoder:

```bash
sed -i 's/align_corners=True/align_corners=False/g' mlsd_pytorch/models/mbv2_mlsd_tiny.py
sed -i 's/MobileNetV2(pretrained=True)/MobileNetV2(pretrained=False)/g' mlsd_pytorch/models/mbv2_mlsd_tiny.py
```

## Run

```bash
python build_mlsd.py all          # op-check (banned NONE / >4D 0) + fp16 + tflite-vs-torch parity
python device_gate_mlsd.py        # real-image torch-vs-tflite parity + line decode fixture (optional)
```

`build_mlsd.py all` emits `mlsd.tflite` (fp32) and `mlsd_fp16.tflite`; push the fp16 file with
`../kotlin_cpu_gpu/android/install_to_device.sh`.

## Recipe (single fix)

Pure CNN encoder-decoder (MobileNetV2 + bilinear-upsample decoder). The only GPU re-authoring is the decoder's
`F.interpolate(mode='bilinear', align_corners=True)` → **`align_corners=False`** (the Mali delegate bans
`align_corners=True` + half-pixel). MobileNetV2 has no max-pool (strided convs → no `PADV2`), and the upsample
is `RESIZE_BILINEAR`, not a transposed conv → fully GPU-clean.

Result: banned ops NONE, all tensors ≤4D, fp16 1.4 MB, tflite-vs-torch corr 1.0, device-vs-torch corr 0.997.
The output is a TP map; the host decodes lines (sigmoid + NMS + displacement → endpoints).
