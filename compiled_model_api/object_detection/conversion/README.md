# SSDLite320-MobileNetV3 → LiteRT conversion

Self-contained scripts that reproduce the model used by this sample.

## `convert_ssdlite.py`

Converts torchvision `ssdlite320_mobilenet_v3_large` to a GPU-clean LiteRT `.tflite` via
**litert-torch**, patch-free:

- **4D-head-tap** — returns each feature level's raw head conv outputs (`cls [1, A*91, H, W]`,
  `box [1, A*4, H, W]`) instead of the model's built-in `DefaultBoxGenerator` + NMS postprocess
  (which lowers to `GATHER_ND` / `TOPK` / `>4D`). Decode + NMS run in the app.
- **NCHW I/O** (no `to_channel_last_io`) — channel-last would turn MobileNetV3's
  SqueezeExcitation global-pools into `GATHER_ND`. Keeping NCHW converts stock-clean.

It prints the op histogram + GPU-compatibility check (BANNED / Flex / max-ndim), parity vs
PyTorch, and emits the fp16 (`float_casting`) model.

## `validate_decode.py`

Proves the Kotlin decode (`ObjectDetectionHelper.decode`) is correct: it rebuilds the 3234
default boxes, runs the fp16 `.tflite`, decodes with the same math the app uses (softmax →
`BoxCoder(10,10,5,5)` → per-class NMS), and box-matches the result against stock torchvision —
**298/300 boxes @ IoU 0.99**.

## Run

```bash
pip install "torch>=2.11" torchvision litert-torch ai-edge-litert ai-edge-quantizer numpy pillow
python conversion/convert_ssdlite.py
python conversion/validate_decode.py
```
