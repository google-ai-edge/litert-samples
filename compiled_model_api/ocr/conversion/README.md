# PP-OCRv5 → LiteRT conversion

Scripts that produce the two `.tflite` graphs used by the Android sample, from PP-OCRv5 (PaddleOCR 2025), with [litert-torch](https://github.com/google-ai-edge/litert).

## Environment

```bash
pip install litert-torch ai-edge-litert ai-edge-quantizer torch numpy safetensors pillow
git clone https://github.com/frotms/PaddleOCR2Pytorch   # Apache-2.0 PyTorch port (no PaddlePaddle dep)
# weights (safetensors, bit-exact from PaddleOCR) from HF JoyCN/PaddleOCR-Pytorch:
#   ptocr_v5_mobile_det.safetensors, ptocr_v5_mobile_rec.safetensors, dicts/ppocrv5_dict.txt
# On macOS, `import _stub_propack` first (guards the broken scipy _propack dlopen the port pulls in).
```

## Run

```bash
python build_det.py all       # detector + ZeroStuffConvT2d (ConvTranspose2d -> GPU-clean), fp16
python build_rec.py all       # recognizer + 4D-QKV attention, fp16
```

Emits `ppocr_det_fp16.tflite` + `ppocr_rec_fp16.tflite`; push with `../kotlin_cpu_gpu/android/install_to_device.sh`.

## Files

| File | What |
|---|---|
| `build_det.py` | detector (DBNet) + `ZeroStuffConvT2d` for the DB-head `ConvTranspose2d`; op-check + parity. |
| `build_rec.py` | recognizer (PPLCNetV3 + SVTR + CTC) + 4D-QKV attention split; op-check + parity. CTC decode is host-side. |
| `probe_det.py`, `probe_rec.py` | raw op-checks (show the single blocker each). |
| `_stub_propack.py` | macOS guard: narrow stub of scipy `_propack` (leaves scipy.optimize/signal real). |

## Re-authoring → GPU-clean (parity corr 1.0)

PP-OCRv5 is a CNN OCR pipeline with **no autoregressive decoder** (CTC recognition head), so both stages ride the GPU. Two blockers, both numerically-equivalent rewrites:

1. **Detector DB-head `ConvTranspose2d` (2× k2s2)** → **`ZeroStuffConvT2d`** = the 2D generalization of the DAC/DA3 1D zero-stuff trick: `F.interpolate` nearest ×stride × a zero-stuff mask + flipped `conv2d(padding=k-1)` + crop. `TRANSPOSE_CONV` is Mali-rejected; this lowers to `RESIZE_NEAREST` + `MUL`
   + `CONV_2D`. (Guard: skip the training-only DB `thresh` branch, not hit at inference.)
2. **Recognizer SVTR `Attention`** fused-QKV 5D reshape `(B,N,3,heads,hd)` → split q/k/v to 4D `(B,heads,N,hd)`. The port already drops the NRTR autoregressive branch → pure CTC. char_num = dict (18383) + blank + space = 18385; CTC layout = `['blank'] + dict + [' ']`.

Preprocessing: detector = ImageNet mean/std, /255, NCHW, 640×640. recognizer = resize to h=48 keep-aspect, pad to width 320, `(img/255−0.5)/0.5`. DB box postprocess (threshold + connected components + unclip) and CTC greedy decode run host-side (see the Kotlin app).
