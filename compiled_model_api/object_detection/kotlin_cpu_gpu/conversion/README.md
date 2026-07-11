# YOLOX → GPU-native LiteRT (conversion)

These scripts re-author the official **Megvii YOLOX** (Apache-2.0) weights into GPU-native LiteRT `.tflite` models with [`litert-torch`](https://github.com/google-ai-edge/ai-edge-torch), and verify them. The pre-built models are published at [`litert-community/yolox-{nano,tiny,s,m}-litert`](https://huggingface.co/litert-community) and are downloaded automatically by the Android app — you only need these scripts to reproduce or modify them.

## The one trick: fold the Focus stem

YOLOX is a pure CNN, but its **Focus stem** (stride-2 space-to-depth slicing) lowers to `GATHER_ND`, which the LiteRT GPU delegate rejects. `build_yolox.py` folds Focus + its following 3×3 conv into a single, numerically-exact **6×6 stride-2 conv** (`FoldedStem`). The result has **zero GATHER_ND / TopK / Cast** ops and **no >4-D tensors** — full `LITERT_CL` residency. The heads are tapped raw (`decode_in_inference=False`), so grid-decode + NMS run in the app (see the Kotlin `ObjectDetectorHelper`), keeping meshgrid/exp out of the graph.

## Setup

```bash
pip install torch torchvision litert-torch ai-edge-litert ai-edge-quantizer opencv-python loguru tabulate
git clone https://github.com/Megvii-BaseDetection/YOLOX.git   # model definitions (Apache-2.0)
# official weights:
for v in nano tiny s m; do
  curl -L -o yolox_$v.pth \
    https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_$v.pth
done
```

Place `build_yolox.py` / `validate_decode.py` next to the cloned `YOLOX/` directory and the `.pth` files.

## Run

```bash
# Re-author + convert + op-check + fp16 + parity (NHWC input = drop-in for the app):
python build_yolox.py yolox-s --nhwc
python build_yolox.py yolox-tiny --nhwc
python build_yolox.py yolox-nano --nhwc
python build_yolox.py yolox-m --nhwc

# Prove the host decode == YOLOX built-in decode, and end-to-end detection on a real image:
python validate_decode.py yolox-s
```

`build_yolox.py` prints, per model: folded-stem-vs-original parity (corr 1.000000), the op histogram with `GATHER_ND: 0 / banned: NONE / >4D: 0`, and tflite-vs-PyTorch parity (fp32 corr 1.0, fp16 corr ≈ 0.9999). Output: `yolox_<v>_nhwc_fp16.tflite`.

## I/O

- **Input** `[1, S, S, 3]` NHWC, **BGR, 0-255, no normalization** (letterbox: uniform scale, gray-114 pad). `S` = 416 (nano/tiny) or 640 (s/m).
- **Output** `[1, A, 85]` raw heads, anchor-major. `85 = 4 box (cx,cy,w,h grid units) + 1 obj + 80 class`; obj/class are sigmoid'd, boxes are **not** decoded.

## Verified on device (Pixel 8a)

All four convert GPU-clean and run with full `LITERT_CL` residency; device-GPU vs desktop-CPU output correlation ≥ 0.999, detections match the CPU reference.
