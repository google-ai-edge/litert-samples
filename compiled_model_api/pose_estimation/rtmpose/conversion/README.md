# RTMPose-s → LiteRT conversion

Produces the `rtmpose_s_fp16.tflite` graph used by the RTMPose app, from mmpose's RTMPose-s (CSPNeXt +
RTMCC/SimCC head), with [litert-torch](https://github.com/google-ai-edge/litert).

## Environment (no compiled mmcv needed)

The RTMPose model definition lives in mmpose. It can be built with `mmcv-lite` (pure Python) — RTMPose has no
DCN / deformable-attention, so the compiled `mmcv._ext` ops are not required. `build_rtmpose.py` sets up
lightweight import stubs for the unused heavy heads (see its header).

```bash
pip install litert-torch ai-edge-litert ai-edge-quantizer torch
pip install mmengine mmcv-lite mmpose --no-deps munkres json_tricks
# build_rtmpose.py stubs xtcocotools (COCO-eval) + mmdet + mmcv.ops (pulled by RTMOHead/EDPoseHead, both unused).
```

## Run

```bash
python build_rtmpose.py        # downloads the RTMPose-s checkpoint; op-check + fp16 + tflite-vs-torch parity
python device_gate_rtm.py      # real-image torch-vs-tflite parity + SimCC keypoint decode fixture (optional)
```

`build_rtmpose.py` emits `rtm.tflite` (fp32) and `rtmpose_s_fp16.tflite`; push the fp16 file with
`../kotlin_cpu_gpu/android/install_to_device.sh`.

## Recipe (numerically exact)

The CSPNeXt backbone (SiLU, pure CNN) and the RTMCC head convert clean, with two **on-device-only** Mali
re-authorings applied in `build_rtmpose.py` (both monkey-patch mmpose in place):

- **`ScaleNorm` (RMS norm) → SafeRMSNorm.** Its input reaches ≈ |274|, so `Σ x²` ≈ 3.6M overflows fp16 (65504)
  on the Mali delegate → `norm = ∞` → `x/∞ = 0` (all-zero head). Scaling `x` down by S=64 before squaring keeps
  the reduction in fp16 range and is numerically identical.
- **GAU `act@act` BMM → broadcast-multiply + reduce-sum** (`q@kᵀ`, `kernel@v` over K=17 tokens), which the Mali
  delegate computes correctly where the activation×activation batch-matmul is mis-computed.

Output is two 1D SimCC distributions per keypoint; argmax over the bins (÷ split=2) gives the pixel coordinate
in the app. Result: banned ops NONE, all tensors ≤4D, tflite-vs-torch corr 1.0, device-vs-torch corr 0.999.
