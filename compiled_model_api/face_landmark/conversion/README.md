# RTMPose-Face (WFLW) → LiteRT conversion

Produces `rtm_face_fp16.tflite` (98 WFLW facial landmarks) from [mmpose](https://github.com/open-mmlab/mmpose) RTMPose-m (Apache-2.0), with [litert-torch](https://github.com/google-ai-edge/litert).

## Environment (the "mm-stack", no compiled mmcv)

```bash
pip install litert-torch ai-edge-litert ai-edge-quantizer torch torchvision numpy pillow
pip install mmengine mmcv-lite mmpose --no-deps munkres json_tricks
```

`build_rtm_face.py` stubs `mmdet`/`mmdet.utils`/`mmcv.ops`/`xtcocotools` (the heads' `__init__` eagerly imports them but RTMPose doesn't use them) and monkey-patches the two Mali fixes (below). The WFLW config + checkpoint are pulled from the mmpose model zoo automatically.

## Run

```bash
python build_rtm_face.py          # op-check + fp16 + tflite-vs-torch parity
python device_gate_face.py        # real-face torch-vs-tflite parity + 98-landmark draw (optional)
```

Emits `rtm_face_fp16.tflite`; push with `../kotlin_cpu_gpu/android/install_to_device.sh`.

## Recipe (the RTMPose recipe, unchanged from the human-pose sample)

Same model family as RTMPose; only the config/checkpoint change to WFLW. Two on-device-only Mali fixes:
1. **`ScaleNorm` (RMS norm) → SafeRMSNorm** — the RTMCC `ScaleNorm` channel `Σx²` overflows fp16 on the Mali delegate (→ `norm=∞` → `x/∞=0`, an all-zero head). Scale `x` down by 64 before squaring, then rescale.
2. **GAU `act@act` BMM → broadcast-reduce** — `(q[:,:,None,:]·k[:,None,:,:]).sum(-1)`.

Output = `simcc_x[1,98,512]`, `simcc_y[1,98,512]` (output[0]=x, output[1]=y); each landmark = argmax / 2. Result: banned ops NONE, ≤4D, tflite-vs-torch corr 1.0, device-vs-torch corr 0.9995.
