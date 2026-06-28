# L2CS-Net → LiteRT conversion

Produces the `gaze_fp16.tflite` graph used by the Gaze Estimation sample, from
[L2CS-Net](https://github.com/Ahmednull/L2CS-Net) (Ahmednull, MIT, ResNet50, Gaze360), with
[litert-torch](https://github.com/google-ai-edge/litert).

## Environment

```bash
pip install litert-torch ai-edge-litert ai-edge-quantizer torch torchvision numpy pillow huggingface_hub
git clone https://github.com/Ahmednull/L2CS-Net      # provides l2cs/model.py (loaded via importlib)
```

`build_gaze.py` expects `L2CS-Net/` next to it, and downloads the `L2CSNet_gaze360.pkl` weights from
`tianfxc/l2cs` on the Hugging Face Hub (this avoids the upstream Google-Drive *folder* download, which
`gdown` silently fails on). The `l2cs` package `__init__` pulls `face_detection`, so the model definition is
loaded directly via `importlib` from `L2CS-Net/l2cs/model.py`.

## Run

```bash
python build_gaze.py all          # op-check (banned NONE / >4D 0) + fp16 + tflite-vs-torch parity
python device_gate_gaze.py        # real-image torch-vs-tflite parity + gaze decode (optional)
```

`build_gaze.py all` emits `gaze.tflite` (fp32) and `gaze_fp16.tflite`; push the fp16 file with
`../kotlin_cpu_gpu/android/install_to_device.sh`.

## Recipe (two ResNet fixes)

Pure CNN (ResNet50 + 2 FC heads → 90 yaw / 90 pitch angle bins). Two on-device-only re-authorings:

1. **Stem `MaxPool2d(3, s2, p1)` → zero-pad + valid max-pool.** PyTorch's max-pool pads with `-inf`, which
   lowers to a `PADV2` op the Mali ML Drift delegate **won't delegate** (it splits the graph and fails to
   compile). Because the pool follows a ReLU (inputs ≥ 0), padding with **0** is exactly equivalent and emits a
   delegatable `PAD` → full GPU residency.
2. **Global `AdaptiveAvgPool2d(1)` → `mean(3).mean(2)`** (two single-axis means; the Mali multi-axis-pool fix).

The softmax over the 90 angle bins is baked in; the host does the expectation `deg = Σ p_i·i·4 − 180`. Result:
banned ops NONE, all tensors ≤4D, tflite-vs-torch corr 1.0, device-vs-torch corr 0.9999.
