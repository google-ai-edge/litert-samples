# UniSal → LiteRT conversion

Produces `unisal_fp16.tflite` (visual saliency) from [UniSal](https://github.com/rdroste/unisal) (rdroste,
Apache-2.0, MobileNetV2 + bilinear decoder), with [litert-torch](https://github.com/google-ai-edge/litert).

## Environment

```bash
pip install litert-torch ai-edge-litert ai-edge-quantizer torch torchvision numpy pillow tensorboardX
git clone https://github.com/rdroste/unisal        # ships training_runs/pretrained_unisal/weights_best.pth
```

`build_unisal.py` expects `unisal/` next to it.

## Run

```bash
python build_unisal.py all          # op-check + fp16 + tflite-vs-torch parity
python device_gate_unisal.py        # real-image torch-vs-tflite parity + heatmap (optional)
```

Emits `unisal_fp16.tflite` (6.5 MB); push with `../kotlin_cpu_gpu/android/install_to_device.sh`.

## Recipe (three numerically-exact fixes)

UniSal is a unified image+video model; for static images the **Bypass-RNN** path is used and the **SALICON**
domain pinned (its domain-specific BatchNorm/smoothing fold to constants). Three GPU fixes:

1. **Strided subsample `x[..., ::2, ::2]` → `F.avg_pool2d(x, kernel_size=1, stride=2)`** — the MobileNetV2 stride
   slice lowers to `GATHER_ND`; a kernel-1 stride-2 average-pool selects the exact same pixels → `AVERAGE_POOL_2D`.
2. **Bake the 16 Gaussian prior maps** — `_get_gaussian_maps` (meshgrid + exp) emits `GATHER_ND`/`BROADCAST_TO`;
   they depend only on the (fixed) feature size, so precompute once and concatenate the constant.
3. **`F.pad(mode="replicate")` → 0-pad** for the 41×41 Gaussian smoothing (replicate → `GATHER_ND`). The smoothing
   is kept — it suppresses border artifacts (dropping it anti-correlates the output with the reference).

The final spatial log-softmax / normalization runs in the app. Result: banned ops NONE, ≤4D, tflite-vs-torch corr
1.0, device-vs-torch corr 0.9998.
