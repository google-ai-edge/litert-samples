# NAFNet → LiteRT conversion

Scripts that produce the `nafnet_fp16.tflite` graph used by the Android sample, from the official
[NAFNet-GoPro-width32](https://github.com/megvii-research/NAFNet) checkpoint, with
[litert-torch](https://github.com/google-ai-edge/litert).

## Environment

```bash
pip install litert-torch ai-edge-litert ai-edge-quantizer
pip install torch numpy pillow scipy
# Download the official checkpoint (keys match the arch defined in build_nafnet.py):
#   huggingface-cli download nyanko7/nafnet-models NAFNet-GoPro-width32.pth --local-dir .
```

## Run

```bash
python build_nafnet.py all          # re-author + op-check (banned NONE / >4D 0) + fp16 + tflite-vs-torch parity
```

`build_nafnet.py all` emits `nafnet.tflite` (fp32) and `nafnet_fp16.tflite`; push the fp16 file with
`../kotlin_cpu_gpu/android/install_to_device.sh`.

## Files

| File | What |
|---|---|
| `build_nafnet.py` | the NAFNet arch (keys match the checkpoint) + the three re-authorings + op-check + fp16 + parity. |
| `device_gate_naf.py` | builds a real-image fixture for an on-device parity check. |

## Recipe (all numerically exact)

NAFNet is a pure CNN (no activations; SimpleGate = channel-split multiply). Three GPU re-authorings:

1. **`LayerNorm2d` (custom autograd Function) → fp16-safe channel LayerNorm.** The deep residual stream
   reaches |x|≈175, so the channel reductions `Σ_c x` (~90k) and `Σ_c (x−μ)²` (~15M) **overflow fp16 (max
   65504)** on the Mali GPU delegate — which computes in fp16 regardless of the model dtype (so a "fp32 model"
   does not help). Fix: reduce in a down-scaled `x/S` domain (S=128), then rescale `var` and `(x−μ)` back —
   exact (LayerNorm is scale-invariant), eps in the original domain. Diagnosis: a shallow block (|x|≈6) is
   correct on device, the deep middle (|x|≈175) is corrupted → divergence ∝ magnitude = fp16 overflow.
2. **Simplified Channel Attention `AdaptiveAvgPool2d(1)` → `mean(3).mean(2)`** (two single-axis means; a single
   multi-axis global pool overflows / is mis-computed on Mali).
3. **Upsample `Conv2d(1×1)+PixelShuffle(2)` → Conv2d + depth-to-space `ZeroStuffConvT2d`** (PixelShuffle lowers
   to a 6D reshape; ZeroStuffConvT2d is `RESIZE_NEAREST` + `MUL` + `CONV_2D`).

Result: banned ops NONE, all tensors ≤4D, fp16 38 MB, tflite-vs-torch corr 1.0, device-vs-torch corr 1.0.


## Denoising variant (NAFNet-SIDD)

`build_sidd.py` is the same recipe for the **SIDD-width32 denoising** model (config width32 / enc[2,2,4,8] /
mid12 / dec[2,2,2,2], weights `NAFNet-SIDD-width32.pth` from `nyanko7/nafnet-models`). It produces
`sidd_fp16.tflite` — push it in place of the deblur model to run denoising in the same app. Device-verified
on a Pixel 8a: `2179/2179` LITERT_CL, ~46 ms, device-vs-torch corr **0.999999**, no NaN. Model:
[litert-community/NAFNet-SIDD-width32-LiteRT](https://huggingface.co/litert-community/NAFNet-SIDD-width32-LiteRT).
