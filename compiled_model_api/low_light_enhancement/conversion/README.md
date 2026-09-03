# CPGA-Net → LiteRT conversion

Produces `cpga_fp16.tflite` (low-light enhancement) from [CPGA-Net](https://github.com/Shyandram/CPGA-Net-Pytorch) (Shyandram, IJPRAI, MIT), with [litert-torch](https://github.com/google-ai-edge/litert).

## Environment

```bash
pip install litert-torch ai-edge-litert ai-edge-quantizer torch numpy pillow guided-filter-pytorch
git clone https://github.com/Shyandram/CPGA-Net-Pytorch    # ships weights/enhance_color-llie-ResCBAM_g.pkl
```

`build_cpga.py` expects `CPGA-Net-Pytorch/` next to it (it stubs the unused `guided_filter_pytorch` import if missing).

## Run

```bash
python build_cpga.py all          # op-check + fp16 + tflite-vs-torch parity
python device_gate_cpga.py        # dark-image torch-vs-tflite parity + before/after (optional)
```

Emits `cpga_fp16.tflite` (0.1 MB — the smallest model in the zoo); push with `../kotlin_cpu_gpu/android/install_to_device.sh`.

## Recipe (three numerically-exact fixes)

A tiny channel-prior + gamma-correction CNN (0.025 M params). Three GPU fixes:

1. **Gamma correction `torch.pow(x, γ)` → `exp(γ · log x)`** — `POW` is banned on Mali; the identity is exact (clamp the base to [1e-9, 1] first) and emits native `EXP`/`LOG`.
2. **CBAM / gamma global pools** — `AdaptiveAvgPool2d(1)` → `mean(3).mean(2)`; `AdaptiveMaxPool2d(1)` → `F.max_pool2d(x, kernel_size=(H,W))` (max-pool, not `torch.amax`, which has no NHWC rewriter in litert-torch).
3. The dark/bright **channel prior** (`max`/`min` over RGB) stays as `REDUCE_MAX`/`REDUCE_MIN` (GPU-clean).

`isdgf=False` (no guided filter → no bicubic). Result: banned ops NONE, ≤4D, tflite-vs-torch corr 1.0, device-vs-torch corr 0.99999.
