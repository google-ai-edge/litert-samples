# CREPE → LiteRT conversion

Produces the `crepe_full_fp16.tflite` graph used by the Android sample, from [CREPE](https://github.com/marl/crepe) via the [torchcrepe](https://github.com/maxrmorrison/torchcrepe) PyTorch port, with [litert-torch](https://github.com/google-ai-edge/litert).

## Environment

```bash
pip install litert-torch ai-edge-litert ai-edge-quantizer
pip install torch numpy torchcrepe   # torchcrepe bundles the pretrained weights (assets/full.pth)
```

## Run

```bash
python build_crepe.py     # -> crepe_full_fp16.tflite, op-check + parity, 220/440/880 Hz self-test
```

Push the tflite with `../kotlin_cpu_gpu/android/install_to_device.sh`.

## How it converts — the simplest in this collection, zero patches

The whole `torchcrepe.Crepe` network is a **pure CNN** and converts directly with no rewrites:

- input `x[:, None, :, None]` → `[1, 1, 1024, 1]`
- 6× `layer` = `F.pad(zero, asymmetric "same")` → `Conv2d` → `ReLU` → `BatchNorm2d` → `MaxPool2d((2,1))` (conv1 = kernel (512,1) / stride (4,1), the rest (64,1))
- `permute(0, 2, 1, 3).reshape(-1, 2048)` (stays ≤4D) → `Linear(2048 → 360)` → `sigmoid`

Why it's GPU-clean with **no patches** (unlike most models in this repo):

| concern | why it's fine |
|---|---|
| padding | the "same" padding is a **constant zero-pad** → native `PAD`, not the reflect-pad `GATHER_ND` |
| activations | no GELU (no `Erf`), just ReLU/sigmoid; no TransposeConv, no dilated conv |
| rank | the head `permute/reshape` stays **≤4D** (no 5D/6D) |
| fp16 on Mali | **per-frame zero-mean/unit-var normalization** keeps activations ~O(1) → no overflow/precision wall |

op-check: banned NONE, >4D 0; fp16 tflite-vs-torch corr **1.000000**; self-test 220/440/880 Hz → 219.9/440.2/881.4 Hz.

## Preprocessing & decode (host-side, in the app)

Mono 16 kHz, framed into 1024-sample windows; per frame subtract the mean and divide by the std. Decode the 360 activations: peak bin ± 4, activation-weighted average → `cents = 20·bin + 1997.3794…` → `Hz = 10·2^(cents/1200)`; peak activation = confidence. Nearest note: `midi = 69 + 12·log2(Hz/440)`.

## Files

| File | What |
|---|---|
| `build_crepe.py` | converts the full model to fp16, op-check + parity, synthesized-tone self-test. |

Upstream: [marl/crepe](https://github.com/marl/crepe) (MIT); weights via [torchcrepe](https://github.com/maxrmorrison/torchcrepe) (MIT).
