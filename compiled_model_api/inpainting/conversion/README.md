# MI-GAN → LiteRT conversion

Produces `migan_fp16.tflite` (image inpainting / object removal) from [MI-GAN](https://github.com/Picsart-AI-Research/MI-GAN) (Picsart, ICCV 2023, MIT), with [litert-torch](https://github.com/google-ai-edge/litert).

## Environment

```bash
pip install litert-torch ai-edge-litert ai-edge-quantizer torch numpy pillow gdown
git clone https://github.com/Picsart-AI-Research/MI-GAN
# weights: migan_512_places2.pt from the repo's Google-Drive folder (or HF mirror d4rkk3y/migan for 256)
```

`build_migan.py` expects `MI-GAN/` next to it and `migan_models/migan_512_places2.pt`. The `.pt` is the **inference**-model state_dict — load it directly into `migan_inference.Generator(resolution=512)` (the repo's `export_inference_model.py` is only for converting a source `.pkl` → inference model; not needed here).

## Run

```bash
python build_migan.py all          # op-check + fp16 + tflite-vs-torch parity
python device_gate_migan.py        # mask + torch-vs-tflite parity + composite (optional)
```

Emits `migan_fp16.tflite` (16.3 MB); push with `../kotlin_cpu_gpu/android/install_to_device.sh`.

## Recipe (zero re-authoring — the norm-free generator lane)

The MI-GAN **inference** generator is already GPU-friendly: depthwise-separable `Conv2d`, `nn.Upsample(nearest)`
+ a fixed FIR-filter grouped conv (**no transposed conv** → no ZeroStuff), leaky-ReLU with clamp (→ `MAXIMUM`/
`MINIMUM`), and **no normalization** (StyleGAN-style). No fixes needed: banned ops NONE, ≤4D, tflite-vs-torch corr 1.0, device-vs-torch corr 0.99998.

**I/O**: input is 4ch `concat(mask − 0.5, rgb · mask)` (rgb ∈ [-1,1], mask = 1 keep / 0 erase); output [-1,1]; composite as `rgb·mask + out·(1-mask)`.
