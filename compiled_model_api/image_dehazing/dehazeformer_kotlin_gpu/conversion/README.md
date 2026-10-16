# DehazeFormer-MCT — conversion

Converts [DehazeFormer](https://github.com/IDKiro/DehazeFormer) (MIT, TIP 2023; MCT
curve-mapping variant trained on a mixed dataset for real-world haze) to a LiteRT
`CompiledModel`-GPU `.tflite` with litert-torch. Model code and checkpoint are fetched
from the author's Hugging Face Space (IDKiro/DehazeFormer_Demo) automatically.

## Setup

```bash
pip install torch litert-torch huggingface_hub numpy
```

## Run

```bash
python build_dehaze.py       # -> dehazeformer_base.tflite (17 MB) + ref fixtures
python validate_dehaze.py    # flatbuffer op scan + CompiledModel parity vs fixtures
```

The basenet runs at a fixed 256×256 and emits `[1,72,256,256]` curve parameters; the app
applies them to the full-resolution image host-side (`Dehazer.applyCurves`, the exact
official grid_sample mapping).

All re-authors are **exact** (desktop corr vs PyTorch 1.0000000):

- reflect pads → slice+concat — litert-torch lowers `reflection_pad2d` to `GATHER_ND`
  (rejected by the GPU delegate), including `padding_mode='reflect'` convs with padding=0
- Swin window partition/reverse → ≤4D reshape/permute; qkv → channel slices; relative
  position bias baked from the meta MLP
- RLN global norm + SKFusion global pool → **hierarchical means**: a single `MEAN` over
  C·H·W (1.5M elements) overflows the Mali fp16 accumulator → NaN output; equal-window
  `avg_pool` stages + small MEANs are mathematically identical and fp16-safe
- SKFusion 5D view+softmax → 4D pairwise softmax; Conv+PixelShuffle → zero-stuff
  ConvTranspose (per-subpixel conv bias re-added as a constant map)

Verified on a Pixel 8a: **2042/2042 nodes on the GPU delegate, 1 partition**, device corr
0.999998; end-to-end (device curves + host mapping) vs the official pipeline corr 0.999997.
