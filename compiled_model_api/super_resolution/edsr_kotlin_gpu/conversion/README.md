# EDSR (×4) — conversion

Converts [EDSR](https://arxiv.org/abs/1707.02921) (eugenesiow/edsr-base via super-image,
Apache-2.0, DIV2K) to a LiteRT `CompiledModel`-GPU `.tflite` with litert-torch.

## Setup

```bash
pip install torch litert-torch super-image numpy
```

## Run

```bash
python build_edsr.py    # loads the Apache-2.0 EDSR-base x4 weights
# -> edsr.tflite  (7.7 MB, [1,3,128,128] -> [1,3,512,512])
```

EDSR is a pure CNN, but its **PixelShuffle** upsampler lowers to rank-5/6 reshapes the
Mali delegate rejects. Exact fix: **PixelShuffle(r) = a fixed-weight `ConvTranspose2d(stride=r)`**
→ ZeroStuffConvT2d (nearest-upsample + stride zero-stuff mask + flipped conv). Result:
68/68 nodes on the delegate, device corr 0.999946. Also unblocks other PixelShuffle SR models.
