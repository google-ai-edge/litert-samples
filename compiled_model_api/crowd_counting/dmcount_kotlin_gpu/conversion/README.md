# DM-Count — conversion

Converts [DM-Count](https://github.com/cvlab-stonybrook/DM-Count) (MIT, NeurIPS 2020 crowd
counting) to a LiteRT `CompiledModel`-GPU `.tflite` with litert-torch.

## Setup

```bash
pip install torch litert-torch numpy
git clone https://github.com/cvlab-stonybrook/DM-Count.git DM-Count   # weights via git-lfs
```

## Run

```bash
python build_dmcount.py      # loads the MIT UCF-QNRF weights bundled in the repo
# -> dmcount.tflite  (86 MB, [1,3,512,512] -> [1,1,64,64] density map)
python validate_dmcount.py   # flatbuffer op scan + CompiledModel parity vs fixtures
```

DM-Count is a pure CNN (VGG19 + conv regression head) → fully GPU-compatible: **30/30 nodes
on the delegate, 1 partition**; Pixel 8a corr 0.9998–1.0 vs PyTorch with the count within
0.4% on real crowd images, ~79 ms/frame.

The one graph change is **exact**: `F.upsample_bilinear` (align_corners=True
`RESIZE_BILINEAR`, banned on the GPU delegate) is a linear operator, re-authored as two
constant-matrix multiplies — with the constant on the **RHS** so it lowers to
`FULLY_CONNECTED` (the delegate rejects `BATCH_MATMUL` with a constant LHS). Desktop corr
vs PyTorch is 1.000000 with an identical count.

Preprocessing: RGB, resize 512×512, ImageNet-normalize, NCHW. Output: non-negative density
map; `sum(map)` = person count, normalize per frame for a heatmap overlay.
