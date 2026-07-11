# BiSeNet face parsing — conversion

Converts [zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) (BiSeNet, 19-class CelebAMask-HQ face parsing, MIT) to a LiteRT `CompiledModel`-GPU `.tflite` with litert-torch.

## Setup

```bash
pip install torch litert-torch huggingface_hub
git clone https://github.com/zllrunning/face-parsing.PyTorch.git fp
```

## Run

```bash
FP_REPO=./fp python build_faceparsing.py
# -> faceparsing.tflite  (53 MB, [1,3,512,512] -> [1,19,512,512])
```

Three GPU re-authoring patches (all baked into the exported graph → 74/74 nodes on the delegate, 1 partition):
1. **`align_corners=True` → `False`** — the output upsamples use `align_corners=True`, the resize form the GPU delegate rejects.
2. **global `avg_pool2d(x, x.size()[2:])` → `mean([2,3])`** — the context/attention modules pool with a full-spatial kernel, which the Mali delegate rejects as an `AVERAGE_POOL_2D`; a MEAN reduce is supported.
3. **zero-pad maxpool** — the ResNet stem `MaxPool2d(padding=1)` lowers to a PADV2 with `-inf` padding (`PADV2: src has wrong size` on Mali); an explicit 0-pad + unpadded maxpool is exact (input is post-ReLU ≥ 0).

The trained weights are loaded from the HF mirror `AI2lab/face-parsing.PyTorch`. Input is RGB, ImageNet-normalized, NCHW; output is class logits (argmax → face-part map).
