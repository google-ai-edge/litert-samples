# PIDNet-S — conversion

Converts [XuJiacong/PIDNet](https://github.com/XuJiacong/PIDNet) PIDNet-S
(Cityscapes, MIT) to a LiteRT `CompiledModel`-GPU `.tflite` with litert-torch.
PIDNet is a pure CNN with `align_corners=False` interpolation, so it converts with
**zero GPU patches** (0 tensors of rank > 4, 0 GPU-incompatible ops).

## Setup

```bash
pip install torch litert-torch onnx huggingface_hub
git clone https://github.com/XuJiacong/PIDNet.git
```

## Run

```bash
PIDNET_REPO=./PIDNet python build_pidnet.py
# -> pidnet_s.tflite  (30 MB, [1,3,1024,1024] -> [1,19,128,128])
```

The trained PIDNet-S weights are loaded from an ONNX mirror whose initializer names
match the original repo's PyTorch keys (all 453 load 1:1), then converted directly.
Input is RGB, ImageNet-normalized, NCHW; output is class logits at 1/8 resolution
(argmax over the 19 classes per pixel, then upscale).
