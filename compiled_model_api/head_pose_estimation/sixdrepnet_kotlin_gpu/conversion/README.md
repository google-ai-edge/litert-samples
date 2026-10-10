# 6DRepNet — conversion

Converts [6DRepNet](https://github.com/thohemp/6DRepNet) (ICIP 2022, MIT, 300W-LP) to a
LiteRT `CompiledModel`-GPU `.tflite` with litert-torch.

## Setup

```bash
pip install torch litert-torch sixdrepnet huggingface_hub numpy
```

## Run

```bash
python build_6drepnet.py    # loads the MIT deploy-mode RepVGG weights (osanseviero HF mirror)
# -> 6drepnet.tflite  (157 MB, [1,3,224,224] face crop -> [1,6] 6D rotation)
```

6DRepNet in **deploy** mode (RepVGG re-parameterized to plain 3x3 convs + ReLU) is a pure
CNN → fully GPU-compatible with **zero patches**: 36/36 nodes on the delegate, device corr
0.9993. Use the deploy weights (fused `rbr_reparam`), not the training-mode branches. The
6D → Gram-Schmidt rotation matrix → Euler decode runs host-side.
