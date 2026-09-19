# U²-Net Portrait — conversion

Converts the [U²-Net](https://github.com/xuebinqin/U-2-Net) `u2net_portrait` model
(Apache-2.0) to a LiteRT `CompiledModel`-GPU `.tflite` with litert-torch.

## Setup

```bash
pip install torch litert-torch huggingface_hub numpy
git clone https://github.com/xuebinqin/U-2-Net.git u2net-src
```

## Run

```bash
python build_portrait.py    # loads the Apache-2.0 u2net_portrait weights
# -> portrait.tflite  (176 MB, [1,3,512,512] -> [1,1,512,512])
```

U²-Net is a pure CNN → fully GPU-compatible with one defensive patch (`align_corners=False`):
893/893 nodes on the delegate, device corr 0.998683. Preprocessing: RGB, `x/255` then
ImageNet-normalize. Output: min-max normalize, then invert (`1−x`) for dark strokes on white.
