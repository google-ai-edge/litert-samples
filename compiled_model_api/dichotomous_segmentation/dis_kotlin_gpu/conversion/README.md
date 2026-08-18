# DIS (IS-Net) — conversion

Converts [DIS](https://github.com/xuebinqin/DIS) (`isnet-general-use`, ECCV 2022, Apache-2.0)
to a LiteRT `CompiledModel`-GPU `.tflite` with litert-torch.

## Setup

```bash
pip install torch litert-torch huggingface_hub numpy
git clone https://github.com/xuebinqin/DIS.git dis-src
```

## Run

```bash
python build_dis.py    # loads the Apache-2.0 isnet-general-use weights
# -> dis.tflite  (176 MB, [1,3,1024,1024] -> [1,1,1024,1024] alpha)
```

DIS is a pure CNN (IS-Net RSU blocks) → fully GPU-compatible with one defensive patch
(`align_corners=False` on the bilinear upsamples): 247/247 nodes on the delegate, device
max|diff| 0.00034. Preprocessing `x/255 - 0.5`; output is a sigmoid alpha (0..1).
