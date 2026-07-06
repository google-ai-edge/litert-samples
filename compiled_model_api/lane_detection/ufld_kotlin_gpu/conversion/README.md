# Ultra-Fast-Lane-Detection — conversion

Converts [Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection)
(ResNet18, CULane, ECCV 2020, MIT) to a LiteRT `CompiledModel`-GPU `.tflite` with
litert-torch.

## Setup

```bash
pip install torch litert-torch gdown numpy
git clone https://github.com/cfzd/Ultra-Fast-Lane-Detection.git
```

## Run

```bash
python build_ufld.py    # fetches the MIT CULane ResNet18 weights
# -> ufld.tflite  (178 MB, [1,3,288,800] -> [1,201,18,4])
```

UFLD is a pure CNN (ResNet18 + row-classification head). The only patch is
ZeroPadMaxPool for the ResNet18 stem (its `MaxPool2d(padding=1)` lowers to a `-inf`
PADV2 that the Mali delegate rejects; the 0-pad + unpadded maxpool is exact post-ReLU).
Result: 41/41 nodes on the delegate, device corr 0.999982.
