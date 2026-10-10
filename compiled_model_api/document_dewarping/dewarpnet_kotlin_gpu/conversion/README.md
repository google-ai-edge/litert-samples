# DewarpNet — conversion

Converts [DewarpNet](https://github.com/cvlab-stonybrook/DewarpNet) (ICCV 2019, MIT,
doc3d) to a LiteRT `CompiledModel`-GPU `.tflite` with litert-torch.

## Setup

```bash
pip install torch litert-torch gdown numpy
git clone https://github.com/cvlab-stonybrook/DewarpNet.git dewarp-src
```

## Run

```bash
python build_dewarp.py    # fetches the MIT doc3d weights (WCNet + BMNet)
# -> dewarp.tflite  (189 MB, [1,3,256,256] BGR/255 -> [1,2,128,128] backward map)
```

DewarpNet is a pure CNN (WCNet UNet + BMNet DenseNet). Two exact patches make it
GPU-compatible: (1) `ConvTranspose2d` → ZeroStuffConvT2d (nearest-upsample + stride
zero-stuff mask + flipped conv; the Mali delegate rejects `TRANSPOSE_CONV`), and (2)
`Hardtanh(0,1)` → `relu(x) - relu(x-1)` (the delegate rejects `RELU_0_TO_1`). Result:
371/371 nodes on the delegate, device corr 0.999866. Apply `grid_sample` host-side.
