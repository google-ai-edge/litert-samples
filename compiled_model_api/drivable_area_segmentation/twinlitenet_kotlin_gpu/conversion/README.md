# TwinLiteNet — conversion

Converts [TwinLiteNet](https://github.com/chequanghuy/TwinLiteNet) (2023, MIT, BDD100K)
to a LiteRT `CompiledModel`-GPU `.tflite` with litert-torch.

## Setup

```bash
pip install torch litert-torch numpy
git clone https://github.com/chequanghuy/TwinLiteNet.git
```

## Run

```bash
python build_twinlite.py    # uses the bundled MIT weights (pretrained/best.pth)
# -> twinlite.tflite  (3.1 MB, [1,3,360,640] -> 2x [1,2,360,640])
```

TwinLiteNet is a pure CNN (ESPNet-C encoder + two seg heads). The only patch is
ZeroStuffConvT2d for the `ConvTranspose2d` upsamplers (nearest-upsample + stride
zero-stuff mask + flipped conv; the Mali delegate rejects `TRANSPOSE_CONV`). Result:
270/270 nodes on the delegate, device corr 0.99997/0.99998.
