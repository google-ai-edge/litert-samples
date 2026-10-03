# YOLACT-ResNet50 — conversion

Converts base [YOLACT](https://github.com/dbolya/yolact) (ResNet50, ICCV 2019, MIT) to a LiteRT `CompiledModel`-GPU `.tflite` with litert-torch. The network runs on the GPU; the decode (NMS + linear-combination masks) is host-side.

## Setup

```bash
pip install torch litert-torch huggingface_hub pycocotools
git clone https://github.com/dbolya/yolact.git
```

## Run

```bash
python build_yolact.py    # loads dbolya/yolact-resnet50 weights from HF
# -> yolact.tflite  (125 MB, [1,3,550,550] -> loc/conf/mask/proto)
```

Notes: base YOLACT only (NOT YOLACT++ = DCNv2 deformable conv, GPU-incompatible). The script stubs YOLACT's CUDA assumptions and sets `use_jit=False` so the scripted FPN is traceable, bypasses the built-in NMS, and applies ZeroPadMaxPool (the ResNet50 stem `MaxPool2d(padding=1)` lowers to a `-inf` PADV2 that the Mali delegate rejects; the 0-pad + unpadded maxpool is exact post-ReLU). Result: 138/138 nodes on the delegate, device corr 0.99999–1.0 on all four raw outputs. The 19248 SSD priors are baked to `priors.bin` for the host-side box decode.
