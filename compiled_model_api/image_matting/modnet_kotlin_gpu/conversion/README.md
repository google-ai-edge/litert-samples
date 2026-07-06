# MODNet — conversion

Converts [ZHKKKe/MODNet](https://github.com/ZHKKKe/MODNet) (trimap-free portrait
matting, Apache-2.0) to a LiteRT `CompiledModel`-GPU `.tflite` with litert-torch.

## Setup

```bash
pip install torch litert-torch huggingface_hub
git clone https://github.com/ZHKKKe/MODNet.git
```

## Run

```bash
MODNET_REPO=./MODNet python build_modnet.py
# -> modnet.tflite  (26 MB, [1,3,512,512] -> [1,1,512,512] alpha)
```

Two GPU re-authoring patches (both baked into the exported graph → 0 tensors of
rank > 4, 0 GPU-incompatible ops):
1. **SE block `Linear` → `1×1 conv`** — the 2D-Linear→4D-reshape confuses the
   NCHW↔NHWC layout (`mul` broadcast mismatch); 1×1 convs on the pooled tensor are
   identical and NCHW-clean.
2. **fp16-safe hierarchical-mean `InstanceNorm`** — MODNet's IBNorm runs
   `InstanceNorm2d` over up to 512×512 spatial; the variance `sum(dd²)` overflows
   fp16 on the Mali delegate (matte degrades). Computing the spatial mean via a
   cascade of `/2` average-pools (magnitude-bounded, exact for power-of-2) restores
   GPU correlation to 0.99994.

The trained weights are loaded from the HF mirror `DavG25/modnet-pretrained-models`.
Input is RGB normalized to [-1, 1], NCHW; output is a soft alpha matte in [0, 1].
