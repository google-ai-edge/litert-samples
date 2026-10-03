# ormbg — conversion

Converts [ormbg](https://huggingface.co/schirrmacher/ormbg) (open background removal, Apache-2.0, ISNet) to a LiteRT `CompiledModel`-GPU `.tflite` with litert-torch.

## Setup

```bash
pip install torch litert-torch huggingface_hub numpy
```

## Run

```bash
python build_ormbg.py    # downloads schirrmacher/ormbg weights (Apache-2.0)
# -> ormbg.tflite  (176 MB, [1,3,1024,1024] -> [1,1,1024,1024] alpha)
```

ormbg is a pure CNN (ISNet RSU blocks). The only patch is a defensive `align_corners=True -> False` on the bilinear upsamples (the GPU delegate rejects `align_corners=True`). Result: 246/246 nodes on the delegate, device corr 0.999881.
