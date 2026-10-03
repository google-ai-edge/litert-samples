# Cloth Segmentation (U²-Net) — conversion

Converts [cloth-segmentation](https://github.com/levindabhi/cloth-segmentation) (U²-Net,
MIT, iMaterialist-Fashion) to a LiteRT `CompiledModel`-GPU `.tflite` with litert-torch.

## Setup

```bash
pip install torch litert-torch huggingface_hub numpy
git clone https://github.com/levindabhi/cloth-segmentation.git
```

## Run

```bash
python build_clothseg.py    # loads the MIT U2NET cloth weights
# -> clothseg.tflite  (176 MB, [1,3,768,768] -> [1,4,768,768])
```

U²-Net is a pure CNN → fully GPU-compatible with one defensive patch (`align_corners=False`
on the bilinear upsamples): 254/254 nodes on the delegate, device corr 0.999798. ⚠ The
checkpoint has a `module.` prefix — strip it before `load_state_dict` (else random weights
→ garbage output). `argmax` the 4-class output: 0 bg, 1 upper, 2 lower, 3 full body.
