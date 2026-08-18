# Silent-Face Anti-Spoofing (MiniFASNetV2) — conversion

Converts [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)
(`2.7_80x80_MiniFASNetV2`, Apache-2.0) to a LiteRT `CompiledModel`-GPU `.tflite` with litert-torch.

## Setup

```bash
pip install torch litert-torch numpy
git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.git silentface-src
```

## Run

```bash
python build_silentface.py    # uses the bundled Apache-2.0 MiniFASNetV2 weights
# -> silentface.tflite  (1.85 MB, [1,3,80,80] -> [1,3] softmax)
```

MiniFASNetV2 is a pure CNN → fully GPU-compatible with **zero patches** (PReLU lowers to
GPU-clean relu ops): 168/168 nodes on the delegate, device corr 1.0. Input BGR `x/255`, face
crop 80×80. Output softmax: class 1 = live, 0 & 2 = spoof (print / replay).
