# TIGER-DnR → LiteRT conversion

Converts [JusperLee/TIGER-DnR](https://huggingface.co/JusperLee/TIGER-DnR) (3 sibling TIGER graphs:
dialog / effect / music) to GPU-native LiteRT `.tflite` with
[litert-torch](https://github.com/google-ai-edge/LiteRT-Torch), fp16.

```bash
git clone https://github.com/JusperLee/TIGER          # model code (MIT)
hf download JusperLee/TIGER-DnR --local-dir ckpt-dnr  # weights (Apache-2.0)
python build_tiger.py dialog                          # + effect, music
```

`build_tiger.py` verifies each stage: re-authored-torch vs original-torch (waveform corr ≥ 0.9999 on
a real 12 s mixture) → fp32 tflite op-check (banned ops NONE, all tensors ≤4D) + CPU parity → fp16
(FLOAT_CASTING) parity, and writes `tiger_<stem>_fp16.tflite` + device fixtures.

`gpu_tiger.py` holds the numerically-equivalent re-authoring (see the sample README for the list:
in-graph DFT-conv STFT, folded-batch→4D Conv2d, per-position SafeNorm, uniform pool/resize chain at
T=1040, one-hot FC band resizes, per-head 3D-BMM attention, PReLU/6-D-mask rewrites, fp16-safe eps,
broadcast-free mask head). `_stub_tiger.py` shims optional deps (`torch_complex`, `typeguard`,
`distutils`) so the upstream package imports without them.
