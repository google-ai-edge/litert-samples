# CMGAN → LiteRT conversion

`build_cmgan.py` converts [ruizhecao96/CMGAN](https://github.com/ruizhecao96/CMGAN) (MIT, ckpt
in-repo) to a GPU-native LiteRT `.tflite` with
[litert-torch](https://github.com/google-ai-edge/LiteRT-Torch), with parity gates at every stage:
re-authored-torch vs original-torch (corr ≥ 0.9999 on a real noisy chunk) → fp32 tflite op-check
(banned ops NONE, ≤4D) + CPU parity → fp16 (FLOAT_CASTING) parity → device fixtures.

`gpu_cmgan.py` holds the numerically-equivalent re-authoring (in-graph hamming-DFT STFT +
`exp/ln` power compression, algebraic phase cancellation, Shaw rel-pos constant bake + skew
realignment, batch-1 4D conformer, exact 4D SPConvTranspose chain, safe norms with fp16-safe eps).
`ref_cmgan.py` reproduces the reference enhancement pipeline (RMS normalization, compression,
iSTFT) and writes the parity fixtures.

```bash
git clone https://github.com/ruizhecao96/CMGAN     # code + best_ckpt (MIT)
python ref_cmgan.py                                # reference + fixtures
python build_cmgan.py                              # -> cmgan_fp16.tflite
```
