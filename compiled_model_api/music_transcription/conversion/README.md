# Basic Pitch → LiteRT conversion

Rebuilds Spotify's basic-pitch `nmp` model as a GPU-native LiteRT `.tflite` **from the official ONNX artifact's constants** (no TensorFlow required):

- `extract_weights.py` — dumps all 102 initializers from `nmp.onnx` to `bp_weights.npz`.
- `gpu_bp.py` — torch re-implementation (bit-exact vs the official ONNX: corr 1.000000 on all three outputs): 9-octave conv-CQT with the per-bin norm folded into per-octave kernel copies, reflect-pad as an anti-diagonal-constant matmul, fp16-safe post-log clamp in NormalizedLog, harmonic stacking as static slices, and the 3-branch CNN.
- `build_bp.py` — parity gates (torch vs official ONNX ≥0.9999 per output) → litert-torch convert → op-check (banned NONE, ≤4D) → fp32 tflite parity (1.0) → device fixtures.

```bash
curl -LO https://github.com/spotify/basic-pitch/raw/main/basic_pitch/saved_models/icassp_2022/nmp.onnx
mv nmp.onnx nmp_official.onnx
python extract_weights.py
python build_bp.py            # -> basicpitch.tflite (fp32, 0.84 MB)
```
