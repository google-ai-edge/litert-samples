# 3DDFA_V2 → LiteRT conversion

Converts the [3DDFA_V2](https://github.com/cleardusk/3DDFA_V2) (MIT) MobileNetV1 3DMM regressor to
fp16 tflite. The regressor is a pure CNN (crop [1,3,120,120] → 62 params), so it converts through
litert-torch with no re-authoring, GPU-clean.

- `build_tddfa.py` — loads the `mb1_120x120` checkpoint, converts to fp16 tflite, checks parity
  (62-param corr + reconstructed-landmark pixel error), and exports the tiny BFM 68-keypoint bases +
  parameter mean/std as raw f32 bins for the host-side reconstruction.
- `ref_tddfa.py` — the reference pipeline (FaceBoxes detect → regressor → 68 landmarks). Shims a
  pure-python NMS so FaceBoxes runs without building its Cython op.

```bash
pip install torch numpy opencv-python pyyaml
git clone https://github.com/cleardusk/3DDFA_V2   # bundles the weights + BFM configs
python build_tddfa.py            # -> tddfa_mb1_fp16.tflite + tddfa_*.bin
```

Notes for the host code: the model was trained on **cv2 BGR** input; the BFM bases are **interleaved**
`[x0,y0,z0,x1,…]` (the reference reconstructs with `reshape(3,-1, order='F')`); the 62 parameters are
de-normalized with the exported mean/std before parsing R / offset / α_shp / α_exp.
