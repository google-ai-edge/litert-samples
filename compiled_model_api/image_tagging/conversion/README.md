# RAM++ → LiteRT conversion

Converts [RAM++ (`xinyu1205/recognize-anything-plus-model`)](https://github.com/xinyu1205/recognize-anything)
(Apache-2.0) into four graphs with parity checks:

- `build_swin.py` — re-authors the Swin-L encoder GPU-clean (window-partition decomposed to ≤4D,
  qkv 3D-BMM attention, baked relative-position bias, cyclic-shift as slice+concat, PatchMerging
  without strided slices, tanh-GELU, adaptive SafeLayerNorm) and can emit a per-stage-tapped graph
  for the on-device fp16 bisect.
- `build_hybrid.py` — splits it into `ram_swin_s012_fp16.tflite` (GPU, stages 0-2) and
  `ram_stage3_tail_fp16.tflite` (CPU, the fp16-fragile last stage + norm + projection).
- `build_head.py` — the multi-grained `reweight` (bakes the 4585×51 tag bank once as fp16 →
  `ram_reweight_fp16.tflite`) and the Query2Label `taghead` (`ram_taghead_fp16.tflite`, GPU).
- `ram_load.py` — loads the upstream checkpoint (with small transformers-5.x shims).

```bash
pip install torch timm transformers pillow einops safetensors
# download ram_plus_swin_large_14m.pth from xinyu1205/recognize-anything-plus-model
python build_swin.py convert     # + ram_load.py, needs the checkpoint
python build_hybrid.py           # -> ram_swin_s012_fp16.tflite + ram_stage3_tail_fp16.tflite
python build_head.py convert     # -> ram_taghead_fp16.tflite
python build_head.py reweight    # -> ram_reweight_fp16.tflite
```

The `opcheck` / `to_fp16` helpers (op-compatibility check + FLOAT_CASTING fp16 export) are shared
across the LiteRT-torch conversion samples.
