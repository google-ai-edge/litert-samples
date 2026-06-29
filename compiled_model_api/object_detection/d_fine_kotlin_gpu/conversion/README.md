# D-FINE-S → LiteRT conversion (2-graph split)

Produces the two GPU graphs used by the D-FINE sample — `dfine_graphA_fp16.tflite` and
`dfine_graphB_fp16.tflite` — plus the host-tail weights `host_params.bin`, from
[D-FINE](https://github.com/Peterande/D-FINE) (`ustc-community/dfine-small-coco`) with
[litert-torch](https://github.com/google-ai-edge/litert).

## Environment

```bash
pip install transformers litert-torch ai-edge-litert ai-edge-quantizer torch numpy pillow
```

`transformers` downloads the `ustc-community/dfine-small-coco` weights on first use.

## Files

- `build_dfine_split.py` — the GPU re-authoring library + **Graph B** (the FDR deformable decoder).
  Monkeypatches: deformable `grid_sample` → tent-matmul, multi-level MSDeformAttn ≤4D, the FDR LQE
  `prob.topk` → an iterative max-and-mask (3D), `distance2bbox` `stack`→`cat`, SafeLayerNorm, tanh-GELU,
  `inverse_sigmoid` `clamp(0,1)` drop, AIFI sine-embed bake, anchors `finfo.max` → ±1e4 clamp.
- `build_dfine_fix3.py` — **the device-correct Graph A** and the host-side selection. Imports
  `build_dfine_split`. Graph A emits only the two fp16-clean leaves (`enc_class` + `memory_raw*2`); the
  per-token tail (`enc_output` + `enc_bbox_head`) is computed on the host over the 300 topk-selected
  tokens. Saves `dfine_host3_w.npz` (the host-tail weights).
- `pack_assets.py` — renames the two tflites to the app names and writes the bundled
  `host_params.bin` (host-tail weights, fixed layout) + `coco_labels.txt` (80 contiguous classes).

## Run

```bash
python build_dfine_split.py fp16     # -> dfB_fp16.tflite   (Graph B), op-check + parity vs torch
python build_dfine_fix3.py  fp16     # -> dfA_fix3_fp16.tflite + dfine_host3_w.npz (Graph A), parity vs torch
python pack_assets.py . ..            # rename tflites + write android/.../assets/{host_params.bin,coco_labels.txt}
./../android/install_to_device.sh .   # push dfine_graphA/B_fp16.tflite to the device
```

## Why two graphs

D-FINE is a two-stage DETR; the query selection (top-300 proposals by class score) is `TOPK_V2` +
`GATHER`, which have no GPU op. The proposal **grid is image-independent**, so the model splits at exactly
that point: **Graph A** emits `enc_class` + `memory_raw`, the host does top-300 + the per-token tail, and
**Graph B** runs the plain decoder on `(memory_raw, target, ref)`.

## The on-device gate — a Mali 3D-sequence fan-out bug (not an fp16 wall)

Both graphs convert GPU-clean and are fully `LITERT_CL`-resident, but a naïve Graph A (emitting
`enc_class`/`enc_coord`/`output_memory`/`memory_raw` together) silently produced wrong boxes on device —
large objects vanished while small ones stayed perfect. The cause is a Mali delegate bug: a **3D token
tensor `[1,N,256]`** (from `conv.flatten(2).transpose(1,2)`) that is both a graph output and consumed by
another node — or that fans out to several consumers — gets clobbered on the longer branch. (4D conv-map
outputs with the same fan-out are fine.) Here `output_memory` fed both heads; the 3-layer box head lost,
so its reference-box deltas collapsed to ~0. The corruption was masked in correlation by the ±1e4 baked
anchors ("corr 1.0" while the real valid-row corr was 0.88).

**Fix** (`build_dfine_fix3.py`): emit only the two fp16-clean leaves (`enc_class` survives; `memory_raw`
is scaled ×2 to force a separate output buffer, undone on the host) and move the per-token tail
(`enc_output` + `enc_bbox_head`) to the host in fp32 over the 300 selected tokens — exact, because
per-token ops commute with the gather (`gather(f(x),i) = f(gather(x,i))`). Device output then matches
PyTorch (COCO val giraffe 7/7, cats 6/6, IoU 0.98–1.00).
