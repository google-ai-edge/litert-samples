# RF-DETR Nano → LiteRT conversion (2-graph split)

Produces the two GPU graphs used by the RF-DETR sample — `rfdetr_graphA_fp16.tflite` and
`rfdetr_graphB_fp16.tflite` — from [RF-DETR](https://github.com/roboflow/rf-detr) Nano, with
[litert-torch](https://github.com/google-ai-edge/litert).

## Environment

```bash
pip install rfdetr litert-torch ai-edge-litert ai-edge-quantizer torch numpy pillow
```

`rfdetr` downloads the `rf-detr-nano.pth` checkpoint on first use.

## Files

- `build_rfdetr_bb.py` — backbone class patches (windowed DINOv2: 6D window-partition → ≤4D, SDPA →
  manual attention, baked pos-encoding, tanh-GELU, fp16-safe projector LayerNorm).
- `build_rfdetr_full.py` — full-model patches (deformable `grid_sample` → tent-matmul, MSDeformAttn ≤4D,
  baked sine pos-embed, torch.export friction fixes, adaptive fp16-safe `nn.LayerNorm`). Imports
  `build_rfdetr_bb`.
- `build_rfdetr_split.py` — **the entry point**. Imports `build_rfdetr_full` (which applies every patch),
  builds **Graph A** and **Graph B**, op-checks both (GPU-clean: no banned ops, no >4D tensors), validates
  per-graph and end-to-end correlation vs PyTorch, writes the fp16 tflites + device-probe fixtures.

## Run

```bash
python build_rfdetr_split.py all      # convert both graphs, op-check, fp16, parity vs torch
```

Outputs `rfA_fp16.tflite` / `rfB_fp16.tflite` (rename to `rfdetr_graphA_fp16.tflite` /
`rfdetr_graphB_fp16.tflite` for the app) plus `.bin`/`.npy` fixtures for an on-device probe.

## Why two graphs

RF-DETR is a two-stage DETR; the query selection (top-300 proposals by class score) is `TOPK_V2` +
`GATHER`, which have no GPU op. The proposal **grid is image-independent**, so the model splits at exactly
that point: **Graph A** emits `enc_class`/`enc_coord`/`memory`, the host does top-300 + a coord gather, and
**Graph B** runs the decoder on `(memory, refpoint_ts)`. `memory_ts`/`boxes_ts` (the enc auxiliary outputs)
are dead at inference — the decoder query is a learned embedding and the top-k only feeds the reference
points — so the host step is a topk + coord-gather only.

## fp16 hardening

The Mali delegate computes in fp16 regardless of model dtype. Two LayerNorm sites overflow it and are
replaced with a down-scaled, scale-invariant **SafeLayerNorm** (numerically exact): the projector
(`ConvX` outputs reach |x|≈440) and the decoder (layer-0 self-attention output |x|≈1068). The decoder
version is **adaptive** (`S = max(1, amax/8)` per row) — a fixed down-scale would squash the
small-magnitude norms. See the comments in `build_rfdetr_split.py`.
