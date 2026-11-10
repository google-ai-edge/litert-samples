# RF-DETR-Seg Nano Conversion

Model recipes, instructions, scripts, and utilities for converting RF-DETR-Seg Nano instance segmentation to LiteRT.

These scripts convert [roboflow/rf-detr](https://github.com/roboflow/rf-detr) `RFDETRSegNano` (tag 1.9.3, Apache-2.0; 33.6M params, COCO seg AP<sub>50</sub> 63.0) into **two `.tflite` graphs that both run 100% on the LiteRT CompiledModel GPU**, and numerically verify every step against the PyTorch reference. The published artifacts live at [litert-community/RF-DETR-Seg-Nano-LiteRT](https://huggingface.co/litert-community/RF-DETR-Seg-Nano-LiteRT).

The model is a DETR-family instance segmenter: DINOv2-S/12 backbone (312×312, all-global attention), single-level deformable-attention decoder (4 layers), and a ConvNeXt-style mask head that produces a full-image 78×78 mask per query. The two-stage query selection (`TOPK` + `GATHER`) has no GPU op, so the model splits at exactly that point; the host step between the graphs is pure elementwise math plus a topk.

## Pipeline

```
image [1,3,312,312] + clspos [1,1,384] + pospatch [1,676,384]
  ──> [GPU Graph A] ──> enc_class [1,676,91], enc_delta [1,676,4], memory×2 [1,676,256]
  ──> host: ÷2 → proposal-grid combine → top-100 by max class score
            → gather → reparam(refpoint_embed) ──> refpoint [1,100,4]
  ──> [GPU Graph B] (memory, refpoint, query_feat [1,100,256])
  ──> boxes [1,100,4] (cxcywh), logits [1,100,91], masks [1,100,78,78]
  ──> host: sigmoid + threshold + per-class NMS ──> instances (mask inside = logit > 0)
```

- **Input**: square resize to 312×312, RGB, ImageNet mean/std, NCHW, plus the two constant embedding inputs (`clspos.bin`, `pospatch.bin` — see below for why they are inputs).
- **Proposal grid**: image-independent (26×26, `cxcy = (grid+0.5)/26`, `wh = 0.05`) — regenerated on the host from the formula, no artifact needed.
- **Output**: `boxes` cxcywh normalized to [0,1]; `logits` 91-way (index = COCO category id, id 0 unused); `masks` per-query full-image 78×78 raw logits, upsampled bilinearly to the frame.

## ⭐ The baked-constant execution bug (why three tensors are graph inputs)

The ML Drift GPU delegate **silently mis-executes compute chains that consume large baked-constant tensors — the same graph with the constant fed as a runtime input is exact.** This is not a precision issue (fp32 and fp16 flatbuffers return identical wrong numbers) and it is compilation-context dependent (a subgraph can probe clean in isolation and still break in the full build). Minimal pair: `tgt + FFN(tgt)` with `tgt` a baked `[1,100,256]` weight → device corr 0.966; identical ops with `tgt` as a graph input → 0.99999.

Three instances in this one model, all fixed exactly:

| site | symptom (device corr) | fix |
|---|---|---|
| DINOv2 LayerScale `h + λ·f(h)` | 0.62 | fold λ into the preceding Linear (exact) |
| decoder `tgt` = baked `query_feat.weight` | 0.97 | feed `query_feat` as an input; move the reparam combine (baked `refpoint_embed`) to the host |
| ViT pos-embed add `patches + POS[1,677,384]` | 0.55–0.99, varies per build | host-feed `clspos` (cls token + pos) and `pospatch` as inputs |

Graph A additionally emits `memory×2`: a `[1,N,C]` tensor that is both consumed in-graph and a graph output comes back zeroed on the delegate; ×2 forces a separate buffer (exact in fp16) and the host halves it.

## Re-authoring → GPU-clean

Converted with **litert-torch** (NCHW preserved). Both graphs pass the static op-check (no banned ops, no >4D tensors) and reproduce the patched PyTorch model:

| construct | rewrite |
|---|---|
| SDPA / `nn.MultiheadAttention` | manual **rank-4** head-split matmuls (the delegate mis-executes rank-3 batched matmuls; MHA also lowers to rank-3) |
| deformable `grid_sample` | GATHER/CAST-free **tent-matmul** bilinear sampler: weights `relu(1-|i-p|)` × rank-4 BMM, exact incl. zeros-padding OOB |
| `MSDeformAttn` (6D sampling tensors) | re-authored ≤4D for n_levels=1 with the tent-matmul sampler |
| `nn.LayerNorm` (fp16 overflow on device) | **SafeLayerNorm v2**: adaptive per-row down-scale `S = max(1, amax/8)`, never reconstructs the large variance — every intermediate stays O(amax) |
| channels-first LayerNorms (projector, mask-head blocks) | same math via a `[B,HW,C]` **3D detour** (litert-torch's NHWC layout pass cannot rewrite `amax` on layout-tracked 4D tensors) |
| GELU | tanh-GELU (ERF has no GPU lowering) |
| sine pos-embed | `dim_t` baked (no POW/FLOOR_DIV); strided interleave → reshape+stack (no GATHER_ND) |
| seg-head einsum `qc,chw→qhw` | rank-4 matmul |
| two-stage topk/gather + reparam | host (the split above) |

Desktop parity (random input, fp32 tflite vs patched torch): Graph A corr ≥ 0.9995, Graph B corr ≥ 0.9986, E2E chain vs the unsplit `forward_export` reference corr ≥ 0.998 — residuals are fp16-scale rounding.

## Environment

```bash
pip install litert-torch ai-edge-litert ai-edge-quantizer
pip install torch torchvision supervision pyDeprecate transformers
pip install rfdetr==1.9.3 --no-deps    # weights auto-download on first use
```

Verified with `litert-torch==0.10.0`, `ai-edge-litert==2.1.5`, `torch==2.12.1`, `torchvision==0.27.1`, rfdetr tag 1.9.3. `rfdetr` is installed `--no-deps` to keep the pinned torch (its requirements pin an older one); `tfm_compat.py` shims the transformers 4.57↔5.x API moves and is a no-op on ≥5.1.

## Run

```bash
python build_rf_detr_seg_nano.py all       # patches + convert + parity + fp16 + artifacts
python verify_rf_detr_seg_nano.py photo.jpg 0.5
```

`build_rf_detr_seg_nano.py all` emits `rfdetrseg_graphA{,_fp16}.tflite`, `rfdetrseg_graphB{,_fp16}.tflite` and the four host-side `.bin` constants, printing parity numbers at every stage. The fp16 graphs (47.0 MB + 14.8 MB) are the deployment files published on Hugging Face. `verify_rf_detr_seg_nano.py` then runs a real photo end-to-end through the CompiledModel API and, when `rfdetr` is installed, reports per-detection box/mask IoU and class agreement against the official PyTorch model.

## Files

| File | What |
|---|---|
| `build_rf_detr_seg_nano.py` | the re-authoring recipe: GPU patches → 2-graph split → litert-torch convert → op-check → desktop parity → fp16 → host-constant artifacts |
| `verify_rf_detr_seg_nano.py` | real-image end-to-end check through the CompiledModel API (+ box/mask IoU vs the PyTorch reference when `rfdetr` is installed) |
| `tfm_compat.py` | transformers 4.57↔5.x compat shims for rfdetr 1.9.x (no-op on ≥5.1) |

## On-device (Pixel 8a, Tensor G3)

| graph | nodes on GPU | time |
|---|---|---|
| Graph A — backbone + encoder + proposal heads | `1293/1293` LITERT_CL, 1 partition | 17.5 ms |
| Graph B — decoder + heads + mask head | `884/884` LITERT_CL, 1 partition | 9.1 ms |

Real-image device chain vs the official PyTorch `RFDETRSegNano` (threshold 0.5): every detection matches with **box IoU ≥ 0.99, mask IoU ≥ 0.995 and identical classes** on the test images.
