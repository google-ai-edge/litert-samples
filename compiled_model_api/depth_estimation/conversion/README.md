# Converting the depth models to LiteRT

This sample ships two monocular depth models; each `convert_*` script is
self-contained — it converts with `litert-torch`, prints the op histogram, and
checks numerical fidelity vs the original PyTorch model before FP16-quantizing.

| Model | Script | Output (used by app) | Path |
|---|---|---|---|
| MiDaS v2.1 small | `convert_midas_litert.py` | `midas_small_256_fp16.tflite` | clean, no patches |
| Depth Anything 3 (Small) | `convert_da3_litert.py` | `da3_small_gpu_fp16.tflite` | ViT, re-authored |

---

## MiDaS_small

```bash
pip install litert-torch ai-edge-quantizer torch timm matplotlib pillow
python convert_midas_litert.py out 256
```

Produces:

- `midas_small_256.tflite` — fp32 (~66 MB)
- `midas_small_256_fp16.tflite` — fp16 (~33 MB, **used by the app**)

### Why this model converts cleanly

`MiDaS_small` is the CNN MiDaS (EfficientNet-Lite3 backbone), **not** the DPT/ViT
variants. It has no attention, no `PixelShuffle`, no `Focus`/strided-slice stems,
no `grid_sample`, and no `LayerScale` — i.e. none of the ops that force the
`litert-torch` converter into GPU-hostile primitives (`GATHER_ND`, `>4D`
reshapes, Flex). It therefore lowers entirely to GPU-clean builtins with no
patches:

```
CONV_2D, DEPTHWISE_CONV_2D, ADD, RELU, RESIZE_BILINEAR, RESHAPE
```

### Notes

- **Channel-last I/O** (`to_channel_last_io`): the exported model takes NHWC
  `1x256x256x3`, which removes the input transpose and matches the interleaved
  RGB the Android app writes.
- **fp16** (AI Edge Quantizer `FLOAT_CASTING`): half size, native on the GPU
  delegate, ~0.27 % difference vs fp32. Dynamic-range int8 is intentionally not
  used — it favors the CPU/XNNPACK path, not the ML Drift GPU delegate.
- Verified on-device (Pixel 8a): the fp16 model compiles to **234/234 nodes on
  the LiteRT GPU delegate (LITERT_CL)** with no CPU fallback, ~1–3 ms/inference.
  `RESIZE_BILINEAR align_corners=True` is GPU-supported as-is — no change needed.

---

## Depth Anything 3 (Small)

```bash
# needs the upstream DA3 source on the path:
git clone https://github.com/ByteDance-Seed/Depth-Anything-3 && cd Depth-Anything-3
pip install litert-torch ai-edge-quantizer torch timm safetensors huggingface_hub pillow
python /path/to/convert_da3_litert.py [image] [H] [W]   # H,W default to 896x504
```

Produces:

- `da3_small_gpu.tflite` — fp32
- `da3_small_gpu_fp16.tflite` — fp16 (~55 MB, **used by the app**)

### Why this model needs re-authoring

Unlike MiDaS, DA3 is a **DINOv2 ViT-S + RoPE** backbone with a DPT/DualDPT head and
does **not** ride the GPU delegate out of the box. The script applies exact,
weights-verbatim rewrites — it measures depth `corr` vs the all-stock model after
each, ending at **corr 0.99948** vs the official PyTorch pipeline:

| # | Wall | Rewrite |
|---|---|---|
| 1 | checkpoint key prefix | strip the leading `model.` |
| 2 | RoPE data-dependent `int(max_position)` (aborts `torch.export`) | constant frequency table |
| 3 | fused-qkv 5D head-split (the "C12" wall) | separate q/k/v Linears, manual 4D attention |
| 4 | `LayerScale` MUL mis-lays-out the token dim on GPU (`{1,1,N,C}` vs `{N,1,1,C}`) | fold gamma into the preceding Linear, `ls` → `Identity` |
| 5 | interpolating the constant `pos_embed` emits a runtime-less `RESIZE_BILINEAR` | bicubic-resize and bake `pos_embed` to a constant buffer |
| 6 | `TRANSPOSE_CONV` rejected by the Pixel 8a delegate | `ConvTranspose2d` → zero-stuff nearest-upsample + `Conv2d` (exact ~1e-7) |
| 7 | camera-token in-place index-assign → `SELECT_V2` (broadcast 'else' rejected) | equivalent `torch.cat` |
| 8 | DPT head `align_corners=True` resize (banned) + UV sincos `BROADCAST_TO` | `align_corners=False`; drop the ratio-0.1 UV pos-embed refinement |

### Notes

- **Native aspect matters.** DA3 runs at the image's native aspect; a square
  letterbox drops fidelity to corr ~0.977 (padding leaks through global attention).
  This build is fixed to **896×504** (portrait) — re-convert at your aspect
  (`python convert_da3_litert.py <image> <H> <W>`, multiples of 14) for other shapes.
- **Honest residual: corr 0.99948, not 1.0.** FP16 is not the cause (FP32 ≡ FP16).
  The ~0.05 % is rewrite #8 — the DPT-head `align_corners=True→False` change, forced
  because the GPU delegate bans `align_corners=True` resize. An irreducible mobile-GPU
  constraint, not a bug; structure and edge sharpness are visually identical.
- **Heavy.** DA3-Small is a ViT — ~1.8 s / image on Pixel 8a GPU (on-demand, not
  live). The sample's live path is MiDaS; DA3 is the high-quality on-demand option.
