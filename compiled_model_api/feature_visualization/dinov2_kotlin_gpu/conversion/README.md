# DINOv2 ViT-S/14 → LiteRT conversion

Converts the Apache-2.0 [DINOv2](https://github.com/facebookresearch/dinov2) ViT-S/14 backbone (via `timm`) into a graph that runs **entirely on the LiteRT `CompiledModel` GPU delegate**, emitting the dense patch tokens, then validates it against the fp32 PyTorch reference.

## GPU re-authoring (proven ViT recipes)

- **4D attention:** the fused-qkv attention is split into q/k/v and reshaped to `[1, heads, N, d]` (≤4D) with a manual `softmax(qkᵀ/√d)·v`. The native head-split reshapes to 5-D, which the delegate rejects.
- **SafeLayerNorm:** the deviation is scaled by 1/64 before squaring so the per-token sum of squares stays within fp16 range on DINOv2's massive activations, then rescaled — mathematically identical to the plain variance. Without it the deep-ViT variance overflows the fp16 accumulator on the delegate.
- **LayerScale** (`ls1`/`ls2`) is baked into the following projection weights.
- **tanh-GELU** (`0.5x(1+tanh(0.79788(x+0.044715x³)))`) — near-exact and delegate-friendly. The cheaper sigmoid-GELU approximation drifts to feature corr 0.968 over 12 blocks; the tanh form gives 0.99999.
- The pos_embed is baked at a fixed 448 grid by timm at model creation, so there is no runtime interpolation (a bicubic `F.interpolate` of the pos_embed otherwise lowers to `GATHER_ND`).

## Files

- `build_dinov2.py` — re-authors DINOv2 from the timm weights, parity vs stock timm, litert-torch conversion, fp16 cast (`ai_edge_quantizer` FLOAT_CASTING).
- `validate_dinov2.py` — runs the fp16 tflite through the **CompiledModel API** and checks its patch features against fp32 PyTorch.

## Run

```bash
# deps: torch, timm, numpy, pillow, litert-torch, ai-edge-quantizer, ai-edge-litert
# input: test.jpg next to the scripts (timm downloads the DINOv2 weights)
python build_dinov2.py all      # parity -> convert -> fp16
python validate_dinov2.py       # CompiledModel vs fp32 torch (patch-feature corr)
```

Expected: parity corr ≈ 0.99999 with stock timm; validate patch-feature corr > 0.99. On a Pixel 8a the fp16 model runs 864/864 nodes on the GPU delegate (1 partition, ~8 ms), device features vs fp32 corr 0.996.
