# Converting the zero-shot image encoders to LiteRT

Reproduces the two image towers this sample runs on device. Each script loads the timm model, re-authors the GPU-hostile blocks (numerically verbatim — weights are copied, eager parity asserted `corr > 0.9999` before convert), converts with `litert-torch`, op-checks the graph for GPU-banned ops / `>4D` tensors, and FP16-quantizes.

```bash
pip install litert-torch ai-edge-quantizer torch timm
python convert_pecore.py    # -> out/pe_core_base_224_fp16.tflite   (PE-Core, 1024-d)
python convert_siglip2.py   # -> out/siglip2_base_224_fp16.tflite   (SigLIP 2, 768-d)
```

Each produces an fp32 `*.tflite` and the fp16 `*_fp16.tflite` the app downloads (`download_model.gradle` fetches them from [`litert-community/PE-Core-base-patch16-224`](https://huggingface.co/litert-community/PE-Core-base-patch16-224) and [`litert-community/SigLIP2-base-patch16-224`](https://huggingface.co/litert-community/SigLIP2-base-patch16-224)).

## Why these models need re-authoring

A CLIP-style ViT tower does **not** ride the ML Drift GPU delegate out of the box, and full GPU residency does **not** imply a correct result. The scripts apply four weights-exact rewrites (the first three for residency, the last for numerical correctness — see the sample's top-level README "GPU compatibility notes" for the full discussion):

1. **Fused-qkv → 4D manual attention** — the fused `qkv` reshape emits a 5D head-split the GPU delegate rejects. Decompose into separate q/k/v projections; self-attention runs through `scaled_dot_product_attention` (its lowering keeps the batch-matmul 3D with a materialized transpose, which the delegate accepts).
2. **Interleaved 2D-RoPE → rotate-half** *(PE-Core only)* — the interleaved rotary uses a strided slice that lowers to `GATHER_ND` (banned). Bake an even→odd channel permutation into the q/k weights (preserves `q·k` exactly) and apply the gather-free rotate-half form with constant cos/sin.
3. **Attention-pool single-query attention → broadcast-multiply + reduce-sum** — the pooling query is a constant latent, so a batch-matmul there is `const @ non-const` (rejected at compile, and the const-RHS reorder is mis-computed on device). `(q·k).sum` → softmax → `(attn·v).sum` is exact for `latent_len = 1` and GPU-correct.
4. **Overflow-safe LayerNorm** — the delegate reduces the LayerNorm variance in fp16 even for an fp32 graph; deep-ViT massive activations make `sum((x-mean)²)` exceed the fp16 max (65504), corrupting normalization (output correlation collapses to ~0.28 over 12 blocks while still reporting full residency). Scaling by 1/32 before squaring keeps the sum in range — mathematically identical to `nn.LayerNorm`.

**SigLIP 2** needs the same set **minus** RoPE (it has no rope and no class token): rewrites 1, 3 and 4. The overflow-safe LayerNorm (#4) is a general deep-ViT-on-GPU fix, not model-specific.

## Notes

- **I/O.** Both export `[1,3,224,224]` NCHW float32 → L2-normalized embedding (`[1,1024]` PE-Core, `[1,768]` SigLIP 2). PE-Core normalizes input with CLIP mean/std; SigLIP 2 normalizes to `[-1,1]` (`(x/255-0.5)/0.5`). The app matches this in `ZeroShotImageClassificationHelper`.
- **fp16** (AI Edge Quantizer `FLOAT_CASTING`): half size, native on the GPU delegate, negligible drift vs fp32. Dynamic-range int8 is intentionally not used — it favors the CPU/XNNPACK path, not the ML Drift GPU delegate.
- **Verified on-device (Pixel 8a, Mali-G615):** both compile to full LITERT_CL GPU residency (no CPU fallback) and produce embeddings matching PyTorch — PE-Core ~66 ms, SigLIP 2 ~60 ms / image.
- **Text embeddings.** Only the image encoder is converted. The candidate-label `text_embeddings_*.bin` assets are pre-computed on the host with each model's text encoder (`open_clip`'s `PE-Core-B-16` / `ViT-B-16-SigLIP2`, prompt `"a photo of a {label}"`, L2-normalized) — see the top-level README "Customizing the label set".
