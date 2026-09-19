# TIPSv2-B/14 DPT Conversion

Model recipes, instructions, scripts, and utilities for converting the TIPSv2-B/14 DPT dense-prediction model to LiteRT.

These scripts convert [google/tipsv2-b14-dpt](https://huggingface.co/google/tipsv2-b14-dpt) (Google DeepMind, CVPR 2026, Apache-2.0) — the TIPSv2 ViT-B/14 vision backbone with its three DPT heads — into a **single `.tflite` graph** that returns metric depth, surface normals and ADE20K semantic segmentation from one 448×448 image, and verify every stage against the official PyTorch model. The published artifact lives at [litert-community/TIPSv2-B14-DPT-LiteRT](https://huggingface.co/litert-community/TIPSv2-B14-DPT-LiteRT).

The backbone is a DINOv2-style ViT-B/14 (86M params, 12 blocks, one register token, LayerScale); the depth and normals heads were trained on NYU Depth V2 and the segmentation head on ADE20K (150 classes), all on the frozen backbone (~72M params for the three heads). The whole model runs as one GPU graph on the LiteRT CompiledModel API — a multi-task dense-prediction model on the mobile GPU.

## Pipeline

```
RGB image ── resize 448×448, /255 (no mean/std) ──> [1,3,448,448] ─> [GPU] ViT-B/14 + 3× DPT ─┬─> depth   [1,1,448,448]   metres (NYU scale, 0.001–10 m)
                                                                                                ├─> normals [1,3,448,448]   unit vectors
                                                                                                └─> seg     [1,150,256,256] ADE20K logits → argmax (host)
```

- **input**: RGB in **[0, 1]**, NCHW — TIPSv2 does not apply ImageNet normalization. The 448 input matches the native 32×32 `pos_embed` grid, so there is no runtime interpolation.
- **segmentation** comes out at the DPT head's native 256×256 grid; `argmax` over the 150 channels on the host, then upscale (the official pipeline bilinearly upsamples the logits to 448 first — argmax agreement between the two is 99.96 %).

## Environment

```bash
pip install litert-torch ai-edge-litert ai-edge-quantizer
pip install torch numpy pillow transformers safetensors huggingface_hub
```

Verified with `litert-torch==0.10.0`, `ai-edge-litert==2.1.5`, `ai-edge-quantizer==0.7.0`, `torch==2.12.1`, `transformers==5.8.0`. The weights (`model.safetensors`, backbone + heads) are fetched from `google/tipsv2-b14-dpt` automatically; the `ref` stage loads the official model through `transformers` (`trust_remote_code=True`) to produce the parity reference.

## Run

```bash
KMP_DUPLICATE_LIB_OK=TRUE JAX_PLATFORMS=cpu \
    python build_tipsv2_b14_dpt.py all path/to/image.jpg   # ref + parity + convert + fp16
python verify_tipsv2_b14_dpt.py tipsv2_b14_dpt_fp16.tflite path/to/image.jpg
```

`build_tipsv2_b14_dpt.py all` emits `tipsv2_b14_dpt.tflite` (fp32, 633 MB) and `tipsv2_b14_dpt_fp16.tflite` (318 MB) — the deployment file published on Hugging Face — and prints the parity numbers at every stage. Any photo works as the test image.

## Files

| File | What |
|---|---|
| `build_tipsv2_b14_dpt.py` | the re-authoring recipe: official reference -> GPU-clean re-author -> parity -> litert-torch convert -> op-check -> fp16, each stage verified against the reference. |
| `verify_tipsv2_b14_dpt.py` | end-to-end check of a converted `.tflite` through the CompiledModel API: image -> depth / normals / segmentation panel png, plus correlation against the saved reference. |

## Re-authoring → GPU-clean

The model is rebuilt from the HF weights with exact rewrites (the only approximation is tanh-GELU). The converted fp32 graph matches the re-authored torch model exactly; vs the **official** model: depth corr 0.999998, normals corr 0.999999, seg argmax agreement 99.96 %.

| construct | rewrite |
|---|---|
| fused-qkv attention, 5D head split | q/k/v split + `[1, heads, N, d]` manual `softmax(qkᵀ/√d)·v` (the delegate rejects the 5D reshape) |
| LayerScale `ls1`/`ls2` | γ baked into `attn.proj` / `mlp.fc2` weights and biases |
| `nn.LayerNorm` | SafeLayerNorm — deviation pre-scaled by 1/64 before squaring so the fp16 variance cannot overflow on the ViT's large activations (algebraically identical) |
| exact GELU | tanh-GELU (no `ERF` kernel on the GPU) |
| DPT readout `cat(patch, cls.expand) @ W` | `patch @ W_a + cls @ W_b` (exact; the `expand` emits `BROADCAST_TO`) |
| `ConvTranspose2d(k=s, stride=s)` (reassemble ×4 / ×2) | nearest-up × constant zero-stuff mask + `Conv2d(flipped)` (exact; Pixel 8a rejects `TRANSPOSE_CONV`) |
| fusion-block bilinear ×2 with `align_corners=True` | two constant-RHS matmuls `U·X·Uᵀ` (exact; the GPU bans `align_corners=True` resize — flipping the flag, as other DPT ports do, costs accuracy) |
| **depth decoder fp16 range** | activations reach ~1e8 at the logits (fp16 max 65504) — on the GPU the depth came out constant while normals/seg were fine. The decoder after the readout GELU is affine + ReLU + residual adds ending in `relu(l)/Σrelu(l)` (scale-invariant), so power-of-2 scales are folded in: per-level convs ×1/4096…1/64, each fusion `out_conv` ×1/4, `project` ×1/32, head ×1/16 — weights carry the local factor, biases the absolute running scale, residual adds see matching scales. Bit-exact in fp32 (max diff 0.0 vs the unscaled head); every stage ≲100 on device. |

## Verification results

- Re-running this script reproduces the published fp16 file **bit-exactly** (sha256 match; litert-torch 0.10.0).
- Re-authored torch vs official PyTorch: depth corr 0.999998 (max 5 mm on a 1.2–2.8 m scene), normals corr 0.999999, seg argmax agreement 99.96 %.
- fp32 tflite vs official: identical to the torch numbers; op-check GPU-clean (no banned ops, no >4D tensors, no FFT family; 1434 ops).
- fp16 tflite vs official (CPU): depth corr 0.999998, normals 0.999999, seg argmax 99.97 %.
- On device (Pixel 8a, Tensor G3, CompiledModel GPU): **1434/1434 nodes on `LITERT_CL`, 1 partition**; ~0.9 s per image for all three heads (compile + load ≈ 5 s). Device fp16 vs desktop fp32: depth corr 0.99986, normals corr 0.99990, seg argmax agreement 99.3 %.

## One on-device finding (general)

A head that passes the desktop op-check and compiles on the GPU can still return a **constant** output with no NaN and no error when its activations exceed the fp16 range — here the depth decoder (~1e8) while the other two heads (≲40) were fine. Check the per-stage maxima of each head with forward hooks before blaming the converter; when the chain is positively homogeneous (affine + ReLU) and ends in something scale-invariant, the fix is model-side and exact (fold power-of-2 scales as above).
