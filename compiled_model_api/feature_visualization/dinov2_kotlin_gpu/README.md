# Dense Feature Visualization with LiteRT — DINOv2 (ViT-S/14 on GPU)

An Android sample that runs the self-supervised [DINOv2](https://github.com/facebookresearch/dinov2) ViT-S/14 backbone **fully on the LiteRT `CompiledModel` GPU delegate** and visualizes its **dense patch features** — a top-3 PCA of the tokens mapped to RGB. Semantically similar patches (object parts vs background) land near each other in feature space, so they share a color and the object "pops out" with no labels or segmentation.

## Model

| Graph | In → Out | Delegate (Pixel 8a) |
| :-- | :-- | :--: |
| DINOv2 ViT-S/14 | `image[1,3,448,448]` → `tokens[1,1024,384]` | GPU (864/864 nodes, 1 partition) |

fp16 weights, 45 MB, ~8 ms/inference (32×32 = 1024 patch tokens). Converted with [litert-torch](https://github.com/google-ai-edge/litert) — see [`conversion/`](conversion/).

## Pipeline

```
photo --host: resize-448 + ImageNet-norm (NCHW)--> image[1,3,448,448]
                             DINOv2 (GPU) -> tokens[1024,384]
      --host: top-3 PCA (power iteration) -> per-patch RGB -> 32x32 overlay
```

`Dinov2Features.kt` runs the CompiledModel and computes the PCA on the 384×384 token covariance (power iteration + deflation), which is far smaller than the token Gram matrix.

## GPU re-authoring (proven ViT recipes)

- **4D attention:** the fused-qkv attention is split into q/k/v and reshaped to `[1, heads, N, d]` (≤4D) with a manual `softmax(qkᵀ/√d)·v`; the delegate rejects the native 5D head-split reshape.
- **SafeLayerNorm:** the deviation is scaled by 1/64 before squaring so the per-token sum of squares stays in fp16 range on DINOv2's massive activations, then rescaled (algebraically identical).
- **LayerScale** (`ls1`/`ls2`) is baked into the following projection weights.
- **tanh-GELU** (`0.5x(1+tanh(…))`) — near-exact; the sigmoid-GELU approximation drifts to feature corr 0.968 over 12 blocks, tanh → 0.99999.
- The pos_embed is baked at a fixed 448 grid by timm at model creation, so there is no runtime interpolation (no `GATHER_ND`).

## Verification

- Re-authored torch vs stock timm: corr 0.999992.
- fp16 tflite vs fp32 PyTorch through the CompiledModel API: patch-feature corr > 0.99 (`conversion/validate_dinov2.py`).
- On device (Pixel 8a): **864/864 nodes on the GPU delegate, 1 partition**, ~8 ms; device fp16 patch features vs fp32 corr 0.996. The host PCA (power iteration + deflation) matches a reference SVD (|corr| 1.0 for all 3 components).

## Build & run

```bash
cd android
./gradlew :app:installDebug
```

The 45 MB `dinov2_s_fp16.tflite` is downloaded from Hugging Face ([`litert-community/DINOv2-ViT-S14-LiteRT`](https://huggingface.co/litert-community/DINOv2-ViT-S14-LiteRT)) into the app assets at build time (`app/download_model.gradle`); a sample image is bundled. Pick a photo (or use the sample) to see the image and its DINOv2 feature-PCA side by side.

## License

Model: Apache-2.0 (DINOv2 / Meta).
