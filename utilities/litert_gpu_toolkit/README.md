# litert_gpu_toolkit

Pre-conversion patches that rewrite common PyTorch patterns into forms the
LiteRT GPU delegate accepts, plus a post-conversion checker.

Every entry below came out of converting a real model, hitting a wall on
device, and finding the rewrite that clears it. They were carried per-script
until now; this is the shared copy.

## Use

```python
from litert_gpu_toolkit import convert_for_gpu

path = convert_for_gpu(
    model,                                    # nn.Module, will be set to eval()
    dummy_input=torch.randn(1, 3, 1024, 1024),
    output_path="model.tflite",
)
```

`convert_for_gpu` applies the general patches, converts via litert-torch, and
runs `check_gpu_compatibility` on the result. The patches are also importable
individually from `litert_gpu_toolkit.patches` when a model only needs one.

`check_gpu_compatibility(tflite_path)` verifies through the LiteRT
CompiledModel API: it compiles the model for the GPU accelerator, runs every
signature on random inputs, and compares the outputs against a CPU-compiled
reference (`rtol`/`atol` default to 1e-2 — fp16 accumulation on GPU makes
bit-exactness unrealistic). A static op scan (suspect ops, Flex ops, op
distribution, rank-5+ tensors) is included as advisory diagnostics to point
at the right patch when compilation fails, but the verdict comes from the
CompiledModel run. Use it as a gate before spending a device run — it
exercises the host GPU, so still compare on-device output against CPU before
shipping.

## What each patch is for

| Patch | Pattern it rewrites | Why |
|---|---|---|
| `patch_safe_layernorm` | LayerNorm variance | The delegate reduces `Σ(x−mean)²` in fp16 even for an fp32 graph. On deep ViTs and on deep-residual CNNs the activations get large enough to overflow fp16, and the error compounds with depth while the model still reports full delegation. The fix reduces in a down-scaled domain `x/S`, which LayerNorm's scale-invariance makes safe. Three modes: `adaptive_v2` (default, per-row `S = max(1, amax/8)`, stays in the scaled domain — cheapest, but eps then acts at the scaled magnitude, so it is not bit-faithful to stock LayerNorm), `adaptive` (same `S`, rescales before the rsqrt — bit-faithful), `fixed` (constant `S`) |
| `patch_rmsnorm` / `safe_rms` | RMSNorm | Same overflow, different symptom: `Σx²` overflows, `norm` becomes `inf`, and the whole head outputs exactly zero |
| `patch_instance_norm` / `SafeInstanceNorm2d` | InstanceNorm | Same class |
| `hierarchical_mean` | `x.mean((2, 3))` | A single global reduction over a large map overflows the fp16 accumulator. Replaced by a cascade of ÷2 average-pools, so each stage averages at most four values, and it traces to a static chain of `AVERAGE_POOL_2D`. **Exact only for power-of-two spatial dims** — with odd extents the `ceil_mode` edge windows average fewer elements (max error ~0.2 measured on a 37×53 map). Pad first, or keep the map pow2 |
| `ZeroStuffConvT1d` / `2d`, `patch_conv_transpose`, `pixelshuffle_to_conv_transpose` | `ConvTranspose`, `PixelShuffle` | `PixelShuffle` lowers through a rank-6 reshape; `TRANSPOSE_CONV` is emitted at a version the delegate does not accept. Zero-stuff plus a plain conv is exact (~1e-7) and stays rank 4 |
| `patch_grid_sample` | `F.grid_sample` | Lowers to `GATHER_ND`, which the delegate rejects. Rewritten as a bilinear tent-matmul, exact against `F.grid_sample` including zeros-padding out of bounds (error ~2e-7), all rank ≤ 4. Cost is O(HW²), so it suits sparse sampling — it is what carries RF-DETR Nano's deformable cross-attention — not dense warps |
| `patch_window_attention`, `patch_patch_merging`, `patch_einops` | Windowed attention, patch merging, `einops.rearrange` | These build rank-5/6 tensors, or stride-2 slices that become `GATHER_ND` |
| `patch_maxpool_zeropad` / `ZeroPadMaxPool` | `MaxPool` padding | Padding lowers to `GATHER_ND` |
| `patch_groupnorm` / `ManualGroupNorm` | `GroupNorm` | Manual 4-D form |
| `patch_normalize` | `F.normalize` | The div-broadcast form fails on the delegate |
| `patch_interpolate` | `F.interpolate(align_corners=True)`, bicubic | The delegate bans half-pixel with align_corners; bicubic becomes `GATHER_ND` |
| `patch_weight_standardization` | `Conv2d` weight standardization | Bake the standardized weights instead of normalizing at runtime |
| `patch_gelu`, `patch_swish` | GELU, Swish | Defensive, and off by default in `convert_for_gpu`'s general set where the native op is already correct. Note that an approximation choice can matter: a wide-output-range regression head needs the tanh form, not the sigmoid one |

## Verification

The patches are device-derived, not theoretical — the rewrites here are the
ones that took specific models to full GPU residency with correct output on a
Pixel 8a (for example a CLIP ViT-B/32 at 691/691 nodes, CPU-identical).

Two cautions that apply to all of them:

- **Full residency does not mean correct output.** Several of these patches
  exist for bugs where the delegate reports `N/N` nodes and still returns wrong
  numbers. Always compare the on-device output against CPU or the source model.
- The rewrites are numerically exact by construction where stated (zero-stuff
  conv, scale-before-square) and approximate where stated (grid_sample, the
  GELU/Swish substitutions). Check the table before assuming.

## Requirements

`torch`, `litert-torch`, and `ai-edge-litert` (the checker verifies through
the CompiledModel API; its advisory op scan uses the LiteRT interpreter,
falling back to `tensorflow` if installed).
