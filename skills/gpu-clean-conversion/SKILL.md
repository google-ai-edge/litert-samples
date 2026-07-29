---
name: gpu-clean-conversion
description: Convert a PyTorch or Hugging Face model into a LiteRT model that runs fully on the GPU delegate with verified-correct output, and lay it out as a model recipe. Use when converting a new model, or when a converted model is rejected by the delegate, falls back to CPU, or returns wrong numbers on device.
---

# GPU-clean conversion

A conversion is done when three things hold, in this order:

1. it converts,
2. every node runs on the GPU delegate,
3. **the on-device output matches the source model.**

Step 3 is not implied by step 2. The delegate can report full residency and
still return wrong numbers — that is the failure mode most of the rewrites
below exist for. Never call a model done without a numerical check against
CPU or PyTorch on the actual device.

## Loop

**1. Convert plain first.** No patches. This tells you what the model
actually needs rather than what you assumed.

**2. Check the flatbuffer before touching a device.**

```python
from litert_gpu_toolkit import check_gpu_compatibility, print_report
print_report(check_gpu_compatibility("model.tflite"))
```

It reports GPU-incompatible ops, delegate-only ops, Flex ops, and tensors
above rank 4. Fix everything it lists before spending a device run.

**3. Map each symptom to a rewrite.** From `utilities/litert_gpu_toolkit`:

| What you see | Rewrite |
|---|---|
| `GATHER_ND` in the op list | Find its source. Stride-2 slicing (`x[:, ::2]`, Focus stems, patch merging), `grid_sample`, `MaxPool` padding, bicubic `interpolate`, and reflect-mode `F.pad` all lower to it. `patch_grid_sample`, `patch_maxpool_zeropad`, `patch_interpolate`, `patch_patch_merging` |
| Tensors above rank 4 | `PixelShuffle` (rank-6 reshape), windowed attention, `einops.rearrange`, packed-QKV attention head splits. `pixelshuffle_to_conv_transpose`, `patch_window_attention`, `patch_einops` |
| `TRANSPOSE_CONV` rejected | Version skew, not a missing op. `ZeroStuffConvT1d` / `ZeroStuffConvT2d` — zero-stuff plus a plain conv, exact to ~1e-7 |
| `SELECT` / `SELECT_V2` | `PReLU`, `ELU`, in-place index assignment, and `torch.where` masking. Replace with arithmetic: `x*(1-m) + v*m` |
| `BROADCAST_TO` | An outer product or `.expand` on compile-time constants that did not fold. Bake the result as a constant at its target shape |
| Compiles, runs, output is wrong or NaN | The fp16 reduction family. `patch_safe_layernorm`, `patch_rmsnorm`, `patch_instance_norm`, `hierarchical_mean`. Two caveats worth knowing before you apply them: `patch_safe_layernorm`'s default mode is not bit-faithful to stock LayerNorm (use `scale="adaptive"` when you need that), and `hierarchical_mean` is exact only for power-of-two spatial dims |
| Head outputs exactly zero | RMSNorm `Σx²` overflowed fp16 to `inf`. `patch_rmsnorm` |

**4. Re-convert and re-check.** Repeat 2–3 until the checker is clean.

**5. Verify on device, numerically.** Run the same input through the source
model and the on-device graph and compare. Correlation on the output tensor,
plus a task-level check where one exists (argmax match, IoU, mask foreground
count). Record the device and the residency line (`N/N` nodes) alongside the
number — a result without both is not reproducible.

If residency is full and the output is wrong, bisect by intermediate: dump
tensors at block boundaries and find the first one that diverges. The cause is
usually a reduction, not the op you suspect.

## Watch for

- **fp16 is used even for an fp32 graph.** The delegate reduces in fp16
  regardless of tensor dtype. Anything that sums many large values — variance,
  `Σx²`, multi-axis mean — is a candidate.
- **Approximation choices are not free.** Substituting a GELU or Swish
  approximation changes numerics. A head with a wide output range can lose
  real accuracy to the coarser form.
- **`inplace=True` activations mutate a residual.** `out = self.act(x)` then
  `x + out` adds `relu(x)`, not `x`. Swapping the activation silently changes
  the math.
- **A silent CPU fallback looks like success.** If the GPU output is
  bit-identical to CPU fp32, it probably did not run on the GPU. Genuine fp16
  execution drifts in the last digits.

## Output layout

Ship the result as a model recipe, not a one-off script:

```
models/<family>/<model>/
  README.md                 what the model is, what was produced
  converted/
    README.md               environments, pipeline, verification results
    export_*.py             one script per graph
    dump_*_ref.py           reference dumps from the source model
    verify_*.py             parity check against those references
```

Keep the export, the reference dump, and the verification separate so each
can be re-run alone. State the verification numbers in the README with the
device they came from. Weights are not committed.

Do the model first. A demo can follow, and reusable inference code matters
more in it than a full UI.
